# RA-LAW: Reputation-Augmented Learnable Aggregation for Byzantine-Robust FL

Master's thesis codebase. Builds on the **ByzFL** library (EPFL/INRIA) for
robust aggregators, attacks, non-IID partitioning, and an FL benchmark; adds
two custom aggregators on top:

- **FedLAW** — Learnable Aggregation Weights (Wang et al., arXiv 2511.03529)
- **RA-LAW** — Reputation-Augmented LAW (this thesis' contribution)

Both are **stubbed** in this commit; the next milestone implements them.

## Setup

```bash
pyenv install 3.12.10        # if you don't have it
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

(Project pins `byzfl==0.0.11`, `torch==2.12.1`. Python 3.14 is currently too
new for the `torch` wheels we need.)

## Layout

```
configs/
  baseline_mnist.json     # ByzFL benchmark config (Week-1 smoke run)
  loop_mnist.yaml         # custom loop config (Step 2, next)
src/
  loop.py                 # custom synchronous FL round loop (Step 2)
  aggregators.py          # Aggregator interface + FedAvg + FedLAW/RALAW stubs
  run_loop.py             # entrypoint for the custom loop
```

## Week-1 baseline (ByzFL benchmark)

```bash
cp configs/baseline_mnist.json config.json   # benchmark reads ./config.json
.venv/bin/python -c "from byzfl import run_benchmark; run_benchmark(nb_jobs=1)"
```

Config: MNIST, 20 honest + 2 Byzantine (≈10%), Dirichlet α=0.5, FedAvg
(`proportion_selected_clients=1.0`, `local_steps_per_client=5`), aggregators
`{TrMean, Krum}` (both with `f=2`), attack `SignFlipping`, `nb_steps=100`,
`size_train_set=1.0` (fixed hyperparameters — no inner sweep).

Expectation: SignFlipping is a weak attack; TrMean and Krum should land
accuracy-under-attack close to clean (high-90s on MNIST). That confirms the
stack is wired correctly — not a dramatic robustness gap. The interesting
gaps come later with IPM / Sybil.

## Step 2 — custom loop (next commit)

Reason: FedLAW/RA-LAW need per-client loss, stable client IDs, and
persistent aggregator state across rounds. ByzFL's plain `__call__(vectors)`
aggregator slot can't carry any of that, so we run our methods through a
small custom loop that **reuses ByzFL's `DataDistributor`, `Client`,
`ByzantineClient`, and attack classes** verbatim.

`src/aggregators.py` defines:

```python
class Aggregator:
    def aggregate(self, updates, losses, client_ids, global_state) -> np.ndarray: ...

class FedAvg(Aggregator):  # implemented; sample/loss-uniform weighting
class FedLAW(Aggregator):  # STUB — alternating-min weight step from 2511.03529
class RALAW(FedLAW):       # STUB — decayed per-client reputation memory
```

## Reproducibility

Global seed set for Python / NumPy / Torch; deterministic flags on; every
config logged into the results directory; commits in logical steps.

## API surprises caught during Step 0

(These contradict the original brief — the installed package wins.)

- `IPM` is exported as `InnerProductManipulation`, not `IPM`.
- Standalone aggregators are `__call__(vectors)` only — they cannot take
  losses or client IDs. (Confirms the decision to bypass that slot for
  FedLAW/RA-LAW.)
- Attack classes' `__call__` returns a **single** vector; the list-of-`f`
  faulty vectors API lives on `ByzantineClient.apply_attack`.
- Benchmark config's `size_train_set` defaults to `0.8`, which silently
  triggers a per-config hyperparameter sweep. We set `1.0` (fixed HPs)
  for both the smoke run and the main experiments — cleaner, faster, and
  removes a confound for a 6-week timeline.
- ByzFL's `LabelFlipping` is incompatible with the `FedAvg` training
  algorithm in the benchmark (raises). Doesn't bite us yet.
- `f` is a list in the benchmark config (sweep dimension). Single int
  still works because it expands to one combination.
- ByzFL benchmark auto-sets `aggregator.parameters.f = tolerated_f`; you
  only need to set `tolerated_f` (we do both for clarity).

## Credits & license

This work builds on **ByzFL** (Gonzalez et al., *ByzFL: A Library for
Robust and Byzantine-Resilient Federated Learning*, arXiv:2505.24802),
MIT license. We use ByzFL's `DataDistributor`, `Client`, `ByzantineClient`,
`RobustAggregator`, attack classes, and benchmark entrypoint as published.
This project is also MIT-licensed; see `LICENSE`.
