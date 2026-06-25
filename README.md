# RA-LAW: Reputation-Augmented Learnable Aggregation for Byzantine-Robust FL

Master's thesis codebase. Builds on the **ByzFL** library (EPFL/INRIA) for
robust aggregators, attacks, non-IID partitioning, and an FL benchmark; adds
two custom aggregators on top:

- **FedLAW** — Learnable Aggregation Weights (Wang et al., arXiv 2511.03529) — **implemented and validated**
- **RA-LAW** — Reputation-Augmented LAW (this thesis' contribution) — *upcoming*

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
  baseline_mnist.json         # ByzFL benchmark config (smoke run)
  baseline_mnist_clean.json   # clean baseline (no attack)
  loop_mnist.yaml             # custom loop — FedAvg clean consistency check
  fedlaw_mnist.yaml           # FedLAW — no attack
  fedlaw_signflipping.yaml    # FedLAW — SignFlipping attack
  fedlaw_ipm.yaml             # FedLAW — InnerProductManipulation attack
src/
  models.py       # mlp3_mnist (784→200→100→10) + ByzFL injection shim
  projections.py  # sparse unit-capped simplex projection (FedLAW Alg. 3)
  aggregators.py  # Aggregator interface + FedAvg + ByzFLPureAggregator
  loop.py         # custom synchronous FL round loop
  fedlaw.py       # FedLAW two-round training loop (Algorithm 2)
  run_loop.py     # entrypoint: python -m src.run_loop --config <yaml>
  run_fedlaw.py   # entrypoint: python -m src.run_fedlaw --config <yaml>
tests/
  test_projections.py   # 17 unit tests — all passing
docs/
  FL_Thesis_Plan_RA-LAW.pdf
results/
  fedlaw_clean/           # metrics.csv, weights.npy
  fedlaw_signflipping/
  fedlaw_ipm/
  plots/                  # fedlaw_accuracy.png, fedlaw_SignFlipping.png, …
VALIDATION.md             # reproduced behaviours + discrepancies from paper
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

## Custom FL loop (Step 2)

```bash
.venv/bin/python -m src.run_loop --config configs/loop_mnist.yaml
```

Reason: FedLAW/RA-LAW need per-client losses, stable client IDs, and
persistent aggregator state across rounds — none of which ByzFL's plain
`__call__(vectors)` aggregator slot supports. The custom loop **reuses ByzFL's
`DataDistributor`, `Client`, `ByzantineClient`, and attack classes** verbatim.

## FedLAW

```bash
.venv/bin/python -m src.run_fedlaw --config configs/fedlaw_mnist.yaml
.venv/bin/python -m src.run_fedlaw --config configs/fedlaw_signflipping.yaml
.venv/bin/python -m src.run_fedlaw --config configs/fedlaw_ipm.yaml

.venv/bin/python -m src.plot_fedlaw   # writes results/plots/
```

### Two-round communication structure

Each FedLAW round uses **two model broadcasts**, not one:

1. **Round A** — broadcast `θ_k`; each client returns gradient `g_i(θ_k)` and
   loss `f_i(θ_k)`. Stack into matrix `G_k` (shape `n × d`) and vector `f_k`.
2. Compute the **test point** `θ̃ = θ_k − α · G_k · w_k`.
3. **Round B** — broadcast `θ̃`; clients return `G̃` and `f̃` at the test point.
4. **Weight update** — `h = w + α·β · G_kᵀ G̃ w − β · f̃`; then project `h`
   onto the sparse capped simplex `Δ(s, t)`.
5. **Model update** — `θ_{k+1} = θ_k − α · G_k · w_{k+1}`.

The cross term `G_kᵀ G̃ w` measures gradient alignment: Byzantine gradients
are anti-aligned with the consensus direction → negative cross-product → weight
decrease → projection to zero. Honest gradients stay aligned → weight survives.

### Loss-at-current-model

`Client.compute_gradients()` is called **before** `compute_model_update()`.
This captures `f_i(θ_k)` — the loss at the global model, not after a local SGD
step. The paper's weight update requires this; calling in the other order gives
`f_i(θ_k − α·g_i)` (post-step loss), which does not correspond to Algorithm 2.

Byzantine clients have no ground-truth loss; the server imputes `mean(f_honest)`
for them, which keeps the loss term from artificially highlighting any honest
client as an outlier.

### Projection function

`project_sparse_capped_simplex(h, s, t)` (`src/projections.py`) projects onto

```text
Δ(s, t) = { w ≥ 0, Σwᵢ = 1, wᵢ ≤ t, ‖w‖₀ ≤ s }
```

Implementation: (1) select the `s` largest components of `h`; (2) project the
`s`-dimensional sub-vector onto the capped simplex via bisection on the
Lagrange multiplier `λ` (`w_i = clip(h_i − λ, 0, t)`, solve `Σw_i = 1`);
(3) embed back. Feasibility requires `s · t ≥ 1`. For `t=0` the cap is
auto-set to `1/s` (uniform bound). All 17 unit tests pass.

See `VALIDATION.md` for reproduced behaviours and hyperparameter discrepancies
relative to the paper.

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
