# FedLAW under Partial Participation — Reproduction & Coordinated Cohort Dormancy Attack

Master's thesis codebase. Two phases:

**Phase 1 — Reproduction.** Reproduces **FedLAW** (Wang et al., ICLR 2026,
"Byzantine-Robust Federated Learning with Learnable Aggregation Weights",
arXiv 2511.03529). Mechanism verified paper-faithful; three characterized
gaps between our runs and the paper's Table 3 numbers, each with a
diagnosed mechanism.

**Phase 2 — Contribution.** Extends FedLAW to partial participation.
Develops the *coordinated stealth cohort dormancy attack* — a novel
construction targeting caching-based Byzantine-resilient FL under
Bernoulli-p sampling. Empirically shows the attack defeats every standard
defender family tested: FedLAW's cross-product detector, DeMoA cached
momentum with TrMean / Median / CenteredClipping / Bulyan.

Builds on the **ByzFL** library (Gonzalez et al., arXiv 2505.24802) for
client/server primitives and robust aggregators; adds a paper-faithful
FedLAW trainer, a partial-participation harness with three admission
policies, five DeMoA-style baseline trainers, and a battery of diagnostic
experiments documenting exactly what reproduces, what doesn't, and why.

---

## Status (2026-07-04)

| Component | State |
|---|---|
| FedLAW v2 trainer (`src/fedlaw_v2.py`) | **CANONICAL** — paper-faithful + participation modes |
| Partial-participation harness (Bernoulli-p) | ✓ sanity gate 90.61% at p=1.0 |
| Design A (naive) / B(i) (weight cache) / B(ii) (gradient cache) | ✓ all three built + validated |
| Coordinated dormancy attack | ✓ 3 payloads (inverse_mean, stealth_lie, stealth_honest) |
| DeMoA baselines (`src/baselines.py`) | ✓ TrMean, Median, CenteredClipping, Bulyan |
| Krum baseline | ✗ unstable on q=0.9 (whipsaws on class-specific gradients) |
| Multi-seed statistical sweep | pending |
| Design C (proposed defense) | pending — motivated by cohort dormancy result |
| Dissertation writeup | pending |

**Living summary of the whole arc:** `WORK_SYNTHESIS.md` (single-document
map through both phases).

---

## Reproduction status vs paper

**Clean baseline at n=200 q=0.9:** 90.58% at 200 rounds (paper implies
~91–92% — within 1pp of the paper's implied clean number).

**Paper-scale runs at n=200 seed=0:**

| Attack | Config | Our acc | Paper Table 3 | Status |
|---|---|---|---|---|
| flipping_label | q=0.9, f=0.1 | 86.3% | ~89–90% | reproduced within band |
| flipping_label | q=0.9, f=0.4 | **65.4%** | 87.45% | **characterized gap** — multi-group cancellation |
| inverse_gradient | q=0.9, f=0.1 | **81.0%** | ~89.5% | **characterized gap** — detection decay |
| inverse_gradient | q=0.6, f=0.4 | 92.2% | 91.62% | reproduced (+0.6pp) |
| backdoor | q=0.9, f=0.1 | 87.2% | ~89.5% | reproduced within band |
| double | q=0.9, f=0.1 | 87.7% | ~89.5% | reproduced within band |
| LIE (τ=1.5 default) | q=0.9, f=0.4 | 10.2% | 70.10% | **characterized gap** — τ unresolvable |
| LIE (z=0.9346 Baruch) | q=0.9, f=0.4 | 33.3% | 70.10% | **characterized gap** |

Full audit against paper: `PAPER_FAITHFULNESS.md`.
Detailed diagnostics for the three gaps: `results/paper_fixes/REPORT.md`.
Framing (not "paper wrong", not "our code broken"): `REPRODUCTION_STATUS.md`.

---

## Cross-defender dormancy result (Phase 2 contribution)

Coordinated stealth cohort dormancy attack (cohort=80 = 4 groups, T_dark=20,
stealth_lie payload) at n=200 q=0.9 p=0.5 seed=0 200 rounds:

| Defender | Clean | Under attack | Δ |
|---|---|---|---|
| DeMoA + CCLIP (τ ∈ {20,50,100}) | 90.79–90.81% | 81.71–81.96% | −8.85 to −9.08pp |
| FedLAW cache_grad_B_ii | 90.57% | 80.60% | **−9.97pp** |
| DeMoA + TrMean(f=20) | 89.72% | 78.14% | −11.58pp |
| DeMoA + TrMean(f=80, matched) | 89.03% | 63.97% | −25.06pp |
| DeMoA + Median | 88.98% | 60.03% | −28.95pp |
| **DeMoA + Bulyan(f=49)** | **88.83%** | **8.24%** | **−80.59pp** |
| DeMoA + Krum | (unstable on q=0.9 — not viable) | | |

**Every standard aggregator falls.** Bulyan collapses catastrophically to
chance because its iterative-Krum selection actively prefers the identical
dormant cluster (79 zero-distance mutual neighbours dominate their Krum
score), then its coordinate-wise trim removes the honest spread — the
protective mechanism inverts.

---

## The three fixed implementation gaps (Phase 1)

Identified during Phase 1 reproduction. Each references the paper section
that pinned the correct interpretation.

### Gap 1 — pseudo-gradient definition (Algorithm 1)

**Symptom (v1).** Byzantine detection failed at α = 0.01 — cross-product
signal was 200× too small vs the loss term; Byzantine weights never
zeroed.

**Cause.** v1 used raw single-batch gradient `∇f(θ; batch)`.

**Fix (v2).** g_i = (θ − ψ_i) / α from E local SGD epochs (Algorithm 1,
paper p. 60). `src/fedlaw_v2.py:266–298`.

### Gap 2 — server-side ℓ2 clipping (Assumption E1)

**Cause.** v1 did not clip; paper's theorem requires bounded gradients.

**Fix (v2).** Clip to C = max_{i ∈ honest} ‖g_i‖ per round.
`src/fedlaw_v2.py:71–86`. Doesn't change ALIE/LIE (magnitude vs direction).

### Gap 3 — cap arithmetic (Table 1)

**Cause.** v1 misread Table 1 as t = 1/s (exact-exclusion); paper is
t = 1/(s − 10) (adaptive weighting).

**Fix (v2).** t = 1/(s − slack) with slack = min(10, s − 2).
`src/fedlaw_v2.py:233–235`. Reproduces paper Figure 1 adaptive honest
weighting.

---

## Project layout

```
src/
  fedlaw_v2.py        CANONICAL — paper-faithful FedLAW trainer + partial-participation
                      modes (naive_A, cache_weight_B_i, cache_grad_B_ii) + dormancy attack
  run_fedlaw_v2.py    CANONICAL — CLI
  baselines.py        DeMoA-cache trainer + Krum/TrMean/Median/CCLIP/Bulyan aggregators
  attacks.py          all 6 paper attacks (flipping_label, backdoor, inverse_gradient,
                      global_parameter, double, LIE + LIE-raw diagnostic variant)
  data_partition.py   Cao q-partition + group-oriented Byzantine selection
  projections.py      sparse-capped-simplex projection (paper Algorithm 3)
  models.py           mlp3_mnist (784→200→100→10)

  # DEPRECATED / reference-only (retained for historical comparison):
  fedlaw.py, run_fedlaw.py, aggregators.py, loop.py, run_loop.py

configs/
  fedlaw_v2_mnist.yaml   CANONICAL — paper-scale (n=200)
  fedlaw_v2_small.yaml   CANONICAL — small-n smoke test (n=20)
  # v1 configs (DEPRECATED): fedlaw_*.yaml, loop_mnist.yaml, baseline_mnist*.json

tests/
  test_projections.py    17 tests — sparse-capped-simplex projection
  test_data_partition.py 7 tests — Cao partition + selection
  test_attacks.py        15 tests — 6 attack classes
  test_fedlaw_v2.py      9 tests — clip + collect + run loop
  # 48 total, all pass

scripts/
  # Reproduction diagnostics (Phase 1)
  diag_flip_n200.py       flipping_label f=0.4 co-alignment root cause
  diag_flip_seedsweep.py  5-seed sweep — confirms cohort dormancy finding
  diag_flip_coalign.py    per-group cancellation breakdown
  diag_wfreeze.py         inverse_gradient f=0.1 detection-decay verdict
  diag_clean_baseline.py  200-round clean baseline (undertraining excluded)

  # Reproduction batches (Phase 1)
  run_batch_sequential.sh, overnight_queue.sh
  step2_naive_A.py, step2_horizon.py

  # Contribution batches (Phase 2)
  sanity_p10.py, smoke_p05.py, smoke_both_modes.py       — Step 1 sampling harness
  dormancy_smoke.py, stealth_smoke.py                    — dormancy attack smoke
  dormancy_batch.sh, dormancy_cohort_batch.sh            — dormancy full runs
  demoa_protocol_runs.sh, demoa_plot.py                  — DeMoA-protocol comparison
  demoa_ftuned_batch.sh, demoa_median_batch.sh           — DeMoA baseline variants
  baseline_batch.sh, cclip_batch.sh, bulyan_batch.sh     — cross-aggregator sweep

docs/
  PARTIAL_PARTICIPATION_DESIGN.md   §1 participation model, §2 A→B→C design ladder,
                                    §3 dormancy attack, §4 evaluation plan
  Byzantine Robust Federated (1).pdf   ICLR 2026 FedLAW paper (Wang et al.)
  Delayed Momentum Aggregation.pdf     DeMoA baseline reference (Karimireddy et al.)
  FedLAW_Handbook.pdf                  handbook generated during this project
  superpowers/specs/2026-06-26-fedlaw-v2-design.md    v2 design spec
  superpowers/plans/2026-06-26-fedlaw-v2.md           v2 9-task implementation plan

results/
  paper_fixes/REPORT.md            master diagnostic log — all 4 Gaps, LIE Check 1+2,
                                   raw-grad falsification, flipping_label diagnosis,
                                   inverse_gradient detection-decay verdict, clean baseline
  paper_fixes/plots/               9 PNGs — gap1/2/3 weight + cross-product evidence
  validation_v2/REPORT.md          v2 diagnostic — calibration, slack regimes
  validation_v2/plots/             11 PNGs — step1/2/3b/4 figures
  v2/{attack}/q*/frac*/seed0/      paper-scale n=200 runs
  v2_small/                        n=20 smoke-test outputs
  v2/dormancy/                     single-client dormancy (evades re-scoring, no damage)
  v2/dormancy_cohort/              coordinated cohort_20/80 (LANDS at −9.97pp)
  v2/baselines/                    DeMoA + TrMean/Median/CCLIP/Bulyan cross-defender runs
  v2/demoa/                        DeMoA-protocol A vs B(ii) comparison plots
  v2/demoa_option_i_corrected/     Option (i) diagnostic (under-trains, why)
  # archived under archive/:
  baseline_mnist/, baseline_mnist_clean/    pre-FedLAW ByzFL baselines
  fedlaw_clean/, fedlaw_ipm/,
  fedlaw_signflipping/, plots/              v1 trainer outputs

# Top-level docs (read in this order for the whole arc):
README.md                  this file — status snapshot + how to run
WORK_SYNTHESIS.md          full-arc synthesis (Phase 1 + Phase 2 in one document)
REPRODUCTION_STATUS.md     verified-faithful components + 3 characterized gaps + open items
PAPER_FAITHFULNESS.md      code-to-paper audit per numbered component
VALIDATION.md              v3 reproduction report
```

---

## Setup

```bash
pyenv install 3.12.10        # if you don't have it
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Pins `byzfl==0.0.11` and `torch==2.12.1`. Python 3.14 lacks compatible torch
wheels.

---

## How to run experiments

### Reproduction (Phase 1)

**Small-n smoke (n=20, ~30 rounds):**

```bash
source .venv/bin/activate
python -m src.run_fedlaw_v2 --config configs/fedlaw_v2_small.yaml \
    --attack flipping_label --q 0.9 --frac-malicious 0.4 --seed 0
```

Attacks: `flipping_label backdoor inverse_gradient global_parameter double lie lie_raw`.
Results under `results/v2_small/{attack}/q{q}/frac{frac}/seed{seed}/`.

**Paper-scale (n=200, 200 rounds, ~35 min CPU per run):**

```bash
python -m src.run_fedlaw_v2 --config configs/fedlaw_v2_mnist.yaml \
    --attack flipping_label --q 0.9 --frac-malicious 0.4 --seed 0
```

LIE-specific: `--lie-tau 0.9346` uses Baruch's stealth bound for n=200 f=80
(vs the ByzFL default of 1.5). See `PAPER_FAITHFULNESS.md` §7e.

### Partial participation (Phase 2)

**Sanity gate — p=1.0 must reproduce clean baseline (~90.6%):**

```bash
python scripts/sanity_p10.py
```

**Under participation p=0.5, choose admission mode:**

```bash
python -m src.run_fedlaw_v2 --config configs/fedlaw_v2_mnist.yaml \
    --attack backdoor --q 0.9 --frac-malicious 0.4 --seed 0 \
    --p 0.5 --participation-mode cache_grad_B_ii
```

Modes:
- `naive_A` — no persistence (design ladder control)
- `cache_weight_B_i` — persist weight, absent g_i=0 (broken under §2.2 projection)
- `cache_grad_B_ii` — cache gradient + weight with DeMoA-style decay (canonical §2.2 B)

### Coordinated cohort dormancy attack

```bash
python -m src.run_fedlaw_v2 --config configs/fedlaw_v2_mnist.yaml \
    --attack flipping_label --q 0.9 --frac-malicious 0.0 --seed 0 \
    --p 0.5 --participation-mode cache_grad_B_ii \
    --dormancy-T-dark 20 \
    --dormancy-client-indices 40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199 \
    --dormancy-payload stealth_lie
```

The 80-index list is the cohort matching the reproduction's group-oriented
Byzantine assignment (groups {2,4,8,9} at seed=0). Payloads: `stealth_lie`
(LIE-style μ+τσ, evades re-scoring), `stealth_honest` (leaves honest gradient
cached), `inverse_mean` (anti-aligned control, trivially caught).

### DeMoA baselines

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.baselines import BaselineConfig, BaselineTrainer
cfg = BaselineConfig(n_clients=200, q=0.9, T=200, p=0.5,
                     aggregator='cclip', cclip_tau=100.0,
                     use_demoa_cache=True, seed=0,
                     results_dir='results/v2/baselines/my_run')
BaselineTrainer(cfg).run()
"
```

Aggregators: `krum` (unstable on q=0.9), `trmean` (needs `aggregator_f`),
`median`, `cclip` (needs `cclip_tau`), `bulyan` (needs `aggregator_f ≤ 49`
at n=200).

### Diagnostic — flipping_label f=0.4 co-alignment (paper-gap root cause)

```bash
python -u scripts/diag_flip_n200.py 2>&1 | tee diag_output.txt
```

Prints per-round sum_byz / max_byz / max_hon, support sizes, and (at round
5) `‖mean honest g‖`, `‖mean Byzantine g‖`, `cos(byz, honest)`. Runs both
frac=0.4 and frac=0.1 back-to-back (~15 min CPU). Produced
`results/paper_fixes/REPORT.md` §"flipping_label n=200 frac=0.4 diagnosis".

---

## Two-round FedLAW structure

Each FedLAW round uses **two model broadcasts**, not one:

1. **Round A** — broadcast `θ_k`; each client returns pseudo-gradient
   `g_i = (θ_k − ψ_i)/α` and loss `f_i(θ_k)`. Stack into `G_k ∈ ℝ^{n×d}`,
   `f_k ∈ ℝ^n`.
2. Compute the test point `θ̃ = θ_k − α · G_k^T · w_k`.
3. **Round B** — broadcast `θ̃`; clients return `G̃` and `f̃` at θ̃.
4. **Weight update** — `h = w + α·β · G_k^T G̃ · w − β · f̃`; project onto
   the sparse capped simplex `Δ(s, t)`.
5. **Model update** — `θ_{k+1} = θ_k − α · G_k^T · w_{k+1}`.

The cross-product `G_k^T G̃` measures Byzantine alignment: anti-aligned
clients get negative scores → projection zeros them. Documented failure
modes (see `results/paper_fixes/REPORT.md`):

- Co-aligned attacks (LIE/ALIE) — evade cross-product detection.
- Multi-group label-flipping at high Byzantine fraction — group-level
  poisons partially cancel (67% at f=0.4), leaving a small co-aligned
  residual that also evades detection.
- Coordinated cohort dormancy under partial participation — 80 identical
  cached poisons create a positive-feedback loop in the detector via
  self-similar cross-product terms.

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

48 tests. All pass.

---

## Reproducibility

Global seed set for Python / NumPy / Torch; deterministic flags on; every
config copied into its results directory; per-round CSV evals flushed
incrementally so partial runs are usable. Commits are stepwise in `git log`.
p=1.0 fast path is byte-equivalent to the pre-participation trainer
(sampling RNG is not consumed).

---

## Credits & license

Builds on **ByzFL** (Gonzalez et al., arXiv:2505.24802), MIT license. We
use ByzFL's `Client`, `Server`, robust aggregators (`Krum`, `TrMean`,
`Median`, `CenteredClipping`), and attack classes (`ALittleIsEnough` for
LIE). The FedLAW algorithm is by Wang et al. (ICLR 2026, arXiv:2511.03529).
DeMoA is by Karimireddy et al. (see `docs/Delayed Momentum Aggregation.pdf`);
this repo re-implements its cache + decay per §3.1/§A.1 for the baseline
comparison. Bulyan is by Mhamdi et al. (2018), implemented locally as
`src/baselines.py:Bulyan` since ByzFL doesn't provide it. This project
is MIT-licensed; see `LICENSE`.
