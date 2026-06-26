# RA-LAW: Reputation-Augmented Learnable Aggregation for Byzantine-Robust FL

Master's thesis codebase. Reproduces and audits **FedLAW** (Wang et al.,
ICLR 2026, "Byzantine-Robust Federated Learning with Learnable Aggregation
Weights", arXiv 2511.03529), and develops **RA-LAW** (Reputation-Augmented
LAW), this thesis' contribution, on top.

Builds on the **ByzFL** library (Gonzalez et al., arXiv 2505.24802) for
client/server primitives, attacks, and the non-IID data benchmark; adds a
paper-faithful FedLAW trainer and a battery of diagnostic experiments
documenting which paper numbers reproduce, which do not, and why.

---

## Status (2026-06-26)

| Component | State |
|---|---|
| FedLAW v2 trainer (`src/fedlaw_v2.py`) | **CANONICAL** — paper-faithful, in use |
| FedLAW v1 (`src/fedlaw.py`, `src/aggregators.py`, `src/loop.py`) | **DEPRECATED** — kept for historical comparison only; not used in current validation |
| All 6 paper attacks | implemented (`src/attacks.py`) |
| Cao q-partition + group-oriented Byzantine selection | implemented (`src/data_partition.py`) |
| RA-LAW | not started |

Validation against ICLR 2026 paper:

| Attack | n=20 detection | n=200 paper-scale |
|---|---|---|
| SignFlipping, InverseGradient, IPM | Byzantine zeroed round 1 ✓ | reproduced |
| flipping_label / backdoor / double, frac=0.1 | ✓ | reproduced (≈88% vs paper ~89–90%) |
| flipping_label, frac=0.4 | ✓ | **gap ~20pp** (~65% vs 87.45%) — diagnosed as detector failure due to mild co-alignment of multi-group label-flip pseudo-gradients (cos +0.16). Audit (`PAPER_FAITHFULNESS.md`) excludes architecture, gradient definition, cap, and selection scheme as causes. Open question: q-split/flipping-label interaction or genuine detector property at multi-group corruption. |
| LIE | evasion reproduced (Byzantine pinned at cap) | 33.3% (z=0.9346, Baruch) / 10.2% (τ=1.5, ByzFL default) vs paper 70.10% — **characterized partial reproduction** |

Full evidence chain: `results/paper_fixes/REPORT.md` and `VALIDATION.md`.
Code-to-paper mapping: `PAPER_FAITHFULNESS.md`.

---

## The three fixed implementation gaps

These are the gaps identified during validation that, once fixed, brought
the small-n behaviour into agreement with the paper. Each is referenced to
the paper section that pinned the correct interpretation.

### Gap 1 — pseudo-gradient definition (Algorithm 1)

**Symptom (v1).** Byzantine detection failed at α = 0.01 — the cross-product
signal was 200× too small relative to the loss term, and Byzantine weights
were never driven to zero.

**Cause.** v1 used the raw single-batch gradient `∇f(θ; batch)` as g_i.

**Fix (v2).** g_i = (θ − ψ_i) / α, where ψ_i is the local model after E full
SGD epochs (Algorithm 1, paper p. 60). Implemented in
`src/fedlaw_v2.py:266–298`.

**Verification.** With this fix, detection succeeds at α = 0.01 (paper's
value); Byzantine weights → 0 at round 1 for SignFlipping/IPM at n=20.

### Gap 2 — server-side ℓ2 clipping (Assumption E1)

**Symptom.** Theorem's bounded-gradient assumption was not enforced.

**Cause.** v1 did not clip submitted gradients.

**Fix (v2).** Clip every submitted gradient to C = max_{i ∈ honest} ‖g_i‖
per round. Implemented in `src/fedlaw_v2.py:71–86`, applied at L397 and L403.

**Verification.** Theorem's gradient-norm bound is now satisfiable. Clipping
does NOT change ALIE/LIE behaviour — they pass through unchanged in direction
(documented in `VALIDATION.md`).

### Gap 3 — cap arithmetic (Table 1)

**Symptom.** At s = n_honest with t = 1/s (exact-exclusion cap), honest
clients were forced to uniform 1/n_honest weights — no adaptive weighting
possible (contradicts paper Figure 1).

**Cause.** v1 misread Table 1 as t = 1/s. The actual paper specifies
t = 1/(s − 10).

**Fix (v2).** t = 1/(s − slack) with slack = min(10, s − 2) (the slack
guards small-n where s − 10 ≤ 0). Implemented in `src/fedlaw_v2.py:233–235`.

**Verification.** Paper's adaptive honest weighting is reproduced; matches
paper Figure 1 qualitative shape. Small-n configurations have a documented
slack-guard extension noted in `PAPER_FAITHFULNESS.md`.

---

## Project layout

```
src/
  fedlaw_v2.py       CANONICAL — paper-faithful FedLAW trainer
  run_fedlaw_v2.py   CANONICAL — CLI for v2 experiments
  attacks.py         all 6 paper attacks
  data_partition.py  Cao q-partition + group-oriented Byzantine selection
  projections.py     sparse-capped-simplex projection (Algorithm 3)
  models.py          mlp3_mnist (784→200→100→10) + ByzFL injection shim
  run_paper_fixes.py diagnostic runner that produced results/paper_fixes/
  run_validation_v2.py diagnostic runner that produced results/validation_v2/
  plot_fedlaw.py     plot generator for accuracy/weight figures

  # DEPRECATED / reference-only:
  fedlaw.py          v1 trainer
  run_fedlaw.py      v1 CLI
  aggregators.py     v1 aggregator helpers
  loop.py            v1 custom synchronous loop
  run_loop.py        v1 loop CLI

configs/
  fedlaw_v2_mnist.yaml      CANONICAL — paper-scale (n=200)
  fedlaw_v2_small.yaml      CANONICAL — small-n smoke test (n=20)

  # DEPRECATED / reference-only:
  fedlaw_mnist.yaml, fedlaw_signflipping.yaml, fedlaw_ipm.yaml
  loop_mnist.yaml, baseline_mnist.json, baseline_mnist_clean.json

tests/
  test_projections.py    17 tests for sparse-capped-simplex projection
  test_data_partition.py 7 tests for Cao partition + selection
  test_attacks.py        15 tests for all 6 attack classes
  test_fedlaw_v2.py      9 tests for clip + collect + run loop
  diagnose_fedlaw.py     4-part mechanism audit (generates v2 evidence)

scripts/
  diag_flip_n200.py      flipping_label n=200 detector diagnostic
                         (weight trajectory + round-5 grad stats at both fracs)

docs/
  Byzantine Robust Federated (1).pdf   ICLR 2026 FedLAW paper (Wang et al.)
  Delayed Momentum Aggregation.pdf     DeMoA paper (external reference)
  FedLAW_Handbook.pdf                  handbook generated during this project
                                       (notes & derivations, not external)
  superpowers/specs/2026-06-26-fedlaw-v2-design.md  v2 design spec
  superpowers/plans/2026-06-26-fedlaw-v2.md         v2 9-task implementation plan

results/
  paper_fixes/REPORT.md            master diagnostic log — all 4 Gaps,
                                   LIE Check 1+2, raw-grad falsification,
                                   flipping_label n=200 frac=0.4 diagnosis
  paper_fixes/plots/               9 PNGs — gap1/2/3 weight + cross-product evidence
  validation_v2/REPORT.md          v2 diagnostic — calibration, slack regimes
  validation_v2/plots/             11 PNGs — step1/2/3b/4 figures
  v2/{attack}/q*/frac*/seed0/      paper-scale n=200 runs (CANONICAL)
  v2_small/.../                    n=20 smoke-test outputs (CANONICAL)

  # archived (see archive/):
  baseline_mnist/                  pre-FedLAW ByzFL baseline
  baseline_mnist_clean/            pre-FedLAW clean baseline
  fedlaw_clean/, fedlaw_ipm/,
  fedlaw_signflipping/             v1 trainer outputs
  plots/                           v1 trainer plots

VALIDATION.md         top-level v3 reproduction report
PAPER_FAITHFULNESS.md code-to-paper mapping audit (read this first)
README.md             this file
```

---

## Setup

```bash
pyenv install 3.12.10        # if you don't have it
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Project pins `byzfl==0.0.11` and `torch==2.12.1`. Python 3.14 is currently
too new for the torch wheels we need.

---

## How to run experiments

### Small-n smoke test (n=20, ~30 rounds)

```bash
source .venv/bin/activate
python -m src.run_fedlaw_v2 --config configs/fedlaw_v2_small.yaml \
    --attack flipping_label --q 0.9 --frac-malicious 0.4 --seed 0
```

Substitute any of: `flipping_label backdoor inverse_gradient global_parameter
double lie lie_raw` for the `--attack` flag. Results land in
`results/v2_small/{attack}/q{q}/frac{frac}/seed{seed}/` as
`metrics.csv`, `weights.npy`, `config.yaml`.

### Paper-scale (n=200, 200 rounds, ~35 min CPU per run)

```bash
python -m src.run_fedlaw_v2 --config configs/fedlaw_v2_mnist.yaml \
    --attack flipping_label --q 0.9 --frac-malicious 0.4 --seed 0
```

Results land in `results/v2/{attack}/q{q}/frac{frac}/seed{seed}/`.

LIE-specific flag:

```bash
python -m src.run_fedlaw_v2 --config configs/fedlaw_v2_mnist.yaml \
    --attack lie --lie-tau 0.9346 ...
```

`--lie-tau` defaults to 1.5 (ByzFL default); set to 0.9346 to use the Baruch
stealth bound for n=200, f=80 (the theory-grounded value — see
`PAPER_FAITHFULNESS.md` §7e).

### Diagnostic — flipping_label n=200 detector audit

```bash
python -u scripts/diag_flip_n200.py 2>&1 | tee diag_output.txt
```

Prints per-round `sum_byz`, `max_byz`, `max_hon`, support sizes for the
first 22 rounds; at round 5 prints `||mean honest g||`, `||mean Byzantine g||`,
and `cos(byz, honest)`. Runs both frac=0.4 and frac=0.1 back-to-back
(~15 min total CPU). Produced the evidence in
`results/paper_fixes/REPORT.md` §"flipping_label n=200 frac=0.4 diagnosis".

---

## Two-round FedLAW structure

Each FedLAW round uses **two model broadcasts**, not one:

1. **Round A** — broadcast `θ_k`; each client returns pseudo-gradient
   `g_i = (θ_k − ψ_i)/α` and loss `f_i(θ_k)`. Stack into `G_k ∈ ℝ^{n×d}`,
   `f_k ∈ ℝ^n`.
2. Compute the test point `θ̃ = θ_k − α · G_k^T · w_k`.
3. **Round B** — broadcast `θ̃`; clients return `G̃` and `f̃` at θ̃.
4. **Weight update** — `h = w + α·β · G_k^T G̃ · w − β · f̃`; project onto the
   sparse capped simplex `Δ(s, t)`.
5. **Model update** — `θ_{k+1} = θ_k − α · G_k^T · w_{k+1}`.

The cross-product `G_k^T G̃` measures Byzantine alignment: anti-aligned
clients get negative scores → projection zeros them. The mechanism's
**failure mode** is co-aligned attacks (LIE/ALIE; multi-group label-flip at
high Byzantine fraction) — extensively documented in
`results/paper_fixes/REPORT.md`.

---

## Reproducibility

Global seed set for Python / NumPy / Torch; deterministic flags on; every
config logged into the results directory; per-round CSV evals flushed
incrementally so partial runs are usable. Commits are stepwise in `git log`.

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

48 unit tests covering projection, partition, attacks, and trainer. All
pass on the canonical code path.

---

## Credits & license

Builds on **ByzFL** (Gonzalez et al., arXiv:2505.24802), MIT license. We use
ByzFL's `Client`, `Server`, attack classes (`ALittleIsEnough` for LIE),
and benchmark scaffolding. The FedLAW algorithm is by Wang et al.
(ICLR 2026, arXiv:2511.03529). This project is MIT-licensed; see `LICENSE`.
