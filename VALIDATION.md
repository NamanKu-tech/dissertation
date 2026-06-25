# FedLAW Validation Report

Model: `mlp3_mnist` (784→200→100→10)  |  Seed: 0  |  Rounds: 100

## Validated configuration (v2)

| Config | Clients | Byzantine | Data distribution | α (model lr) | β (weight lr) | s / t |
|---|---|---|---|---|---|---|
| SignFlipping | 18 honest + 2 byz | 2 (10%) | Dirichlet α=0.5 | 0.5 | 0.001 | 18 / 1/18 |
| IPM (τ=2) | 18 honest + 2 byz | 2 (10%) | Dirichlet α=0.5 | 0.5 | 0.001 | 18 / 1/18 |
| ALIE (τ=1.5) | 18 honest + 2 byz | 2 (10%) | Dirichlet α=0.5 | 0.5 | 0.001 | 18 / 1/18 |

*(v1 used Dirichlet α=5.0; that was over-conservative — see Discrepancy 1 below.)*

## Results (v2, multi-seed at Dirichlet α=0.5)

| Attack | Seeds | Final accuracy (mean±std) | Byz weights (all seeds) | False excl. |
|---|---|---|---|---|
| SignFlipping | {0,1,2} | **94.2 ± 0.3%** | **0.000 from round 1** | 0 |
| IPM (τ=2) | {0,1,2} | **94.2 ± 0.3%** | **0.000 from round 1** | 0 |
| ALIE (τ=1.5) | {0,1,2} | 91.6 ± 0.4% | **never excluded** | **100/100 rounds** |

## Reproduced behaviours

**Weight collapse:** Byzantine client weights drop to exactly 0.000 at the first weight
update (round 1) and stay at zero for all 100 rounds across all three seeds. This matches
the paper's qualitative claim (Figure 1). Collapse is immediate because α=0.5 gives
a strong gradient alignment signal at the scale of raw mini-batch gradients.

**Accuracy under attack (SignFlipping, IPM):** Both attacks reach ≈94% at round 100,
equalling or exceeding the clean run — once Byzantine clients are excluded, training
proceeds as if no attack were present. This is the core FedLAW claim.

**No false exclusions (SignFlipping, IPM):** Zero honest clients are ever zeroed across
all 300 rounds (3 seeds × 100 rounds). Cross-term separation is clean and persistent.

**Exact-exclusion forces uniform weights:** With s=18, t=1/18 (i.e., s·t=1 exactly),
the feasible set of the projection is a single point: all 18 active clients at weight
1/18. Honest clients carry identical weights throughout — there is no adaptive
within-honest weighting in this regime.

## Discrepancies from paper

### 1. Learning-rate scale: α=0.5 vs paper's α=0.01 — this is the only calibration issue

The paper states α=0.01. With raw mini-batch gradients (||g|| ≈ 1.4 for `mlp3_mnist`,
batch_size=64), the cross term in the weight update is:

    α·β·|E[cross_w]| ≈ α × 0.001 × 0.187 = 0.000187·α

The loss term is:
    β·|f̃| ≈ 0.001 × 2.3 ≈ 0.0023

At α=0.01: cross_term ≈ 1.9e-6, loss_term ≈ 2.3e-3 — the loss term is **1200×
larger**. The weight update is driven entirely by per-client losses. Since Byzantine
clients are imputed with mean(honest losses), the loss term cannot differentiate them.

The balance condition is α > σ(f̃_honest) / |E[cross_w]| ≈ 0.065. We use α=0.5
(7× headroom). Batch averaging does not rescue α=0.01 because the issue is α magnitude,
not gradient noise: confirmed empirically at 16 batches (Part 4 diagnostic).

α=1.0 upper bound: model diverges to NaN within 1–2 rounds (step ≈ α·||g||≈1.4
in parameter space is too large for the MLP).

**Validated minimum viable band: 0.3 ≤ α < 1.0.**

### 2. Data heterogeneity: α=5.0 (v1) was over-conservative — α=0.5 works

The original v1 validation ran at Dirichlet α=5.0 (near-IID). This was due to a
confounding: the failure observed at Dirichlet α=0.5 was actually at α_lr=0.01
(undetectable by the weight update for the reason above), not α_lr=0.5.

The v2 validation (Discrepancy 1 resolved, α_lr=0.5) confirms:
- **Dirichlet α=0.5**: PASS — clean detection, 94.2% accuracy
- **Dirichlet α=0.3**: PASS — clean detection (30-round check)
- **Dirichlet α=0.1**: FAIL — false exclusions every round, Byzantine never zeroed

The practical failure boundary is **Dirichlet α ≈ 0.1–0.2** for the current config.
This matters for partial participation: if participating clients are drawn from extreme
data distributions, the effective heterogeneity of the selected subset can be lower
than the population Dirichlet parameter.

### 3. s,t regime: exact-exclusion is required, slack actively harms robustness

Testing s=20 with t=1/16 or t=1/10 (allowing all 20 clients to survive with a looser
cap) produces an unexpected reversal:

| Regime | Byzantine suppression | Hon weight std | Final acc |
|---|---|---|---|
| Exact (s=18, t=1/18) | 0.000 from round 1 ✓ | 0.000 (uniform) | 94.2% |
| Slack-2 (s=20, t=1/16) | **hits cap 0.0625 by round 20** ✗ | 0.011 | 93.9% |
| Slack-4 (s=20, t=1/10) | **hits cap 0.1 by round 50** ✗ | 0.013 | 93.7% |

In slack regimes, Byzantine clients decrease slightly in rounds 1–5 (cross_w is
negative), but as honest weights differentiate, the correlation of Byzantine gradients
with the evolving weighted consensus flips. By round 10–20, Byzantine weights reverse
and climb to the cap. This is a **feedback loop**: Byzantine gradients corrupt the
model direction, which reorders h values, which increases Byzantine weight, which
increases corruption.

Conclusion: the paper's exact-exclusion constraint s=n−f, t=1/s is not an arbitrary
choice — it is necessary. The sparsity constraint must force Byzantine clients out
before they can corrupt the model direction.

### 4. ALIE defeats the mechanism entirely — structural limitation

A Little Is Enough (ALIE, τ=1.5) submits `g_byz = mean(honest) + τ·σ(honest)·dir`.
The mean(honest) component makes Byzantine gradients **co-aligned** with consensus:

| Attack | cos(g_byz, mean_honest) | cross_w[byz] | h[byz] vs honest |
|---|---|---|---|
| SignFlipping | −1.000 | −0.171 | h[byz] < all honest → zeroed ✓ |
| ALIE τ=1.5 | +0.222 | **+0.347** | h[byz] > all 18 honest → all honest below |
| ALIE τ=0.5 | +0.594 | +0.164 | h[byz] > 5 honest clients |

With exact-exclusion (s=18), the bottom 2 h values are always zeroed. Under ALIE,
these are always honest clients, not Byzantine. False exclusions occur every round;
Byzantine clients are never excluded. Accuracy drops by ≈2.6pp (91.6% vs 94.2%).

This is not a calibration issue. FedLAW's theorem requires anti-aligned gradients;
ALIE violates this assumption by construction. No tuning of α, s, or t can fix it.

**Implication for RA-LAW:** A time-averaged reputation based on cross_w history will
also give Byzantine clients higher accumulated score under ALIE (cross_w[byz]=+0.347
consistently exceeds the honest mean ≈ 0.204). The reputation signal must be derived
from a different source than instantaneous gradient alignment to be robust to ALIE.

### 5. Convergence noise (unchanged from v1)

With α=0.5, the per-round step is large (≈0.1–0.3 in parameter space), producing
visible noise in the accuracy curve. The paper's smoother curves reflect smaller
effective step size. This is a presentation difference, not an algorithm issue.

## s,t regime recommendation

Use **exact-exclusion (s=n−f, t=1/s)** for all experiments. For partial participation
with participation rate ρ: set s=⌊ρ·n_honest⌋ each round (the number of actually
participating honest clients), t=1/s. This maintains the exact-exclusion property and
the SNR analysis carries over.

Step 3a (subsample diagnostic) confirms cross-term separation holds down to **k=3
honest clients** at Dirichlet α=0.5 with d=178,110 MNIST parameters. Below k=3
the Byzantine signal floor has not been tested.

## Conclusion

FedLAW's core mechanism is correctly implemented:
- Algorithm 2 (two-round gradient collection, weight update, sparse capped simplex
  projection) executes correctly — confirmed by 17/17 projection unit tests and
  multi-seed full-length validation.
- Byzantine detection (SignFlipping, IPM) works at the paper's target regime
  (Dirichlet α=0.5) with α_lr=0.5, verified across seeds {0,1,2}, all 100 rounds.

Two confirmed limitations relative to the paper:
1. α=0.01 is too small by a factor ≥6.5× for raw mini-batch gradients. Minimum
   viable α ≈ 0.3–0.5.
2. ALIE defeats the gradient-alignment detection mechanism entirely. This is a
   known structural gap in FedLAW (and any cross-product-based aggregation rule)
   that RA-LAW's design must address through a different signal.

## Plots (v2)

- `results/validation_v2/plots/step1_seed{0,1,2}_weights.png` — weight trajectories
- `results/validation_v2/plots/step2_*_weights.png` — regime comparison
- `results/validation_v2/plots/step2_honest_weight_dist.png` — weight distributions
- `results/validation_v2/plots/step3b_dirichlet_sweep.png` — failure boundary map
- `results/validation_v2/plots/step4_*_weights.png` — attack comparisons
- `results/validation_v2/REPORT.md` — full running diagnostic report
