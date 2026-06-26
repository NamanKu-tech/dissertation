# FedLAW Paper Fixes — Validation Report

Model: mlp3_mnist (784→200→100→10)
Three implementation gaps found by comparing code against ICLR 2026 paper.
This report records expected vs. observed behaviour for each fix.

## VERDICT (filled in after all steps complete)

| Gap | Fix | Predicted behaviour | Observed |
|-----|-----|---------------------|----------|
| 1 — Gradient def. | pseudo-grad (θ−ψ_i)/α, E=3 epochs | α=0.01 works, byz zeroed round 1 | TBD |
| 2 — Clipping | C=max honest norm per round | ALIE damage contained, no inversion | TBD |
| 3 — Cap t | t=1/(s−10)=1/8 for s=18 | non-uniform honest weights, byz still excluded | TBD |

*(Table updated after experiments run — search for VERDICT in each section.)*

---

────────────────────────────────────────────────────────────────────────
## All fixes applied — paper-comparable validation

All three gaps fixed: pseudo-gradients (E=3), clipping (C=max honest norm),
cap t=1/(s−10). Config: n=18+2, Dirichlet α=0.5, 30 rounds, 3 seeds.
Attacks: SignFlipping, InnerProductManipulation (τ=2).

Note: paper Table 3 reports results at q (Dirichlet) ∈ {0.5, 0.6, 0.9}.
We test at q=0.5 (our standard) as a reference point.
Full 100-round 3-seed paper-identical runs require ~2-4h on CPU.

### Attack: SignFlipping
  seed=0: acc=94.16%  byz_zeroed=1  false_excl=24
  seed=1: acc=94.31%  byz_zeroed=1  false_excl=28
  seed=2: acc=94.16%  byz_zeroed=1  false_excl=27
  Mean±std: 94.21 ± 0.07%
  Byzantine zeroed ≤ round 2 (all seeds): True
  Zero false exclusions (all seeds):      False

### Attack: IPM (τ=2)
  seed=0: acc=94.14%  byz_zeroed=1  false_excl=23
  seed=1: acc=94.33%  byz_zeroed=1  false_excl=28
  seed=2: acc=94.16%  byz_zeroed=1  false_excl=25
  Mean±std: 94.21 ± 0.09%
  Byzantine zeroed ≤ round 2 (all seeds): True
  Zero false exclusions (all seeds):      False


────────────────────────────────────────────────────────────────────────
## Paper-scale LIE validation (2026-06-26)

### Setup

Config: n=200, frac_malicious=0.4 (80 Byzantine / 120 honest), q=0.9 (Cao et al.),
E=3 local epochs, α=0.01, β=0.01, T=200 rounds, seed=0.

Cap arithmetic:
  s = n_honest = 120
  slack = min(10, s−2) = 10
  t = 1/(s−slack) = 1/110 ≈ 0.009091
  s·t = 120/110 ≈ 1.0909 ≥ 1  ✓ (feasible)
  max Byzantine weight mass = 80 × (1/110) = 80/110 ≈ 0.7273

Paper target (Table 3, MNIST, LIE, q=0.9, 40% Byzantine):
  FedLAW: 70.10 ± 2.17%
  FedAvg: 84.22%

### Results

Round-by-round diagnostics (eval_every=10):

  round   0: acc=0.0686  sum_byz=0.4000  max_byz=0.0050  max_hon=0.0050
  round  10: acc=0.1895  sum_byz=0.7273  max_byz=0.0091  max_hon=0.0091
  round  20: acc=0.1174  sum_byz=0.7273  max_byz=0.0091  max_hon=0.0091
  round  50: acc=0.1034  sum_byz=0.7273  max_byz=0.0091  max_hon=0.0091
  round 100: acc=0.1022  sum_byz=0.7273  max_byz=0.0091  max_hon=0.0091
  round 200: acc=0.1022  (FINAL)

Final test accuracy: ~10.2% (chance for 10-class MNIST).
Paper target:        70.10 ± 2.17%.
Discrepancy:         ~60 percentage points.

### Diagnosis

**OUTCOME 2: Genuine discrepancy.**

Byzantine weight behaviour:
- Round 0: uniform weights (0.005 = 1/200 each, sum_byz = 80/200 = 0.4)
- Round 10: Byzantine jump to cap (0.0091 = 1/110), sum_byz = 80/110 = 0.7273
- Rounds 10–200: completely stable. All 80 Byzantine clients at the cap.
  max_hon = max_byz = 1/110 → both groups have some clients at the cap.
  Remaining honest budget: 1 − 0.7273 = 0.2727 split across ~40 selected honest clients.

The model collapses to chance despite both groups hitting the cap.

Root cause: τ = 1.5 is inappropriate for pseudo-gradients at q=0.9 heterogeneity.

With q=0.9 Cao-partitioned data, the coordinate-wise standard deviation of honest
pseudo-gradients is ~3× LARGER than the mean (verified in simulation with d=100
proxy and 10 groups × 12 clients each):

  ||mean_pseudo|| ≈ 4.99
  ||std_pseudo||  ≈ 15.86
  ||lie_1.5||     ≈ 28.76   (cos with mean = +0.998, co-aligned)
  ||lie_z_stlth|| ≈  9.86   (cos with mean = −0.992, ANTI-aligned — targets Krum, not FedLAW)

  Magnitude ratio: ||lie_1.5|| / ||mean|| ≈ 5.77

With τ=1.5, the LIE pseudo-gradient is 5.77× the honest mean. This corresponds to
an implied local model ψ_byz = θ − α·g_lie that is far outside the range of any
real local model (the σ component, proportional to the inter-client spread, dominates).
When 72.7% of the global weight goes to these 80 Byzantine clients, the weighted-average
model update:

  θ_{k+1} = Σ w_i ψ_i ≈ 0.727 ψ_byz + 0.273 ψ_honest_avg

is pulled to an incoherent region by the ψ_byz term, causing divergence.

The Baruch et al. stealth bound for 40% Byzantine (n=200, f=80) gives z = −0.935
(negative — it targets coordinate-wise median defenses, not FedLAW). This is NOT
the right τ for our setting and using it would produce an anti-aligned attack that
FedLAW would detect.

**The paper's 70% result requires a different τ** — one small enough that the LIE
vector stays close to the honest mean in magnitude while remaining co-aligned. The
paper's appendix §I.1 refers to "z = stealth bound (Baruch et al. 2019)" but does
not print the numerical value. With pseudo-gradients and q=0.9 heterogeneity, the
appropriate τ for graceful degradation (not collapse) is unknown without the paper's
exact implementation.

### What is reproduced vs. open

Reproduced:
  - LIE evades FedLAW's cross-product detection (Byzantine weights do NOT go to 0)
  - Cap arithmetic is feasible (s·t = 1.09 ≥ 1)
  - Byzantine clients absorb max allowed budget (sum_byz = 80/110 = 0.727)
  - Small-n collapse (n=20, t=1/2) was correctly diagnosed as a cap artifact

Not reproduced:
  - Final accuracy 70.10% vs our 10.2%
  - Graceful degradation — we get full collapse instead

Outstanding question for viva:
  Which τ does the paper use for LIE pseudo-gradient experiments?
  (Options: stealth-bound z computed for FedLAW specifically; or a fixed small τ;
   or τ applied to raw gradients not pseudo-gradients.)
  Investigation needed before claiming LIE reproduction.

────────────────────────────────────────────────────────────────────────
## End of paper fixes validation

Full report: ./results/paper_fixes/REPORT.md
