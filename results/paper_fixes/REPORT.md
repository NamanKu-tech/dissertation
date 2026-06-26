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
## LIE Check 1 + Check 2 diagnostics (2026-06-26)

### Check 1 — what does LIE compute μ/σ over?

Object: honest pseudo-gradients g_i = (θ − ψ_i)/α  (correct — no wiring bug)
Measured at round 0, MNIST scale (d = 178,110), n=200, f=80, q=0.9:

  ||μ_pseudo||    = 20.24
  ||σ_pseudo||    = 42.94    (ratio σ/μ = 2.12)

  b_lie (τ=1.5):  ||b|| = 56.29   cos(b, μ) = −0.2502   (ANTI-aligned)
  b_lie (z=0.935): ||b|| = 33.98   cos(b, μ) = −0.0337   (nearly orthogonal)

  Honest client norms: mean = 47.20,  std = 3.25

Key finding: at MNIST scale with pseudo-gradients and q=0.9, the LIE vector is
NOT co-aligned with the honest mean for either τ value. The earlier proxy
simulation (d=100) gave cos=+0.998 — this was a dimensionality artefact.

### Check 2 — Baruch et al. stealth bound

n=200, f=80:
  m = ⌊200/2 + 1⌋ − 80 = 21
  z = Φ⁻¹((200 − 80 − 21) / (200 − 80)) = Φ⁻¹(0.8250) = 0.9346

LIE rerun with τ = z = 0.9346 (n=200, q=0.9, frac_mal=0.4, seed=0, T=200):

  round   0: acc=0.0686  sum_byz=0.4000
  round  10: acc=0.1950  sum_byz=0.7273  (cap reached, stable thereafter)
  round  50: acc=0.2391  sum_byz=0.7273
  round 100: acc=0.2650  sum_byz=0.7273
  round 150: acc=0.2955  sum_byz=0.7273
  round 200: acc=0.3328  [FINAL]

  Final accuracy: 33.3%   (τ=1.5 gave 10.2%;  paper target: 70.10 ± 2.17%)

### Verdict

τ=1.5 → collapse (10.2%): LIE vector anti-aligned, Byzantine mass = 72.7%,
  weighted update pulled to incoherent region.

z=0.9346 → partial learning (33.3%): model is NOT collapsed, Byzantine weights
  stable at cap, but accuracy plateaus ~30-33% with high noise. Still 37pp
  below paper target.

LIE evasion of FedLAW's cross-product detector: CONFIRMED in both runs.
  Byzantine weights jump to cap at round 10 and remain at 0.7273 for all 200
  rounds — FedLAW never zeros them.

Reproduction status:
  Behaviour (LIE evades detection):        REPRODUCED
  Paper accuracy (70.10 ± 2.17%):          NOT REPRODUCED (best: 33.3% with z)

Outstanding question: the 37pp residual gap with z=0.9346 is unexplained. The
paper's appendix §I.1 states "z = stealth bound (Baruch et al. 2019)" without
printing τ numerically. Possible explanations not yet ruled out:
  (a) Paper uses raw gradients, not pseudo-gradients, for LIE μ/σ computation.
  (b) Paper uses a different heterogeneity setup for MNIST LIE experiments.
  (c) Paper reports a different seed / more rounds than T=200.
Investigation would require the paper's source code or correspondence with
authors — beyond the scope of this dissertation reproduction.

────────────────────────────────────────────────────────────────────────
## End of paper fixes validation

Full report: ./results/paper_fixes/REPORT.md
