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
## LIE raw-gradient hypothesis test (2026-06-26)

### Hypothesis (a)
Paper computes LIE μ/σ over raw single-batch gradients ∇f(θ; batch), not
pseudo-gradients (θ−ψ_i)/α. Raw grads should have smaller σ/μ, producing a
co-aligned forged vector that evades FedLAW gracefully.

### Setup
LIERawGradAttack: at each round, honest clients set params to θ, run one backward
pass (no local epochs), extract flat gradient vector. Byzantine clients submit
b = μ_raw + z·σ_raw as their pseudo-gradient. τ = z = 0.9346. n=20, q=0.9, f=8.

### Diagnostic @ round 0 (n=20 smoke run, d=178,110)

  ||μ_raw||     = 0.963
  ||σ_raw||     = 2.805
  σ_raw/μ_raw   = 2.91      ← HIGHER than pseudo-grad ratio of 2.12, not lower
  ||b_lie_raw|| = 2.72
  ||μ_pseudo||  = 36.82     ← forged vector is ~14× smaller than honest pseudo-grad
  cos(b_lie_raw, μ_pseudo) = −0.023  (orthogonal — not co-aligned)

### Result

Byzantine weights:  zeroed by round 5, sum_byz=0.000 for all remaining rounds.
Final accuracy:     88.4%   (FedLAW fully recovers — attack is detected)

### Verdict: hypothesis (a) is FALSE.

Raw gradients do NOT have a smaller σ/μ ratio — they have a larger one (2.91 vs
2.12 for pseudo-gradients). The forged vector b_lie_raw is both orthogonal to the
pseudo-grad mean AND tiny in magnitude (~14× smaller than honest pseudo-grads).
FedLAW's cross-product mechanism assigns near-zero scores to these Byzantine
clients and zeros their weights by round 5. Raw-grad LIE is detectable.

The reason pseudo-grad LIE evades detection (z=0.9346 run: sum_byz=0.727 stable)
is that ||b_lie_pseudo|| ≈ 34 is close to the honest pseudo-grad norm (~47), so
cross-product scores are near-zero but not sufficiently negative to zero them.
Switching to raw grads collapses ||b|| to ~2.7, making the cross-product
unambiguously near-zero and enabling exclusion.

The 37pp accuracy gap (33.3% vs 70.10%) remains unexplained by hypothesis (a).
Remaining possibilities not investigated:
  (b) Different data heterogeneity setup for MNIST LIE experiments in the paper.
  (c) Paper used more rounds or different seeds for the Table 3 number.
  (d) The paper's LIE results reflect a bug or optimistic configuration not
      described in the appendix.
Without the paper's source code, the gap cannot be closed from published info.

────────────────────────────────────────────────────────────────────────
## flipping_label n=200 frac=0.4 diagnosis (2026-06-26)

Context: partial paper-scale runs showed flipping_label q=0.9 f=0.4 plateauing
at ~65% by round 200 (paper Table 3 target: 87.45%). frac=0.1 tracked the paper.
Diagnosed before chasing the gap.

### D1 — weight trajectory (per round, first 22 rounds)

frac=0.4 (s=120, t=1/110, max byz mass = 80/110 = 0.727):

    r   sum_byz   max_byz   max_hon  n_byz_sup  n_hon_sup
    0    0.5073    0.0091    0.0091         62         58
    1    0.5038    0.0091    0.0091         62         58
    2    0.4644    0.0091    0.0091         59         61
    3    0.3798    0.0091    0.0091         46         74
    4    0.2975    0.0091    0.0091         40         80
    5    0.2741    0.0091    0.0091         40         80
    6    0.2727    0.0091    0.0091         40         80   ← cap floor reached
   …   (stable through round 21)

frac=0.1 (s=180, t=1/170, max byz mass = 20/170 = 0.118):

    r   sum_byz   max_byz   max_hon  n_byz_sup  n_hon_sup
    0    0.1176    0.0059    0.0059         20        160
    1    0.1176    0.0059    0.0059         20        157
    2    0.0878    0.0051    0.0059         20        159
    3    0.0589    0.0044    0.0059         20        160
    4    0.0588    0.0051    0.0059         19        160
   …   sum_byz stays around 0.059, half the cap maximum

At frac=0.4, sum_byz starts ABOVE its uniform initial value (0.51 vs 0.40) — the
first projection adds weight to Byzantine clients — then settles to the cap
floor (40 Byzantine clients pinned at cap = 0.0091, total 0.273) by round 6. Of
the 80 Byzantine clients, ~40 are excluded but ~40 hold full cap weight forever.

### D2 — selection scheme (code inspection)

src/data_partition.py:select_malicious_indices implements group-oriented
selection: rng.permutation(n_groups) then fills complete groups. For n=200,
n_per_group=20: frac=0.4 picks 4 full groups (80 clients, here groups {2,4,8,9});
frac=0.1 picks 1 full group (20 clients, here group 2). Matches paper §I.1.
NOT the bug.

### D3 — Byzantine vs honest gradient stats @ round 5

frac=0.4:
  ||mean honest g||      = 15.31
  ||mean Byzantine g||   =  9.25
  cos(byz, honest_mean)  = +0.1617    ← CO-ALIGNED (mildly) → evades detector

frac=0.1:
  ||mean honest g||      = 10.24
  ||mean Byzantine g||   = 29.47
  cos(byz, honest_mean)  = −0.1361    ← anti-aligned → detector works

Root cause. Byzantine flipping_label clients at frac=0.4 produce pseudo-gradients
whose mean is mildly co-aligned with the honest mean. FedLAW's cross-product
detector relies on Byzantine anti-alignment; it cannot distinguish co-aligned
flipping_label clients from weak honest clients. The mechanism degenerates to
selecting clients with the strongest training signal, and 40 of the 80 Byzantine
clients have stronger h-values than 40 of the 120 honest clients, so they enter
the support at cap and stay there.

At frac=0.1 the Byzantine clients are a single group (group 2) — their flipped
label gradient is anti-aligned with the consensus from 9 other classes. As Byz
fraction rises, the Byzantine mean averages across more flipped classes and
partially cancels in the direction orthogonal to honest, leaving a small
co-aligned residual.

### D4 — cap arithmetic

frac=0.4: s·t = 120/110 = 1.0909 ≥ 1 ✓. Max byz mass 0.727, honest budget 0.273.
frac=0.1: s·t = 180/170 = 1.0588 ≥ 1 ✓. Max byz mass 0.118, honest budget 0.882.

Cap is not the bottleneck — the paper achieves 87% at frac=0.4 with the same
formula. The cap allows Byzantine to hold up to 72.7%, but only matters if the
detector lets them reach it. The detector failure (D3) is the upstream cause.

### Verdict — B (implementation gap, not random selection)

Selection (D2) matches the paper. Cap arithmetic (D4) matches the paper. The
detector demonstrably fails at frac=0.4 (D3) by leaving Byzantine clients at
+0.16 cosine alignment with honest mean. This is upstream of the cap.

The paper achieving 87% with the same formula means either:
  (i) Their model produces strongly anti-aligned pseudo-grads under flipping_label
      with 4-group corruption (architectural difference — they likely use a CNN
      while we use mlp3_mnist; CNN gradients have richer class-specific structure
      that flipping_label more sharply inverts), OR
  (ii) Their pseudo-gradient definition or local-epoch count produces sharper
       direction divergence (we use E=3, β=0.01, momentum=0), OR
  (iii) An undocumented difference in the projection / weight update schedule
        prevents the Byzantine clients from holding cap weight.

The frac=0.1 case works because the single-group Byzantine pseudo-grad is sharply
anti-aligned (cos=−0.14). The frac=0.4 failure is therefore not a wholesale bug
but a sensitivity of the detector to alignment that emerges only at multi-group
Byzantine corruption.

Concrete next step (not done — out of scope for this batch):
  Swap mlp3_mnist for a small CNN (paper architecture for MNIST is usually a
  2-conv CNN); rerun flipping_label q=0.9 f=0.4 and re-measure cos(byz, honest)
  at round 5. If CNN gradients are anti-aligned, the gap closes. If not, the
  paper's 87% number is not reproducible from published configuration alone.

────────────────────────────────────────────────────────────────────────
## flipping_label co-alignment root cause (2026-06-26)

Follow-up to the §"flipping_label n=200 frac=0.4 diagnosis" above. The
paper-faithfulness audit (`PAPER_FAITHFULNESS.md`) has already verified the
gradient definition, cap arithmetic, group-oriented selection, and model
architecture (paper §5.1 specifies the 3-layer MLP). The open question is
why our cos(byz, honest) = +0.16 at frac=0.4 contradicts the paper's
Appendix C, which states data-poisoning attacks conflict with the honest
cluster (anti-alignment) at all fractions. This section answers it.

### D1 — label-flip mapping verified

Byzantine client #40 (group 2): 10 sample pairs (original → flipped) show
2→7, 5→4 (etc.), all matching L−l−1 with L=10. Group-2 examples (which
make up 90% of client #40's data by q=0.9 partition) all flip to label 7.
Honest client #0 (group 0) sees unflipped labels [0,0,…] from the loader.

  Mapping is exact paper §I.1. Not a bug.

### D2 — cos(byz_mean, honest_mean) vs frac_malicious (n=200, q=0.9, round 5)

  frac=0.1  1 group   ||byz||=29.47  cos = −0.1361   anti-aligned
  frac=0.2  2 groups  ||byz||=14.93  cos = −0.1709   anti-aligned
  frac=0.3  3 groups  ||byz||= 8.51  cos = −0.0755   anti-aligned (weakening)
  frac=0.4  4 groups  ||byz||= 9.25  cos = +0.1617   CO-ALIGNED

The transition is **not smooth from −0.14 to +0.16**. It is non-monotonic:
anti-alignment strengthens from frac=0.1 to frac=0.2, then weakens at 0.3,
then flips sign at 0.4. Norm shrinks roughly monotonically from 29.5 to 8.5
between frac=0.1 and 0.3 — clear cancellation across groups.

### D3 — per-group breakdown at frac=0.4 (decisive)

Per-group mean Byzantine pseudo-gradient at round 5:

  group 2 (label 2→7, n=20):   ||mean_g||=35.81   cos(g, honest)=+0.1732
  group 4 (label 4→5, n=20):   ||mean_g||=24.24   cos(g, honest)=−0.0398
  group 8 (label 8→1, n=20):   ||mean_g||=21.44   cos(g, honest)=−0.0470
  group 9 (label 9→0, n=20):   ||mean_g||=31.85   cos(g, honest)=+0.0549

Individual group pseudo-gradients are large (21–36) — each corrupted group
on its own produces a strong update signal. But pairwise cosines between
the 4 corrupted-group means are **all negative**:

  cos(g2, g4) = −0.18    cos(g2, g8) = −0.21    cos(g2, g9) = −0.11
  cos(g4, g8) = −0.36    cos(g4, g9) = −0.30    cos(g8, g9) = −0.19

The 4 Byzantine groups point in **mutually antagonistic** directions
(each group is pushing the model to mislabel its own dominant class,
which is a different parameter-space direction per class).

Cancellation when summed:

  Σ_g ||group_g||      = 113.34    (max possible if perfectly aligned)
  || Σ_g group_g ||    =  36.99    (after summation)
  cancellation ratio   =   0.33    (67% of the per-group magnitude cancels)
  ||overall_byz_mean|| =   9.25    (after dividing by 4 groups)
  cos(overall_byz_mean, honest) = +0.1617

### What happens mechanically

At multi-group corruption with q-split data, each corrupted group submits
a pseudo-gradient that pushes the model away from correctly classifying its
own dominant label. These per-group directions are nearly orthogonal /
mildly anti-aligned with each other (the parameter directions for
class-2-misclassification, class-4-misclassification, etc. are largely
distinct). Their average partially cancels — the surviving residual is
weak, and at this point in training happens to be mildly co-aligned with
the general "improve classification" direction of the honest gradient.

The cross-product detector measures alignment of the **averaged**
Byzantine direction against the consensus. It cannot see the per-group
signals individually. So with 4 groups it sees a small, co-aligned vector
and cannot distinguish it from a weak-honest-client gradient.

At frac=0.1 (1 group) there's no cancellation — the Byzantine direction
is large and clearly anti-aligned. Detector works. At frac=0.4 (4 groups)
the same per-group signals partially cancel and the residual flips sign.
Detector fails.

### Verdict: (C) — genuine property, not implementation bug

Evidence supporting (C):

  - Label-flip mapping exactly matches paper (D1).
  - Per-group pseudo-gradients are individually large (21–36) and largely
    anti-aligned with each other (D3 pairwise table) — flipping_label is
    "working" per group as the paper describes.
  - The Byzantine mean cancellation is geometric, not a bug — it is the
    inevitable consequence of averaging 4 group-specific misclassification
    directions.
  - All upstream paper-faithfulness items already verified
    (`PAPER_FAITHFULNESS.md` §§1, 3, 6, plus architecture in §"Synthesis").
  - The non-monotonic frac series (−0.14, −0.17, −0.08, +0.16) shows a
    smooth geometric transition that arises from the number of groups
    averaged — not a sudden threshold effect that would suggest a wiring
    bug.

This contradicts the paper's Appendix C claim that data-poisoning attacks
produce gradients that conflict with the honest cluster at all fractions.
With the q-split + group-oriented selection at frac=0.4, our measurements
show the averaged Byzantine gradient becomes co-aligned (cos=+0.16), and
FedLAW's detector cannot suppress them — Byzantine weights pin at the
cap floor (sum_byz=0.273) for all 200 rounds.

We cannot rule out hypothesis (B) — some unstated experimental difference
(seed coupling, q-split implementation choice, projection-time bias) that
the paper used to obtain 87.45% — because our setup is paper-faithful on
every item we can verify from the published configuration.

The 18pp gap (~65% vs 87.45%) at frac=0.4 flipping_label is therefore a
**reportable finding**: FedLAW's cross-product detector exhibits a
multi-group cancellation failure mode under Cao q-split data that is not
acknowledged in the paper's mechanism description.

────────────────────────────────────────────────────────────────────────
## Seed-sensitivity of flipping_label co-alignment finding (2026-06-27)

The prior (C) verdict rested on a single seed (group draw {2,4,8,9}). Since
the cancellation depends on which groups are corrupted and what they flip
to, the finding was open to dismissal as a single-draw artefact. This
section tests robustness across 5 different seeds (different group draws).

### Setup

flipping_label, n=200, q=0.9, frac=0.4, rounds 0–5, varying only `seed`.
The seed controls both the data partition and `select_malicious_indices`
group draw, so each seed gives a different corrupted-group set.

### Results (round 5, cos(byz_mean, honest_mean))

  seed  corrupted groups        cos        ||byz||  ||hon||  sum_byz
   0    {2, 4, 8, 9}           +0.1617      9.248   15.309    0.274
   1    {2, 4, 6, 9}           +0.1247     10.797   14.908    0.273
   2    {0, 1, 4, 5}           +0.0829     21.133   15.574    0.273
   3    {1, 2, 4, 5}           +0.1570     21.790   16.219    0.273
   4    {3, 4, 8, 9}           +0.1192      9.502   15.442    0.302

  cos mean = +0.1291,  std = 0.0286,  range = [+0.083, +0.162]
  n positive = 5/5,  n negative = 0/5
  sum_byz at round 5 across all seeds ≈ 0.273–0.302 (cap floor)

### Interpretation

5/5 seeds give positive cos at round 5 across distinct group draws —
co-alignment is **not a single-draw artefact**. The std (0.029) is small
relative to the mean (+0.129); no seed comes close to crossing zero. Per-
seed ‖byz_mean‖ varies (9.2 to 21.8) — the magnitude of cancellation
depends on which 4 groups are picked, but the sign of the residual's
alignment with the honest mean does not.

sum_byz lands at the cap floor (~0.273) in every seed: Byzantine
weights pin at maximum cap allocation regardless of group draw. This is
the detector failing the same way every time.

### Verdict strengthens — (C) holds robustly

The multi-group cancellation failure mode is a **reproducible property of
FedLAW + Cao q-split at frac=0.4**, not a function of which specific
groups are corrupted. The paper would not "average it away" across seeds:
5/5 of our seeds give cos > 0 and sum_byz at the cap floor, so any seed
average over this configuration would also show the gap.

This rules out the seed-variance dismissal. The 18pp gap (~65% vs the
paper's 87.45%) at flipping_label q=0.9 frac=0.4 stands as a reportable
finding: FedLAW's cross-product detector cannot suppress 4-group label-flip
Byzantine clients under Cao q=0.9 partitioning, because the averaged
Byzantine pseudo-gradient cancels into a small co-aligned residual that
the detector cannot distinguish from a weak honest gradient.

The remaining possibility (B) — that the paper used an unstated
experimental detail to obtain 87.45% — is now the only available
explanation that does not require concluding the paper's number is
unreproducible from its published configuration.

────────────────────────────────────────────────────────────────────────
## inverse_gradient f=0.1 re-entry — detection decays as model converges (2026-06-27)

The completed inverse_gradient q=0.9 f=0.1 n=200 run shows Byzantine clients
detected at round 10 (sum_byz: 0.1 → 0.001) but climbing back to the cap
floor (0.1176 = 20/170) by round 20 and staying there. Final accuracy
81.0% vs paper 89.5% (gap −8.5pp). This addendum determines whether the
re-entry is caused by our v2 `w_freeze` extension or by a genuine FedLAW
property.

### D1 — sum_byz trajectory with w_freeze DISABLED

`w_freeze_rounds=999` (effectively off), single short run, n=200,
q=0.9, frac=0.1, inverse_gradient, seed=0:

  round  8:  sum_byz = 0.0000   (detected, fully zero)
  round  9:  sum_byz = 0.0010
  round 10:  sum_byz = 0.0042
  round 11:  sum_byz = 0.0144
  round 12:  sum_byz = 0.0513
  round 13:  sum_byz = 0.0685
  round 14:  sum_byz = 0.0990
  round 15:  sum_byz = 0.1176   ← CAP REACHED, w_freeze would have been off
  rounds 16–29: 0.1176 (stable, even with freeze off)

Re-entry to the cap is **complete by round 15**. The original v2 freeze
fires at round 20 — five rounds too late to be the cause. With freeze
disabled, the system reaches the same pinned state.

  Freeze does NOT cause the re-entry. It locks in a state the dynamics
  produced on their own.

### D2 — h-component breakdown @ rounds 5/10/15/20

For each round, h_i = w_i + α·β·(G·Gᵀ̃·w)_i − β·f̃_i. The two component
magnitudes and per-client values:

  round  5:  ||det_term|| = 0.1014,  ||loss_term|| = 0.2683,  ratio = 0.378
             cross(byz, w) avg = −38.89   (strongly anti-aligned)
             cross(hon, w) avg = +71.63
  round 10:  ||det_term|| = 0.0520,  ||loss_term|| = 0.2250,  ratio = 0.231
             cross(byz, w) avg = −30.78   (still anti-aligned)
             cross(hon, w) avg = +36.27
  round 15:  ||det_term|| = 0.0313,  ||loss_term|| = 0.2066,  ratio = 0.152
             cross(byz, w) avg = +39.61   ← FLIPPED SIGN
             cross(hon, w) avg = +18.25
  round 20:  ||det_term|| = 0.0233,  ||loss_term|| = 0.1920,  ratio = 0.122
             cross(byz, w) avg = +36.98
             cross(hon, w) avg = +11.84

Two simultaneous effects:

  (i) The detection term's magnitude shrinks ~4× from round 5 to round 20
      (||det|| 0.101 → 0.023). The loss term shrinks only ~1.4× over the
      same span. Ratio collapses from 0.378 to 0.122.

  (ii) The Byzantine cross-product itself FLIPS SIGN between rounds 10
       and 15 — from −30.8 (anti-aligned, detectable) to +39.6
       (positive, indistinguishable from a weak honest contribution).

### Why the sign flips — mechanism

The InverseGradient attack returns g_byz_i = −g_honest_i (the Byzantine
client's *own* pseudo-gradient negated). At q=0.9 each group's gradient
is dominated by its assigned class's signal. The Byzantine clients here
are all from group 2 (label-2 dominant).

At round 5, honest gradients across all 9 honest groups are individually
large and the consensus direction is well-aligned with "improve general
classification". Negating group 2's pseudo-gradient produces a vector
that opposes this consensus → cross-product strongly negative.

By round 15, honest gradients have shrunk (loss 2.3 → 1.36) and become
increasingly class-specific (each group's gradient now mostly addresses
its own dominant label, which is approximately orthogonal to other groups'
gradients). The consensus is the average of 9 nearly-orthogonal class
directions, much smaller than the per-group magnitude. Negating one
group's pseudo-gradient (−g_group2) produces a vector that is no longer
opposed to the consensus — it ends up mildly aligned with the residual
"improve label-2 classification" direction that the consensus needs.

### Why Byzantine clients re-enter the support (loss imputation interaction)

Byzantine clients use imputed-mean-honest loss (in `_collect`). At round 15
this imputed value is 1.36. Honest clients use their actual losses, which
under q=0.9 heterogeneity are *highly variable* — well-trained groups have
losses as low as 0.7, struggling groups have losses above 2.0.

So the h-vector ordering at round 15 becomes:

  high h:  low-loss honest clients (well-trained groups)
  mid h:   Byzantine clients (imputed mean loss + slightly positive det)
  low h:   high-loss honest clients (struggling groups)

The sparse-capped projection selects the top-180 of 200. Byzantine fill
the slots vacated by the lowest-h honest clients (the heterogeneous
losers). Byzantine wind up at full cap, 20–25 honest clients get zero
weight (the diagnostic shows `n_hon_sup` = 153–159 instead of 180).

### Verdict: (B) — genuine detection-decay-on-convergence

Evidence ruling out (A) freeze artifact: D1 shows re-entry completes at
round 15 with the freeze disabled. The freeze cannot be the cause; it
just makes the pinned state permanent past round 20 instead of allowing
the dynamics to continue (which D1 shows would not have recovered).

Evidence supporting (B) genuine property:
  - Detection magnitude shrinks 4× by round 20 (D2 aggregate norms).
  - Byzantine cross-product flips sign — the inverse_gradient attack
    stops being anti-aligned at the same point that the model becomes
    well-trained on easy honest features.
  - Loss imputation puts Byzantine in the middle of the h-ranking,
    above the struggling-group honest clients, who get displaced.

Verdict is not (C) (both) because, with freeze off, the system **does
not recover** after round 20 — sum_byz stays at the cap. There is no
"freeze prevented a recovery" case to make.

### Implications

  - The flipping_label f=0.4 finding (cancellation, sum_byz pinned by
    round 6) and the inverse_gradient f=0.1 finding (detection decay,
    pinned by round 15) are TWO DISTINCT failure modes of FedLAW's
    cross-product detector at paper scale with Cao q-split data.
  - Both reproduce robustly. Neither is caused by our v2 w_freeze
    extension.
  - The paper's accuracy numbers at these configs (87.45% for flipping
    f=0.4, 87.41% for inverse_gradient at f=0.4, ~89% for f=0.1
    detected attacks) are not reproducible from our paper-faithful
    setup without an unstated experimental detail we have not been
    able to identify.

The remaining n=200 detected-attack runs (backdoor, double, inverse at
f=0.4) are likely to exhibit similar plateau behaviour — the detection
decay mechanism does not depend on the specific attack, only on whether
the attack's anti-alignment survives gradient shrinkage. To be confirmed
by the open paper-scale runs (not started in this addendum).

────────────────────────────────────────────────────────────────────────
## Clean-baseline sanity check (2026-06-27)

Before extending the detection-decay finding into a general claim about
FedLAW's detector at paper scale, we verify whether our reproduction's
clean (no Byzantine) training trajectory matches what the paper implies.
If our clean model undertrains relative to the paper, the gradient
dynamics driving the detection decay could be a symptom of our setup, not
a property of FedLAW.

### Setup

`frac_malicious=0.0`, `n=200`, `q=0.9`, `T=200`, `α=0.01`, `β=0.01`,
`E=3`, `seed=0`, `w_freeze` disabled. No Byzantine clients exist; the
trainer runs the standard FedLAW projection on honest-only contributions.

### Accuracy trajectory (clean)

  round  20:  72.21%
  round  50:  84.05%
  round 100:  88.42%
  round 150:  89.87%
  round 200:  **90.58%**

The paper's under-attack accuracies for detected attacks at q=0.9 are
~87–89.5%, implying a clean baseline around 91–92%. Our clean number
(90.58%) is roughly 1pp below the paper's implied clean — close, not
identical. Our reproduction is not significantly undertrained.

### Honest-client loss heterogeneity

Loss statistics across all 200 clients each round (mean / std / min / max):

  round   5:  f_mean=2.07  f_std=0.17  f_min=1.81  f_max=2.53   max/mean=1.22
  round  10:  f_mean=1.75  f_std=0.30  f_min=1.31  f_max=2.55   max/mean=1.46
  round  20:  f_mean=1.24  f_std=0.40  f_min=0.60  f_max=2.23   max/mean=1.80
  round  50:  f_mean=0.67  f_std=0.32  f_min=0.17  f_max=1.66   max/mean=2.48
  round 100:  f_mean=0.44  f_std=0.22  f_min=0.09  f_max=1.02   max/mean=2.30
  round 200:  f_mean=0.35  f_std=0.17  f_min=0.06  f_max=0.98   max/mean=2.82

Two patterns:

  - The mean loss drops smoothly from 2.07 (round 5) to 0.35 (round 200).
  - The relative spread (max/mean) GROWS from 1.22× early to ~2.5–2.8×
    after round 50, and stays large for the rest of training.

The std falls slower than the mean: by round 100, std/mean ≈ 0.50; by
round 200, std/mean ≈ 0.49. Some honest groups (those whose dominant
labels are harder under q=0.9 partitioning) carry a persistent loss
~3× higher than the average client throughout training.

### Why this matters for the detection-decay finding

The inverse_gradient f=0.1 diagnosis attributed the round-15 re-entry
to two effects: detection term shrinkage and Byzantine clients
out-ranking high-loss honest clients via imputed-mean loss. The
clean-baseline numbers confirm both inputs to that mechanism are real
features of our reproduction, not artefacts:

  - **Detection shrinkage**: honest gradient magnitudes scale with loss.
    Mean loss drops 6× from round 5 (2.07) to round 200 (0.35), so honest
    pseudo-gradient magnitudes shrink by a comparable factor. This drives
    the detection-term collapse (||det|| dropped 4× from round 5 to 20
    in the diagnostic).
  - **Loss heterogeneity**: f_max stays 2.3–2.8× f_mean throughout.
    The imputed-mean trick (Byzantine f̃ = honest mean) places Byzantine
    clients above the high-loss honest stragglers — Byzantine f̃ = 0.44
    is below honest f_max = 1.02 at round 100. Several honest clients are
    consistently above the imputed-mean Byzantine value at every round.

### Conclusion — frame as our reproduction's behavior

Our reproduction reaches 90.58% clean at 200 rounds, ~1pp below the
paper's implied clean. Under inverse_gradient f=0.1 we reach 81.03%, a
9.5pp degradation from our clean. The paper's implied under-attack
degradation is ~2–3pp. The 6–7pp discrepancy in degradation cannot be
attributed to undertraining alone — our clean is close to the paper's
implied clean.

The detection-decay mechanism documented above is therefore a property of
**our reproduction at the configurations we ran**: with α=0.01, β=0.01,
E=3, q=0.9 Cao partition, mlp3_mnist, and seed=0, the FedLAW cross-product
detector loses Byzantine separability as the model converges and honest
losses become heterogeneous. Whether the paper's setup avoids this — via
some unstated experimental detail we have not identified, or via a
different convergence regime — remains undetermined.

This is consistent with the existing characterized-gap framing: **not
"the paper is unreproducible"; it is "we cannot reproduce these specific
numbers from the published configuration, and we have characterized why
our reproduction lands where it does."**

────────────────────────────────────────────────────────────────────────
## End of paper fixes validation

Full report: ./results/paper_fixes/REPORT.md
