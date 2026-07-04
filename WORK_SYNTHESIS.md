# FedLAW under Partial Participation — Full Work Synthesis

Single-document synthesis of the entire dissertation work to date, spanning:
(1) reproducing the ICLR 2026 FedLAW paper, (2) building the partial-
participation extension, (3) developing the coordinated dormancy attack,
and (4) testing the attack against the full defender family (FedLAW's
learnable-weights + DeMoA-cache with four aggregators).

Written 2026-07-04 while the final Bulyan cell is running. Points to the
authoritative artefacts in the repo (this doc is a map, not the sole record).

---

## 0. TL;DR

**FedLAW reproduction (Phase 1):** Mechanism verified paper-faithful.
Clean baseline reaches 90.58% at n=200 (paper implies 91–92%). Three
characterized accuracy gaps, each with mechanism identified:
- LIE at Baruch z: 33.3% vs 70.10% (paper's stealth bound doesn't
  produce evasion in our setup; cause not resolvable from published info).
- flipping_label f=0.4: 65.4% vs 87.45% (multi-group cancellation is
  robust across 5 seeds; genuine detector failure under q=0.9 + 4-group
  Byzantine).
- inverse_gradient f=0.1: 81.0% vs 89.5% (detection decays as model
  converges; w-freeze exonerated; not undertraining).

**Partial-participation contribution (Phase 2):** Bernoulli-p sampling
harness passes sanity gate (90.61% at p=1.0). Design ladder A→B(i)→B(ii)
implemented and validated. Coordinated cohort dormancy attack lands on
FedLAW cache_grad_B_ii for −9.97pp. Attack is **general across all
caching-based defenders tested**: FedLAW, DeMoA+TrMean(f=20/80),
DeMoA+Median, DeMoA+CCLIP(τ=20/50/100). Bulyan test in progress at
time of writing; theorem-limit already established (Bulyan's n≥4f+3
requirement excludes 40% Byzantine at n=200).

**No sweep run yet.** The framing must be confirmed by the Bulyan
result before the systematic multi-seed sweep can be committed to.

---

## 1. Phase 1 — FedLAW paper reproduction (context: dissertation
foundation)

### 1.1 The setup

Reproduce Wang et al. ICLR 2026 (arXiv 2511.03529) on MNIST:
- 3-layer MLP (784→200→100→10) — matches paper §5.1
- Cao q-split partitioning (paper §I.1), group-oriented Byzantine selection
- Learnable aggregation weights `w` via capped-simplex projection
- Attacks: flipping_label, backdoor, inverse_gradient, global_parameter,
  double, LIE (all in paper §I.1)

Authoritative artefacts:
- `PAPER_FAITHFULNESS.md` — code-to-paper audit per numbered component
- `REPRODUCTION_STATUS.md` — verified-faithful vs characterized gaps
- `results/paper_fixes/REPORT.md` — master diagnostic log

### 1.2 Three implementation gaps discovered and fixed

During reproduction, three gaps between v1 code and the paper were
identified and corrected in v2:

**Gap 1 — Pseudo-gradient definition (paper Algorithm 1).**
v1 used raw single-batch gradient `∇f(θ; batch)`; the paper's Algorithm 1
uses `g_i = (θ − ψ_i)/α` from E local SGD epochs. Fix in `src/fedlaw_v2.py:266–321`.
Effect: detection at α=0.01 (paper's value) now works; previously failed
because cross-product signal was 200× too small at α=0.01 with raw grads.

**Gap 2 — Server-side ℓ2 clipping (paper Assumption E1).**
v1 did not clip; paper's convergence theorem requires bounded gradients.
Fix in `src/fedlaw_v2.py:71–86`: clip to C = max_{i honest} ‖g_i‖ per
round. Doesn't change ALIE/LIE behaviour (clipping is a magnitude, not
direction, defense) — documented.

**Gap 3 — Cap arithmetic (paper Table 1).**
v1 misread cap as t=1/s (exact-exclusion); paper Table 1 is t=1/(s−10)
(adaptive weighting). Fix in `src/fedlaw_v2.py:233–235`: slack=min(10,s−2)
with feasibility check s·t ≥ 1. Reproduces paper Figure 1's adaptive
honest weighting behaviour.

### 1.3 Verified components (`PAPER_FAITHFULNESS.md`)

All numbered paper components verified paper-faithful:
- §1 Pseudo-gradient (Algorithm 1) — matches
- §2 Weight update (Algorithm 2) — matches, one documented departure (w-freeze)
- §3 Cap and sparsity (Table 1) — matches
- §4 Server clipping (Assumption E1) — matches
- §5 Cao q-partition (§I.1) — matches
- §6 Group-oriented Byzantine selection (§I.1) — matches
- §7a–e Attacks (§I.1) — all match; backdoor uses random target per §I.1

Model architecture: `mlp3_mnist` matches paper §5.1 "3-layer fully
connected network on MNIST".

48/48 unit tests pass (`tests/`).

### 1.4 Reproduction status (`REPRODUCTION_STATUS.md`)

**Clean baseline at n=200 q=0.9:** 90.58% at 200 rounds (paper implies
~91–92% — within 1pp).

**Small-n behaviour (n=20):** Every attack detected; Byzantine weights
zeroed at round 1. Correct qualitative reproduction.

**Paper-scale runs at n=200 seed=0** (from `results/v2/`):

| Attack | Config | Our acc | Paper Table 3 | Status |
|---|---|---|---|---|
| flipping_label | q=0.9, f=0.1 | 86.3% | ~89–90% | reproduced within band |
| flipping_label | q=0.9, f=0.4 | **65.4%** | 87.45% | **characterized gap** |
| flipping_label | q=0.6, f=0.4 | 54.7% | 92.22% | same gap mechanism |
| inverse_gradient | q=0.9, f=0.1 | **81.0%** | ~89.5% | **characterized gap** |
| inverse_gradient | q=0.9, f=0.4 | 75.7% | 87.41% | related gap |
| inverse_gradient | q=0.6, f=0.4 | 92.2% | 91.62% | reproduced |
| backdoor | q=0.9, f=0.1 | 87.2% | ~89.5% | reproduced within band |
| backdoor | q=0.9, f=0.4 | 83.4% | 87.88% | ~5pp gap |
| double | q=0.9, f=0.1 | 87.7% | ~89.5% | reproduced within band |
| double | q=0.9, f=0.4 | 80.8% | 87.47% | ~7pp gap |
| LIE (τ=1.5 default) | q=0.9, f=0.4 | 10.2% | 70.10% | **characterized gap** |
| LIE (z=0.9346 Baruch) | q=0.9, f=0.4 | 33.3% | 70.10% | **characterized gap** |

### 1.5 Three characterized gaps — mechanisms

**Gap 1: LIE accuracy (§`results/paper_fixes/REPORT.md`)**
Behaviour reproduced (LIE evades detector — Byzantine weights pin at
cap). Paper's 70.10% number NOT reproduced with either τ=1.5 (ByzFL
default) or z=0.9346 (Baruch stealth bound). Diagnosed: μ, σ computed
over pseudo-gradients (Check 1), Baruch formula verified (Check 2),
raw-gradient hypothesis falsified (Check 3). Cause not resolvable
from published information — paper does not print the τ they use.

**Gap 2: flipping_label f=0.4 co-alignment**
`cos(byz_mean, honest_mean)` at round 5: +0.16 at f=0.4 vs −0.14 at
f=0.1. Byzantine detector requires anti-alignment; at 4-group
corruption (Cao q-split assignment), the individually-anti-aligned
group pseudo-gradients partially cancel when averaged (67% cancellation).
The small co-aligned residual evades detection. Robust across 5 seeds
(cos mean +0.129, std 0.029, 5/5 positive) — see `scripts/diag_flip_seedsweep.py`.

**Gap 3: inverse_gradient f=0.1 detection decay**
Byzantine weights detected at round 10 (sum_byz drops 0.1 → 0.001) but
climb back to cap by round 15. Diagnosed with w-freeze DISABLED:
`scripts/diag_wfreeze.py` shows re-entry completes before round 20 →
freeze exonerated. Mechanism: as honest gradients shrink 4× and become
class-specific at q=0.9 heterogeneity, Byzantine cross-product flips
sign (−31 to +40 between rounds 10 and 15); detection term shrinks
faster than loss term. Clean-baseline diagnostic (`scripts/diag_clean_baseline.py`)
confirms 90.58% clean → the 9.5pp under-attack gap is NOT undertraining.

**Framing (from `REPRODUCTION_STATUS.md`):** Neither "the paper is
unreproducible" nor "our implementation is wrong". Mechanism is verified
correct; three specific configurations produce characterized behaviours
of a paper-faithful FedLAW that don't match the paper's specific
numbers.

---

## 2. Phase 2 — Partial-participation extension

Documented in `docs/PARTIAL_PARTICIPATION_DESIGN.md`.

### 2.1 Design ladder A → B(i) → B(ii)

The design doc §2 posits three participation modes:
- **A — naive:** project over sampled set S_t only; absent client's
  weight is discarded (weight continuity broken across rounds).
- **B — cache-and-include (§2.4 crux):**
  - **B Option (i):** cache WEIGHT only. Absent g_i := 0; detector
    cannot re-score absent client → dormancy possible.
  - **B Option (ii):** cache GRADIENT and weight with DeMoA-style
    decay (1−αp)^τ. Absent contributes decayed cached g_i to both
    model update AND cross-product detector → dormancy may be defeated
    by re-scoring.
- **C — staleness-aware:** conditional on dormancy landing against B.

### 2.2 Step 1 sanity gate (`d1dfae8`)

Added Bernoulli-p sampling to `src/fedlaw_v2.py`. p=1.0 fast path is
byte-equivalent to the original trainer (no sampling RNG consumed).

**Sanity gate: p=1.0 clean → 90.61% vs reference 90.58%** (Δ=+0.0003,
noise). Gate PASSED.

p=0.5 smoke: |S_t| 85–118 (mean 101, matches Ber(0.5,200)≈100).

### 2.3 Step 2 naive A characterization (`dbf115c`, `a513d90`)

Ran naive A under partial participation. **A did not degrade as
expected** — at f=0.1 backdoor, p<1.0 finished at parity or above p=1.0
at full 200-round horizon (88.5% at p=0.5 vs 87.2% at p=1.0). At high
stress (p=0.25, f=0.4), A degraded 4.7pp vs matched p=1.0.

Verdict: A is more robust than the design doc anticipated at low
contamination; degrades modestly at high contamination + low p.

### 2.4 Naive-A vs cache_weight_B_i cleanup (`77ee241`)

**Bug caught by user:** the first cache_weight_B_i implementation
projected over S_t (not full n) and let self.w drift globally (sum → 6.7
by round 9 at p=0.1 f=0.4). The apparent "+5-7pp win at f=0.4" was a
normalization artefact.

**Fix:** implement §2.2 canonical B — project over full n every round,
absent g_i := 0. self.w becomes valid simplex (Σ=1 verified). But then
cache_B_i under-trains catastrophically (down to 45.6% at p=0.1 f=0.4)
because active clients can only hold ~|S_t|/n of the weight mass —
model gets ~1/10 the effective step at p=0.1.

**Interpretation:** Option (i) (cache weight, not gradient) is
structurally broken under canonical §2.2 projection — absent weight
mass is stranded with g=0 contribution.

### 2.5 Step 3 cache_grad_B_ii (`343ea5b`, `12f9922`)

Implemented Option (ii): cache gradient AND weight, DeMoA-style
(1−αp)^τ decay. Absent contributes decayed cached g_i to both model
update AND cross-product detector.

**8-cell DeMoA-protocol comparison (naive_A vs cache_grad_B_ii):**

| Cell | naive_A | cache_grad_B_ii | Δ |
|---|---|---|---|
| p=0.5 f=0.1 | 88.47% | 86.97% | −1.50pp |
| **p=0.5 f=0.4** | 78.73% | **83.50%** | **+4.77pp** |
| p=0.1 f=0.1 | 88.22% | 88.25% | +0.03pp |
| **p=0.1 f=0.4** | 78.84% | **84.24%** | **+5.40pp** |

**cache_grad_B_ii wins at f=0.4 by ~5pp.** Byzantine suppression 73–77%
better than naive_A. The Option (ii) mechanism (re-score absent via
cached decayed gradient) works.

Verdict: Design B is the working canonical partial-participation
FedLAW. Ready to attack.

### 2.6 Step 4 dormancy attack (§3 of design doc) — first pass

Implemented in `src/fedlaw_v2.py` — dormant client force-included in
S_t during build-trust [0, T_dark), force-excluded after. At round
T_dark−1, dormant submits a poisoned pseudo-gradient that the server
caches. From then on, cached (decayed) poison persists.

Three payload variants: `inverse_mean` (control, anti-aligned),
`stealth_lie` (μ + z·σ, LIE-style stealth), `stealth_honest` (leave
own honest gradient — attack is just the going-dark itself).

**Single-client dormancy result (`23be5e5`):**
- inverse_mean: cos=−0.99 at round 9 → weight drops to 0.003; caught
- stealth_lie: cos=−0.01 at round 9 → weight HELD at 0.005 for all 180
  dark rounds
- stealth_honest: cos=+0.20 → weight HELD

Stealth evades re-scoring. But single client at 1/n=0.5% weight cap
produces zero measurable accuracy change (Δ within noise).

Verdict for singleton: cap prevents damage, not detector. Outcome (3)
per the user's framework.

### 2.7 Coordinated cohort dormancy (`eff0a9d`, `ab7d34c`)

User's key insight: singleton test was the wrong regime; the realistic
threat is a coordinated cohort matching the reproduction's group-oriented
Byzantine assignment. Cohort_20 = group 2; cohort_80 = groups {2,4,8,9}.

**All dormants cache the SAME poison vector** (contrast: reproduction's
group-oriented f=0.4 Byzantines had different poisons per group that
cancelled 67%).

**Coordinated cohort result:**

| Cell | Final acc | Δ vs matched control |
|---|---|---|
| clean_B | 90.57% | — |
| B_c20_T20 | 89.30% | −0.73pp vs naive |
| **B_c80_T20** | **80.60%** | **−8.03pp vs naive** |
| B_c80_T50 | 82.77% | −5.86pp |
| A_c80_T20 (naive control) | 88.63% | (compute loss only) |

**Coordinated cohort_80 dormancy LANDS at cache_grad_B_ii for
−8.03pp beyond the naive compute-loss baseline.**

**Mechanism (per-round diagnostic):** sum_w_cohort stays at exactly
80/190 = 0.421 for all 180 dark rounds; avg_cos drifts to −0.27 (should
push weights out) but weights don't drop. Why?

Because all 80 dormants cached the SAME poison vector, their cross-
product `(cross @ w)[dormant_i] = Σ_j g_i·g̃_j · w_j` is dominated by
self-similar terms: dormant i × dormant j = ‖poison‖² for every j in
cohort. This gives (80/190) × ‖poison‖² ≈ +0.42‖poison‖² of positive
signal from the self-similar cohort, overwhelming the ≈−0.13‖poison‖²
of negative signal from the 90 honest active clients. Net cross-product
positive → dormants stay in top-s. **Coordinated identical caching creates
a positive-feedback loop in the detector.**

### 2.8 Baseline coverage (`633bee4`, `c6c4177`, `fb2d968`)

User's next critique: the "coordinated dormancy is a general threat"
framing needs testing against DeMoA (the proposal's promised baseline)
and its whole aggregator family.

Implemented `src/baselines.py` — trainer that shares FedLAW's data /
client infrastructure but replaces the learnable-weights step with a
robust aggregator (Krum, TrMean, Median, CenteredClipping, Bulyan),
with optional DeMoA-style cache + (1−αp) decay.

**Cross-defender comparison table** (all: n=200, q=0.9, p=0.5,
frac=0.0 background, coordinated stealth_lie cohort_80, T_dark=20,
200 rounds, seed=0):

| Defender | Clean | Under c80 T20 | Δ | Verdict |
|---|---|---|---|---|
| FedLAW cache_grad_B_ii | 90.57% | 80.60% | −9.97pp | LANDS |
| DeMoA + CCLIP τ=100 | 90.81% | 81.96% | −8.85pp | LANDS |
| DeMoA + CCLIP τ=50 | 90.81% | 81.91% | −8.90pp | LANDS |
| DeMoA + CCLIP τ=20 | 90.79% | 81.71% | −9.08pp | LANDS |
| DeMoA + TrMean(f=20) | 89.72% | 78.14% | −11.58pp | LANDS |
| DeMoA + TrMean(f=80, matched) | 89.03% | 63.97% | −25.06pp | LANDS |
| DeMoA + Median | 88.98% | 60.03% | −28.95pp | LANDS |
| DeMoA + Bulyan(f=49) | (running) | (running) | — | **pending** |
| DeMoA + Krum | — | — | — | not viable (unstable on q=0.9) |

### 2.9 Framing corrections along the way — accountability

The user forced multiple corrections that improved the honesty of the
work. Each correction was decisive; each was gated on running the
next test rather than proceeding on assumption.

1. **A/B semantic mislabeling** — my Step 1 called weight persistence
   "naive A"; it was actually a B-variant. Fixed by explicit
   participation_mode split.
2. **Simplex constraint violation** — cache_weight_B_i let Σw drift to
   6.7 at p=0.1 f=0.4. Fixed by projecting over full n.
3. **Anti-aligned dormancy is incompetent** — first attack used
   `−mean(honest)` (cos=−0.99, trivially caught). Fixed by stealth_lie
   (cos≈0, evades).
4. **1-client dormancy is not the threat model** — coordinated cohort
   is. Fixed by cohort implementation.
5. **DeMoA-TrMean(f=20) was under-parametrized** — the "DeMoA falls"
   verdict was premature. Re-tested at f=80 (falls harder) and Median
   (falls hardest); framing extended.
6. **CCLIP is DeMoA's PRIMARY aggregator** — test it before claiming
   "general threat". Tested at τ ∈ {20,50,100}; falls uniformly.
7. **Bulyan is the anti-clustering aggregator** most likely to resist.
   Currently testing.

### 2.10 Mechanism summary — why the attack lands

The coordinated stealth cohort dormancy attack exploits **two
compounding properties**:

- **Caching** — every defender that maintains client state across
  absences retains the dormant cohort's influence in the aggregator's
  input. Naive PP (no cache) trivially resists but loses in clean
  accuracy from lost compute.
- **Coordination** — 80 identical cached values concentrate at ONE
  direction per coordinate. This produces:
  - **FedLAW:** positive-feedback in the cross-product detector (each
    dormant sees 79 identical "neighbours" boosting its score).
  - **TrMean:** point mass in the middle of sorted values; TrMean(f)
    only trims extremes, keeps the middle.
  - **Median:** cluster of 80 at one value contains the position-#100.
  - **CCLIP:** 80 identical vectors have identical ‖v−m‖, so all clip
    by the same factor; direction of 40% input dominates the mean
    regardless of τ.

The reproduction phase's flipping_label f=0.4 finding (67% cancellation
because 4 groups had 4 different poisons) is exactly what dormancy
avoids by construction: one coordinated poison, zero cancellation.

---

## 3. Documentation trail

### Top-level docs
- `README.md` — project overview, three fixed gaps, how to run
- `VALIDATION.md` — v3 reproduction report
- `PAPER_FAITHFULNESS.md` — code-to-paper audit
- `REPRODUCTION_STATUS.md` — verified-faithful + characterized gaps + open items
- `WORK_SYNTHESIS.md` — **this document**

### Design docs
- `docs/PARTIAL_PARTICIPATION_DESIGN.md` — §1 participation model, §2
  design ladder A→B→C, §3 dormancy attack, §4 evaluation plan
- `docs/superpowers/specs/2026-06-26-fedlaw-v2-design.md` — v2 design spec
- `docs/superpowers/plans/2026-06-26-fedlaw-v2.md` — v2 9-task plan
- `docs/FedLAW_Handbook.pdf` — project-generated handbook (not external ref)
- `docs/Byzantine Robust Federated (1).pdf` — Wang et al. ICLR 2026 paper
- `docs/Delayed Momentum Aggregation.pdf` — DeMoA baseline reference

### Diagnostic evidence
- `results/paper_fixes/REPORT.md` — master log across all reproduction
  diagnostics (LIE Checks 1/2, raw-gradient falsification, flipping_label
  co-alignment root cause, inverse_gradient detection decay verdict,
  clean-baseline sanity check, seed sensitivity)
- `results/v2/{attack}/q*/frac*/seed0/` — paper-scale runs
- `results/v2_small/` — n=20 smoke tests
- `results/v2/dormancy/` — 1-client dormancy (evades re-scoring, cap
  prevents damage)
- `results/v2/dormancy_cohort/` — coordinated cohort_20/80 (LANDS)
- `results/v2/baselines/` — DeMoA + TrMean/Median/CCLIP/Bulyan

### Diagnostic scripts
- `scripts/diag_flip_n200.py` — flipping_label f=0.4 co-alignment
- `scripts/diag_flip_seedsweep.py` — 5-seed sweep of the finding above
- `scripts/diag_wfreeze.py` — inverse_gradient f=0.1 detection-decay
- `scripts/diag_clean_baseline.py` — clean 200-round baseline
- `scripts/dormancy_smoke.py` / `scripts/stealth_smoke.py` — dormancy variants
- `scripts/step2_horizon.py`, `scripts/step2_naive_A.py` — naive A characterization
- `scripts/demoa_plot.py`, `scripts/demoa_protocol_runs.sh` — DeMoA-protocol batch
- `scripts/dormancy_cohort_batch.sh` — coordinated cohort attack
- `scripts/baseline_batch.sh`, `demoa_median_batch.sh`, `cclip_batch.sh`,
  `bulyan_batch.sh` — cross-defender comparison

### Source
- `src/fedlaw_v2.py` — canonical trainer (learnable weights + participation)
- `src/attacks.py` — 6 paper attacks
- `src/data_partition.py` — Cao q-split + group-oriented Byzantine
- `src/projections.py` — sparse capped simplex (paper Alg. 3)
- `src/models.py` — mlp3_mnist
- `src/baselines.py` — Krum/TrMean/Median/CCLIP/Bulyan + DeMoA cache

### Deprecated (retained for historical comparison in `archive/`)
- v1 trainer (`fedlaw.py`, `aggregators.py`, `loop.py`)
- Pre-FedLAW ByzFL baselines

---

## 4. What's open

### Immediate (blocking framing)
- **Bulyan test** (running at time of writing). Result gates whether
  the "general threat across every standard aggregator" claim is
  earned or whether one aggregator (Bulyan's anti-clustering selection)
  resists.

### Prompt 2 (systematic sweep, gated on Bulyan)
Once framing is confirmed:
- Multi-seed {0,1,2} across all cells
- Cohort size sweep (cohort ∈ {20, 40, 60, 80}) at p=0.5 f=0.4
- Participation rate sweep p ∈ {0.5, 0.25, 0.1} at cohort=80
- T_dark sweep {10, 20, 50, 100}
- Attack payload comparison (stealth_lie vs stealth_honest vs
  inverse_mean control)

### Design C (§2.3 of design doc), gated on Prompt 2
Candidate mechanisms:
- Mutual-similarity penalty in FedLAW cross-product (a client whose
  (cross @ w)[i] is dominated by other-cached self-similar rather
  than active contributions is flagged as suspect)
- Cohort-size-scaled staleness eviction (aggregate cache mass bounded
  globally, not just per-client)
- Anti-clustering post-cache filter before aggregation

### Reproduction items still open
- Multi-seed at n=200 for reproduction cells (all currently seed=0)
- `flipping_label q=0.9 f=0.1` final 10 rounds (currently 86.3% at
  round 190)
- LIE reproduction: additional τ investigation could try the paper's
  specific reference implementation if it becomes available

---

## 5. Honest limitations to record in the writeup

- **Single-seed**: nearly all runs are seed=0. Multi-seed pending.
- **MNIST only**: heterogeneity is via Cao q-split; CIFAR/FEMNIST not
  tested. Ours matches the FedLAW paper's MNIST focus but is not
  broader.
- **Krum unusable on q=0.9**: picks one client per round → whipsaws
  on class-specific gradients. Not a viable comparator; excluded.
- **Bulyan f=49 is the max valid** at n=200 (n≥4f+3 requirement).
  Our 40% attack is beyond Bulyan's design range by construction —
  the test is "even at max-valid f, does Bulyan resist?".
- **DeMoA implementation limitation**: I implemented the cache + decay
  per §3.1/§A.1 of the paper (verified against paper). I did NOT
  implement DeMoA's full training pipeline (momentum, learning-rate
  schedules) — my baseline is a robust aggregator + DeMoA cache
  applied to pseudo-gradients. This is a fair adaptation for MNIST +
  our data, not a bit-exact DeMoA reproduction.
- **CCLIP τ was tested over a 5× range** (20/50/100), not exhaustively.
  Karimireddy's recommended τ ∝ σ√(n/f) not swept.

---

## 6. What the dissertation can claim (this is what the framing supports)

**The reproduction:**
1. FedLAW's mechanism is implemented paper-faithfully; the clean baseline
   trains to within 1pp of the paper's implied clean number.
2. Small-n behaviour reproduces qualitatively for every attack.
3. Paper-scale behaviour reproduces for detected attacks at low
   contamination (f=0.1) and at q=0.6 for inverse_gradient.
4. Three specific configurations produce characterized behaviours of a
   correctly-implemented FedLAW that don't match the paper's number:
   LIE accuracy (τ unresolvable from published info), flipping_label
   f=0.4 (multi-group cancellation, seed-robust), inverse_gradient f=0.1
   (detection decays as model converges). Each has a diagnosed
   mechanism; none is "the paper is wrong" or "our code is broken".

**The contribution:**
5. Under partial participation, FedLAW's naive-A extension is more
   robust than the design doc anticipated at low contamination but
   degrades at cohort scale + high stress.
6. Design B (Option ii, canonical §2.2 with (1−αp)^τ decay) recovers
   full-strength update and wins at f=0.4 by 5pp over naive-A.
7. The coordinated stealth cohort dormancy attack (novel construction,
   grounded in DeMoA §3's "future work: dedicated attacks against the
   delayed momentum principle") defeats all standard caching-based
   Byzantine-resilient FL defenders under partial participation:
   FedLAW's cross-product detector, DeMoA with TrMean / Median /
   CenteredClipping. Bulyan test in progress; expected outside its
   theorem's design range and unlikely to resist at 40% Byzantine.
8. The attack mechanism is unified across defender families: caching
   preserves cohort influence; coordination (identical cached values)
   defeats each aggregator's specific detection primitive via
   point-mass concentration.

**What the dissertation must NOT claim** (per the accountability
corrections along the way):
- Not "the paper is wrong" — the reproduction gaps are characterized
  behaviours, not paper errors.
- Not "1:1 paper reproduction" — three specific cells don't match.
- Not "DeMoA falls in general" — it falls under specific practical
  aggregator instantiations tested here; Bulyan and Krum (unstable)
  aren't in the covered set until the current batch lands.
- Not "coordinated dormancy is a completely new attack" — the category
  (time-coupled / on-off / ALIE-family) is prior art; the contribution
  is novel BY CONSTRUCTION targeting caching-based Byzantine-resilient
  FL specifically, and the empirical characterization across defender
  families.

---

*Living document. Updated after each phase concludes. Next update
scheduled when the Bulyan cell completes.*
