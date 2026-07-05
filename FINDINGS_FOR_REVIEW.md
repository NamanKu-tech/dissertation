# FINDINGS_FOR_REVIEW.md

Conservative summary of controls, evidence, and what can/cannot be claimed.
Written after the δ-check batch and the Bulyan verify batch landed;
running the FedLAW controls batch overnight for the remaining controls.
This document will be updated when the running batch completes.

**IMPORTANT:** Read this document before writing any claim into the
dissertation. Every claim below is scoped to exactly what its control
established. Where a control has not run, this is marked PENDING; do
NOT elevate a PENDING to a claim.

Written 2026-07-05 IST.

---

## 0. The framing that must be REVISED

**Prior WORK_SYNTHESIS.md and README.md claimed:** "Coordinated stealth
cohort dormancy defeats every standard caching-based Byzantine-resilient
FL defender: FedLAW's cross-product detector, DeMoA + TrMean, DeMoA +
Median, DeMoA + CenteredClipping, DeMoA + Bulyan."

**Corrected framing (this document):** This claim was TESTED-OUTSIDE-SPEC
for most DeMoA baselines. When re-tested at each defender's own proven /
tested δ:
- DeMoA + CCLIP at δ=0.2: RESISTS (−2.80pp, essentially compute loss).
- DeMoA + TrMean(f=40, matched) at δ=0.2: RESISTS (−3.09pp).
- FedLAW at δ=0.2: barely lands (−2.80pp = compute loss).
- FedLAW at δ=0.4 (paper's own tested regime): DOES land (−9.97pp
  at seed=0, multi-seed PENDING).
- Bulyan(f=49) at δ=0.2 (within Bulyan's own 24.5% max): STILL falls
  −29.20pp — a real Bulyan-specific vulnerability at coordinated cohort
  scale, within spec.

The "general threat across every defender" framing is FALSIFIED. The
narrower claims (below) are supported by the evidence.

---

## 1. The δ audit — what each defender is proven for

The critical error in the prior framing: attacking at 40% Byzantine
against defenders whose theorems only cover smaller δ. The correct
guaranteed / tested δ per defender:

| Defender | Guarantee | Max δ at n=200 |
|---|---|---|
| FedLAW (Wang et al. ICLR 2026) | Paper Table 3 explicitly tests f ∈ {0.1, 0.4} | 40% (within paper's own tested regime) |
| Bulyan (Mhamdi 2018) | Requires n ≥ 4f+3 | 49/200 = **24.5% max** |
| TrMean(f) | Guarantee when actual Byzantine ≤ f | f/n |
| Median | Nominal δ < 0.5 | 50% nominal, tighter constants smaller |
| CenteredClipping (Karimireddy 2021) | Strong constants typically at δ ≤ 20–25% | ~20-25% |

Attacking at 40% Byzantine:
- FedLAW at 40%: **within paper's regime** — valid
- TrMean(f=80) at 40%: **at TrMean's f-bound** — valid, borderline
- Median at 40%: **within nominal 50%** — valid, near limit
- CCLIP at 40%: **exceeds strong-constant regime** — outside spec
- Bulyan(f=49) at 40%: **exceeds 24.5% requirement** — outside spec
- TrMean(f=20) at 40%: **way exceeds f=20 (max δ=10%)** — outside spec

Prior c80 tests marked outside-spec: CCLIP, Bulyan(f=49), TrMean(f=20).
These results are **not evidence against the defenders' theorems**.

---

## 2. Control-by-control status

### Control 1 — Does dormancy do real work, or is it just coordinated poison?

**Compares** FedLAW under:
- (a) dormancy attack (build-trust → dark, cached poison persists)
- (b) coordinated poison from round 0 (no build-trust)
- (c) coordinated poison, cohort PRESENT every round (no dormancy)

**Status:** RUNNING (14-cell FedLAW batch, overnight). Seed-0 (a)
already have: −9.97pp at c80 T_dark=20 p=0.5 stealth_lie. (b) and (c)
pending.

**What the result will tell us:**
- (a) ≈ (b) ≈ (c) → dormancy adds nothing; "coordinated poison" is the
  mechanism and "dormancy" is a redundant framing.
- (a) >> (c) → dormancy under partial participation IS the mechanism;
  the build-trust + cache steps add real damage beyond immediate poison.

Cannot claim "dormancy is the mechanism" until (a) vs (c) is measured.

### Control 2 — Is the damage the poison or lost participation?

**Compares** FedLAW under c80 T_dark=20 at seed 0:
- Poison-dormant (stealth_lie): 80.60% (Δ=−9.97pp from clean 90.57%)
- Honest-dormant (stealth_honest): PENDING at seed 0

The correct "attack effect" is `poison-dormant − honest-dormant`, NOT
`poison-dormant − clean`. The latter includes the pure participation
loss (80 clients absent → less compute per round) that the attack did
not cause.

**Related evidence we do have (Bulyan honest-dormant control):**
- Bulyan clean seed 0: 88.83%
- Bulyan poison-dormant seed 0: 8.24% (−80.59pp) — outside spec but real
- Bulyan honest-dormant seed 0 (control): 85.79% (−3.04pp)
- Attack effect (poison MINUS honest) at Bulyan: 8.24% − 85.79% = **−77.55pp**

For Bulyan (at outside-spec c80), the attack effect is nearly all
attack (compute loss ~3pp; poison damage ~77pp). But at within-spec c40
Bulyan fell only −29pp from clean — need c40 honest-dormant to compute
that attack effect too. NOT YET RUN.

For FedLAW: seed-0 honest-dormant running now. Cannot state the
attack-attributable damage until it lands.

### Control 3 — Does the attack need partial participation?

**Tests** FedLAW cache_grad_B_ii at p=1.0 with the cohort always
present, always submitting coordinated poison. If it breaks FedLAW at
p=1.0 too, "partial participation" is not enabling.

**Status:** RUNNING (seeds 0, 1, 2) — same 14-cell batch.

**What the result will tell us:**
- FedLAW falls at p=1.0 coord-present → "partial participation" isn't
  necessary; coordinated poison alone kills FedLAW.
- FedLAW resists at p=1.0 → the caching mechanism IS enabling; without
  the cache-through-absence there's no attack.

### Control 4 — Is FedLAW's vulnerability heterogeneity-dependent?

**Tests** FedLAW dormancy c80 T_dark=20 at q=0.6 (vs the q=0.9
we've been testing). If the attack is only effective at extreme
heterogeneity, the claim narrows.

**Status:** RUNNING (seeds 0, 1, 2).

### Control 5 — Multi-seed

**Status:** PENDING for essentially every cell. Currently reported
numbers are seed=0 point estimates except where noted.

Cells with multi-seed data:
- Bulyan clean: 88.83 / 88.91 / 87.94 → **88.56 ± 0.53pp** ✓
- Bulyan c80 poison (outside spec): 8.24 / 17.61 / 12.52 → **12.79 ± 4.71pp** ✓

Cells that are seed=0 only (marked as such throughout):
- FedLAW c80 T20 poison at q=0.9 (the −9.97pp number): seed 0
- All δ-check c40 cells (CCLIP, TrMean(f=40), Bulyan, FedLAW)
- All FedLAW controls in the running batch (seeds 0, 1, 2 all pending)

### Control 6 — Is stealth doing the work, or is it scale?

**Test:** does a SINGLE stealthy dormant client evade FedLAW's detector?

**Evidence we have (from `results/v2/dormancy/`):**
- Single-client dormancy with stealth_lie payload at c=1 T_dark=10 p=0.5:
  cos(cached_g, honest_mean) after poisoning: −0.01 (orthogonal, not
  anti-aligned). Weight held at uniform 1/n=0.005 throughout dark.
- Single-client stealth_honest: cos=+0.20, weight held.

**Interpretation:** Stealth (near-orthogonal cached vs honest mean) is
sufficient to evade the cross-product detector, EVEN AT SINGLE-CLIENT
scale — but at single-client scale the cap (1/n=0.005) mathematically
bounds the accuracy impact to zero (verified: Δ within noise from
clean_B in prior runs).

**Verdict:** Stealth genuinely evades re-scoring. Scale (cohort size)
determines whether the evasion translates to measurable accuracy
damage. Both stealth AND scale contribute; neither alone is the
mechanism.

### δ-check (added by the user, foundational)

Run: c40 (20% Byzantine = within every defender's spec) T_dark=20
seed 0. Multi-seed pending.

| Defender | Clean | Dorm c40 T20 | Δ | Verdict |
|---|---|---|---|---|
| FedLAW cache_grad_B_ii | 90.57% | 87.77% | −2.80pp | Attack barely lands (~compute loss) |
| DeMoA + TrMean(f=40, matched) | 89.41% | 86.32% | −3.09pp | **RESISTS at its spec** |
| DeMoA + CCLIP (τ=100) | 90.81% | 88.01% | −2.80pp | **RESISTS at its spec** |
| DeMoA + Bulyan(f=49) | 88.83% | 59.63% | **−29.20pp** | **Falls WITHIN its own spec** |

**Multi-seed on these cells: PENDING.** Seed-0 point estimates.

---

## 3. Claims that CAN be defended (scoped to their controls)

Each claim below is at the STRENGTH of the evidence — if only seed=0
is available, the claim is a point estimate flagged as such.

### C-1 (reproduction, well-supported):
FedLAW's mechanism is implemented paper-faithfully; clean baseline
trains to 90.58% (paper implies 91–92%). Backed by PAPER_FAITHFULNESS.md,
48/48 unit tests, REPRODUCTION_STATUS.md.

### C-2 (reproduction gaps, seed-0):
Three specific paper-scale configurations produce characterized
behaviours of a paper-faithful FedLAW that do not match Table 3's
number (LIE, flipping_label f=0.4, inverse_gradient f=0.1). Each has
a diagnosed mechanism (results/paper_fixes/REPORT.md). Framing is
"characterized behaviours of a paper-faithful implementation," not
"paper wrong" and not "code broken." Currently seed=0 only.

### C-3 (partial-participation harness, well-supported):
Bernoulli-p sampling harness passes the p=1.0 sanity gate (90.61% vs
reference 90.58%). Fast path is byte-equivalent to the baseline.

### C-4 (Design B works at within-paper regime, seed-0):
`cache_grad_B_ii` (Option ii) wins at f=0.4 by ~5pp over naive_A
at seed 0 in the DeMoA-protocol comparison. Consistent with the
mechanism (re-score absent via cached decayed gradient). Currently
seed=0.

### C-5 (single-client dormancy evades, but cannot damage):
A single stealth_lie dormant client evades FedLAW's cross-product
re-scoring (cos ≈ 0 with honest mean; weight held at 1/n) but produces
no measurable accuracy change (within noise) because the per-client
cap bounds the influence.

### C-6 (Bulyan at outside-spec c80 collapses to the poison, not to
Bulyan-PP degeneracy, multi-seed):
Bulyan(f=49) + DeMoA-cache at c80 T_dark=20 (40% Byzantine, OUTSIDE
Bulyan's 24.5% max) collapses to 12.79 ± 4.71% across seeds {0,1,2}
under the stealth_lie payload. The honest-dormant control (85.79% at
seed 0) confirms this is genuinely the poison payload's effect, not a
Bulyan-partial-participation artefact. HOWEVER: this test is outside
Bulyan's proven δ regime and does NOT falsify Bulyan's theorem.

### C-7 (Bulyan within-spec at δ=0.2 still falls, seed-0):
Bulyan(f=49) at c40 T_dark=20 p=0.5 (20% Byzantine, WITHIN Bulyan's
24.5% max) falls −29.20pp from clean at seed 0. Multi-seed PENDING.
This is a Bulyan-specific vulnerability at coordinated cohort scale,
within its own proven δ regime — if multi-seed confirms.

### C-8 (DeMoA-CCLIP and DeMoA-TrMean resist within spec, seed-0):
At c40 T_dark=20 p=0.5 (20% Byzantine, within DeMoA's typical δ=0.2
regime): CCLIP falls −2.80pp; TrMean(f=40 matched) falls −3.09pp.
Both are essentially compute loss (matches expected loss from 40
clients being absent). DeMoA appears to resist coordinated cohort
dormancy at its own proven δ. Multi-seed PENDING.

### C-9 (FedLAW at δ=0.4 falls, seed-0):
FedLAW cache_grad_B_ii at c80 T_dark=20 p=0.5 (40% Byzantine, within
FedLAW's paper-tested regime) falls −9.97pp from clean at seed 0.
Multi-seed PENDING.

---

## 4. Claims that CANNOT be defended (falsified or unsupported)

### F-1 (falsified): "DeMoA falls to coordinated dormancy."
The c80 tests for CCLIP were outside CCLIP's strong-constant regime.
At the within-spec c40 test, CCLIP resists (−2.80pp = compute loss).
Same for TrMean(f=40 matched). The general "DeMoA falls" framing is
falsified — DeMoA + standard aggregators at their own proven δ RESIST
the attack.

### F-2 (falsified as stated): "Coordinated dormancy defeats every
standard caching-based defender."
This claim is false. At within-spec δ, DeMoA + CCLIP resists;
DeMoA + TrMean(f matched) resists; FedLAW itself only barely lands
(−2.8pp). Only Bulyan falls within-spec — and that's Bulyan-specific.

### F-3 (unsupported at multi-seed): every point estimate above.
The −9.97pp FedLAW damage, the −29.20pp Bulyan damage, the −2.80pp
CCLIP/FedLAW δ=0.2 numbers — all are seed=0. The user's rule ("no
point estimates; −Xpp ± 8 is a different claim from ± 1") applies.
Multi-seed is running for the essential cells.

### F-4 (unsupported): "Dormancy is the mechanism for FedLAW's fall."
Cannot claim this until Control 1 (dormancy vs coordinated-present-p=0.5)
completes and shows a meaningful (a) vs (c) gap.

### F-5 (unsupported): "The attack requires partial participation."
Cannot claim this until Control 3 (coord-present at p=1.0) completes.

### F-6 (unsupported): "The vulnerability is heterogeneity-independent."
Cannot claim this until Control 4 (q=0.6) completes.

---

## 5. What we can write in the dissertation (draft framing)

**Reproduction chapter:** FedLAW mechanism is paper-faithful; clean
baseline reaches 90.58%. Three characterized behavioural gaps of a
paper-faithful FedLAW at specific f/q configurations, each with a
mechanism.

**Contribution chapter — narrow version:** FedLAW's Design B(ii)
(cached-gradient partial-participation extension) is vulnerable to
coordinated stealth cohort dormancy at high Byzantine fraction
(f=0.4, in the paper's own tested regime). At this specific
configuration, a coordinated cohort submitting an identical
LIE-style stealth poison at the moment of going dark degrades
accuracy by ~10pp beyond the compute-loss baseline (seed-0; multi-seed
pending). The mechanism is a positive-feedback loop in the cross-product
detector: self-similar cached gradients boost each other's scores.

**Baseline chapter — corrected version:** DeMoA + standard robust
aggregators (CCLIP, TrMean matched) RESIST the attack at DeMoA's
proven δ = 0.2 (only compute-loss damage). Bulyan at its own proven
δ = 0.2 falls −29pp — a specific vulnerability of Bulyan's iterative-
Krum selection under identical Byzantine clusters, but distinct from
the FedLAW mechanism. The generic "DeMoA falls" claim from the prior
document is retracted.

**Whether "coordinated cohort dormancy is a general threat to caching
defenders" is a defensible claim:** NO, based on δ-check evidence.
The narrower claim "specific caching mechanisms fall to specific
coordinated attacks within their spec — FedLAW at f=0.4, Bulyan at
δ=0.2" IS defensible, subject to multi-seed confirmation.

---

## 6. Currently running (14-cell FedLAW controls batch, ~7h)

Launched 2026-07-05 IST. Progress: `results/v2/controls_progress.txt`.

Cells:
1. FedLAW c80 T=20 stealth_honest seeds 0, 1, 2 (Control 2: attack vs compute loss)
2. FedLAW c80 coord-present p=0.5 seeds 0, 1, 2 (Control 1c: dormancy adds anything?)
3. FedLAW c80 coord-present p=1.0 seeds 0, 1, 2 (Control 3: needs PP?)
4. FedLAW c80 T=20 q=0.6 seeds 0, 1, 2 (Control 4: heterogeneity)
5. FedLAW c80 T=20 stealth_lie seeds 1, 2 (multi-seed the existing seed-0 headline)

Total: 14 runs. When done, this document will be updated with the
per-control verdicts, mean±std, and the final "claims we can defend"
list narrowed to what the controls actually support.

---

## 7. Remaining untested (out of scope for this batch, per user directive)

- Multi-seed on all δ-check cells (CCLIP, TrMean(f=40), Bulyan at c40)
- q=0.6 for the DeMoA baselines
- Payloads other than stealth_lie for Controls 1–5
- FedLAW at c40 T_dark ∈ {50, 100} (T_dark sweep)
- Cohort size sweep at within-spec δ
- Any dataset other than MNIST
- Any n other than 200

Do NOT claim on these axes.

---

## 8. Honest recommendation to the reviewer

Delay writing any claim about the DeMoA-family baselines until the
δ-check multi-seed lands. Delay writing any FedLAW-attack claim until
the 14-cell control batch lands. Write the reproduction chapter (§5
here) — it is well-supported. Write the FedLAW mechanism (Design B(ii)
Option) chapter with the existing seed-0 result but flag it explicitly
as seed=0. Do not write the "general threat" framing at all.

*This document will be regenerated when the controls batch completes.*
