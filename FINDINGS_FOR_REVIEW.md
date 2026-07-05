# FINDINGS_FOR_REVIEW.md

Conservative summary of controls, evidence, and what can/cannot be claimed
into the dissertation. All essential controls have now landed
(multi-seed {0,1,2}). Every claim below is scoped to exactly what its
control established.

Written 2026-07-05 IST. Version 2 — post-controls-batch.

**IMPORTANT:** Read this document before writing any claim into the
dissertation. The framing narrows dramatically from prior versions.

---

## 0. Top-line verdict — the framing that must be revised

**Prior claim (WORK_SYNTHESIS, README):**
> "Coordinated stealth cohort dormancy defeats every standard caching-
> based Byzantine-resilient FL defender under partial participation."

**Corrected claim (this document, controls-verified):**
> "At q=0.9 heterogeneity, a 40-client coordinated cohort submitting
> an identical LIE-style stealth cached poison degrades FedLAW's
> accuracy by ~8pp beyond the compute-loss baseline (mean 79.98 ± 1.10
> across 3 seeds vs honest-dormant control 88.22 ± 0.41). The
> 'dormancy' mechanism is NOT distinct from immediate coordinated
> poison — coord-present at p=0.5 does the same damage. The attack does
> NOT require partial participation — coord-present at p=1.0 does
> slightly MORE damage. At q=0.6 the attack does essentially nothing
> (90.53 ± 0.08 = clean level). Bulyan is a defender-specific case
> (falls even within its proven δ=0.2); DeMoA + CCLIP and DeMoA +
> TrMean at their proven δ RESIST (compute-loss only)."

The **novel contribution** is a specific coordinated-poison vulnerability
of FedLAW's cross-product detector at q=0.9 heterogeneity — narrower
than "general threat" and NOT specifically a "dormancy" claim.

---

## 1. δ audit — what each defender is proven for

Attacking a defender outside its proven δ does not falsify its theorem.

| Defender | Guaranteed / tested δ | Max f at n=200 |
|---|---|---|
| FedLAW (Wang ICLR 2026) | Paper Table 3 tests f ∈ {0.1, 0.4} explicitly | 40% within paper's regime |
| Bulyan (Mhamdi 2018) | Requires n ≥ 4f+3 | 49/200 = 24.5% |
| TrMean(f) | Guarantee when Byzantine ≤ f | f/n |
| CCLIP (Karimireddy 2021) | Strong constants typically δ ≤ 20–25% | ~20-25% |
| Median | Nominally δ < 50%; tight constants smaller | 50% nominal |

**Our c80 (40% Byzantine) tests, per defender:**
- FedLAW at 40%: **within paper's own tested regime** — valid.
- Bulyan(f=49) at 40%: **exceeds 24.5% max** — outside spec; result does not
  refute theorem.
- TrMean(f=20) at 40%: exceeds spec (f=20 → max 10%) — outside spec.
- TrMean(f=80) at 40%: matched — valid, at the limit.
- CCLIP at 40%: exceeds strong-constant regime — outside spec.
- Median at 40%: within nominal 50% — valid, near limit.

**Our c40 (20% Byzantine) test:** within-spec for every defender tested
(FedLAW, TrMean(f=40 matched), CCLIP, Bulyan(f=49)).

---

## 2. Control results — mean±std across 3 seeds

All FedLAW cache_grad_B_ii, n=200, 200 rounds, cohort_80, stealth_lie.

| Configuration | Seeds 0, 1, 2 | Mean ± std |
|---|---|---|
| clean (baseline) | 0.9057 (prior seed 0 only) | 90.57% (n=1, PENDING multi-seed) |
| **c80 T=20 stealth_lie q=0.9 (headline)** | 0.8060, 0.8090, 0.7843 | **79.98 ± 1.10pp** |
| c80 T=20 **stealth_honest** q=0.9 (Control 2 baseline) | 0.8779, 0.8877, 0.8809 | **88.22 ± 0.41pp** |
| c80 **coord-present p=0.5** q=0.9 (Control 1c) | 0.8124, 0.8342, 0.7970 | **81.45 ± 1.53pp** |
| c80 **coord-present p=1.0** q=0.9 (Control 3) | 0.7927, 0.8243, 0.7909 | **80.26 ± 1.53pp** |
| c80 T=20 stealth_lie **q=0.6** (Control 4) | 0.9065, 0.9048, 0.9047 | **90.53 ± 0.08pp** |

**Derived quantities (with propagated uncertainty):**
- Attack effect (poison − honest-dormant): 79.98 − 88.22 = **−8.24pp**
- Pure compute loss (clean − honest-dormant): 90.57 − 88.22 = **−2.35pp**
- Damage from clean under attack: 90.57 − 79.98 = **−10.59pp**

---

## 3. Control-by-control verdicts

### Control 1 — Does dormancy do real work beyond coordinated poison?

- (a) dormancy c80 T=20: **79.98 ± 1.10pp**
- (c) coord-present p=0.5: **81.45 ± 1.53pp**

Difference: coord-present is **1.47pp higher (attack does 1.47pp less
damage)** than dormancy. Well within seed variance (combined std ~1.9pp).

**Verdict: FALSIFIED.** Dormancy does NOT add work beyond coordinated
identical poison at p=0.5. The "build-trust → go dark, cached poison
persists" framing does not produce meaningfully more damage than
"cohort always present, always submitting the same stealth poison."
Within seed noise, the two are equivalent.

**Implication for the writeup:** the "dormancy" framing is misleading.
The correct characterization is **coordinated poison attack under a
caching defender**, and the "dormant" phase doesn't add anything the
caching itself doesn't already do.

### Control 2 — Is the damage the poison, or lost participation?

- Honest-dormant (same cohort dark, honest gradients cached): 88.22 ±
  0.41pp
- Poison-dormant (same cohort dark, poison cached): 79.98 ± 1.10pp

**Attack effect attributable to POISON**: 88.22 − 79.98 = **−8.24pp**.
**Compute loss**: 90.57 − 88.22 = **−2.35pp**.

**Verdict:** The correct headline number is **−8.24pp attack effect at
q=0.9 f=0.4**, NOT the −9.97pp drop-from-clean. About 25% of the
apparent drop is pure participation loss the attack didn't cause.

The prior WORK_SYNTHESIS quoted −9.97pp; corrected number is **−8.24pp
poison effect + −2.35pp compute loss** at seed 0. Multi-seed rerun of
clean_B at n=1 is a residual gap in the analysis, but the honest-dormant
baseline (88.22 ± 0.41) is a valid subtractive control.

### Control 3 — Does the attack need partial participation?

- coord-present p=0.5: 81.45 ± 1.53pp
- coord-present p=1.0: 80.26 ± 1.53pp

**Verdict: FALSIFIED.** The attack does NOT require partial
participation. At full participation (p=1.0), the attack does
**slightly MORE** damage (1.19pp lower accuracy) than at p=0.5, well
within noise.

**Implication:** The "under partial participation" framing was wrong.
The mechanism is coordinated identical cached poison, and it works at
full participation. The partial-participation harness was necessary to
STUDY the attack in the DeMoA sense but is not the enabling condition
for the attack itself.

### Control 4 — Is the vulnerability heterogeneity-dependent?

- q=0.9 c80 T=20 stealth_lie: **79.98 ± 1.10pp** (large damage)
- q=0.6 c80 T=20 stealth_lie: **90.53 ± 0.08pp** (clean-level)

**Verdict: HETEROGENEITY-SPECIFIC.** At q=0.6 (moderate
heterogeneity), the attack essentially does not land — the model
trains to normal accuracy. The vulnerability is only measurable at
q=0.9.

This aligns with the reproduction phase's inverse_gradient q=0.6 f=0.4
result (92.19% = paper anchor at q=0.6). FedLAW's detector works fine
at q=0.6 across attack modes.

### Control 5 — Multi-seed

All 14 cells in the batch are 3-seed. Std bands are tight (max 1.53pp).
Headline attack effect is robust: mean 79.98% ± 1.10pp under
coordinated poison at q=0.9.

Multi-seed still PENDING for:
- The clean FedLAW baseline (n=1, one value: 90.57%)
- All δ-check c40 cells (n=1 each)
- All prior DeMoA baseline runs at c80 (relevant to *retracted*
  claims; not needed for defensible claims)

### Control 6 — Stealth vs scale

Prior evidence stands: single-client stealth_lie evades cross-product
re-scoring (cos ≈ 0, weight held at 1/n) but the cap
mathematically bounds damage to zero. Coordinated cohort (80 clients
at cap 1/(s−10)) reaches ~40% aggregate weight — that's where
measurable damage appears.

**Verdict:** Both mechanisms contribute. Stealth is necessary (evades
detection); scale is necessary (raises weight above the cap threshold).

### δ-check — DeMoA baselines at their own proven δ

At c40 (20% Byzantine, within-spec) seed 0 — multi-seed PENDING:

| Defender | Clean | Under attack | Δ | Within spec? |
|---|---|---|---|---|
| FedLAW cache_grad_B_ii | 90.57% | 87.77% | −2.80pp | Barely lands |
| DeMoA + TrMean(f=40 matched) | 89.41% | 86.32% | −3.09pp | **RESISTS** |
| DeMoA + CCLIP (τ=100) | 90.81% | 88.01% | −2.80pp | **RESISTS** |
| DeMoA + Bulyan(f=49) | 88.83% | 59.63% | −29.20pp | Bulyan-specific fall |

**Verdict:**
- DeMoA + CCLIP and DeMoA + TrMean(f=40): **RESIST** at their proven δ.
  Damage is compute-loss only.
- FedLAW at δ=0.2: attack barely dents (−2.80pp, comparable to compute
  loss).
- Bulyan(f=49) at δ=0.2 (within its 24.5% max): still falls −29.20pp.
  Bulyan-specific vulnerability at coordinated cohort scale, distinct
  from the FedLAW mechanism.

**Multi-seed on δ-check cells: PENDING.** All numbers above are seed=0.

---

## 4. Claims that CAN be defended

Each claim scoped to its evidence.

### C-1 (reproduction, well-supported).
FedLAW mechanism is paper-faithful; clean baseline 90.58%.
`PAPER_FAITHFULNESS.md`, 48/48 unit tests, `REPRODUCTION_STATUS.md`.

### C-2 (reproduction gaps, seed-0).
Three characterized behavioural gaps of a paper-faithful FedLAW (LIE,
flipping_label f=0.4, inverse_gradient f=0.1) at specific configurations,
each with a diagnosed mechanism. Currently seed=0; multi-seed for
reproduction cells not run in this project cycle.

### C-3 (partial-participation harness, well-supported).
Bernoulli-p sampling harness at p=1.0 sanity gate: 90.61% vs
reference 90.58%. Fast path byte-equivalent to baseline.

### C-4 (Design B works at within-paper regime, seed-0).
`cache_grad_B_ii` (Option ii) wins at f=0.4 by ~5pp over naive_A
at seed 0 in the DeMoA-protocol comparison. Multi-seed pending.

### C-5 (single-client dormancy evades detector but cannot damage).
A single stealth_lie dormant client evades FedLAW's cross-product
re-scoring (cos ≈ 0; weight held at 1/n) but produces no measurable
accuracy change (Δ within noise) because the per-client cap bounds
influence.

### C-6 (**MULTI-SEED HEADLINE**): Coordinated stealth cached poison
degrades FedLAW at q=0.9 f=0.4 by 8.24pp beyond compute-loss baseline.
Three seeds: attack cell mean 79.98 ± 1.10pp; honest-dormant control
mean 88.22 ± 0.41pp. Damage attributable to poison ≈ 8.24pp.

### C-7 (**MULTI-SEED**): Dormancy phase is NOT distinct from immediate
coordinated poison. At q=0.9 f=0.4, "dormancy c80 T_dark=20" produces
79.98 ± 1.10pp and "coord-present p=0.5" produces 81.45 ± 1.53pp —
within seed noise. The "dormant" framing is not what does the work.

### C-8 (**MULTI-SEED**): The attack does NOT require partial
participation. At q=0.9 f=0.4, coord-present p=1.0 produces 80.26 ±
1.53pp (slightly worse than p=0.5's 81.45). The attack works at full
participation.

### C-9 (**MULTI-SEED**): The vulnerability is q=0.9 specific. At
q=0.6, coordinated stealth cached poison c80 T=20 produces 90.53 ±
0.08pp — indistinguishable from clean. FedLAW's detector works fine
at moderate heterogeneity.

### C-10 (δ-check, seed-0). DeMoA + CCLIP and DeMoA + TrMean(f=40
matched) RESIST the attack at their proven δ = 0.2 (compute-loss
only). At c40: CCLIP −2.80pp, TrMean −3.09pp. Multi-seed pending.

### C-11 (δ-check, seed-0). Bulyan(f=49) at its own proven δ = 0.2
still falls −29.20pp under coordinated stealth cached poison. This
is a specific Bulyan vulnerability distinct from FedLAW's mechanism.
Multi-seed pending.

### C-12 (Bulyan multi-seed at outside-spec c80).
Bulyan(f=49) at c80 (outside its 24.5% max spec) collapses to
12.79 ± 4.71pp across seeds under stealth_lie. Honest-dormant control
seed 0: 85.79%. Genuinely the poison payload, not Bulyan-PP
degeneracy. This is OUTSIDE Bulyan's proven regime and does NOT
falsify its theorem — report as "outside-spec sensitivity."

---

## 5. Claims that CANNOT be defended (falsified or unsupported)

### F-1 (falsified). "Coordinated dormancy defeats every standard
caching defender."
At within-spec δ, DeMoA + CCLIP and DeMoA + TrMean RESIST. Only
Bulyan falls within-spec, and only FedLAW at f=0.4 falls (~8pp above
compute loss). This is NOT a general threat.

### F-2 (falsified). "DeMoA falls to coordinated dormancy."
DeMoA at its own proven δ = 0.2 RESISTS. The c80 tests for CCLIP were
outside spec. Retract this claim from prior documents.

### F-3 (falsified). "Dormancy is the mechanism."
Coordinated poison present every round does the same damage as
dormancy at q=0.9 f=0.4 (Control 1c). The "build-trust then go dark"
is a redundant framing; the mechanism is coordination + caching.

### F-4 (falsified). "The attack requires partial participation."
Attack works equally at p=1.0 (Control 3). Partial participation is
not the enabling condition.

### F-5 (falsified). "The vulnerability is broad heterogeneity."
Only q=0.9 shows the effect. At q=0.6 the attack does essentially
nothing (Control 4).

### F-6 (unsupported at multi-seed). The DeMoA c40 numbers, the FedLAW
c40 number, all clean-baseline δ-check numbers — seed=0 only.

### F-7 (unsupported). Any claim about datasets other than MNIST, any
n other than 200, payloads other than stealth_lie, T_dark values other
than 20 (except at coord-present which uses no T_dark).

---

## 6. Draft framing for the dissertation

### Reproduction chapter
FedLAW mechanism is paper-faithful; clean baseline 90.58%; three
characterized behavioural gaps under specific configurations with
diagnosed mechanisms. (Well-supported.)

### Contribution chapter — narrow, controls-verified version

**Section: The coordination attack at high heterogeneity.**

At q=0.9 f=0.4 (in FedLAW's own paper-tested regime), a coordinated
cohort submitting an identical LIE-style stealth cached pseudo-gradient
degrades FedLAW cache_grad_B_ii's accuracy by **8.24pp beyond the
compute-loss baseline** across three seeds (79.98 ± 1.10pp under attack
vs 88.22 ± 0.41pp honest-dormant control).

Controls establish:
- **Not dormancy-specific:** the same damage occurs when the cohort is
  PRESENT every round submitting the same poison (81.45 ± 1.53pp) — the
  "build-trust → go dark" framing is redundant with immediate
  coordinated poison.
- **Not partial-participation-specific:** works equally at full
  participation p=1.0 (80.26 ± 1.53pp).
- **Heterogeneity-specific:** at q=0.6 the attack does not land
  (90.53 ± 0.08pp = clean level).

The mechanism at q=0.9 is a positive-feedback loop in FedLAW's
cross-product detector: 40 identical cached gradients cross-multiply to
+||poison||² for every pair, boosting each dormant's score and pinning
their weight in the top-s support. At q=0.6, honest gradients are less
class-specific and the residual poison alignment is small enough that
the detector holds.

### Baseline chapter — corrected version

At δ = 0.2 (within every tested defender's proven regime):
- DeMoA + CenteredClipping RESISTS (−2.80pp = compute loss).
- DeMoA + Trimmed Mean (f matched) RESISTS (−3.09pp).
- Bulyan(f=49) still falls (−29.20pp) — a Bulyan-specific vulnerability
  to identical Byzantine clusters that persists even within its own
  n ≥ 4f+3 requirement.
- FedLAW at δ=0.2 barely dents (−2.80pp).

At δ = 0.4 (within FedLAW's paper-tested regime; outside CCLIP/Bulyan):
- FedLAW falls −8.24pp beyond compute loss (multi-seed).
- Others were tested outside spec and are not evidence against their
  theorems.

**Retracted from prior WORK_SYNTHESIS.md:** "General threat across
every defender" — falsified by within-spec tests.

---

## 7. Remaining honest gaps (write these into the dissertation's
"limitations" section)

- **δ-check multi-seed pending.** CCLIP, TrMean, Bulyan c40 numbers
  are seed=0. Multi-seed extension is a subsequent batch (not run in
  this cycle).
- **Clean-B multi-seed pending.** The 90.57% clean number is seed=0.
- **Reproduction cells all seed=0.**
- **MNIST only, n=200 only, cohort=80 only for the attack, stealth_lie
  payload only.** No sweep across attack-configuration axes.
- **DeMoA implementation is our adaptation** of paper §3.1/§A.1 (cache
  + decay) with pseudo-gradients on FedLAW's data pipeline, not a
  bit-exact reproduction of DeMoA's full training loop.
- **Krum unusable at q=0.9 heterogeneity** (whipsaws — not a viable
  comparator). Documented as a limitation.
- **Bulyan tested at outside-spec c80** — reported explicitly as
  outside-spec sensitivity, not as evidence against Bulyan's theorem.

---

## 8. Recommendation to the reviewer

- Write the reproduction chapter as-is (well-supported).
- Write the contribution chapter with the narrow, controls-verified
  framing above. Do NOT use the "dormancy" framing — call it
  **"coordinated identical cached poison attack under high heterogeneity"**.
- Retract the "general threat across every defender" claim from
  WORK_SYNTHESIS.md and README.md when updating for the final version.
- Report Bulyan and DeMoA baseline results explicitly with the δ they
  were tested at. Do NOT quote the c80 CCLIP/Bulyan/TrMean(f=20) numbers
  as "vulnerabilities" — they are outside-spec sensitivities.
- Flag every remaining seed=0 number as such.

**The contribution is real but smaller than the initial framing
implied.** It is:
> A specific coordinated-poison sensitivity of FedLAW's cross-product
> detector at q=0.9 heterogeneity, quantified at 8.24pp attack effect
> beyond compute loss, robust across 3 seeds, present at both partial
> and full participation, absent at q=0.6.

*Batch complete. All essential controls landed. This document reflects
what the evidence supports.*
