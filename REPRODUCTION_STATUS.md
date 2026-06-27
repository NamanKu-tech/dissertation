# REPRODUCTION_STATUS.md

Consolidated test and reproduction status for the FedLAW reproduction.
Pulled from existing reports (`PAPER_FAITHFULNESS.md`,
`results/paper_fixes/REPORT.md`, `VALIDATION.md`) and one fresh `pytest`
invocation. No new training runs.

Last updated: 2026-06-27.

## Three-line honest summary

1. **Mechanism: VERIFIED-FAITHFUL** — every numbered FedLAW component
   (gradient definition, weight update, cap, clipping, q-partition,
   selection, attacks, architecture) is paper-faithful per
   `PAPER_FAITHFULNESS.md` and supported by 48/48 passing unit tests.
2. **Detected attacks reproduce behaviourally** — at n=20 (small-n) and at
   n=200 frac=0.1 for flipping_label; Byzantine weights driven to or near
   zero, model recovers. No clean paper-scale completed run with multi-seed
   means.
3. **Two characterized gaps** — LIE accuracy (33% with Baruch z vs paper
   70%, cause "could not be resolved from published information") and
   flipping_label at frac=0.4 (~65% vs 87.45%, multi-group cancellation
   reproducible across 5 seeds, "genuine FedLAW property OR unidentified
   config difference — undetermined").

**No claim of 1:1 numerical reproduction.**

---

## Step 1 — Unit test suite

> *Unit tests verify code-correctness against our own specification; they
> do NOT by themselves establish reproduction of the paper's numbers.*

`pytest tests/ -v` (run 2026-06-27, single invocation):

```
48 passed in 48.34s
```

Per-file breakdown:

| File | Tests | Passed | Scope |
|---|---|---|---|
| `tests/test_attacks.py` | 15 | 15 | All 6 attack classes — mapping correctness, shape, count, round-gating |
| `tests/test_data_partition.py` | 7 | 7 | Cao q-split (q=1 concentration, q=0.1 spread, sizes) + group-oriented selection |
| `tests/test_fedlaw_v2.py` | 9 | 9 | `_clip_gradients`, `_collect`, end-to-end `run()` invariants |
| `tests/test_projections.py` | 17 | 17 | Capped-simplex + sparse-capped-simplex (sum=1, bounds, sparsity, feasibility) |

Not part of pytest:

- `tests/diagnose_fedlaw.py` — 4-part mechanism audit script, run-once
  diagnostic (not pass/fail). Generated `results/validation_v2/REPORT.md`
  evidence.

**Status: all tests pass. No failures to list.**

---

## Step 2 — Reproduction status table

Status legend (from your spec):

- **VERIFIED-FAITHFUL** — code matches paper spec, confirmed by audit + test
- **REPRODUCED (behaviour, small-n)** — qualitative match at n=20, not paper scale
- **REPRODUCED (single-seed, paper-scale)** — matches Table 3 at n=200, seed=0 only
- **CHARACTERIZED GAP** — does not match paper number; cause documented
- **OPEN / INCOMPLETE** — run not finished or multi-seed not done

### Mechanism components

| Component | Paper ref | Status | Evidence |
|---|---|---|---|
| Gradient definition (pseudo-grad) | Algorithm 1 | **VERIFIED-FAITHFUL** | `PAPER_FAITHFULNESS.md` §1 (`fedlaw_v2.py:266–321`); `test_fedlaw_v2::test_collect_returns_correct_shapes`; `results/paper_fixes/REPORT.md` §Gap 1 |
| Weight update (two-round) | Algorithm 2 | **VERIFIED-FAITHFUL** *(with w-freeze departure)* | `PAPER_FAITHFULNESS.md` §2 (`fedlaw_v2.py:395–414`); w-freeze documented as v2 efficiency extension; sum_byz stabilises by round 6 in all observed configs, so freeze at round 20 has no behavioural effect on results we report |
| Cap and sparsity | Table 1 | **VERIFIED-FAITHFUL** | `PAPER_FAITHFULNESS.md` §3 (`fedlaw_v2.py:233–239`); `test_projections` (17 tests); small-n slack guard noted |
| Server-side ℓ2 clipping | Assumption E1 | **VERIFIED-FAITHFUL** | `PAPER_FAITHFULNESS.md` §4 (`fedlaw_v2.py:71–86`); `test_fedlaw_v2::test_clip_*` (4 tests) |
| Data partition (Cao q-split) | §I.1 | **VERIFIED-FAITHFUL** | `PAPER_FAITHFULNESS.md` §5 (`data_partition.py:8–46`); `test_data_partition` (4 tests) |
| Malicious selection (group-oriented) | §I.1 | **VERIFIED-FAITHFUL** | `PAPER_FAITHFULNESS.md` §6 (`data_partition.py:49–73`); `test_select_malicious_group_oriented`; runtime-verified in diagnostic D2 |
| Architecture (3-layer MLP) | §5.1 | **VERIFIED-FAITHFUL** | `mlp3_mnist` (784→200→100→10) matches paper §5.1 "3-layer fully connected network on MNIST" |

### Attacks (implementation only)

| Attack | Status | Evidence |
|---|---|---|
| flipping_label | **VERIFIED-FAITHFUL** | `PAPER_FAITHFULNESS.md` §7a; D1 mapping verified at runtime (`results/paper_fixes/REPORT.md`) |
| inverse_gradient | **VERIFIED-FAITHFUL** | `PAPER_FAITHFULNESS.md` §7b; `test_inverse_gradient_negates` |
| backdoor | **VERIFIED-FAITHFUL** | `PAPER_FAITHFULNESS.md` §7c (per-example random target matches paper §I.1 "randomly changed to a label between 0 and L−1") |
| double | **VERIFIED-FAITHFUL** | `PAPER_FAITHFULNESS.md` §7d; `test_double_attack_round*` (3 tests) |
| LIE | **VERIFIED-FAITHFUL** | `PAPER_FAITHFULNESS.md` §7e; pseudo-gradient μ/σ confirmed |

### Per-attack experimental status

Configs that have actually been **run** (read from `results/v2/.../metrics.csv`
and `results/v2_small/.../`):

| Attack | Config | n | Rounds completed | Final acc | Paper Table 3 | Status |
|---|---|---|---|---|---|---|
| flipping_label | q=0.9, f=0.4, seed=0 | 200 | 200 / 200 | 65.4% | 87.45% | **CHARACTERIZED GAP** (see §3 below) |
| flipping_label | q=0.6, f=0.4, seed=0 | 200 | 200 / 200 | 54.7% | 92.22% | **CHARACTERIZED GAP** (same mechanism) |
| flipping_label | q=0.9, f=0.1, seed=0 | 200 | 190 / 200 | 86.3% | ~89–90% | **REPRODUCED (single-seed, paper-scale)** — ~3pp short; trajectory still climbing at last eval |
| flipping_label | all attacks, small-n | 20 | 30 / 30 | n/a | n/a | **REPRODUCED (behaviour, small-n)** — detection works; Byzantine weights driven to 0 at n=20 |
| inverse_gradient | q=0.9, f=0.4, seed=0 | 200 | 20 / 200 (interrupted) | n/a | 87.41% | **OPEN / INCOMPLETE** |
| inverse_gradient | q=0.6, f=0.4, seed=0 | 200 | 20 / 200 (interrupted) | n/a | 91.62% | **OPEN / INCOMPLETE** |
| inverse_gradient | q=0.9, f=0.1, seed=0 | 200 | 20 / 200 (interrupted) | n/a | ~89–90% | **OPEN / INCOMPLETE** |
| inverse_gradient | small-n smoke | 20 | done | n/a | n/a | **REPRODUCED (behaviour, small-n)** |
| backdoor | q=0.9, f=0.4, seed=0 | 200 | 10 / 200 (interrupted) | n/a | 87.88% | **OPEN / INCOMPLETE** |
| backdoor | q=0.6, f=0.4, seed=0 | 200 | 10 / 200 (interrupted) | n/a | (not anchored) | **OPEN / INCOMPLETE** |
| backdoor | q=0.9, f=0.1, seed=0 | 200 | 20 / 200 (interrupted) | n/a | ~89–90% | **OPEN / INCOMPLETE** |
| backdoor | small-n smoke | 20 | done | n/a | n/a | **REPRODUCED (behaviour, small-n)** |
| double | q=0.9, f=0.4, seed=0 | 200 | 10 / 200 (interrupted) | n/a | 87.47% | **OPEN / INCOMPLETE** |
| double | q=0.6, f=0.4, seed=0 | 200 | 10 / 200 (interrupted) | n/a | (not anchored) | **OPEN / INCOMPLETE** |
| double | q=0.9, f=0.1, seed=0 | 200 | 10 / 200 (interrupted) | n/a | ~89–90% | **OPEN / INCOMPLETE** |
| double | small-n smoke | 20 | done | n/a | n/a | **REPRODUCED (behaviour, small-n)** |
| LIE | q=0.9, f=0.4, τ=1.5, seed=0 | 200 | 200 / 200 | 10.2% | 70.10% | **CHARACTERIZED GAP** (see §3 below) |
| LIE | q=0.9, f=0.4, z=0.9346, seed=0 | 200 | 200 / 200 | 33.3% | 70.10% | **CHARACTERIZED GAP** (best of two principled τ values) |
| LIE | small-n smoke | 20 | done | 9.6% (collapse) | n/a | small-n cap artefact, not paper-comparable |

**No multi-seed paper-scale runs of any cell. All single-seed (seed=0).**

---

## Step 3 — The two characterized gaps

### Gap 1: LIE accuracy

- Behaviour reproduced: LIE evades the cross-product detector — Byzantine
  weights pin at the cap (sum_byz ≈ 0.727 = 80/110) for all 200 rounds in
  both τ runs.
- Numbers not reproduced: 33.3% (with the Baruch stealth bound z=0.9346)
  and 10.2% (with the ByzFL default τ=1.5) vs the paper's 70.10%.
- Hypotheses ruled out (`results/paper_fixes/REPORT.md`):
  - τ choice — both the principled stealth bound and the ByzFL default
    leave the gap.
  - LIE μ/σ computed over the wrong object — Check 1 confirmed
    pseudo-gradients; Check 2 confirmed the Baruch formula; the raw-grad
    hypothesis (`LIERawGradAttack`) was tested and falsified (the
    raw-gradient forged vector is ~14× smaller than honest pseudo-grads
    and FedLAW detects it).
- **Frame**: "the paper's 70.10% LIE accuracy could not be resolved from
  published information alone." Resolving it would likely require the
  paper's reference implementation or correspondence with the authors.

### Gap 2: flipping_label at frac=0.4

- Behaviour reproduced: Byzantine weights are non-zero (FedLAW does not
  fully suppress 4-group label-flip Byzantine clients) — but unlike LIE,
  this is unexpected for a *data-poisoning* attack per the paper's
  Appendix C, which predicts anti-alignment at all fractions.
- Numbers: 65.4% (q=0.9) and 54.7% (q=0.6) vs paper's 87.45% and 92.22%.
- Cause documented (`results/paper_fixes/REPORT.md` §"flipping_label
  co-alignment root cause", §"Seed-sensitivity"):
  - Per-corrupted-group pseudo-gradients are individually large (21–36 in
    norm) and largely anti-aligned with the honest mean.
  - Pairwise cosines between the 4 corrupted groups' mean gradients are
    all negative (range −0.36 to −0.11).
  - Averaging these 4 mutually anti-aligned vectors causes ~67%
    cancellation; the small residual happens to be mildly co-aligned with
    the honest mean (cos = +0.13 ± 0.03 across 5 seeds, range +0.08 to
    +0.16, 5/5 positive).
  - The cross-product detector cannot see the per-group signals and
    interprets the cancelled residual as a weak honest gradient. Byzantine
    weights pin at cap (sum_byz ≈ 0.27, the cap floor).
- Robust across 5 different group draws (different seeds give different
  corrupted-group sets {2,4,8,9}, {0,1,4,5}, etc., but all give cos > 0
  and sum_byz ≈ 0.27).
- **Frame**: "genuine FedLAW property OR unidentified config difference
  with the paper — undetermined." The paper's 87.45% under a paper-faithful
  configuration is currently unreproduced; whether closing the gap requires
  an unstated experimental detail (B) or whether the paper's number is
  unreproducible from the published spec (C) cannot be decided without
  the paper's reference implementation.

Neither gap should be read as "the paper is wrong." Both should be read as
"the published configuration alone does not allow us to reproduce the
specific number; the mechanism and behaviour we do reproduce."

---

## Step 4 — Genuinely open items

These are unfinished work, not failure modes:

| Item | Current state | What would close it |
|---|---|---|
| Multi-seed paper-scale runs | All n=200 cells are seed=0 only. Paper Table 3 reports 5-seed mean ± std. | Re-run each completed cell with seeds {1, 2, 3, 4}, report mean ± std. ~35 min CPU per run × 12 cells × 4 seeds ≈ 28 CPU-hours. |
| inverse_gradient n=200 (3 configs) | Interrupted at round 20 / 200 | Resume to 200 rounds (paper anchors: 87.41%, 91.62%, ~89–90%). |
| backdoor n=200 (3 configs) | Interrupted at round 10–20 / 200 | Resume to 200 rounds. |
| double n=200 (3 configs) | Interrupted at round 10 / 200 | Resume to 200 rounds. |
| flipping_label q=0.9 f=0.1 | 190 / 200 rounds complete | Resume final 10 rounds. |

For the open detected-attack cells (inverse_gradient, backdoor, double),
the small-n smoke evidence and the flipping_label f=0.1 trajectory both
strongly suggest reproduction at paper scale — but **we do not have those
numbers in hand**. They are open, not reproduced.

For multi-seed: every paper-scale number in this repo currently rests on
a single seed. The flipping_label f=0.4 finding has the supporting 5-seed
*diagnostic* sweep (cos value), but not 5-seed accuracy at 200 rounds.

---

## What this repo currently allows you to defensibly say

- The FedLAW mechanism is implemented exactly as the paper specifies, with
  one documented departure (w-freeze) that is empirically inert at all
  configs we ran.
- Detection works at small-n for every attack; at paper-scale with f=0.1
  for flipping_label.
- LIE evades the detector (behaviour reproduced); the paper's 70.10%
  accuracy is a characterized gap.
- flipping_label at f=0.4 exposes a multi-group cancellation failure mode
  of the cross-product detector that is robust across seeds; the paper's
  87.45% is a characterized gap.
- The remaining detected-attack n=200 numbers are open work, not
  reproduced.
- No multi-seed paper-scale means.
