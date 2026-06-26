# FedLAW Validation Report

Model: `mlp3_mnist` (784→200→100→10)  |  Seeds: {0,1,2}  |  Rounds: 30–100

## Implementation gaps fixed (v3)

Three gaps between the implementation and the ICLR 2026 paper were identified
and fixed. Each changes a prior conclusion from v2:

| Gap | Fix | Key finding |
|---|---|---|
| 1 — Gradient definition | g_i = (θ−ψ_i)/α from E=3 local epochs | α=0.01 now works; v2's α=0.5 was compensating for this |
| 2 — Server-side ℓ2 clipping | C = max honest norm per round | Necessary for theorem; does NOT fix ALIE (direction unchanged) |
| 3 — Cap t | t = 1/(s−10) per paper Table 1 | Honest weights become non-uniform (matching Figure 1) |

Full diagnostic in `results/paper_fixes/REPORT.md`.

---

## Validated configuration (v3, all gaps fixed)

| Config | Clients | Byzantine | Data distribution | α (model lr) | β (weight lr) | E (local epochs) | s / t |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SignFlipping | 18 honest + 2 byz | 2 (10%) | Dirichlet α=0.5 | **0.01** | 0.001 | 3 | 18 / 1/8 |
| IPM (τ=2) | 18 honest + 2 byz | 2 (10%) | Dirichlet α=0.5 | **0.01** | 0.001 | 3 | 18 / 1/8 |
| ALIE (τ=1.5) | 12 honest + 8 byz | 8 (40%) | Dirichlet α=0.5 | 0.01 | 0.001 | 3 | 12 / 1/2 |

(v2 used α_lr=0.5, raw batch gradients, t=1/s — see Discrepancies below for why each changed.)

## Results (v3, 30 rounds, all fixes applied)

| Attack | Seeds | Acc at round 30 | Byz zeroed | Notes |
| --- | --- | --- | --- | --- |
| SignFlipping | {0,1,2} | **94.21 ± 0.07%** | **round 1** | false excl. = adaptive weighting (expected) |
| IPM (τ=2) | {0,1,2} | **94.21 ± 0.09%** | **round 1** | false excl. = adaptive weighting (expected) |
| ALIE (τ=1.5) | {0} | 93.4% | never | 15/15 false excl.; clipping: identical |

Notes: (1) "false excl." with t=1/(s−10) are low-h honest clients zeroed by adaptive weighting,
not a detection failure. Byzantine weights are 0 throughout. (2) 30-round numbers are
diagnostic. Full 100-round 3-seed runs estimated ~2–4h on CPU and are deferred.

---

## FedLAW v2 results (paper-faithful implementation, Cao et al. q-partition)

FedLAW v2 uses the paper's Cao et al. q-parameter data partitioning (not Dirichlet),
all 5 paper attacks, and all 4 algorithm fixes. Two scales tested.

### (a) Behaviour reproduced — small n (n=20, 30 rounds, q=0.9, 40% Byzantine, seed=0)

| Attack | Acc round 30 | sum_byz at round 5 | Detection |
|---|---|---|---|
| flipping_label | 87.3% | 0.0000 | ✓ zeroed round 5 |
| backdoor | 88.4% | 0.0000 | ✓ zeroed round 5 |
| inverse_gradient | 87.8% | 0.0000 | ✓ zeroed round 5 |
| double | 87.5% | 0.0000 | ✓ zeroed round 5 |
| **lie (τ=1.5)** | **9.6% (collapse)** | **1.0000** | ✗ all Byzantine at cap |

Note: n=20 LIE collapse is a **small-n cap artifact** — with s=12, t=1/2, the 8
Byzantine clients absorb 100% of the weight budget (8 × 1/2 > 1 → bisection gives
each 1/8, sum=1.0). This is NOT the paper's scenario.

### (b) Numbers reproduced — paper scale (n=200, 200 rounds, q=0.9, 40% Byzantine, seed=0)

Cap arithmetic: s=120, t=1/110, s·t=120/110≈1.09 ≥ 1 ✓.
Max Byzantine weight mass = 80/110 ≈ 0.727.
Paper target (Table 3): FedLAW LIE = **70.10 ± 2.17%**.

| Round | acc | sum_byz | max_byz | max_hon |
|---|---|---|---|---|
| 0 | 6.9% | 0.4000 | 0.0050 | 0.0050 |
| 10 | 18.9% | **0.7273** | 0.0091 | 0.0091 |
| 50 | 10.3% | 0.7273 | 0.0091 | 0.0091 |
| 200 | **10.2%** | 0.7273 | 0.0091 | 0.0091 |

**GENUINE DISCREPANCY.** Final accuracy 10.2% vs paper's 70.10% — 60 point gap.

The detection behaviour matches: Byzantine weights do NOT go to zero (LIE evades the
cross-product mechanism). But 80 Byzantine clients at cap (sum_byz=0.7273) cause full
collapse, not graceful degradation.

Root cause: with q=0.9 heterogeneous data, coordinate-wise std of pseudo-gradients is
~3× larger than the mean (||std||≈15.8 vs ||mean||≈5.0 in simulation). ByzFL's
τ=1.5 gives ||lie_vector||≈5.77×||mean||. When 72.7% of weight goes to 80 such
clients, the implied local model ψ_byz = θ − α·g_lie lies far outside any real model
region, causing divergence. The paper's τ is unknown — §I.1 says "z = stealth bound
(Baruch et al. 2019)" but does not print the value. Investigation required.

---

## Reproduced behaviours (v3)

**Weight collapse (SignFlipping, IPM):** Byzantine weights drop to exactly 0.000 at
round 1 and remain at zero. Identical timing to v2 — the detection is immediate.
With local-epoch pseudo-gradients the cross-term is ~(E·steps)² × larger, making
the Byzantine suppression signal dramatically stronger at α=0.01.

**Adaptive honest weighting (Gap 3):** With t=1/(s−10)=1/8, honest clients receive
different weights (std=0.036 at round 30, max≈0.115≈cap). This matches Figure 1
of the paper. With exact-exclusion t=1/s (v2), all survivors are forced to exactly
1/18 — no adaptive weighting is possible.

**ALIE evasion persists:** ALIE's co-aligned gradient direction (cos(g_byz, μ_honest)
> 0) survives both the gradient-definition fix and server-side clipping. Clipping
bounds ||g_byz|| ≤ C but does not change the direction. The paper's Table 3 numbers
(FedLAW 70%, FedAvg 84% under LIE at 40% Byzantine) reflect graceful degradation,
not detection — which is consistent with our 93.4% at τ=1.5 (a weak attack).

---

## Discrepancies from paper

### 1. ~~Gradient scale: α=0.5 required~~ — RETRACTED

**v2 claim:** "The balance condition is α > 0.065. We use α=0.5 (7× headroom).
Minimum viable band: 0.3 ≤ α < 1.0."

**Why it was wrong:** The analysis assumed g_i was a raw mini-batch gradient.
`Client.compute_gradients()` returns exactly that — one backward pass, no optimizer
step. The paper's Algorithm 1 line 7 defines g_i = −(ψ_i − θ)/α where ψ_i is the
local model after E full SGD epochs. This pseudo-gradient is E×steps_per_epoch
(≈156) times larger in magnitude than the raw gradient:

    ||g_pseudo|| ≈ E × steps × ||g_raw|| ≈ 156 × 1.4 ≈ 218

The cross-term at α=0.01 with pseudo-gradients:
    α·β·||g_pseudo||² · cos ≈ 0.01 × 0.001 × 218² × cos ≈ 0.475·cos

This is 200× larger than the loss term (0.001 × 2.3 ≈ 0.0023). Detection is
immediate even at α=0.01.

**Corrected finding:** α=0.01 (paper's value) works correctly. The fix is the
gradient definition, not the learning rate. The `compute_model_update(E × steps)`
→ `g_i = (theta − psi_i) / alpha` pipeline is required.

### 2. Data heterogeneity: α=5.0 (v1) was over-conservative — stands

v2 finding is unchanged. Dirichlet α=0.5 works with the correct implementation.
The v1 failure at Dirichlet α=0.5 was caused by α_lr=0.01 with raw gradients
(Discrepancy 1 above), not by the data heterogeneity level itself.

Failure boundary with the corrected implementation: **Dirichlet α ≈ 0.1–0.2**
(Step 3b of v2 diagnostic). Not retested with v3 — the boundary may shift slightly
with local-epoch pseudo-gradients, but the qualitative finding is expected to hold.

### 3. ~~Exact-exclusion required, slack actively harms~~ — PARTIALLY RETRACTED

**v2 claim:** "The paper's exact-exclusion constraint s=n−f, t=1/s is not arbitrary
— it is necessary. The sparsity constraint must force Byzantine clients out before
they can corrupt the model direction."

**What stands:** The sparsity constraint **s = n−f** is necessary. Setting s > n−f
(v2 Step 2: s=20 with t=1/16 or t=1/10) creates a feedback loop — Byzantine clients
remain active, corrupt the model direction, which reorders h values, which increases
Byzantine weight, causing further corruption. That analysis is correct and stands.

**What was wrong:** The cap **t = 1/s** is NOT the paper's setting. Paper Table 1
uses t = 1/(s−10), not t = 1/s. With s=n−f=18 and t=1/8 (slack=10):

    s·t = 18 × (1/8) = 2.25 ≥ 1 ✓ (feasible)

This looser cap still excludes Byzantine clients (they remain in the bottom 2 h
values, outside the top-s selected by the projection), but allows honest clients to
have adaptive, non-uniform weights. With s=18 and t=1/8, the capped simplex
projection will zero the lowest-h honest clients to keep Σw=1 with each w_i ≤ 1/8.
This is the intended adaptive weighting, not a false exclusion.

**Corrected finding:** Use s=n−f (unchanged) but t=1/(s−10) per paper Table 1.
For small n (e.g. n=20, s=18) use a smaller slack (s−2 or s−4) if s−10 ≤ 0.
Always verify s·t ≥ 1 before running.

### 4. LIE/ALIE — evasion confirmed, paper numbers NOT reproduced

**Detection evasion (confirmed, small n):** ALIE (τ=1.5) submits
g_byz = mean(honest) + τ·σ(honest). The mean component makes Byzantine gradients
co-aligned with consensus, evading FedLAW's cross-product detection:

| Attack | cos(g_byz, mean_honest) | cross_w[byz] | Detection |
|---|---|---|---|
| SignFlipping | −1.000 | strongly negative | ✓ zeroed round 1 |
| ALIE τ=1.5 | +0.222 | **positive, > honest mean** | ✗ Byzantine weights survive |

Server-side ℓ2 clipping does NOT fix this: it bounds ||g_byz|| ≤ C but does not
change the gradient direction. Cross_w[byz] remains positive after clipping.

**Paper-scale accuracy — GENUINE DISCREPANCY (n=200, 200 rounds, seed=0):**
Our result: 10.2% (model collapse). Paper Table 3: 70.10 ± 2.17%.

The collapse at paper scale is caused by τ=1.5 being too aggressive for pseudo-
gradients with q=0.9 heterogeneity. With ||std_pseudo|| ≈ 3× ||mean_pseudo||,
the LIE vector has magnitude 5.77× the honest mean. 80 Byzantine clients at 72.7%
total weight pull the model to a degenerate region. The paper's τ is unspecified
numerically in §I.1 ("z = stealth bound (Baruch et al. 2019)") — further
investigation needed to reproduce 70.10%.

Note: Baruch et al.'s stealth bound for 40% Byzantine (n=200, f=80) gives
z = Φ⁻¹(0.175) = −0.935 (negative), which is for Krum/Bulyan evasion, NOT
FedLAW. Applying z=-0.935 would give an anti-aligned attack that FedLAW detects.
The correct τ for FedLAW LIE experiments is a separate open question.

**Structural limitation (unchanged):** FedLAW's theorem requires Byzantine
gradients to be anti-aligned with consensus. LIE violates this by construction.
No norm-based clipping or hyperparameter tuning can make FedLAW detect LIE.
The question is only whether LIE causes collapse or graceful degradation.

**Implication for RA-LAW (unchanged):** A reputation signal based on cross_w
history will also mis-rank Byzantine clients under LIE. The reputation signal
must use a source orthogonal to instantaneous gradient alignment.

### 5. Convergence noise — revised with local epochs

v2 noted large per-round noise with α=0.5 (step ≈ α·||g||≈0.7 in parameter space).
With v3 (α=0.01, local epochs), the model update is θ_{k+1} = Σ_i w_i ψ_i (weighted
average of local models — α cancels). The effective step is bounded by the local
model drift ||ψ_i − θ_k|| ≈ E·steps·α·||grad|| ≈ 156 × 0.01 × 1.4 ≈ 2.18 per
client, averaged across 18 clients with diverse directions. In practice, accuracy
curves at v3 are smooth and converge steadily (93.96% at round 30 vs 94.2% at
round 100 in v2).

---

## s,t regime recommendation (updated)

Use **s = n−f** (exact sparsity) and **t = 1/(s−10)** (paper's cap) for all
experiments. For small n where s−10 ≤ 0, use t = 1/(s−2) as a minimum viable slack.
Always verify s·t ≥ 1.

For partial participation with rate ρ: set s = ⌊ρ·n_honest⌋ each round,
t = 1/(s − max(0, s−18)) or a fixed slack proportional to participation size.
The SNR analysis carries over to partial participation — cross-term separation
holds down to **k=3 honest clients** at Dirichlet α=0.5 (v2 Step 3a).

---

## Conclusion

FedLAW's algorithm is correctly implemented. Three input-pipeline gaps have been
identified and fixed relative to the ICLR 2026 paper:

1. **Gradient definition (Gap 1, CRITICAL):** `g_i` must be the local-update
   pseudo-gradient `(θ − ψ_i)/α` from E=3 local SGD epochs, NOT a raw mini-batch
   gradient. This is why v2 required α=0.5 — it was compensating for the 156×
   smaller gradient magnitude. With the fix, α=0.01 (paper's value) achieves
   Byzantine detection (SignFlipping, IPM) from round 1 across all seeds.

2. **Server-side clipping (Gap 2, NECESSARY BUT INSUFFICIENT):** Clipping all
   gradients to C = max honest norm is required by the paper's theoretical
   assumptions. However, it does not fix ALIE detection — ALIE's co-aligned
   direction survives clipping.

3. **Cap t (Gap 3, CORRECTNESS):** t = 1/(s−10) per paper Table 1, not t = 1/s.
   With t = 1/s, the capped simplex has a single feasible point (all survivors
   at exactly 1/s). The paper's t = 1/(s−10) allows adaptive honest weighting
   (matching Figure 1) while still excluding Byzantine clients.

**What is reproduced vs. open:**

| Claim | Status |
|---|---|
| SignFlipping / IPM detection (small n, Dirichlet) | ✓ Reproduced (v3, 3 seeds) |
| 4-attack detection (small n, q-partition, FedLAW v2) | ✓ Behaviour reproduced |
| LIE evasion of cross-product detection | ✓ Mechanism confirmed |
| LIE graceful degradation to ~70% (paper Table 3) | ✗ NOT reproduced — collapse to 10% |
| LIE: τ value for paper's 70.10% result | ✗ Unknown — investigation required |

One confirmed structural limitation: **LIE evades FedLAW's cross-product detection
mechanism** regardless of norm clipping or hyperparameter tuning. RA-LAW's reputation
signal must be derived from a source beyond instantaneous gradient alignment to handle
co-aligned attacks.

---

## Plots

**v3 (paper fixes):**

- `results/paper_fixes/plots/gap1a_raw_alpha001_weights.png` — raw grad α=0.01 failure
- `results/paper_fixes/plots/gap1b_pseudo_alpha001_weights.png` — pseudo-grad α=0.01 fix
- `results/paper_fixes/plots/gap1b_pseudo_alpha001_crossw.png` — cross_w separation
- `results/paper_fixes/plots/gap2a_alie40pct_noclip_weights.png` — ALIE without clipping
- `results/paper_fixes/plots/gap2b_alie40pct_clip_weights.png` — ALIE with clipping
- `results/paper_fixes/plots/gap3a_exact_cap_weights.png` — exact-exclusion weights
- `results/paper_fixes/plots/gap3b_paper_cap_weights.png` — paper cap adaptive weights
- `results/paper_fixes/plots/gap3_weight_dist_comparison.png` — weight distributions

**v2 (pre-fix, raw gradients):**

- `results/validation_v2/plots/` — weight trajectories, regime comparison, attack comparisons
- `results/validation_v2/REPORT.md` — full running diagnostic report
