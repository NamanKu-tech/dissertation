# PAPER_FAITHFULNESS.md

Read-only audit mapping each core component of the FedLAW reproduction to the
corresponding part of the ICLR 2026 paper ("Byzantine-Robust Federated Learning
with Learnable Aggregation Weights"). For every component this document states:

- which file/lines implement it
- which paper section / equation / algorithm it implements
- whether the implementation **matches**, **partially matches**, or **diverges**
- the evidence (test or diagnostic run) that backs the verdict

Last updated: 2026-06-26.

---

## Canonical vs deprecated code paths

| Path | Status |
|---|---|
| `src/fedlaw_v2.py`, `src/run_fedlaw_v2.py`, `configs/fedlaw_v2_*.yaml` | **CANONICAL** — current paper-faithful trainer + CLI + configs |
| `src/attacks.py`, `src/data_partition.py`, `src/projections.py`, `src/models.py` | **CANONICAL** — supporting modules used by v2 |
| `src/fedlaw.py`, `src/run_fedlaw.py`, `src/aggregators.py`, `src/loop.py`, `src/run_loop.py` | **DEPRECATED / reference-only** — earlier v1 implementation, retained for historical comparison; not used in current validation |
| `configs/fedlaw_mnist.yaml`, `configs/fedlaw_signflipping.yaml`, `configs/fedlaw_ipm.yaml`, `configs/loop_mnist.yaml` | **DEPRECATED / reference-only** — paired with v1 trainer above |

All audit verdicts below are against the **CANONICAL** code path.

---

## 1 — Gradient definition (Algorithm 1)

**Paper.** Each round k, every client i performs E full local SGD epochs from
the global model θ_k, producing ψ_i. The pseudo-gradient is

  g_i = (θ_k − ψ_i) / α

so that θ_{k+1} = θ_k − α · Σ w_i g_i = Σ w_i ψ_i. Paper uses α = 0.01,
E = 3 for MNIST.

**Code.** `src/fedlaw_v2.py:266–321` (`_collect`):
- L283–286: `client.set_parameters(theta)` then `loss_i = client.compute_gradients()` records the loss at θ.
- L289–291: `steps = cfg.E * len(client.training_dataloader); client.compute_model_update(steps)` runs E full epochs.
- L293–298: extracts ψ_i and computes `g_i = (theta − psi) / cfg.alpha`.

Defaults `alpha=0.01`, `E=3` in `FedLAWV2Config` (`src/fedlaw_v2.py:41–66`).

**Verdict: MATCHES.**

Evidence: `tests/test_fedlaw_v2.py::test_collect_pseudo_grad_definition` confirms
the exact formula; `results/paper_fixes/REPORT.md` §"Gap 1" records the
small-n run that flipped detection from "fails at α=0.01" (raw grads, v1) to
"Byzantine zeroed round 1 at α=0.01" (pseudo-grads, v2).

---

## 2 — Weight update (Algorithm 2)

**Paper.** Two-round update per outer iteration:

  Round A: collect G_k = pseudo-grads at θ_k.
  Round B: take a tentative step θ̃ = θ_k − α · G_kᵀ w_k, collect G̃ at θ̃.
  Compute h_k = w_k + α β · G_kᵀ G̃ · w_k − β · f̃,
  then project w_{k+1} = Π_{Δ(s,t)} (h_k).
  Model: θ_{k+1} = θ_k − α · G_kᵀ w_{k+1}.

**Code.** `src/fedlaw_v2.py:395–414` (inner loop of `run`):
- L396–397: Round A collect at θ_k, clip.
- L401: θ̃ = θ_k − α · G^T w.
- L402: Round B collect at θ̃, clip.
- L406–409: `cross = G @ G_tilde.T; h = w + α β (cross @ w) − β f_tilde`.
- L410–411: `project_sparse_capped_simplex(h, s, t)` (`src/projections.py:45–87`).
- L414: model update θ_{k+1} = θ_k − α · G^T w.

**One documented departure: w-freeze.** After `cfg.w_freeze_rounds` rounds
(default 20), the weight vector is frozen and only the model continues to
update. This is **not in the paper text** — added in v2 design (`docs/superpowers/specs/2026-06-26-fedlaw-v2-design.md`) for compute efficiency
after observing that w stabilises by ~round 5 in our runs.

Empirical impact at n=200, f=0.4 flipping_label: sum_byz already pins to the
cap floor (0.273) by round 6, so freezing at round 20 does not change the
trajectory (verified in `scripts/diag_flip_n200.py` output). At smaller n
the freeze has been informally confirmed harmless because detection completes
in 1–2 rounds.

**Verdict: MATCHES the paper formula. PARTIALLY DIVERGES on w-freeze** (added
optimisation, empirically confirmed not to change behaviour for the configs we
report). Flagged here for transparency; results/paper_fixes/REPORT.md uses the
v2 trainer including this freeze.

Evidence: `tests/test_fedlaw_v2.py::test_run_weights_sum_to_one`,
`results/v2_small/*/seed0/weights.npy` (per-round w history),
`results/paper_fixes/REPORT.md` §"All fixes applied — paper-comparable
validation".

---

## 3 — Cap and sparsity (Table 1)

**Paper.** Sparsity s = (1 − frac_malicious) · n = n_honest; cap
t = 1/(s − 10). For projection feasibility require s · t ≥ 1.

**Code.** `src/fedlaw_v2.py:233–239` (in `__init__`):
- L233: `self.sparsity = self.n_honest`.
- L234: `slack = min(10, self.sparsity − 2)` — guards small-n where s−10 ≤ 0.
- L235: `self.cap = 1.0 / max(self.sparsity − slack, 1)`.
- L236–239: explicit `s·t ≥ 1` feasibility check; raises ValueError otherwise.

For paper-scale n=200, f=0.4: s=120, slack=10, t=1/110, s·t = 120/110 ≈ 1.09 ✓.
For paper-scale n=200, f=0.1: s=180, slack=10, t=1/170, s·t = 180/170 ≈ 1.06 ✓.
For n=20, f=0.4: s=12, slack=10, t=1/2 — small-n degenerate case noted in
`VALIDATION.md` (Byzantine absorb 100% of budget because 8 × 1/2 ≥ 1).

**Verdict: MATCHES** for n where s > 10 (paper's intended regime). Small-n
guard (`slack = s − 2` when s ≤ 12) is documented as an extension, not paper
behaviour.

Evidence: `tests/test_projections.py` confirms the projection enforces Σw=1,
0 ≤ w_i ≤ t, ‖w‖₀ ≤ s. Diagnostic in `scripts/diag_flip_n200.py` prints the
explicit cap arithmetic at run-time.

---

## 4 — Server-side ℓ2 clipping (Assumption E1)

**Paper.** Each round, clip every client's submitted gradient to ℓ2 norm
C = max_{i ∈ honest} ‖g_i‖. The paper's convergence theorem requires bounded
gradients; clipping enforces this.

**Code.** `src/fedlaw_v2.py:71–86` (`_clip_gradients`):
- L78: `C = max ‖g_i‖` over honest indices.
- L80–84: scales any row with ‖g_i‖ > C down to C; honest rows are unchanged.

Applied in `run` at L397 (after Round A collect) and L403 (after Round B collect).

**Important caveat (the paper does not state, our experiments showed).** Clipping
bounds magnitude only. ALIE-style attacks that submit co-aligned vectors of
comparable magnitude pass through unchanged in direction (`VALIDATION.md` §3).
Clipping is necessary for the theorem but not sufficient for ALIE/LIE
detection — which the paper acknowledges by reporting LIE/ALIE as a separate
column (lower than other attacks) in Table 3.

**Verdict: MATCHES.**

Evidence: `tests/test_fedlaw_v2.py::test_clip_max_honest`, repeated in the
"Gap 2" section of `results/paper_fixes/REPORT.md`.

---

## 5 — Data partition (§I.1, Cao et al. q-split)

**Paper.** Each example with label l is assigned to "group l" with probability
q, and to each of the other n_labels − 1 groups with probability
(1 − q) / (n_labels − 1). Clients are arranged group-by-group:
clients [g · n_per_group, (g+1) · n_per_group) belong to group g.

**Code.** `src/data_partition.py:8–46` (`cao_partition`):
- L24: `n_per_group = n_clients // n_labels`.
- L26: extract integer label array.
- L29–35: for each example, sample group g with probability q (same label) or
  uniform over the n_labels − 1 others.
- L37–45: for each group g, shuffle its examples and split into `n_per_group`
  equal chunks; each chunk becomes one client's DataLoader.

**Verdict: MATCHES.**

Evidence: `tests/test_data_partition.py` includes both a q=1 concentration
test and a q=0.1 spread test that lock in the per-example sampling
distribution; the trainer constructs the partition with `cfg.seed` so the
assignment is deterministic across runs.

---

## 6 — Malicious-client selection (§I.1, group-oriented)

**Paper.** Byzantine clients are drawn **as complete groups** (the "hardest
case"): `⌈n_byz / n_per_group⌉` groups are corrupted, the remaining groups
are honest. Group order is randomised.

**Code.** `src/data_partition.py:49–73` (`select_malicious_indices`):
- L60: `n_groups = n_clients // clients_per_group`.
- L61–62: `rng.permutation(n_groups)` — random group order.
- L64–72: fill the chosen groups completely before moving to the next.

For n=200, n_per_group=20: f=0.4 corrupts 4 full groups (80 clients);
f=0.1 corrupts 1 group (20 clients). Verified at runtime in the diagnostic
output (`/tmp/diag_flip_n200.out`: groups {2,4,8,9} at f=0.4, group {2} at
f=0.1).

**Verdict: MATCHES.**

Evidence: `tests/test_data_partition.py::test_group_oriented_selection`;
`results/paper_fixes/REPORT.md` §"flipping_label n=200 frac=0.4 diagnosis" D2.

---

## 7 — Attacks (§I.1)

### 7a — flipping_label

**Paper.** Each example's label l replaced with L − l − 1 (where L = n_labels).

**Code.** `src/attacks.py:18–30` (`FlipLabelDataset.__getitem__`):
`return x, self.n_labels - 1 - int(y)`.

**Verdict: MATCHES.** Test: `tests/test_attacks.py` covers the mapping.

### 7b — inverse_gradient

**Paper.** Byzantine submission g_byz_i = − g_honest_i (flip the sign of the
pseudo-gradient). Equivalently ψ_byz = 2 θ − ψ_honest.

**Code.** `src/attacks.py:70–79` (`InverseGradientAttack`):
`return [-g.copy() for g in pseudo_grads]`.

**Verdict: MATCHES.**

### 7c — backdoor

**Paper.** Add a fixed `trigger_size × trigger_size` patch to the image; the
label is "randomly changed to a label between 0 and L−1" (paper §I.1).

**Code.** `src/attacks.py:33–60` (`BackdoorDataset`):
- L40: `_TRIGGER_VALUE = -0.4242` — the normalised MNIST encoding of pixel
  value 0 ((0 − 0.1307)/0.3081).
- L42–48: `trigger_size=8` default; pre-sampled random labels per example
  (seed-controlled, uniform over `[0, n_labels)`).
- L53–60: clones the image, paints an 8×8 centre patch with the trigger value,
  returns the patched image with the pre-sampled label.

**Verdict: MATCHES.** Per-example uniform random target labels match the
paper's "randomly changed to a label between 0 and L−1".

### 7d — double

**Paper.** Two Byzantine sub-populations: M1 starts inverse_gradient from
round 2; M2 starts global_parameter from round 5. The point is to test
temporal mixing — defences that adapt to one attack pattern can be exposed
to a second.

**Code.** `src/attacks.py:114–141` (`DoubleAttack`):
- L131–132: first half (M1) of Byzantine list runs InverseGradient.
- L135–136: from round 1 (0-indexed) M1 active.
- L138–139: from round 4 (0-indexed) M2 active with GlobalParam.

Activation rounds 1 and 4 are **0-indexed**, matching paper rounds 2 and 5.

**Verdict: MATCHES.**

### 7e — LIE (and the rerouted LIE-raw-grad variant)

**Paper.** Byzantine submission b = μ + z · σ where μ, σ are computed
coordinate-wise over the honest clients' gradients, z is the Baruch et al.
stealth bound z = Φ⁻¹((n − f − m)/(n − f)) with m = ⌊n/2 + 1⌋ − f. For
n=200, f=80: m=21, z = Φ⁻¹(0.825) ≈ 0.9346.

**Code.** `src/attacks.py:144–166` (`LIEAttack`) wraps ByzFL's
`ALittleIsEnough(tau=…)`. The `tau` parameter is exposed in
`FedLAWV2Config.lie_tau` (`src/fedlaw_v2.py:65`) and the CLI
(`src/run_fedlaw_v2.py:35–36`).

**Computation object.** Both the paper and our LIE compute μ, σ over
**pseudo-gradients** g_i = (θ − ψ_i)/α (verified in `src/fedlaw_v2.py:347–348`
where the LIE attack receives `honest_pseudo_grads`).

A second class `LIERawGradAttack` (`src/attacks.py:169–199`) computes μ, σ
over raw single-batch gradients ∇f(θ; batch) instead. This was added to test
a hypothesis about the paper's number — **not paper-faithful**, kept only
for the falsified-hypothesis evidence in
`results/paper_fixes/REPORT.md` §"LIE raw-gradient hypothesis test".

**Verdict: MATCHES** for `LIEAttack` (pseudo-gradient μ/σ matches paper).
`LIERawGradAttack` is an explicitly-labelled diagnostic variant.

---

## Summary table

| # | Component | File | Verdict | Note |
|---|---|---|---|---|
| 1 | Pseudo-gradient (Alg. 1) | `fedlaw_v2.py:266–321` | **matches** | E=3, α=0.01 defaults match paper |
| 2 | Weight update (Alg. 2) | `fedlaw_v2.py:395–414` | **matches**, w-freeze added | freeze is a v2 efficiency extension; empirically harmless |
| 3 | Cap & sparsity (Table 1) | `fedlaw_v2.py:233–239` | **matches** | small-n slack guard documented |
| 4 | Server clipping (E1) | `fedlaw_v2.py:71–86` | **matches** | clipping bounds magnitude only — does not detect ALIE/LIE direction |
| 5 | Cao q-partition (§I.1) | `data_partition.py:8–46` | **matches** | |
| 6 | Group-oriented selection (§I.1) | `data_partition.py:49–73` | **matches** | |
| 7a | flipping_label | `attacks.py:18–30` | **matches** | |
| 7b | inverse_gradient | `attacks.py:70–79` | **matches** | |
| 7c | backdoor | `attacks.py:33–60` | **matches** | per-example uniform random target matches paper §I.1 |
| 7d | double | `attacks.py:114–141` | **matches** | |
| 7e | LIE | `attacks.py:144–166` | **matches** | pseudo-gradient μ/σ confirmed |
| 7e' | LIE-raw-grad | `attacks.py:169–199` | **diagnostic only** | not paper-faithful; falsifies a hypothesis |

## Open audit items

- **w-freeze.** Documented departure (efficiency only, no observed behavioural
  difference at the configs we have run).

## Synthesis — what this audit does and does not explain

With items 1 (pseudo-gradient definition, Algorithm 1), 3 (cap and sparsity,
Table 1), and 6 (group-oriented Byzantine selection, §I.1) all verified
paper-faithful, AND model architecture confirmed to match the paper (paper
§5.1 specifies "a 3-layer fully connected network on MNIST", which is
`mlp3_mnist` — 784→200→100→10), the frac=0.4 flipping_label co-alignment
finding (cos(byz, honest_mean) = +0.16 at f=0.4 vs −0.14 at f=0.1, recorded
in `results/paper_fixes/REPORT.md` §"flipping_label n=200 frac=0.4
diagnosis") is **NOT** explained by any implementation infidelity uncovered
in this audit.

The open question is therefore one of:

- a subtlety in the flipping_label × q-split × group-oriented selection
  interaction that the audit's coarse checks did not catch (e.g. seed
  coupling, a partition-time bias the per-example tests would not detect), or
- a genuine property of FedLAW's cross-product detector at multi-group
  corruption — in which case the paper's 87.45% at frac=0.4 reflects an
  experimental detail not captured by the published configuration alone.

Resolving this open question is for the diagnosis follow-up, not for an
architecture swap (architecture has been excluded).
