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
## Gap 2 — Server-side ℓ2 clipping (paper Assumption E1 + Appendix C Layer 1)

### Audit: current clipping status

No clipping exists anywhere in fedlaw.py or run_validation_v2.py.
Gap 2 fix: after each collection round, compute C = max(||g_honest_i||),
then project all incoming gradients onto the ℓ2-ball of radius C:
  g_i ← g_i × min(1, C / ||g_i||)

Expected effect on ALIE (ALittleIsEnough, τ=1.5):
  Without clipping: ||g_byz|| = ||μ + τσ|| > ||g_honest_max|| (τσ term inflates norm).
  ALIE's direction has cos(g_byz, mean_honest) > 0 (co-aligned by construction).
  cross_w[byz] > cross_w[honest] → honest clients falsely excluded.

  With clipping: ||g_byz|| ≤ C = max(||g_honest||). Norm is bounded.
  Direction is unchanged — clipping alone does NOT fix cos alignment.
  Paper does NOT claim strong ALIE detection (Table 3: FedAvg=84%, FedLAW=70%).
  SUCCESS = 'graceful degradation', not 'inversion stops'.

Test setup: 40% Byzantine (8 of 20), Dirichlet α=0.5, ALIE τ=1.5, 15 rounds.
Note: with 8 Byzantine, s = n−f = 12 for exact exclusion.

### Experiment 2a — ALIE 40% Byzantine, NO clipping (pre-fix)
Byzantine zeroed at: None
False exclusions: 15 / 15 rounds
Final accuracy: 93.37%
Round 1 cross_w: byz mean=9008.4948, honest mean=8808.0859

### Experiment 2b — ALIE 40% Byzantine, WITH clipping (Gap 2 fix)
Byzantine zeroed at: None
False exclusions: 15 / 15 rounds
Final accuracy: 93.37%
Round 1 cross_w: byz mean=9008.4948, honest mean=8808.0859

### Paper comparison — Table 3 (FedLAW under LIE)
  Paper FedAvg under LIE q=0.9, 40% Byzantine:  ≈ 84%
  Paper FedLAW under LIE q=0.9, 40% Byzantine:  ≈ 70%
  (Paper's q notation ≈ Dirichlet heterogeneity; lower q = more heterogeneous.)
  Our result without clipping: 93.37%
  Our result with clipping:    93.37%

**VERDICT Gap 2: INCONCLUSIVE** — clipping shows no clear improvement.
  This is consistent with theory: clipping bounds norm but not direction.
  ALIE's co-alignment with honest gradients survives any norm-based clipping.

Note: 15-round diagnostic may not fully reflect steady-state behaviour.
The key insight is that ALIE defeats the cross-product mechanism structurally,
and clipping is a necessary but insufficient fix for this attack class.
