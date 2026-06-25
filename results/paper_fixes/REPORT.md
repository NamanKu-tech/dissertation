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
## End of paper fixes validation

Full report: ./results/paper_fixes/REPORT.md
