# archive/

Historical artefacts moved out of the canonical project tree on 2026-06-26.
Nothing here is referenced by the current code path, tests, or validation
documents. Retained for provenance only.

## Contents

### `results/baseline_mnist/`  (91 files)
ByzFL benchmark baseline — runs of Krum and TrMean against SignFlipping at
n=22 (20 honest + 2 Byzantine), Dirichlet α=0.5. Pre-dates the FedLAW
focus. Originally produced by:
`python -c "from byzfl import run_benchmark; run_benchmark(nb_jobs=1)"`
with `configs/baseline_mnist.json` as `config.json`. Superseded by the
FedLAW-specific evidence in `results/paper_fixes/` and `results/v2/`.

### `results/baseline_mnist_clean/`  (6 files)
Same ByzFL benchmark setup but with no Byzantine clients (clean baseline).
Used for sanity-checking the loop before introducing attacks. Pre-FedLAW.

### `results/fedlaw_clean/`  (3 files: `config.json`, `metrics.csv`, `weights.npy`)
Output of the **DEPRECATED v1 trainer** (`src/fedlaw.py`) on the clean
(no-attack) MNIST config. Pre-dates the v2 paper-faithful trainer. Superseded
by v2 small-n smoke runs in `results/v2_small/`.

### `results/fedlaw_ipm/`  (3 files)
DEPRECATED v1 trainer output for the IPM attack. Superseded by
`results/v2/*` and `results/v2_small/*`.

### `results/fedlaw_signflipping/`  (3 files)
DEPRECATED v1 trainer output for the SignFlipping attack. Superseded by
v2 runs.

### `results/plots/`  (3 PNGs: `fedlaw_accuracy.png`, `fedlaw_SignFlipping.png`, `fedlaw_IPM.png`)
DEPRECATED v1 trainer plots. Superseded by `results/paper_fixes/plots/`
and `results/validation_v2/plots/`.

## Why not deleted

These outputs are evidence that v1 existed and produced specific results
before v2 corrected the three implementation gaps documented in
`results/paper_fixes/REPORT.md`. Keeping them under `archive/` preserves
that audit trail without polluting the canonical results tree.
