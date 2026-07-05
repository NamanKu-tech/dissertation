#!/bin/bash
# THESIS TEST: does FedLAW fail MORE under partial participation than full,
# to STANDARD attacks? No dormancy, no coordinated poison — just the paper's
# own attack suite at p ∈ {1.0, 0.5, 0.25}, multi-seed.
#
# Attacks: flipping_label (=label-flipping), inverse_gradient (=IPM/sign-flip),
# lie (=ALIE). At q=0.9 f=0.4 (in FedLAW paper's own tested regime).
# 36 cells total = (3 attacks × 3 p × 3 seeds) + (clean × 3 p × 3 seeds).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

PROGRESS=results/v2/thesis_test_progress.txt
mkdir -p results/v2/thesis_test
echo "Thesis test batch started $(date)" > "$PROGRESS"
echo "36 cells. Sequential." >> "$PROGRESS"

# Args: label seed p attack frac lie_tau
run_cell() {
  local label=$1 seed=$2 p=$3 attack=$4 frac=$5 lie_tau=$6
  echo "" >> "$PROGRESS"
  echo "=== START: ${label} $(date) ===" >> "$PROGRESS"

  python -u - <<PYEOF > /tmp/thesis_${label}.log 2>&1
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer
cfg = FedLAWV2Config(
    n_clients=200, n_labels=10, q=0.9,
    frac_malicious=${frac}, attack_name="${attack}",
    alpha=0.01, beta=0.01, E=3, w_freeze_rounds=20,
    T=200, eval_every=10, seed=${seed},
    p=${p}, participation_mode="cache_grad_B_ii",
    lie_tau=${lie_tau},
    results_dir="results/v2/thesis_test/${label}",
)
FedLAWV2Trainer(cfg).run()
PYEOF
  local csv="results/v2/thesis_test/${label}/metrics.csv"
  [ -f "$csv" ] && local final=$(tail -1 "$csv") || local final="MISSING"
  echo "DONE: ${label} → ${final} $(date)" >> "$PROGRESS"
}

# Clean baselines at each p (Byzantine fraction = 0)
for s in 0 1 2; do
  for p in 1.0 0.5 0.25; do
    run_cell "clean_p${p}_seed${s}" $s $p flipping_label 0.0 1.5
  done
done

# Attack cells at f=0.4 q=0.9
for s in 0 1 2; do
  for p in 1.0 0.5 0.25; do
    run_cell "flip_p${p}_seed${s}"    $s $p flipping_label 0.4 1.5
    run_cell "invg_p${p}_seed${s}"    $s $p inverse_gradient 0.4 1.5
    run_cell "lie_p${p}_seed${s}"     $s $p lie 0.4 0.9346
  done
done

echo "" >> "$PROGRESS"
echo "ALL 36 RUNS DONE $(date)" >> "$PROGRESS"
