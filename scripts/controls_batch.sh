#!/bin/bash
# Control experiments — isolate each causal claim before writing findings.
# All n=200, q=0.9 (unless stated), 200 rounds, cache_grad_B_ii, cohort_80.
# Sequential, laptop-safe.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

PROGRESS=results/v2/controls_progress.txt
mkdir -p results/v2/controls
echo "Controls batch started $(date)" > "$PROGRESS"
echo "Multi-seed {0,1,2} on essential cells." >> "$PROGRESS"

COHORT_80="40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199"

# Args: label seed T_dark payload p q coord_present
run_fedlaw() {
  local label=$1 seed=$2 T_dark=$3 payload=$4 p=$5 q=$6 coord=$7
  echo "" >> "$PROGRESS"
  echo "=== START: ${label} $(date) ===" >> "$PROGRESS"
  python -u - <<PYEOF > /tmp/ctrl_${label}.log 2>&1
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer
cfg = FedLAWV2Config(
    n_clients=200, n_labels=10, q=${q},
    frac_malicious=0.0, attack_name="flipping_label",
    alpha=0.01, beta=0.01, E=3, w_freeze_rounds=20,
    T=200, eval_every=10, seed=${seed},
    p=${p}, participation_mode="cache_grad_B_ii",
    dormancy_T_dark=${T_dark},
    dormancy_client_indices=[$COHORT_80],
    dormancy_payload="${payload}",
    coordinated_present=${coord},
    results_dir="results/v2/controls/${label}",
)
FedLAWV2Trainer(cfg).run()
PYEOF
  local final=$(tail -1 results/v2/controls/${label}/metrics.csv 2>/dev/null)
  echo "DONE: ${label} → ${final} $(date)" >> "$PROGRESS"
}

# Control 2 — honest-dormant (attribution: attack vs compute loss)
for s in 0 1 2; do
  run_fedlaw fedlaw_c80_T20_HONEST_seed${s}      $s 20 stealth_honest 0.5 0.9 False
done

# Control 1c — coordinated present at p=0.5 (does dormancy do work?)
for s in 0 1 2; do
  run_fedlaw fedlaw_c80_coord_present_p05_seed${s}  $s -1 stealth_lie 0.5 0.9 True
done

# Control 3 — coordinated present at p=1.0 (does attack need PP?)
for s in 0 1 2; do
  run_fedlaw fedlaw_c80_coord_present_p10_seed${s}  $s -1 stealth_lie 1.0 0.9 True
done

# Control 4 — q=0.6 (heterogeneity dependence)
for s in 0 1 2; do
  run_fedlaw fedlaw_c80_T20_q06_seed${s}         $s 20 stealth_lie 0.5 0.6 False
done

# Multi-seed the existing dormancy c80 (only had seed 0 previously)
for s in 1 2; do
  run_fedlaw fedlaw_c80_T20_stealth_seed${s}      $s 20 stealth_lie 0.5 0.9 False
done

echo "" >> "$PROGRESS"
echo "ALL RUNS DONE $(date)" >> "$PROGRESS"
