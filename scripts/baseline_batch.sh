#!/bin/bash
# Baselines batch — TrMean + DeMoA-cache vs coordinated cohort_80 dormancy.
# n=200, q=0.9, p=0.5, frac=0.0, stealth_lie payload, T_dark=20, seed=0, 200 rounds.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

PROGRESS=results/v2/baselines_progress.txt
mkdir -p results/v2/baselines
echo "Baselines batch started $(date)" > "$PROGRESS"
echo "3 runs × 200 rounds × ~25 min = ~1.25h total." >> "$PROGRESS"

COHORT_80="40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199"

run_one() {
  local label=$1 aggregator=$2 f=$3 cache=$4 dormancy=$5

  echo "" >> "$PROGRESS"
  echo "=== START: ${label} $(date) ===" >> "$PROGRESS"

  python -u - <<PYEOF > /tmp/base_${label}.log 2>&1
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.baselines import BaselineConfig, BaselineTrainer
cfg = BaselineConfig(
    n_clients=200, q=0.9, T=200, eval_every=5, seed=0,
    p=0.5, aggregator="${aggregator}", aggregator_f=${f},
    use_demoa_cache=${cache},
    ${dormancy}
    results_dir="results/v2/baselines/${label}",
)
BaselineTrainer(cfg).run()
PYEOF

  local final=$(tail -1 results/v2/baselines/${label}/metrics.csv 2>/dev/null)
  echo "DONE: ${label} → ${final} $(date)" >> "$PROGRESS"
}

# Baselines
run_one clean_TrMean_DeMoA         trmean 20 True  ""
run_one dorm_TrMean_DeMoA_c80_T20  trmean 20 True  "dormancy_T_dark=20, dormancy_client_indices=[$COHORT_80], dormancy_payload='stealth_lie',"
run_one dorm_TrMean_naive_c80_T20  trmean 20 False "dormancy_T_dark=20, dormancy_client_indices=[$COHORT_80], dormancy_payload='stealth_lie',"

echo "" >> "$PROGRESS"
echo "ALL 3 RUNS DONE $(date)" >> "$PROGRESS"
