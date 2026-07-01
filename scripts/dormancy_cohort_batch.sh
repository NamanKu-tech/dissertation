#!/bin/bash
# Coordinated multi-client dormancy — cohort scale.
# n=200, q=0.9, p=0.5, frac=0.0, stealth_lie payload, seed=0, 200 rounds.
#   cohort_20 = group 2 (matches reproduction f=0.1 group-oriented Byzantine selection)
#   cohort_80 = groups {2,4,8,9} (matches reproduction f=0.4 assignment)
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

PROGRESS=results/v2/dormancy_cohort_progress.txt
mkdir -p results/v2/dormancy_cohort
echo "Coordinated cohort dormancy batch started $(date)" > "$PROGRESS"
echo "6 runs × 200 rounds × ~35 min = ~3.5h total." >> "$PROGRESS"

COHORT_20="40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59"
COHORT_80="${COHORT_20},80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199"

run_one() {
  local mode=$1 tdark=$2 cohort=$3 label=$4
  echo "" >> "$PROGRESS"
  echo "=== START: ${label} $(date) ===" >> "$PROGRESS"

  python -u -m src.run_fedlaw_v2 \
    --config configs/fedlaw_v2_mnist.yaml \
    --attack flipping_label --q 0.9 \
    --frac-malicious 0.0 --seed 0 \
    --p 0.5 --participation-mode "$mode" \
    --dormancy-T-dark "$tdark" \
    --dormancy-client-indices "$cohort" \
    --dormancy-payload stealth_lie \
    > /tmp/dcoh_${label}.log 2>&1

  local out="results/v2/dormancy_cohort/${label}"
  mkdir -p "$out"
  cp results/v2/flipping_label/q0.9/frac0.0/seed0/metrics.csv "$out/metrics.csv" 2>/dev/null || true
  cp results/v2/flipping_label/q0.9/frac0.0/seed0/dormancy_diag.csv "$out/dormancy_diag.csv" 2>/dev/null || true

  local final=$(tail -1 results/v2/flipping_label/q0.9/frac0.0/seed0/metrics.csv 2>/dev/null)
  echo "DONE: ${label} → ${final} $(date)" >> "$PROGRESS"
}

run_one cache_grad_B_ii  20   "$COHORT_20"  B_c20_T20
run_one cache_grad_B_ii  50   "$COHORT_20"  B_c20_T50
run_one cache_grad_B_ii  20   "$COHORT_80"  B_c80_T20
run_one cache_grad_B_ii  50   "$COHORT_80"  B_c80_T50
run_one naive_A          20   "$COHORT_20"  A_c20_T20
run_one naive_A          20   "$COHORT_80"  A_c80_T20

echo "" >> "$PROGRESS"
echo "ALL 6 RUNS DONE $(date)" >> "$PROGRESS"
