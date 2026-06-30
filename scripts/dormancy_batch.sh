#!/bin/bash
# Stealthy dormancy attack batch — 6 runs, sequential.
# n=200, q=0.9, p=0.5, frac=0 (only dormant), seed=0, 200 rounds.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

PROGRESS=results/v2/dormancy_progress.txt
mkdir -p results/v2/dormancy
echo "Stealthy dormancy batch started $(date)" > "$PROGRESS"
echo "6 runs × 200 rounds × ~35 min = ~3.5h total." >> "$PROGRESS"

run_one() {
  local mode=$1 tdark=$2 payload=$3 label=$4
  echo "" >> "$PROGRESS"
  echo "=== START: ${label} $(date) ===" >> "$PROGRESS"

  local dorm_args=""
  if [ "$tdark" != "-1" ]; then
    dorm_args="--dormancy-T-dark $tdark --dormancy-client-idx 40 --dormancy-payload $payload"
  fi

  python -u -m src.run_fedlaw_v2 \
    --config configs/fedlaw_v2_mnist.yaml \
    --attack flipping_label --q 0.9 \
    --frac-malicious 0.0 --seed 0 \
    --p 0.5 --participation-mode "$mode" \
    $dorm_args \
    > /tmp/dorm_${label}.log 2>&1

  local out="results/v2/dormancy/${label}"
  mkdir -p "$out"
  cp results/v2/flipping_label/q0.9/frac0.0/seed0/metrics.csv "$out/metrics.csv" 2>/dev/null || true
  cp results/v2/flipping_label/q0.9/frac0.0/seed0/dormancy_diag.csv "$out/dormancy_diag.csv" 2>/dev/null || true

  local final=$(tail -1 results/v2/flipping_label/q0.9/frac0.0/seed0/metrics.csv 2>/dev/null)
  echo "DONE: ${label} → ${final} $(date)" >> "$PROGRESS"
}

run_one cache_grad_B_ii  -1   none           clean_B
run_one cache_grad_B_ii  20   stealth_lie    dorm_B_lie_T20
run_one cache_grad_B_ii  50   stealth_lie    dorm_B_lie_T50
run_one cache_grad_B_ii  20   stealth_honest dorm_B_honest_T20
run_one naive_A          20   stealth_lie    dorm_A_lie_T20
run_one naive_A          50   stealth_lie    dorm_A_lie_T50

echo "" >> "$PROGRESS"
echo "ALL 6 RUNS DONE $(date)" >> "$PROGRESS"
