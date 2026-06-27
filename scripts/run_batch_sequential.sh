#!/bin/bash
# Sequential paper-scale batch — one job at a time, no parallelism.
# Appends progress to results/v2/batch_progress.txt after each job.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

PROGRESS=results/v2/batch_progress.txt
mkdir -p results/v2
echo "Batch started $(date)" > "$PROGRESS"

run_job() {
  local attack=$1 q=$2 frac=$3 paper=$4
  local label="${attack} q=${q} f=${frac}"
  echo "" >> "$PROGRESS"
  echo "=== START: ${label} (paper ${paper}%) $(date) ===" >> "$PROGRESS"

  python -u -m src.run_fedlaw_v2 \
    --config configs/fedlaw_v2_mnist.yaml \
    --attack "$attack" --q "$q" --frac-malicious "$frac" --seed 0 \
    > /tmp/batch_${attack}_q${q}_f${frac}.log 2>&1

  # Extract final acc from CSV
  csv="results/v2/${attack}/q${q}/frac${frac}/seed0/metrics.csv"
  local final_line=$(tail -1 "$csv")
  local final_acc=$(echo "$final_line" | awk -F',' '{print $2}')

  # Extract final sum_byz from the run log (last "sum_byz=" line)
  local sum_byz=$(grep -oE 'sum_byz=[0-9.]+' /tmp/batch_${attack}_q${q}_f${frac}.log | tail -1 | cut -d= -f2)

  echo "DONE: ${label}  final_acc=${final_acc}  sum_byz=${sum_byz}  paper=${paper}%  $(date)" >> "$PROGRESS"
}

# Order: detected attacks, easy first (f=0.1) within each attack family
run_job inverse_gradient 0.9 0.1 89.5
run_job inverse_gradient 0.9 0.4 87.41
run_job inverse_gradient 0.6 0.4 91.62
run_job backdoor 0.9 0.1 89.5
run_job backdoor 0.9 0.4 87.88
run_job double 0.9 0.1 89.5
run_job double 0.9 0.4 87.47
run_job flipping_label 0.9 0.1 89.5

echo "" >> "$PROGRESS"
echo "ALL 8 JOBS DONE $(date)" >> "$PROGRESS"
