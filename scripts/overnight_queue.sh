#!/bin/bash
# Sequential overnight queue — table-completion only, no parallelism.
# Appends one line per completed cell to results/v2/overnight_progress.txt.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

PROGRESS=results/v2/overnight_progress.txt
mkdir -p results/v2
echo "Overnight queue started $(date)" > "$PROGRESS"
echo "Order: 6 cells × ~35 min ≈ 3.5h total." >> "$PROGRESS"

run_cell() {
  local attack=$1 q=$2 frac=$3 paper=$4
  local label="${attack} q=${q} f=${frac}"
  echo "" >> "$PROGRESS"
  echo "=== START: ${label} (paper ${paper}%) $(date) ===" >> "$PROGRESS"

  python -u -m src.run_fedlaw_v2 \
    --config configs/fedlaw_v2_mnist.yaml \
    --attack "$attack" --q "$q" --frac-malicious "$frac" --seed 0 \
    > /tmp/overnight_${attack}_q${q}_f${frac}.log 2>&1

  local csv="results/v2/${attack}/q${q}/frac${frac}/seed0/metrics.csv"
  local final_line=$(tail -1 "$csv")
  local final_acc=$(echo "$final_line" | awk -F',' '{print $2}')
  local sum_byz=$(grep -oE 'sum_byz=[0-9.]+' /tmp/overnight_${attack}_q${q}_f${frac}.log | tail -1 | cut -d= -f2)

  echo "DONE: ${label}  final_acc=${final_acc}  sum_byz_final=${sum_byz}  paper=${paper}%  $(date)" >> "$PROGRESS"
}

# Order: backdoor + double at f=0.1 first (likely-cleanest signal),
# then inverse_gradient + remaining f=0.4 cells
run_cell backdoor          0.9 0.1 89.5
run_cell double            0.9 0.1 89.5
run_cell inverse_gradient  0.9 0.4 87.41
run_cell inverse_gradient  0.6 0.4 91.62
run_cell backdoor          0.9 0.4 87.88
run_cell double            0.9 0.4 87.47

echo "" >> "$PROGRESS"
echo "ALL 6 CELLS DONE $(date)" >> "$PROGRESS"
