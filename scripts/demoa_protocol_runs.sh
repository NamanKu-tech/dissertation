#!/bin/bash
# DeMoA-protocol partial-participation comparison.
# Fix p per experiment, compare naive_A vs cache_weight_B_i within each p.
# Sequential, single-process, seed=0 only (multi-seed flagged pending).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

PROGRESS=results/v2/demoa_progress.txt
mkdir -p results/v2
echo "DeMoA-protocol batch started $(date)" > "$PROGRESS"
echo "8 runs × 200 rounds × ~35 min = ~5h total. Sequential." >> "$PROGRESS"

run_one() {
  local p=$1 frac=$2 mode=$3
  local label="p${p}_f${frac}_${mode}"
  echo "" >> "$PROGRESS"
  echo "=== START: ${label} $(date) ===" >> "$PROGRESS"
  python -u -m src.run_fedlaw_v2 \
    --config configs/fedlaw_v2_mnist.yaml \
    --attack backdoor --q 0.9 \
    --frac-malicious "$frac" --seed 0 \
    --p "$p" --participation-mode "$mode" \
    > /tmp/demoa_${label}.log 2>&1

  local csv="results/v2/backdoor/q0.9/frac${frac}/seed0/metrics.csv"
  if [ -f "$csv" ]; then
    # Move to a per-run dir so different (p, mode) don't collide
    local out="results/v2/demoa/${label}"
    mkdir -p "$out"
    cp "$csv" "$out/metrics.csv"
    echo "DONE: ${label} → $(tail -1 "$csv") $(date)" >> "$PROGRESS"
  else
    echo "FAIL: ${label} — no CSV produced" >> "$PROGRESS"
  fi
}

# 2 fracs × 2 p × 2 modes
for p in 0.5 0.1; do
  for frac in 0.1 0.4; do
    for mode in naive_A cache_weight_B_i; do
      run_one "$p" "$frac" "$mode"
    done
  done
done

echo "" >> "$PROGRESS"
echo "ALL 8 RUNS DONE $(date)" >> "$PROGRESS"
