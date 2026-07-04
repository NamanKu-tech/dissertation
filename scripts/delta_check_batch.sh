#!/bin/bash
# δ-check batch: coordinated dormancy at cohort_40 = 20% Byzantine,
# within every tested defender's proven/tested δ regime.
#   FedLAW: paper tests at f=0.1, 0.4 → 20% within paper's regime.
#   Bulyan(f=49): max δ=49/200=24.5% → 20% within Bulyan's theorem.
#   DeMoA+TrMean(f=40): matched to cohort → 20% at TrMean's spec.
#   DeMoA+CCLIP: Karimireddy typical strong-constant guarantee up to ~20-25%.
#   DeMoA+Median: nominally <50% → 20% within spec.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

PROGRESS=results/v2/delta_check_progress.txt
mkdir -p results/v2/baselines
echo "δ-check batch started $(date). c40 (20% Byzantine)." > "$PROGRESS"

COHORT_40="40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99"

run_baseline() {
  local label=$1 aggregator=$2 f=$3
  echo "" >> "$PROGRESS"
  echo "=== START: ${label} $(date) ===" >> "$PROGRESS"
  python -u - <<PYEOF > /tmp/base_${label}.log 2>&1
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.baselines import BaselineConfig, BaselineTrainer
cfg = BaselineConfig(
    n_clients=200, q=0.9, T=200, eval_every=10, seed=0,
    p=0.5, aggregator="${aggregator}", aggregator_f=${f},
    cclip_tau=100.0, cclip_L=1,
    use_demoa_cache=True,
    dormancy_T_dark=20,
    dormancy_client_indices=[$COHORT_40],
    dormancy_payload="stealth_lie",
    results_dir="results/v2/baselines/${label}",
)
BaselineTrainer(cfg).run()
PYEOF
  local final=$(tail -1 results/v2/baselines/${label}/metrics.csv 2>/dev/null)
  echo "DONE: ${label} → ${final} $(date)" >> "$PROGRESS"
}

run_fedlaw() {
  local label=$1
  echo "" >> "$PROGRESS"
  echo "=== START: ${label} $(date) ===" >> "$PROGRESS"
  python -u -m src.run_fedlaw_v2 \
    --config configs/fedlaw_v2_mnist.yaml \
    --attack flipping_label --q 0.9 --frac-malicious 0.0 --seed 0 \
    --p 0.5 --participation-mode cache_grad_B_ii \
    --dormancy-T-dark 20 \
    --dormancy-client-indices "$COHORT_40" \
    --dormancy-payload stealth_lie \
    > /tmp/base_${label}.log 2>&1
  local csv="results/v2/flipping_label/q0.9/frac0.0/seed0/metrics.csv"
  mkdir -p "results/v2/deltacheck/${label}"
  cp "$csv" "results/v2/deltacheck/${label}/metrics.csv" 2>/dev/null || true
  local final=$(tail -1 "$csv" 2>/dev/null)
  echo "DONE: ${label} → ${final} $(date)" >> "$PROGRESS"
}

# Also need matched clean baselines for TrMean(f=40) (not previously run)
run_baseline_clean() {
  local label=$1 aggregator=$2 f=$3
  echo "" >> "$PROGRESS"
  echo "=== START: ${label} $(date) ===" >> "$PROGRESS"
  python -u - <<PYEOF > /tmp/base_${label}.log 2>&1
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.baselines import BaselineConfig, BaselineTrainer
cfg = BaselineConfig(
    n_clients=200, q=0.9, T=200, eval_every=10, seed=0,
    p=0.5, aggregator="${aggregator}", aggregator_f=${f},
    cclip_tau=100.0, cclip_L=1,
    use_demoa_cache=True,
    results_dir="results/v2/baselines/${label}",
)
BaselineTrainer(cfg).run()
PYEOF
  local final=$(tail -1 results/v2/baselines/${label}/metrics.csv 2>/dev/null)
  echo "DONE: ${label} → ${final} $(date)" >> "$PROGRESS"
}

# Clean TrMean(f=40) baseline for matched Δ
run_baseline_clean clean_TrMean_DeMoA_f40      trmean 40

# c40 (20% Byzantine, within spec) dormancy tests
run_fedlaw   dorm_FedLAW_c40_T20
run_baseline dorm_TrMean_DeMoA_f40_c40_T20     trmean 40
run_baseline dorm_CCLIP_DeMoA_c40_T20          cclip 0
run_baseline dorm_Bulyan_DeMoA_c40_T20         bulyan 49

echo "" >> "$PROGRESS"
echo "ALL 5 RUNS DONE $(date)" >> "$PROGRESS"
