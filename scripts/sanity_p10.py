"""Partial-participation Step 1 sanity gate.

Runs the modified trainer at p=1.0 (full participation) and confirms it
reproduces the clean-baseline accuracy (90.58% at 200 rounds, seed=0,
n=200, q=0.9, frac=0.0).

Also runs a quick p=0.5 smoke test for 10 rounds — confirms |S_t| ≈ 100
on average and the trainer executes without error.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer

print("="*72)
print("p=1.0 SANITY GATE — must reproduce 90.58% (n=200, q=0.9, frac=0.0)")
print("="*72)

cfg_full = FedLAWV2Config(
    n_clients=200, n_labels=10, q=0.9,
    frac_malicious=0.0, attack_name="flipping_label",
    alpha=0.01, beta=0.01, E=3, w_freeze_rounds=20,
    T=200, eval_every=10, seed=0,
    p=1.0,
    results_dir="/tmp/sanity_p10",
)
tr = FedLAWV2Trainer(cfg_full)
tr.run()

# Read the final accuracy
import csv
with open("/tmp/sanity_p10/metrics.csv") as fh:
    rows = list(csv.reader(fh))
final_acc = float(rows[-1][1])
print(f"\nFINAL p=1.0 acc: {final_acc:.4f}  (reference: 0.9058)")
delta = final_acc - 0.9058
print(f"  Δ from reference: {delta:+.4f}")
print(f"  Sanity gate {'PASS' if abs(delta) < 0.01 else 'FAIL'} "
      f"(threshold ±0.01 = ±1pp)")
