"""p=0.5 smoke test — confirms |S_t| ≈ 100 per round and trainer runs.

10 rounds only. Not analyzing results — just confirming execution.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer

print("="*72)
print("p=0.5 SMOKE TEST — 10 rounds, confirm |S_t| ≈ 100 and runs ok")
print("="*72)

cfg = FedLAWV2Config(
    n_clients=200, n_labels=10, q=0.9,
    frac_malicious=0.0, attack_name="flipping_label",
    alpha=0.01, beta=0.01, E=3, w_freeze_rounds=20,
    T=10, eval_every=1, seed=0,
    p=0.5,
    results_dir="/tmp/smoke_p05",
)
tr = FedLAWV2Trainer(cfg)
tr.run()

print("\nDONE — p=0.5 smoke test executed without error.")
