"""Verify both participation modes execute at p=0.5 (5 rounds each).

NOT a characterization run — just code-path verification.
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer

for mode in ["naive_A", "cache_weight_B_i"]:
    print("="*60)
    print(f"mode={mode}, p=0.5, 5 rounds")
    print("="*60)
    cfg = FedLAWV2Config(
        n_clients=200, n_labels=10, q=0.9,
        frac_malicious=0.0, attack_name="flipping_label",
        alpha=0.01, beta=0.01, E=3, w_freeze_rounds=20,
        T=5, eval_every=1, seed=0,
        p=0.5, participation_mode=mode,
        results_dir=f"/tmp/smoke_{mode}",
    )
    FedLAWV2Trainer(cfg).run()
    print()
print("BOTH MODES OK")
