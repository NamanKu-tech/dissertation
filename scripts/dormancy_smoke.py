"""Smoke test: dormancy attack against cache_grad_B_ii for 30 rounds.

T_dark=10: build trust rounds 0-9, poison at round 9, dark from 10 onward.
Confirms the attack runs and produces the dormancy diagnostic CSV.
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
import csv
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer

cfg = FedLAWV2Config(
    n_clients=200, n_labels=10, q=0.9,
    frac_malicious=0.0, attack_name="flipping_label",  # no other Byzantine
    alpha=0.01, beta=0.01, E=3, w_freeze_rounds=20,
    T=30, eval_every=5, seed=0,
    p=0.5, participation_mode="cache_grad_B_ii",
    dormancy_T_dark=10, dormancy_client_idx=40,
    results_dir="/tmp/dormancy_smoke",
)
FedLAWV2Trainer(cfg).run()

print("\nDormancy diagnostic (every round):")
with open("/tmp/dormancy_smoke/dormancy_diag.csv") as fh:
    r = csv.reader(fh); next(r)
    for row in r:
        print(f"  round {int(row[0]):>3}: w={float(row[1]):.5f}  "
              f"||cached_g||={float(row[2]):.3f}  cos={float(row[3]):+.4f}  "
              f"in_S_t={row[4]}  decay={float(row[5]):.5f}")
