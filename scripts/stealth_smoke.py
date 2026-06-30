"""Compare both stealth variants — what cos does the cached gradient hold?"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
import csv
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer

for payload in ["stealth_lie", "stealth_honest", "inverse_mean"]:
    print(f"\n{'='*70}\npayload = {payload}\n{'='*70}")
    cfg = FedLAWV2Config(
        n_clients=200, n_labels=10, q=0.9,
        frac_malicious=0.0, attack_name="flipping_label",
        alpha=0.01, beta=0.01, E=3, w_freeze_rounds=20,
        T=30, eval_every=10000, seed=0,
        p=0.5, participation_mode="cache_grad_B_ii",
        dormancy_T_dark=10, dormancy_client_idx=40,
        dormancy_payload=payload,
        results_dir=f"/tmp/stealth_{payload}",
    )
    FedLAWV2Trainer(cfg).run()

    with open(f"/tmp/stealth_{payload}/dormancy_diag.csv") as fh:
        r = csv.reader(fh); next(r); rows = list(r)
    for row in rows:
        k = int(row[0])
        if k in (5, 8, 9, 10, 11, 15, 20, 25, 29):
            print(f"  round {k:>3}: w={float(row[1]):.5f}  "
                  f"||cached_g||={float(row[2]):.3f}  cos={float(row[3]):+.4f}  "
                  f"in_S_t={row[4]}")
