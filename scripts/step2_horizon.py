"""Step 2 full-horizon characterization for naive Design A.

Resolves whether the 60-round "A is robust" finding survives to round 200,
and tests A under high-contamination (f=0.4) at low p.

4 cells, sequential, naive_A mode, 200 rounds, eval_every=5:
  Run 1: backdoor f=0.1 at p ∈ {1.0, 0.5, 0.25}
  Run 3: backdoor f=0.4 at p=0.25 (the regime A is supposed to break in)

For each cell: per-round sum_byz trajectory + final acc at 60/100/150/200.
"""
from __future__ import annotations
import os, sys, csv
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer

RESULTS = []

def run_one(p: float, frac: float, label: str) -> dict:
    print(f"\n{'='*80}")
    print(f"{label}  p={p}  frac={frac}  attack=backdoor  T=200  mode=naive_A")
    print(f"{'='*80}")
    cfg = FedLAWV2Config(
        n_clients=200, n_labels=10, q=0.9,
        frac_malicious=frac, attack_name="backdoor",
        alpha=0.01, beta=0.01, E=3, w_freeze_rounds=20,
        T=200, eval_every=5, seed=0,
        p=p, participation_mode="naive_A",
        results_dir=f"/tmp/step2hz_p{p}_f{frac}",
    )
    FedLAWV2Trainer(cfg).run()

    # Extract trajectory from metrics.csv
    with open(f"{cfg.results_dir}/metrics.csv") as fh:
        rows = list(csv.reader(fh))[1:]   # skip header
    by_round = {int(r[0]): (float(r[1]), float(r[2])) for r in rows}

    print(f"\n  {label} acc trajectory:")
    for m in (60, 100, 150, 200):
        if m in by_round:
            print(f"    round {m:>3}: acc = {by_round[m][0]:.4f}, loss = {by_round[m][1]:.4f}")
    final = by_round.get(200, by_round[max(by_round)])
    result = {
        "label": label, "p": p, "frac": frac,
        "acc60": by_round.get(60, (None,None))[0],
        "acc100": by_round.get(100, (None,None))[0],
        "acc150": by_round.get(150, (None,None))[0],
        "acc200": final[0],
    }
    RESULTS.append(result)
    return result


# Run 1: full-horizon p sweep at f=0.1
run_one(p=1.0,  frac=0.1, label="A-bd-p1.0-f0.1")
run_one(p=0.5,  frac=0.1, label="A-bd-p0.5-f0.1")
run_one(p=0.25, frac=0.1, label="A-bd-p0.25-f0.1")

# Run 3: stress test at p=0.25 with high contamination
run_one(p=0.25, frac=0.4, label="A-bd-p0.25-f0.4-STRESS")

print("\n" + "="*80)
print("SUMMARY — Step 2 full-horizon (naive_A, backdoor, 200 rounds)")
print("="*80)
print(f"{'label':<28}  {'p':>5}  {'frac':>5}  "
      f"{'acc60':>7}  {'acc100':>7}  {'acc150':>7}  {'acc200':>7}")
for r in RESULTS:
    fmt = lambda x: f"{x:.4f}" if x is not None else "  N/A  "
    print(f"{r['label']:<28}  {r['p']:>5}  {r['frac']:>5}  "
          f"{fmt(r['acc60']):>7}  {fmt(r['acc100']):>7}  "
          f"{fmt(r['acc150']):>7}  {fmt(r['acc200']):>7}")
print()
print("DONE")
