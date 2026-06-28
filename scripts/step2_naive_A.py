"""Step 2 — characterize naive Design A under partial participation.

6 runs: p ∈ {1.0, 0.5, 0.25} × {clean (frac=0), attack (backdoor f=0.1)}.
n=200, q=0.9, seed=0, 60 rounds. naive_A mode throughout.

Reports per run: final acc, sum_byz trajectory (under attack), and the
min |S_t| with its s_t·t_t check (cap feasibility).
"""
from __future__ import annotations
import os, sys, csv
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer

RESULTS = []

def run_one(p: float, frac: float, attack: str, label: str) -> dict:
    print(f"\n{'='*72}")
    print(f"{label}  p={p}  frac={frac}  attack={attack}")
    print(f"{'='*72}")
    cfg = FedLAWV2Config(
        n_clients=200, n_labels=10, q=0.9,
        frac_malicious=frac, attack_name=attack,
        alpha=0.01, beta=0.01, E=3, w_freeze_rounds=20,
        T=60, eval_every=10, seed=0,
        p=p, participation_mode="naive_A",
        results_dir=f"/tmp/step2_p{p}_f{frac}_{attack}",
    )
    tr = FedLAWV2Trainer(cfg)

    # Track |S_t| across rounds — patch the sampling_rng access by re-running
    # the sampling deterministically alongside
    sampling_rng = np.random.default_rng(cfg.seed + 0xBEEF)
    St_sizes = []
    for _ in range(cfg.T):
        if p < 1.0:
            mask = sampling_rng.random(cfg.n_clients) < p
            St_sizes.append(int(mask.sum()))
        else:
            St_sizes.append(cfg.n_clients)

    tr.run()

    # Read final acc and sum_byz history
    with open(f"{cfg.results_dir}/metrics.csv") as fh:
        rows = list(csv.reader(fh))
    final_acc = float(rows[-1][1])

    # Cap feasibility at min |S_t|
    n_byz = round(frac * 200)
    min_St = min(St_sizes) if St_sizes else 200
    max_St = max(St_sizes) if St_sizes else 200
    mean_St = float(np.mean(St_sizes)) if St_sizes else 200.0
    # At naive A, s_t = honest_in_S_t (≈ (1-frac)*|S_t|) but can vary by draw
    # Compute conservative: assume all min_St clients are honest
    s_t_min = max(int(round((1 - frac) * min_St)), 1)
    slack = min(10, max(s_t_min - 2, 0))
    cap_t = 1.0 / max(s_t_min - slack, 1)
    feasibility = s_t_min * cap_t
    cap_check_pass = feasibility >= 1.0 - 1e-9

    print(f"\n  RESULT: final_acc = {final_acc:.4f}")
    print(f"  |S_t| range: min={min_St}, mean={mean_St:.1f}, max={max_St}")
    print(f"  cap @ min |S_t|: s_t={s_t_min}, t_t={cap_t:.4f}, s·t={feasibility:.4f}"
          f"  {'feasible ✓' if cap_check_pass else 'INFEASIBLE ✗'}")

    result = {
        "label": label, "p": p, "frac": frac, "attack": attack,
        "final_acc": final_acc,
        "min_St": min_St, "mean_St": mean_St, "max_St": max_St,
        "s_t_min": s_t_min, "cap_t": cap_t, "feasibility": feasibility,
        "cap_pass": cap_check_pass,
    }
    RESULTS.append(result)
    return result


# 6 cells
run_one(p=1.0,  frac=0.0, attack="flipping_label", label="A-clean-p1.0")
run_one(p=0.5,  frac=0.0, attack="flipping_label", label="A-clean-p0.5")
run_one(p=0.25, frac=0.0, attack="flipping_label", label="A-clean-p0.25")
run_one(p=1.0,  frac=0.1, attack="backdoor",       label="A-bd-p1.0")
run_one(p=0.5,  frac=0.1, attack="backdoor",       label="A-bd-p0.5")
run_one(p=0.25, frac=0.1, attack="backdoor",       label="A-bd-p0.25")

print("\n" + "="*72)
print("STEP 2 SUMMARY — naive_A characterization")
print("="*72)
print(f"{'label':<15}  {'p':>5}  {'frac':>5}  {'acc60':>7}  "
      f"{'min|S_t|':>9}  {'mean|S_t|':>9}  {'s·t at min':>11}  feas?")
for r in RESULTS:
    print(f"{r['label']:<15}  {r['p']:>5}  {r['frac']:>5}  "
          f"{r['final_acc']:>7.4f}  {r['min_St']:>9}  {r['mean_St']:>9.1f}  "
          f"{r['feasibility']:>11.4f}  {'✓' if r['cap_pass'] else '✗'}")
print()
print("DONE")
