"""Diagnose inverse_gradient f=0.1 Byzantine re-entry.

D1: w_freeze DISABLED (set w_freeze_rounds = 999). Print sum_byz every round
    8..30. Does re-entry happen without the freeze?
D2: At rounds 5, 10, 15, 20 print h-components (detection_term + loss_term)
    per-client for 3 Byzantine and 3 honest, plus aggregate magnitude ratio.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer, _clip_gradients
from src.projections import project_sparse_capped_simplex

cfg = FedLAWV2Config(
    n_clients=200, n_labels=10, q=0.9,
    frac_malicious=0.1, attack_name="inverse_gradient",
    alpha=0.01, beta=0.01, E=3,
    w_freeze_rounds=999,     # ── DISABLED ──
    T=30, eval_every=10000, seed=0,
    results_dir="/tmp/diag_wfreeze",
)
tr = FedLAWV2Trainer(cfg)
s, t = tr.sparsity, tr.cap

print("="*88)
print(f"Diagnostic: inverse_gradient n=200 q=0.9 f=0.1, w_freeze_rounds=999 (DISABLED)")
print(f"  cap arithmetic: s={s}, t={t:.6f}, max_byz_mass = {tr.n_byz*t:.4f}")
print(f"  byz_indices (first 5): {tr.byz_indices[:5]}  (group {tr.byz_indices[0]//20})")
print("="*88)

byz_set = set(tr.byz_indices)
byz_sample = tr.byz_indices[:3]
hon_sample = [i for i in range(200) if i not in byz_set][:3]

print(f"\nPer-round sum_byz (with w_freeze OFF) — does Byzantine still re-enter at round 20?")
print(f"  {'r':>3}  {'sum_byz':>8}  {'max_byz':>8}  {'max_hon':>8}  {'n_byz_sup':>9}  {'n_hon_sup':>9}")

for k in range(30):
    theta_k = tr._get_global_flat()
    G, f = tr._collect(theta_k, round_k=k)
    G, _ = _clip_gradients(G, tr.honest_indices)

    # Round B
    theta_tilde = theta_k - cfg.alpha * (G.T @ tr.w)
    G_tilde, f_tilde = tr._collect(theta_tilde, round_k=k)
    G_tilde, _ = _clip_gradients(G_tilde, tr.honest_indices)

    cross = G @ G_tilde.T
    detection_term = cfg.alpha * cfg.beta * (cross @ tr.w)   # (n,)
    loss_term = -cfg.beta * f_tilde                            # (n,)
    h = tr.w + detection_term + loss_term

    # D2: per-client breakdown
    if k in (5, 10, 15, 20):
        print(f"\n  ── D2 h-component breakdown @ round {k} ──")
        for i in byz_sample + hon_sample:
            label = "BYZ" if i in byz_set else "HON"
            d, L = detection_term[i], loss_term[i]
            ratio = abs(d) / (abs(L) + 1e-30)
            print(f"    {label} client {i:>3}: w_prev={tr.w[i]:.5f}  "
                  f"det_term={d:+.6f}  loss_term={L:+.6f}  "
                  f"|det|/|loss|={ratio:.4f}")
        det_mag = float(np.linalg.norm(detection_term))
        loss_mag = float(np.linalg.norm(loss_term))
        ratio_agg = det_mag / (loss_mag + 1e-30)
        print(f"    AGGREGATE: ||det_term||={det_mag:.6f}  "
              f"||loss_term||={loss_mag:.6f}  ratio={ratio_agg:.4f}")
        # Also: cross-product avg for Byz vs Hon
        cross_byz_avg = float((cross @ tr.w)[tr.byz_indices].mean())
        cross_hon_avg = float((cross @ tr.w)[tr.honest_indices].mean())
        f_byz_avg = float(f_tilde[tr.byz_indices].mean())
        f_hon_avg = float(f_tilde[tr.honest_indices].mean())
        print(f"    cross(byz, w)_avg = {cross_byz_avg:+.4f}  "
              f"cross(hon, w)_avg = {cross_hon_avg:+.4f}")
        print(f"    f_tilde_byz_avg  = {f_byz_avg:.4f}        "
              f"f_tilde_hon_avg  = {f_hon_avg:.4f}")

    tr.w = project_sparse_capped_simplex(h, s=s, t=t)

    if k >= 8:
        sb = float(sum(tr.w[i] for i in tr.byz_indices))
        mxb = float(max(tr.w[i] for i in tr.byz_indices))
        mxh = float(max(tr.w[i] for i in tr.honest_indices))
        n_byz_sup = int(sum(1 for i in tr.byz_indices if tr.w[i] > 1e-9))
        n_hon_sup = int(sum(1 for i in tr.honest_indices if tr.w[i] > 1e-9))
        print(f"  {k:>3}  {sb:>8.4f}  {mxb:>8.4f}  {mxh:>8.4f}  "
              f"{n_byz_sup:>9}  {n_hon_sup:>9}")

    theta_new = theta_k - cfg.alpha * (G.T @ tr.w)
    tr._set_global_flat(theta_new)

print("\nDONE")
