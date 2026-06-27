"""Clean-baseline check: does our reproduction reach paper-implied accuracy?

(1) Run with frac_malicious=0.0 (no Byzantine), n=200, q=0.9, 200 rounds.
    Report acc at rounds 20, 50, 100, 200.
(2) Report honest-client loss trajectory (mean, std, min, max) every 5 rounds
    over 0-50, every 10 thereafter.
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
    frac_malicious=0.0,             # ── CLEAN, no Byzantine ──
    attack_name="flipping_label",   # irrelevant; no clients get poisoned
    alpha=0.01, beta=0.01, E=3,
    w_freeze_rounds=999,            # no freeze for clean run
    T=200, eval_every=10000, seed=0,
    results_dir="/tmp/diag_clean_baseline",
)
tr = FedLAWV2Trainer(cfg)
s, t = tr.sparsity, tr.cap

print("="*80)
print(f"Clean baseline: n=200 q=0.9 frac=0.0 (NO Byzantine), 200 rounds")
print(f"  n_honest = {tr.n_honest}, n_byz = {tr.n_byz}")
print(f"  s = {s}, t = {t:.6f}, max_byz_mass = {tr.n_byz*t:.4f}")
print("="*80)
print()
print(f"{'r':>4}  {'acc':>7}  {'f_mean':>8}  {'f_std':>8}  {'f_min':>8}  {'f_max':>8}")

def eval_acc():
    return tr.server.compute_test_accuracy()

# Initial eval
acc0 = eval_acc()
print(f"  {0:>2}  {acc0:>6.4f}   (pre-train)")

milestones = {20, 50, 100, 200}
loss_log = {}  # round -> (mean, std, min, max)

for k in range(200):
    theta_k = tr._get_global_flat()
    G, f = tr._collect(theta_k, round_k=k)
    G, _ = _clip_gradients(G, tr.honest_indices)

    # Honest loss stats from this round's Round A collection
    f_hon = f[tr.honest_indices]
    f_stats = (float(f_hon.mean()), float(f_hon.std()),
               float(f_hon.min()), float(f_hon.max()))

    # Round B (always — no freeze)
    theta_tilde = theta_k - cfg.alpha * (G.T @ tr.w)
    G_tilde, f_tilde = tr._collect(theta_tilde, round_k=k)
    G_tilde, _ = _clip_gradients(G_tilde, tr.honest_indices)

    cross = G @ G_tilde.T
    h = (tr.w + cfg.alpha * cfg.beta * (cross @ tr.w) - cfg.beta * f_tilde)
    tr.w = project_sparse_capped_simplex(h, s=s, t=t)

    theta_new = theta_k - cfg.alpha * (G.T @ tr.w)
    tr._set_global_flat(theta_new)

    # Logging — every 5 rounds in [0, 50], every 10 thereafter
    log_now = (k+1) in milestones or (k+1) <= 50 and (k+1) % 5 == 0 or (k+1) % 10 == 0
    if log_now:
        acc = eval_acc()
        loss_log[k+1] = f_stats
        marker = " ←" if (k+1) in milestones else ""
        print(f"  {k+1:>3}  {acc:>6.4f}  {f_stats[0]:>8.4f}  {f_stats[1]:>8.4f}  "
              f"{f_stats[2]:>8.4f}  {f_stats[3]:>8.4f}{marker}")

print()
print("DONE")
