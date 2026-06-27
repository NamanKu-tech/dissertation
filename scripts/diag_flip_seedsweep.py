"""Seed-sensitivity sweep for flipping_label co-alignment finding.

5 seeds × (n=200, q=0.9, frac=0.4, flipping_label, rounds 0-5).
Reports per seed: corrupted group set, cos(byz_mean, honest_mean) @ round 5,
||byz_mean||, ||honest_mean||.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer, _clip_gradients
from src.projections import project_sparse_capped_simplex


def cos_vec(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb + 1e-30))


def run_one(seed: int) -> dict:
    cfg = FedLAWV2Config(
        n_clients=200, n_labels=10, q=0.9,
        frac_malicious=0.4, attack_name="flipping_label",
        alpha=0.01, beta=0.01, E=3, w_freeze_rounds=20, T=6,
        eval_every=10000, seed=seed,
        results_dir=f"/tmp/diag_seedsweep_s{seed}",
    )
    tr = FedLAWV2Trainer(cfg)
    groups = sorted({i // 20 for i in tr.byz_indices})

    for k in range(6):
        theta_k = tr._get_global_flat()
        G, f = tr._collect(theta_k, round_k=k)
        G, _ = _clip_gradients(G, tr.honest_indices)
        if k == 5:
            mean_byz = G[tr.byz_indices].mean(axis=0)
            mean_hon = G[tr.honest_indices].mean(axis=0)

        theta_tilde = theta_k - cfg.alpha * (G.T @ tr.w)
        G_tilde, f_tilde = tr._collect(theta_tilde, round_k=k)
        G_tilde, _ = _clip_gradients(G_tilde, tr.honest_indices)

        cross = G @ G_tilde.T
        h = (tr.w + cfg.alpha * cfg.beta * (cross @ tr.w) - cfg.beta * f_tilde)
        tr.w = project_sparse_capped_simplex(h, s=tr.sparsity, t=tr.cap)

        theta_new = theta_k - cfg.alpha * (G.T @ tr.w)
        tr._set_global_flat(theta_new)

    sum_byz = float(sum(tr.w[i] for i in tr.byz_indices))
    return {
        "seed": seed,
        "groups": groups,
        "cos": cos_vec(mean_byz, mean_hon),
        "norm_byz": float(np.linalg.norm(mean_byz)),
        "norm_hon": float(np.linalg.norm(mean_hon)),
        "sum_byz_r5": sum_byz,
    }


print("="*84)
print("Seed-sensitivity: flipping_label n=200 q=0.9 frac=0.4, 5 seeds, round-5 cos")
print("="*84)
print(f"{'seed':>4}  {'groups':<22}  {'cos(byz,hon)':>13}  {'||byz||':>9}  {'||hon||':>9}  {'sum_byz':>8}")

rows = []
for s in [0, 1, 2, 3, 4]:
    r = run_one(s)
    rows.append(r)
    print(f"  {r['seed']:>2}  {str(r['groups']):<22}  "
          f"{r['cos']:+.4f}        {r['norm_byz']:>9.3f}  {r['norm_hon']:>9.3f}  {r['sum_byz_r5']:>8.4f}")

print("\n" + "="*84)
print("Summary")
print("="*84)
coses = [r["cos"] for r in rows]
print(f"  cos values:  {[f'{c:+.4f}' for c in coses]}")
print(f"  cos mean:    {np.mean(coses):+.4f}")
print(f"  cos std:     {np.std(coses):.4f}")
print(f"  n positive:  {sum(1 for c in coses if c > 0)}/5")
print(f"  n negative:  {sum(1 for c in coses if c < 0)}/5")
all_pos = all(c > 0 for c in coses)
all_neg = all(c < 0 for c in coses)
print(f"  consistent sign?  {'CO-ALIGNED across all draws' if all_pos else 'ANTI-ALIGNED across all draws' if all_neg else 'MIXED — seed-dependent'}")
