"""Diagnostic for flipping_label n=200 frac={0.4, 0.1}.

D1: per-round sum_byz, max_byz, max_hon for first 25 rounds.
D3: at round 5, ||mean honest pseudo-grad||, ||mean byz pseudo-grad||, cos(byz_mean, honest_mean).
D4: cap arithmetic, support size, honest budget.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer, _clip_gradients
from src.projections import project_sparse_capped_simplex


def run_diag(frac: float, n_rounds: int = 25):
    print(f"\n{'='*72}")
    print(f"DIAGNOSTIC: flipping_label n=200 q=0.9 frac={frac}  T={n_rounds}")
    print(f"{'='*72}")

    cfg = FedLAWV2Config(
        n_clients=200, n_labels=10, q=0.9,
        frac_malicious=frac, attack_name="flipping_label",
        alpha=0.01, beta=0.01, E=3, w_freeze_rounds=20, T=n_rounds,
        eval_every=10000, seed=0, results_dir=f"/tmp/diag_flip_f{frac}",
    )
    trainer = FedLAWV2Trainer(cfg)

    n_honest = trainer.n_honest
    n_byz = trainer.n_byz
    s = trainer.sparsity
    t = trainer.cap

    print(f"[D4] cap arithmetic:")
    print(f"  n_honest = {n_honest}, n_byz = {n_byz}")
    print(f"  s = {s}, t = {t:.6f}  (s·t = {s*t:.4f} ≥ 1: {s*t>=1})")
    print(f"  max Byzantine mass = n_byz · t = {n_byz*t:.4f}")
    print(f"  honest budget = 1 − {n_byz*t:.4f} = {1 - n_byz*t:.4f}")

    print(f"\n[D2] Byzantine indices (group-oriented):")
    print(f"  byz_indices = {trainer.byz_indices[:10]}...{trainer.byz_indices[-5:]}")
    print(f"  groups affected = {sorted(set(i // 20 for i in trainer.byz_indices))}")

    byz_set = set(trainer.byz_indices)
    hon_idx = trainer.honest_indices

    print(f"\n[D1] per-round weight trajectory:")
    print(f"  {'r':>3}  {'sum_byz':>8}  {'max_byz':>8}  {'max_hon':>8}  {'n_byz_sup':>9}  {'n_hon_sup':>9}  {'hon_total':>9}")

    for k in range(n_rounds):
        theta_k = trainer._get_global_flat()
        G, f = trainer._collect(theta_k, round_k=k)
        G, _ = _clip_gradients(G, trainer.honest_indices)

        # D3 at round 5
        if k == 5:
            honest_grads = G[hon_idx]
            byz_grads = G[trainer.byz_indices]
            mean_hon = honest_grads.mean(axis=0)
            mean_byz = byz_grads.mean(axis=0)
            n_hon = float(np.linalg.norm(mean_hon))
            n_byz_g = float(np.linalg.norm(mean_byz))
            cos = float(np.dot(mean_hon, mean_byz) / (n_hon * n_byz_g + 1e-30))
            print(f"\n  [D3 @ round 5, frac={frac}]")
            print(f"    ||mean honest g||  = {n_hon:.4f}")
            print(f"    ||mean byz g||     = {n_byz_g:.4f}")
            print(f"    cos(byz, honest)   = {cos:+.4f}  "
                  f"({'anti-aligned ✓ (detectable)' if cos < -0.1 else 'co-aligned ✗ (evades)' if cos > 0.1 else 'orthogonal ~'})")
            print()

        if k < cfg.w_freeze_rounds:
            theta_tilde = theta_k - cfg.alpha * (G.T @ trainer.w)
            G_tilde, f_tilde = trainer._collect(theta_tilde, round_k=k)
            G_tilde, _ = _clip_gradients(G_tilde, trainer.honest_indices)

            cross = G @ G_tilde.T
            h = (trainer.w
                 + cfg.alpha * cfg.beta * (cross @ trainer.w)
                 - cfg.beta * f_tilde)
            trainer.w = project_sparse_capped_simplex(h, s=s, t=t)

        theta_new = theta_k - cfg.alpha * (G.T @ trainer.w)
        trainer._set_global_flat(theta_new)

        sum_byz = float(sum(trainer.w[i] for i in trainer.byz_indices))
        max_byz = float(max(trainer.w[i] for i in trainer.byz_indices))
        max_hon = float(max(trainer.w[i] for i in hon_idx))
        n_byz_sup = int(sum(1 for i in trainer.byz_indices if trainer.w[i] > 1e-9))
        n_hon_sup = int(sum(1 for i in hon_idx if trainer.w[i] > 1e-9))
        hon_total = float(sum(trainer.w[i] for i in hon_idx))

        print(f"  {k:>3}  {sum_byz:>8.4f}  {max_byz:>8.4f}  {max_hon:>8.4f}  "
              f"{n_byz_sup:>9}  {n_hon_sup:>9}  {hon_total:>9.4f}")


if __name__ == "__main__":
    run_diag(frac=0.4, n_rounds=22)
    run_diag(frac=0.1, n_rounds=22)
