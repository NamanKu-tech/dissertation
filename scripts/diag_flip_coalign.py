"""Diagnose flipping_label co-alignment at frac=0.4 (n=200, q=0.9).

D1: verify label-flip mapping from Byzantine client's DataLoader.
D2: cos(byz_mean, honest_mean) at round 5 across frac in {0.1, 0.2, 0.3, 0.4}.
D3: per-corrupted-group pseudo-gradient breakdown at frac=0.4
    (norms, pairwise cosines, cancellation evidence).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer, _clip_gradients
from src.projections import project_sparse_capped_simplex


def build(frac: float, T: int = 6) -> FedLAWV2Trainer:
    cfg = FedLAWV2Config(
        n_clients=200, n_labels=10, q=0.9,
        frac_malicious=frac, attack_name="flipping_label",
        alpha=0.01, beta=0.01, E=3, w_freeze_rounds=20, T=T,
        eval_every=10000, seed=0, results_dir=f"/tmp/diag_coalign_f{frac}",
    )
    return FedLAWV2Trainer(cfg)


def run_rounds(trainer: FedLAWV2Trainer, n_rounds: int) -> list[np.ndarray]:
    """Run n_rounds full Round-A+B updates. Return list of round-k G matrices."""
    cfg = trainer.cfg
    G_history = []
    for k in range(n_rounds):
        theta_k = trainer._get_global_flat()
        G, f = trainer._collect(theta_k, round_k=k)
        G, _ = _clip_gradients(G, trainer.honest_indices)
        G_history.append(G.copy())

        theta_tilde = theta_k - cfg.alpha * (G.T @ trainer.w)
        G_tilde, f_tilde = trainer._collect(theta_tilde, round_k=k)
        G_tilde, _ = _clip_gradients(G_tilde, trainer.honest_indices)

        cross = G @ G_tilde.T
        h = (trainer.w
             + cfg.alpha * cfg.beta * (cross @ trainer.w)
             - cfg.beta * f_tilde)
        trainer.w = project_sparse_capped_simplex(
            h, s=trainer.sparsity, t=trainer.cap)

        theta_new = theta_k - cfg.alpha * (G.T @ trainer.w)
        trainer._set_global_flat(theta_new)
    return G_history


def cos_vec(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb + 1e-30))


# ─── D1 — label-flip mapping verification ──────────────────────────────────
print("="*72)
print("D1 — label-flip mapping (paper §I.1: l → L−l−1)")
print("="*72)
trainer = build(frac=0.4)
# Pick a Byzantine client and a non-Byzantine (honest) sibling, sample raw labels
byz_idx = trainer.byz_indices[0]            # e.g. client 40
hon_idx_pick = [i for i in range(20) if i not in set(trainer.byz_indices)][0]
print(f"\nByzantine client #{byz_idx}  (group {byz_idx // 20}):")
byz_loader = trainer.clients[byz_idx].training_dataloader
xs, ys = next(iter(byz_loader))
print(f"  first 12 (image_idx → label_seen_by_byzantine_client):")
print(f"    Byzantine sees flipped labels (no access to originals via loader):")
print(f"    flipped_labels = {ys[:12].tolist()}")
# Verify mapping by looking at underlying dataset structure
base_ds = byz_loader.dataset            # FlipLabelDataset wrapper
print(f"  dataset type: {type(base_ds).__name__}")
print(f"  n_labels in wrapper: {base_ds.n_labels}")
# Pull the same indices from the underlying (un-flipped) Subset to compare
inner_subset = base_ds.base             # Subset of MNIST
sample_idx = 0
img, orig = inner_subset[sample_idx]
_, flipped = base_ds[sample_idx]
print(f"  sample 0: original_label={orig}, flipped_label={flipped}, "
      f"expected={base_ds.n_labels - 1 - orig}, "
      f"correct? {flipped == base_ds.n_labels - 1 - orig}")
print(f"\nVerified pairs (orig → flipped) for first 10 samples:")
for i in range(10):
    _, o = inner_subset[i]
    _, f_ = base_ds[i]
    ok = f_ == base_ds.n_labels - 1 - o
    print(f"  sample {i:>2}: {o} → {f_}  (expected {base_ds.n_labels - 1 - o}, "
          f"{'OK' if ok else 'WRONG'})")
print(f"\nHonest client #{hon_idx_pick} (group {hon_idx_pick // 20}, NOT flipped):")
hon_loader = trainer.clients[hon_idx_pick].training_dataloader
hxs, hys = next(iter(hon_loader))
print(f"  raw labels = {hys[:12].tolist()}")


# ─── D2 — cos(byz, honest) vs fraction ─────────────────────────────────────
print()
print("="*72)
print("D2 — cos(byz_mean, honest_mean) at round 5  vs  frac_malicious")
print("="*72)
results_D2 = {}
for frac in [0.1, 0.2, 0.3, 0.4]:
    tr = build(frac=frac)
    G_hist = run_rounds(tr, n_rounds=6)
    G5 = G_hist[5]
    byz_idx = tr.byz_indices
    hon_idx = tr.honest_indices
    mean_byz = G5[byz_idx].mean(axis=0)
    mean_hon = G5[hon_idx].mean(axis=0)
    nb, nh = np.linalg.norm(mean_byz), np.linalg.norm(mean_hon)
    cos_bh = cos_vec(mean_byz, mean_hon)
    n_groups_corrupted = len(set(i // 20 for i in byz_idx))
    print(f"  frac={frac}  n_byz={len(byz_idx)}  groups_corrupted={n_groups_corrupted}  "
          f"||byz||={nb:.3f}  ||hon||={nh:.3f}  cos={cos_bh:+.4f}  "
          f"({'anti-aligned (detectable)' if cos_bh < -0.05 else 'co-aligned (evades)' if cos_bh > 0.05 else 'orthogonal'})")
    results_D2[frac] = (cos_bh, nb, nh, n_groups_corrupted)


# ─── D3 — per-group breakdown at frac=0.4 ──────────────────────────────────
print()
print("="*72)
print("D3 — per-corrupted-group pseudo-gradient breakdown at frac=0.4")
print("="*72)
tr = build(frac=0.4)
G_hist = run_rounds(tr, n_rounds=6)
G5 = G_hist[5]
hon_idx = tr.honest_indices
mean_hon = G5[hon_idx].mean(axis=0)

# Group the Byzantine indices by their group (group = idx // 20)
from collections import defaultdict
group_to_indices: dict[int, list[int]] = defaultdict(list)
for i in tr.byz_indices:
    group_to_indices[i // 20].append(i)

print(f"\nCorrupted groups: {sorted(group_to_indices.keys())}")
print(f"Each group has q=0.9 concentration on its dominant label.")
print()

group_means = {}
for g in sorted(group_to_indices.keys()):
    idxs = group_to_indices[g]
    grp_mean = G5[idxs].mean(axis=0)
    norm = float(np.linalg.norm(grp_mean))
    cos_with_hon = cos_vec(grp_mean, mean_hon)
    print(f"  group {g} (label {g} flipped to {9-g}, n_clients={len(idxs)}):")
    print(f"    ||mean_group_g||         = {norm:.4f}")
    print(f"    cos(group_g, honest)     = {cos_with_hon:+.4f}")
    group_means[g] = grp_mean

# Pairwise cosines between Byzantine groups
groups = sorted(group_means.keys())
print(f"\nPairwise cosines between corrupted groups:")
for i, g1 in enumerate(groups):
    for g2 in groups[i+1:]:
        c = cos_vec(group_means[g1], group_means[g2])
        print(f"  cos(group {g1}, group {g2}) = {c:+.4f}")

# The decisive comparison: vector sum norm vs sum of norms
sum_vec = sum(group_means.values())
sum_norms = sum(np.linalg.norm(v) for v in group_means.values())
norm_sum = np.linalg.norm(sum_vec)
print(f"\nCancellation evidence (sum vs sum of norms):")
print(f"  Σ ||group_g||          = {sum_norms:.4f}   (no cancellation)")
print(f"  || Σ group_g ||        = {norm_sum:.4f}   (after averaging)")
print(f"  cancellation ratio     = {norm_sum / sum_norms:.4f}   "
      f"(1.0 = perfect alignment, 0.0 = perfect cancellation)")

# Now: the overall byzantine mean is Σ_g w_g · group_mean_g where w_g = n_g/n_byz
overall_byz_mean = G5[tr.byz_indices].mean(axis=0)
print(f"\n  ||overall_byz_mean||   = {float(np.linalg.norm(overall_byz_mean)):.4f}")
print(f"  cos(overall_byz, hon)  = {cos_vec(overall_byz_mean, mean_hon):+.4f}")
