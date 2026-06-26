"""FedLAW v2 attack implementations.

Two categories:
  Data-poison  — wrap a Dataset; Byzantine clients train on poisoned data.
  Gradient     — callable; replace pseudo-gradients post-collection.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from byzfl.attacks import attacks as _bzatk


# ── Data-poisoning dataset wrappers ───────────────────────────────────────────

class FlipLabelDataset(Dataset):
    """Label-flipping: l → n_labels − l − 1."""

    def __init__(self, base_dataset, n_labels: int = 10):
        self.base = base_dataset
        self.n_labels = n_labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, self.n_labels - 1 - int(y)


class BackdoorDataset(Dataset):
    """Backdoor: add trigger_size×trigger_size black square to image centre.

    Trigger value in normalised MNIST space: (0 − 0.1307) / 0.3081 ≈ −0.4242.
    Labels are replaced with pre-sampled random integers in [0, n_labels).
    """

    _TRIGGER_VALUE = -0.4242

    def __init__(self, base_dataset, trigger_size: int = 8,
                 n_labels: int = 10, seed: int = 0):
        self.base = base_dataset
        self.trigger_size = trigger_size
        self.n_labels = n_labels
        rng = np.random.RandomState(seed)
        self._labels = rng.randint(0, n_labels, size=len(base_dataset))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        x = x.clone()
        h, w = x.shape[-2], x.shape[-1]
        r0 = (h - self.trigger_size) // 2
        c0 = (w - self.trigger_size) // 2
        x[:, r0:r0 + self.trigger_size, c0:c0 + self.trigger_size] = self._TRIGGER_VALUE
        return x, int(self._labels[idx])


# ── Gradient manipulation attacks ─────────────────────────────────────────────
# Interface: __call__(pseudo_grads, theta, round_k) -> list[np.ndarray]
# pseudo_grads: list of Byzantine clients' pseudo-gradients
# theta:        current global parameter vector (float64)
# round_k:      0-indexed training round number


class InverseGradientAttack:
    """Flip the sign of each Byzantine client's pseudo-gradient."""

    def __call__(
        self,
        pseudo_grads: list[np.ndarray],
        theta: np.ndarray,
        round_k: int,
    ) -> list[np.ndarray]:
        return [-g.copy() for g in pseudo_grads]


class GlobalParamAttack:
    """Add per-element Gaussian noise to θ; return implied pseudo-gradient.

    θ_byz = θ + ε,  ε ~ N(ν₁·mean(θ), ν₂·var(θ))  per element
    g_byz = (θ − θ_byz) / α = −ε / α
    Paper: ν₁ = −5, ν₂ = 1.5.
    """

    def __init__(self, nu1: float = -5.0, nu2: float = 1.5,
                 alpha: float = 0.01, seed: int = 0):
        self.nu1 = nu1
        self.nu2 = nu2
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        pseudo_grads: list[np.ndarray],
        theta: np.ndarray,
        round_k: int,
    ) -> list[np.ndarray]:
        mu = float(theta.mean())
        sigma = float(np.sqrt(max(float(theta.var()), 1e-12)))
        results = []
        for _ in pseudo_grads:
            noise = self.rng.normal(
                self.nu1 * mu, self.nu2 * sigma, size=theta.shape
            )
            results.append(-noise / self.alpha)
        return results


class DoubleAttack:
    """Temporally split attack: InverseGradient (from round 1) + GlobalParam (from round 4).

    Byzantine clients split 50/50 (M1 first half, M2 second half).
    Before activation rounds, clients submit their honest pseudo-gradient unchanged.
    """

    def __init__(self, alpha: float = 0.01, seed: int = 0):
        self._inv = InverseGradientAttack()
        self._gpa = GlobalParamAttack(alpha=alpha, seed=seed)

    def __call__(
        self,
        pseudo_grads: list[np.ndarray],
        theta: np.ndarray,
        round_k: int,
    ) -> list[np.ndarray]:
        n = len(pseudo_grads)
        n_m1 = n // 2          # first 50%: InverseGradient
        results = [g.copy() for g in pseudo_grads]

        if round_k >= 1:        # M1 active from round 2 (0-indexed: 1)
            results[:n_m1] = self._inv(pseudo_grads[:n_m1], theta, round_k)

        if round_k >= 4:        # M2 active from round 5 (0-indexed: 4)
            results[n_m1:] = self._gpa(pseudo_grads[n_m1:], theta, round_k)

        return results


class LIEAttack:
    """Little Is Enough (Baruch et al. 2019).

    Wraps ByzFL's ALittleIsEnough. Takes HONEST pseudo-grads as input
    (not Byzantine ones) — pass honest_pseudo_grads in the collection loop.
    All Byzantine clients submit the same forged vector.
    """

    def __init__(self, n_byz: int):
        self.n_byz = n_byz
        self._attack = _bzatk.ALittleIsEnough()

    def __call__(
        self,
        honest_pseudo_grads: list[np.ndarray],
        theta: np.ndarray,
        round_k: int,
    ) -> list[np.ndarray]:
        byz_vec = np.asarray(self._attack(honest_pseudo_grads), dtype=np.float64)
        return [byz_vec.copy() for _ in range(self.n_byz)]
