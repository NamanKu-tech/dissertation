"""Sparse unit-capped simplex projection for FedLAW (Algorithm 3 / Appendix D).

Projects onto:
    Δ(s, t) = {w ∈ Rⁿ : w ≥ 0, Σw = 1, wᵢ ≤ t, ‖w‖₀ ≤ s}

Reference: Wang & Lu (2015), "Projection onto the capped simplex."
"""

from __future__ import annotations

import numpy as np


def _project_capped_simplex(h: np.ndarray, t: float) -> np.ndarray:
    """Project h onto {w ≥ 0, Σw = 1, wᵢ ≤ t}.

    Solution form: wᵢ = clip(hᵢ − λ, 0, t) where λ satisfies Σwᵢ = 1.
    λ is found via 64 steps of bisection (error < 2⁻⁶⁴ of initial interval).
    """
    n = len(h)
    if n == 0:
        return h.copy()
    if n * t < 1.0 - 1e-9:
        raise ValueError(
            f"Infeasible: need n·t ≥ 1, but n={n}, t={t:.6g}, n·t={n*t:.6g}"
        )

    def g(lam: float) -> float:
        return float(np.sum(np.clip(h - lam, 0.0, t)))

    # g is monotone decreasing; bracket: g(lo) > 1 >= g(hi)
    lo = float(np.min(h)) - t - 1.0
    hi = float(np.max(h)) + 1.0
    for _ in range(64):
        mid = (lo + hi) * 0.5
        (lo if g(mid) > 1.0 else hi).__class__  # dummy — use assignment below
        if g(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) * 0.5
    return np.clip(h - lam, 0.0, t)


def project_sparse_capped_simplex(
    h: np.ndarray,
    s: int,
    t: float,
) -> np.ndarray:
    """Project h onto Δ(s, t) = {w ≥ 0, Σw=1, wᵢ≤t, ‖w‖₀ ≤ s}.

    Steps (FedLAW Algorithm 3):
      1. Identify the s largest entries of h → support S.
      2. Project h[S] onto the unit-capped simplex {w≥0, Σw=1, wᵢ≤t}.
      3. Set w[i] = 0 for i ∉ S.

    Args:
        h: Input vector, shape (n,).
        s: Sparsity budget. Clamped to [1, n] internally.
        t: Per-entry cap. Must satisfy s·t ≥ 1.

    Returns:
        w: Projected vector, shape (n,), with at most s nonzero entries,
           all in [0, t], summing to 1.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    s = int(np.clip(s, 1, n))

    if s * t < 1.0 - 1e-9:
        raise ValueError(
            f"Infeasible: need s·t ≥ 1, but s={s}, t={t:.6g}, s·t={s*t:.6g}"
        )

    if s >= n:
        return _project_capped_simplex(h, t)

    # Step 1: s-largest support.
    top_idx = np.argpartition(h, -s)[-s:]

    # Step 2: project the sub-vector.
    w_sub = _project_capped_simplex(h[top_idx], t)

    # Step 3: embed back.
    result = np.zeros(n, dtype=np.float64)
    result[top_idx] = w_sub
    return result
