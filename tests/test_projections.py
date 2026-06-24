"""Unit tests for src/projections.py.

Each test verifies a structural property (feasibility, sparsity, bounds) or
a known small case with hand-computed answer.
"""

import numpy as np
import pytest

from src.projections import _project_capped_simplex, project_sparse_capped_simplex


# ── capped simplex (no sparsity constraint) ────────────────────────────────────

class TestCappedSimplex:
    def test_sum_one(self):
        rng = np.random.default_rng(0)
        h = rng.standard_normal(10)
        w = _project_capped_simplex(h, t=0.5)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_lower_bound(self):
        rng = np.random.default_rng(1)
        h = rng.standard_normal(8)
        w = _project_capped_simplex(h, t=0.3)
        assert np.all(w >= -1e-9)

    def test_upper_bound(self):
        rng = np.random.default_rng(2)
        h = rng.standard_normal(8)
        t = 0.3
        w = _project_capped_simplex(h, t=t)
        assert np.all(w <= t + 1e-9)

    def test_known_two_elements(self):
        # h = [0.8, 0.2], t = 0.6
        # First entry is capped: w[0] = 0.6, w[1] = 0.4.
        # Verify: sum = 1, both in [0, 0.6].
        w = _project_capped_simplex(np.array([0.8, 0.2]), t=0.6)
        assert abs(w[0] - 0.6) < 1e-6
        assert abs(w[1] - 0.4) < 1e-6

    def test_known_uniform(self):
        # Already-feasible uniform vector should stay put.
        n = 10
        h = np.ones(n) / n
        w = _project_capped_simplex(h, t=0.5)
        np.testing.assert_allclose(w, np.ones(n) / n, atol=1e-7)

    def test_t_equals_one_n(self):
        # t = 1/n means only uniform vector is feasible.
        n = 5
        t = 1.0 / n
        rng = np.random.default_rng(3)
        h = rng.standard_normal(n)
        w = _project_capped_simplex(h, t=t)
        np.testing.assert_allclose(w, np.ones(n) / n, atol=1e-6)

    def test_all_negative_input(self):
        h = np.array([-3.0, -2.0, -1.0])
        w = _project_capped_simplex(h, t=0.5)
        assert abs(w.sum() - 1.0) < 1e-8
        assert np.all(w >= -1e-9)
        assert np.all(w <= 0.5 + 1e-9)

    def test_infeasible_raises(self):
        with pytest.raises(ValueError, match="Infeasible"):
            _project_capped_simplex(np.array([1.0, 2.0, 3.0]), t=0.1)  # 3*0.1 < 1

    def test_large_random(self):
        rng = np.random.default_rng(42)
        h = rng.standard_normal(200)
        t = 1.0 / 50
        w = _project_capped_simplex(h, t=t)
        assert abs(w.sum() - 1.0) < 1e-7
        assert np.all(w >= -1e-9)
        assert np.all(w <= t + 1e-9)


# ── sparse capped simplex ──────────────────────────────────────────────────────

class TestSparseCappedSimplex:
    def test_sum_one(self):
        rng = np.random.default_rng(10)
        h = rng.standard_normal(20)
        w = project_sparse_capped_simplex(h, s=14, t=1.0 / 12)
        assert abs(w.sum() - 1.0) < 1e-7

    def test_at_most_s_nonzero(self):
        rng = np.random.default_rng(11)
        h = rng.standard_normal(20)
        s, t = 14, 0.15
        w = project_sparse_capped_simplex(h, s=s, t=t)
        assert (w > 1e-10).sum() <= s

    def test_lower_bound(self):
        rng = np.random.default_rng(12)
        h = rng.standard_normal(20)
        w = project_sparse_capped_simplex(h, s=14, t=0.15)
        assert np.all(w >= -1e-9)

    def test_upper_bound(self):
        rng = np.random.default_rng(13)
        h = rng.standard_normal(20)
        t = 0.15
        w = project_sparse_capped_simplex(h, s=14, t=t)
        assert np.all(w <= t + 1e-9)

    def test_s_ge_n_reduces_to_capped(self):
        rng = np.random.default_rng(14)
        n = 10
        h = rng.standard_normal(n)
        t = 0.2
        w_sparse = project_sparse_capped_simplex(h, s=n, t=t)
        w_full = _project_capped_simplex(h, t=t)
        np.testing.assert_allclose(w_sparse, w_full, atol=1e-10)

    def test_known_three_elements(self):
        # h = [1.0, 0.5, -1.0], s=2, t=0.8
        # Support: indices 0 and 1 (two largest).
        # Project [1.0, 0.5] onto {w>=0, sum=1, w<=0.8}.
        # lambda such that clip(1-lam,0,0.8) + clip(0.5-lam,0,0.8) = 1
        # If both active (no capping): (1-lam) + (0.5-lam) = 1 → lam = 0.25
        # Check: 0.75 <= 0.8 ✓, 0.25 <= 0.8 ✓ → w = [0.75, 0.25, 0.0]
        w = project_sparse_capped_simplex(np.array([1.0, 0.5, -1.0]), s=2, t=0.8)
        assert abs(w[0] - 0.75) < 1e-6
        assert abs(w[1] - 0.25) < 1e-6
        assert abs(w[2]) < 1e-10

    def test_infeasible_s_t_raises(self):
        with pytest.raises(ValueError, match="Infeasible"):
            project_sparse_capped_simplex(np.ones(20), s=5, t=0.1)  # 5*0.1 < 1

    def test_fedlaw_typical(self):
        # Typical FedLAW scenario: 20 clients, 2 Byzantine, s=18, t=1/18
        rng = np.random.default_rng(99)
        n, s, t = 20, 18, 1.0 / 18
        h = rng.standard_normal(n)
        w = project_sparse_capped_simplex(h, s=s, t=t)
        assert abs(w.sum() - 1.0) < 1e-7
        assert (w > 1e-10).sum() <= s
        assert np.all(w >= -1e-9)
        assert np.all(w <= t + 1e-9)
