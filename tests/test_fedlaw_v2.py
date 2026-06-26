import pytest
import numpy as np


# ── Clip helper ────────────────────────────────────────────────────────────────

def test_clip_leaves_honest_unchanged():
    from src.fedlaw_v2 import _clip_gradients
    G = np.array([[3.0, 4.0], [1.0, 0.0], [100.0, 0.0]])
    G_c, C = _clip_gradients(G, honest_indices=[0, 1])
    assert C == pytest.approx(5.0, abs=1e-6)
    np.testing.assert_array_almost_equal(G_c[0], [3.0, 4.0])
    np.testing.assert_array_almost_equal(G_c[1], [1.0, 0.0])


def test_clip_clips_byz_to_C():
    from src.fedlaw_v2 import _clip_gradients
    G = np.array([[1.0, 0.0], [0.0, 1.0], [10.0, 0.0]])
    G_c, C = _clip_gradients(G, honest_indices=[0, 1])
    assert C == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(G_c[2]) <= C + 1e-6


def test_clip_returns_copy():
    from src.fedlaw_v2 import _clip_gradients
    G = np.array([[1.0, 0.0], [0.0, 1.0], [10.0, 0.0]])
    G_c, _ = _clip_gradients(G, honest_indices=[0, 1])
    assert G_c is not G


def test_clip_non_contiguous_honest_indices():
    from src.fedlaw_v2 import _clip_gradients
    G = np.array([[3.0, 4.0], [100.0, 0.0], [0.0, 1.0]])
    G_c, C = _clip_gradients(G, honest_indices=[0, 2])
    assert C == pytest.approx(5.0, abs=1e-6)
    assert np.linalg.norm(G_c[1]) <= C + 1e-6


# ── _collect ───────────────────────────────────────────────────────────────────

def test_collect_returns_correct_shapes():
    from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer
    cfg = FedLAWV2Config(
        n_clients=4, n_labels=2, q=0.9, frac_malicious=0.5,
        attack_name="inverse_gradient", T=2, eval_every=1,
        results_dir="/tmp/test_v2_collect",
    )
    trainer = FedLAWV2Trainer(cfg)
    import torch
    theta = trainer.server.get_flat_parameters().detach().cpu().numpy().astype(np.float64)
    G, f = trainer._collect(theta, round_k=0)
    assert G.shape == (4, len(theta))
    assert f.shape == (4,)
    assert np.all(np.isfinite(G))
    assert np.all(np.isfinite(f))


def test_collect_byz_loss_imputed():
    from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer
    # Use gradient attack (not data-poison) to avoid label-range issues with n_labels=2
    cfg = FedLAWV2Config(
        n_clients=4, n_labels=2, q=0.9, frac_malicious=0.5,
        attack_name="inverse_gradient", T=2, eval_every=1,
        results_dir="/tmp/test_v2_collect2",
    )
    trainer = FedLAWV2Trainer(cfg)
    theta = trainer.server.get_flat_parameters().detach().cpu().numpy().astype(np.float64)
    G, f = trainer._collect(theta, round_k=0)
    mean_h = float(np.mean([f[i] for i in trainer.honest_indices]))
    for i in trainer.byz_indices:
        assert f[i] == pytest.approx(mean_h, abs=1e-10)


# ── run() ──────────────────────────────────────────────────────────────────────

def test_run_produces_output_files():
    import os
    from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer
    results_dir = "/tmp/test_v2_run"
    cfg = FedLAWV2Config(
        n_clients=4, n_labels=2, q=0.9, frac_malicious=0.5,
        attack_name="inverse_gradient", T=2, eval_every=1,
        w_freeze_rounds=2, results_dir=results_dir,
    )
    FedLAWV2Trainer(cfg).run()
    assert os.path.exists(os.path.join(results_dir, "metrics.csv"))
    assert os.path.exists(os.path.join(results_dir, "weights.npy"))


def test_run_weights_sum_to_one():
    from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer
    cfg = FedLAWV2Config(
        n_clients=4, n_labels=2, q=0.9, frac_malicious=0.5,
        attack_name="inverse_gradient", T=3, eval_every=3,
        w_freeze_rounds=2, results_dir="/tmp/test_v2_weights",
    )
    FedLAWV2Trainer(cfg).run()
    W = np.load("/tmp/test_v2_weights/weights.npy")
    assert W.shape == (4, 4)  # (T+1, n_clients)
    for row in W:
        assert pytest.approx(row.sum(), abs=1e-6) == 1.0


def test_run_byz_weights_decrease():
    from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer
    cfg = FedLAWV2Config(
        n_clients=4, n_labels=2, q=0.9, frac_malicious=0.5,
        attack_name="inverse_gradient", T=3, eval_every=3,
        w_freeze_rounds=3, results_dir="/tmp/test_v2_byz",
    )
    trainer = FedLAWV2Trainer(cfg)
    trainer.run()
    W = np.load("/tmp/test_v2_byz/weights.npy")
    byz_idx = trainer.byz_indices
    for i in byz_idx:
        assert W[-1, i] <= W[0, i] + 1e-6
