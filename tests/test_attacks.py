import pytest
import numpy as np
import torch
from torch.utils.data import Dataset


class _Tiny(Dataset):
    """10 examples, label = index."""
    def __init__(self):
        self.items = [(torch.ones(1, 28, 28) * i * 0.1, i) for i in range(10)]
    def __len__(self): return 10
    def __getitem__(self, i): return self.items[i]


# ── FlipLabelDataset ──────────────────────────────────────────────────────────

def test_flip_label_flips_correctly():
    from src.attacks import FlipLabelDataset
    ds = FlipLabelDataset(_Tiny(), n_labels=10)
    for i in range(10):
        _, y = ds[i]
        assert y == 9 - i, f"label {i} should flip to {9-i}, got {y}"


def test_flip_label_images_unchanged():
    from src.attacks import FlipLabelDataset
    base = _Tiny()
    ds = FlipLabelDataset(base, n_labels=10)
    x_base, _ = base[3]
    x_flip, _ = ds[3]
    assert torch.equal(x_base, x_flip)


def test_flip_label_len_unchanged():
    from src.attacks import FlipLabelDataset
    assert len(FlipLabelDataset(_Tiny())) == 10


# ── BackdoorDataset ───────────────────────────────────────────────────────────

def test_backdoor_trigger_applied():
    from src.attacks import BackdoorDataset

    class AllOnes(Dataset):
        def __len__(self): return 5
        def __getitem__(self, i): return torch.ones(1, 28, 28), 0

    ds = BackdoorDataset(AllOnes(), trigger_size=8, n_labels=10, seed=0)
    x, _ = ds[0]
    # Centre 8×8 block (rows 10:18, cols 10:18) should be ≈ −0.4242
    assert torch.allclose(x[:, 10:18, 10:18],
                          torch.full((1, 8, 8), -0.4242), atol=1e-3)
    # Pixels outside the trigger should be unchanged (1.0)
    assert torch.all(x[:, 0:10, :] == 1.0)


def test_backdoor_labels_random():
    from src.attacks import BackdoorDataset

    class FixedLabel(Dataset):
        def __len__(self): return 50
        def __getitem__(self, i): return torch.zeros(1, 28, 28), 0

    ds = BackdoorDataset(FixedLabel(), n_labels=10, seed=7)
    labels = [ds[i][1] for i in range(50)]
    assert len(set(labels)) > 1, "Backdoor labels should be random, not all 0"


def test_backdoor_len_unchanged():
    from src.attacks import BackdoorDataset

    class D(Dataset):
        def __len__(self): return 8
        def __getitem__(self, i): return torch.zeros(1, 28, 28), i % 3

    assert len(BackdoorDataset(D())) == 8


# ── Gradient attacks ──────────────────────────────────────────────────────────

def test_inverse_gradient_negates():
    from src.attacks import InverseGradientAttack
    attack = InverseGradientAttack()
    grads = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    result = attack(grads, np.zeros(3), round_k=0)
    np.testing.assert_array_equal(result[0], [-1.0, -2.0, -3.0])
    np.testing.assert_array_equal(result[1], [-4.0, -5.0, -6.0])


def test_inverse_gradient_count_preserved():
    from src.attacks import InverseGradientAttack
    attack = InverseGradientAttack()
    grads = [np.ones(10)] * 5
    assert len(attack(grads, np.zeros(10), round_k=0)) == 5


def test_global_param_attack_shape():
    from src.attacks import GlobalParamAttack
    attack = GlobalParamAttack(nu1=-5.0, nu2=1.5, alpha=0.01, seed=42)
    grads = [np.zeros(100), np.zeros(100)]
    theta = np.random.randn(100)
    result = attack(grads, theta, round_k=0)
    assert len(result) == 2
    assert result[0].shape == (100,)


def test_global_param_attack_differs_per_client():
    from src.attacks import GlobalParamAttack
    attack = GlobalParamAttack(nu1=-5.0, nu2=1.5, alpha=0.01, seed=0)
    grads = [np.zeros(100)] * 3
    theta = np.ones(100)
    result = attack(grads, theta, round_k=0)
    assert not np.array_equal(result[0], result[1])


def test_double_attack_round0_no_override():
    from src.attacks import DoubleAttack
    attack = DoubleAttack(alpha=0.01, seed=0)
    grads = [np.ones(10) * 2.0] * 4
    theta = np.zeros(10)
    result = attack(grads, theta, round_k=0)
    for r in result:
        np.testing.assert_array_equal(r, np.ones(10) * 2.0)


def test_double_attack_round1_m1_active():
    from src.attacks import DoubleAttack
    attack = DoubleAttack(alpha=0.01, seed=0)
    grads = [np.ones(10) * 2.0] * 4   # 4 byz: M1=[0,1], M2=[2,3]
    theta = np.zeros(10)
    result = attack(grads, theta, round_k=1)
    # M1 negated
    np.testing.assert_array_equal(result[0], np.ones(10) * -2.0)
    np.testing.assert_array_equal(result[1], np.ones(10) * -2.0)
    # M2 unchanged
    np.testing.assert_array_equal(result[2], np.ones(10) * 2.0)
    np.testing.assert_array_equal(result[3], np.ones(10) * 2.0)


def test_double_attack_round4_both_active():
    from src.attacks import DoubleAttack
    attack = DoubleAttack(alpha=0.01, seed=0)
    grads = [np.ones(10) * 2.0] * 4
    theta = np.zeros(10)
    result = attack(grads, theta, round_k=4)
    # M1 negated
    np.testing.assert_array_equal(result[0], np.ones(10) * -2.0)
    # M2 replaced by GlobalParamAttack (not equal to original 2.0 ones)
    assert not np.array_equal(result[2], np.ones(10) * 2.0)


def test_lie_attack_count():
    from src.attacks import LIEAttack
    attack = LIEAttack(n_byz=4)
    honest = [np.random.randn(100) for _ in range(16)]
    theta = np.zeros(100)
    result = attack(honest, theta, round_k=0)
    assert len(result) == 4


def test_lie_attack_shape():
    from src.attacks import LIEAttack
    attack = LIEAttack(n_byz=3)
    honest = [np.random.randn(50) for _ in range(10)]
    theta = np.zeros(50)
    result = attack(honest, theta, round_k=0)
    assert all(r.shape == (50,) for r in result)
