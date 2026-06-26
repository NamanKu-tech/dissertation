import pytest
import numpy as np
import torch
from torch.utils.data import Dataset


class _ToyDataset(Dataset):
    """1000 examples, labels cycling 0–9 deterministically."""
    def __init__(self):
        self.x = torch.zeros(1000, 1, 28, 28)
        self.y = torch.tensor([i % 10 for i in range(1000)])

    def __len__(self): return 1000
    def __getitem__(self, i): return self.x[i], int(self.y[i])


def test_cao_partition_returns_correct_count():
    from src.data_partition import cao_partition
    loaders = cao_partition(_ToyDataset(), n_clients=20, n_labels=10, q=0.9,
                            batch_size=32, seed=0)
    assert len(loaders) == 20


def test_cao_partition_total_examples():
    from src.data_partition import cao_partition
    loaders = cao_partition(_ToyDataset(), n_clients=20, n_labels=10, q=0.9,
                            batch_size=32, seed=0)
    total = sum(len(loader.dataset) for loader in loaders)
    assert total == 1000


def test_cao_partition_q1_group_concentration():
    """With q=1.0 every example stays in its own label group."""
    from src.data_partition import cao_partition
    loaders = cao_partition(_ToyDataset(), n_clients=20, n_labels=10, q=1.0,
                            batch_size=32, seed=0)
    # 20 clients / 10 groups = 2 clients per group
    # Group 0 = clients 0,1 → should only see label 0
    for client_idx in range(2):
        for _, y_batch in loaders[client_idx]:
            ys = y_batch.tolist() if hasattr(y_batch, 'tolist') else [y_batch]
            for y in ys:
                assert y == 0, f"Group 0 client {client_idx} got label {y} with q=1"


def test_cao_partition_q_lower_reduces_concentration():
    """With q=0.1 the dominant label should appear less than 90% of the time."""
    from src.data_partition import cao_partition
    loaders = cao_partition(_ToyDataset(), n_clients=20, n_labels=10, q=0.1,
                            batch_size=1000, seed=0)
    labels = []
    for _, y_batch in loaders[0]:
        labels.extend(y_batch.tolist() if hasattr(y_batch, 'tolist') else [y_batch])
    dominant_frac = labels.count(0) / max(len(labels), 1)
    assert dominant_frac < 0.9


def test_select_malicious_count():
    from src.data_partition import select_malicious_indices
    for n_byz in [4, 8, 12]:
        idx = select_malicious_indices(n_clients=20, n_byz=n_byz,
                                       clients_per_group=2, seed=0)
        assert len(idx) == n_byz


def test_select_malicious_group_oriented():
    """Selected clients come from complete groups (pairs for n_per_group=2)."""
    from src.data_partition import select_malicious_indices
    idx = select_malicious_indices(n_clients=20, n_byz=8, clients_per_group=2, seed=0)
    groups = sorted(set(i // 2 for i in idx))
    assert len(groups) == 4   # 8 clients = 4 complete groups of 2


def test_select_malicious_within_range():
    from src.data_partition import select_malicious_indices
    idx = select_malicious_indices(n_clients=20, n_byz=6, clients_per_group=2, seed=42)
    assert all(0 <= i < 20 for i in idx)
