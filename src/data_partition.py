"""Cao et al. (2021) q-parameter non-IID data partitioning for FL."""
from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader, Subset


def cao_partition(
    dataset,
    n_clients: int,
    n_labels: int,
    q: float,
    batch_size: int,
    seed: int,
) -> list[DataLoader]:
    """Partition dataset using Cao et al. q-parameter method.

    Each example with label l is assigned to group l with probability q,
    and to each other group with probability (1-q)/(n_labels-1).
    Clients are ordered group-by-group: clients 0..n_per_group-1 are in group 0,
    clients n_per_group..2*n_per_group-1 in group 1, etc.
    """
    rng = np.random.RandomState(seed)
    n_per_group = n_clients // n_labels

    targets = np.array([dataset[i][1] for i in range(len(dataset))], dtype=int)
    n = len(targets)

    group_of = np.empty(n, dtype=int)
    for i, y in enumerate(targets):
        if rng.random() < q:
            group_of[i] = int(y)
        else:
            others = [g for g in range(n_labels) if g != y]
            group_of[i] = int(rng.choice(others))

    loaders: list[DataLoader] = []
    for g in range(n_labels):
        group_idx = np.where(group_of == g)[0]
        rng.shuffle(group_idx)
        chunks = np.array_split(group_idx, n_per_group)
        for chunk in chunks:
            subset = Subset(dataset, chunk.tolist())
            loaders.append(DataLoader(subset, batch_size=batch_size,
                                      shuffle=True, drop_last=False))
    return loaders


def select_malicious_indices(
    n_clients: int,
    n_byz: int,
    clients_per_group: int,
    seed: int,
) -> list[int]:
    """Group-oriented malicious client selection (hardest-case setup from paper §I.1).

    Selects n_byz clients by filling complete groups first, randomising the
    group order. Returns sorted list of Byzantine client indices.
    """
    n_groups = n_clients // clients_per_group
    rng = np.random.RandomState(seed)
    group_order = rng.permutation(n_groups).tolist()

    chosen: list[int] = []
    for g in group_order:
        for offset in range(clients_per_group):
            if len(chosen) >= n_byz:
                break
            chosen.append(g * clients_per_group + offset)
        if len(chosen) >= n_byz:
            break

    return sorted(chosen[:n_byz])
