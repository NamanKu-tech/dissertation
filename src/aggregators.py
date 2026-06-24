"""Aggregator interface for the custom FL loop (src/loop.py).

ByzFL's standalone aggregators are ``__call__(vectors) -> vector`` and cannot
receive per-client losses, IDs, or carry state. This interface wraps them for
use by the loop, and provides FedAvg for baselines.

FedLAW is NOT here — it has a two-round-per-epoch structure that lives in
``src/fedlaw.py`` and cannot be reduced to a single aggregate() call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class GlobalState:
    """Server-side read-only context passed to each aggregate() call."""
    round_idx: int
    global_params: np.ndarray  # flat parameter vector at the start of the round


class Aggregator:
    """Interface used by src/loop.py.

    Instantiated once before training; may keep internal state across rounds.
    """

    def aggregate(
        self,
        updates: list[np.ndarray],
        losses: list[float],
        client_ids: list[int],
        global_state: GlobalState,
    ) -> np.ndarray:
        """Return a new flat global parameter vector."""
        raise NotImplementedError


class FedAvg(Aggregator):
    """Uniform mean over client weight vectors.

    With proportion_selected_clients=1.0 and equal local_steps_per_client,
    matches ByzFL's FedAvg + Average aggregator path (modulo data-loading
    randomness). Used as the consistency check for src/loop.py.
    """

    def aggregate(self, updates, losses, client_ids, global_state):
        return np.stack(updates, axis=0).mean(axis=0)


@dataclass
class ByzFLPureAggregator(Aggregator):
    """Adapter: wraps any ByzFL standalone aggregator (Krum, TrMean, …).

    Lets the custom loop compare learnable rules against vanilla robust
    baselines without duplicating the loop.
    """
    fn: Callable[[list[np.ndarray]], np.ndarray]
    name: str = "byzfl"

    def aggregate(self, updates, losses, client_ids, global_state):
        return np.asarray(self.fn(updates))


_REGISTRY: dict[str, Callable[..., Aggregator]] = {
    "FedAvg": FedAvg,
}


def build(name: str, **kwargs: Any) -> Aggregator:
    """Construct an aggregator by name. Raises KeyError on unknown names."""
    return _REGISTRY[name](**kwargs)
