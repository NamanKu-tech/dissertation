"""Custom synchronous FL round loop that reuses ByzFL pieces.

Why a custom loop at all? ByzFL's standalone aggregator slot is
``__call__(vectors) -> vector`` — it cannot receive per-client losses, stable
IDs, or carry state across rounds. FedLAW and RA-LAW need all three. So we
keep ByzFL's heavy machinery (data partitioning, client / server objects,
attack classes) verbatim, and write a thin per-round driver around them.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from byzfl import ByzantineClient, Client, DataDistributor, Server

from .aggregators import Aggregator, GlobalState


# ---------- Reproducibility ----------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set seeds + deterministic flags across python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops where available; OK to be best-effort on CPU.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


# ---------- Config -------------------------------------------------------------

@dataclass
class LoopConfig:
    # Data / partition
    dataset: str = "mnist"
    data_folder: str = "./data"
    nb_honest: int = 20
    distribution_name: str = "dirichlet_niid"  # iid | dirichlet_niid | gamma_similarity_niid | extreme_niid
    distribution_parameter: float = 0.5
    batch_size: int = 25
    # Model / optim
    model_name: str = "cnn_mnist"
    nb_labels: int = 10
    loss_name: str = "NLLLoss"
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    learning_rate_decay: float = 1.0
    milestones: tuple[int, ...] = ()
    # Round loop
    nb_steps: int = 100
    local_steps_per_client: int = 5
    eval_every: int = 20
    # Byzantine
    nb_byz: int = 0           # f
    attack_name: str = "SignFlipping"
    attack_params: Optional[dict[str, Any]] = None
    # Misc
    device: str = "cpu"
    seed: int = 0
    results_dir: str = "./results/loop_run"
    # Aggregator (resolved by run_loop, not used here)
    aggregator_name: str = "FedAvg"
    aggregator_params: Optional[dict[str, Any]] = None


# ---------- Data helpers -------------------------------------------------------

_DATASET_BUILDERS = {
    "mnist": lambda root: datasets.MNIST(
        root=root, train=True, download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
}
_TEST_BUILDERS = {
    "mnist": lambda root: datasets.MNIST(
        root=root, train=False, download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
}


def _build_client_loaders(cfg: LoopConfig) -> list[DataLoader]:
    """Use ByzFL's DataDistributor to obtain the per-client loaders."""
    train_set = _DATASET_BUILDERS[cfg.dataset](cfg.data_folder)
    full_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    distributor = DataDistributor({
        "data_distribution_name": cfg.distribution_name,
        "distribution_parameter": cfg.distribution_parameter,
        "nb_honest": cfg.nb_honest,
        "data_loader": full_loader,
        "batch_size": cfg.batch_size,
    })
    return distributor.split_data()


def _build_test_loader(cfg: LoopConfig) -> DataLoader:
    return DataLoader(
        _TEST_BUILDERS[cfg.dataset](cfg.data_folder),
        batch_size=128, shuffle=False,
    )


# ---------- The loop -----------------------------------------------------------

class FederatedLoop:
    """Synchronous FL round loop. Aggregator instantiated once; state persists.

    Per round (with ``nb_byz`` Byzantine clients):
      1. Push current global params to every honest client.
      2. Each honest client: ``compute_gradients()`` once at θ_t to obtain
         ``f_i(θ_t)`` (this is the loss FedLAW wants), then
         ``compute_model_update(local_steps)`` to do the local training.
      3. Collect ``(flat_params_i, loss_at_global_i, client_id_i)``.
      4. Construct attack vectors via ``ByzantineClient.apply_attack`` from
         the honest flats; assign them Byzantine IDs (``nb_honest..nb_total-1``)
         and NaN losses.
      5. Call ``aggregator.aggregate(updates, losses, ids, global_state)``.
      6. ``Server.set_parameters(new_flat)``; log eval every ``eval_every``.
    """

    def __init__(self, cfg: LoopConfig, aggregator: Aggregator) -> None:
        self.cfg = cfg
        self.aggregator = aggregator
        self.device = cfg.device
        set_global_seed(cfg.seed)

        os.makedirs(cfg.results_dir, exist_ok=True)

        # Data
        self.client_loaders = _build_client_loaders(cfg)
        self.test_loader = _build_test_loader(cfg)

        # Server (carries the global model + the test loader)
        self.server = Server({
            "model_name": cfg.model_name,
            "device": cfg.device,
            "test_loader": self.test_loader,
            "optimizer_name": "SGD",
            "optimizer_params": {"momentum": cfg.momentum},
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "milestones": list(cfg.milestones),
            "learning_rate_decay": cfg.learning_rate_decay,
            # Dummy: the loop never uses the server's robust aggregator.
            "aggregator_info": {"name": "Average", "parameters": {}},
            "pre_agg_list": [],
        })

        # Honest clients
        self.clients: list[Client] = []
        for i in range(cfg.nb_honest):
            self.clients.append(Client({
                "model_name": cfg.model_name,
                "device": cfg.device,
                "loss_name": cfg.loss_name,
                "LabelFlipping": False,
                "nb_labels": cfg.nb_labels,
                "momentum": cfg.momentum,
                "training_dataloader": self.client_loaders[i],
                "store_per_client_metrics": True,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "milestones": list(cfg.milestones),
                "learning_rate_decay": cfg.learning_rate_decay,
                "optimizer_name": "SGD",
                "optimizer_params": {"momentum": cfg.momentum},
            }))

        # Byzantine client (one object that emits f attack vectors per round)
        if cfg.nb_byz > 0:
            self.byz = ByzantineClient({
                "name": cfg.attack_name,
                "f": cfg.nb_byz,
                "parameters": dict(cfg.attack_params or {}),
            })
        else:
            self.byz = None

    # ---- one round -----------------------------------------------------------

    def _broadcast(self, flat_global: np.ndarray) -> None:
        for c in self.clients:
            c.set_parameters(flat_global)

    def _local_step(self) -> tuple[list[np.ndarray], list[float]]:
        """Run honest clients' local work. Returns (flat_params, losses_at_θ_t).

        Loss is captured at the current global model via a single
        ``compute_gradients()`` call BEFORE ``compute_model_update`` runs the
        actual local SGD. This is what FedLAW needs (see paper); FedAvg
        ignores it. Cheap: one extra batch per round per client.
        """
        flats: list[np.ndarray] = []
        losses_at_global: list[float] = []
        for c in self.clients:
            losses_at_global.append(float(c.compute_gradients()))
            c.compute_model_update(self.cfg.local_steps_per_client)
            flats.append(np.asarray(c.get_flat_parameters()))
        return flats, losses_at_global

    def _attack(self, honest_flats: list[np.ndarray]) -> list[np.ndarray]:
        if self.byz is None:
            return []
        return [np.asarray(v) for v in self.byz.apply_attack(honest_flats)]

    def _eval(self) -> tuple[float, float]:
        """Server-side test accuracy + test loss on the global model."""
        acc = self.server.compute_test_accuracy()
        # Test loss with NLLLoss is the natural pair for cnn_mnist.
        crit = torch.nn.NLLLoss(reduction="sum")
        loss_sum, n = 0.0, 0
        self.server.model.eval()
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.server.model(x)
                loss_sum += float(crit(out, y))
                n += y.numel()
        return acc, loss_sum / max(n, 1)

    # ---- main ----------------------------------------------------------------

    def run(self) -> str:
        cfg = self.cfg
        csv_path = os.path.join(cfg.results_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["round", "test_acc", "test_loss"])

            global_flat = np.asarray(self.server.get_flat_parameters())

            for r in range(cfg.nb_steps):
                if r % cfg.eval_every == 0:
                    acc, loss = self._eval()
                    w.writerow([r, f"{acc:.6f}", f"{loss:.6f}"])
                    fh.flush()
                    print(f"[round {r:4d}] test_acc={acc:.4f}  test_loss={loss:.4f}")

                self._broadcast(global_flat)
                honest_flats, honest_losses = self._local_step()
                byz_flats = self._attack(honest_flats)

                updates = honest_flats + byz_flats
                losses = honest_losses + [float("nan")] * len(byz_flats)
                ids = list(range(cfg.nb_honest)) + list(
                    range(cfg.nb_honest, cfg.nb_honest + len(byz_flats))
                )

                new_flat = self.aggregator.aggregate(
                    updates, losses, ids,
                    GlobalState(round_idx=r, global_params=global_flat),
                )
                self.server.set_parameters(new_flat)
                global_flat = np.asarray(new_flat)

            # final eval
            acc, loss = self._eval()
            w.writerow([cfg.nb_steps, f"{acc:.6f}", f"{loss:.6f}"])
            print(f"[round {cfg.nb_steps:4d}] test_acc={acc:.4f}  test_loss={loss:.4f}")

        print(f"Wrote {csv_path}")
        return csv_path
