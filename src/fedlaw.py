"""FedLAW: Learnable Aggregation Weights (arXiv 2511.03529, Algorithm 2).

FedLAW cannot be implemented behind the Aggregator interface in aggregators.py
because it has a two-communication-round structure per epoch: the server needs
gradients AND losses from two different model points (θ_k and θ̃_k) before it
can form the weight update. The Aggregator.aggregate() interface provides only
a single set of updates. FedLAW therefore owns its entire training loop here.

ByzFL reuse:
- DataDistributor  — Dirichlet non-IID partition (identical config to baselines)
- Client           — forward/backward at a specified parameter vector
- ByzantineClient  — generates f attack vectors per round
- Server           — holds global model, computes test accuracy

Algorithm (per epoch k, Parsa et al. 2025 Algorithm 2):
  1. Broadcast θ_k. Each client i returns g_i = ∇f_i(θ_k) and ℓ_i = f_i(θ_k).
  2. Server: θ̃_k = θ_k − α · Gₖᵀ wₖ  (Gₖ ∈ R^{d×n}, columns = gradients)
  3. Broadcast θ̃_k. Each client i returns g̃_i = ∇f_i(θ̃_k) and ℓ̃_i = f_i(θ̃_k).
  4. Weight update:
       hₖ = wₖ + α·β · Gₖᵀ G̃ₖ wₖ − β · ℓ̃
     In matrix notation with Gₖ_mat (n×d, rows = clients):
       cross = Gₖ_mat @ G̃ₖ_mat.T   (n×n)  ← inner-product between client gradients
       hₖ = wₖ + α·β · cross @ wₖ − β · ℓ̃
  5. wₖ₊₁ = P_s(hₖ)  — project onto sparse unit-capped simplex Δ(s, t)
  6. θₖ₊₁ = θₖ − α · Gₖᵀ wₖ₊₁

Loss-at-current-model choice:
  ByzFL's Client.compute_gradients() runs ONE batch backward at the model's
  CURRENT parameters and returns the batch loss. We call it BEFORE any local
  optimisation so it yields f_i(θ_k), which is what the FedLAW weight update
  requires. This is a stochastic estimate of f_i(θ_k); a full-dataset evaluation
  would be exact but ~20× slower for MNIST. The paper uses full-batch loss in
  the theory; mini-batch is standard in practice.

Byzantine client loss:
  Byzantine clients submit fake gradients but their loss is unknown to the
  server. We impute mean(honest losses) so the loss term −β·ℓ̃ is neutral for
  byz clients; their weights are driven toward zero by the gradient cross term.

TODO (partial participation): all n_honest clients participate every round.
  To add partial participation, replace the full `self.clients` loop in
  _round1() and _round2() with a sampled subset. The weight vector w stays
  size n_total; non-selected clients contribute zero update and their weight
  accumulates stale bias — handle with a participation mask.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from byzfl import ByzantineClient, Client, DataDistributor, Server

import src.models  # registers mlp3_mnist into byzfl namespace (side-effect import)
from .projections import project_sparse_capped_simplex


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


_TRAIN_TRANSFORMS = {
    "mnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
}
_TEST_TRANSFORMS = _TRAIN_TRANSFORMS  # same for MNIST


def _build_loaders(
    dataset: str, data_folder: str, nb_honest: int,
    dist_name: str, dist_param: float, batch_size: int,
) -> tuple[list[DataLoader], DataLoader]:
    if dataset == "mnist":
        train_set = datasets.MNIST(data_folder, train=True,  download=True, transform=_TRAIN_TRANSFORMS["mnist"])
        test_set  = datasets.MNIST(data_folder, train=False, download=True, transform=_TEST_TRANSFORMS["mnist"])
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    full_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    distributor = DataDistributor({
        "data_distribution_name": dist_name,
        "distribution_parameter": dist_param,
        "nb_honest": nb_honest,
        "data_loader": full_loader,
        "batch_size": batch_size,
    })
    client_loaders = distributor.split_data()
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    return client_loaders, test_loader


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class FedLAWConfig:
    # Data
    dataset: str = "mnist"
    data_folder: str = "./data"
    nb_honest: int = 20
    dist_name: str = "dirichlet_niid"
    dist_param: float = 0.5
    batch_size: int = 64
    # Model
    model_name: str = "mlp3_mnist"
    nb_labels: int = 10
    loss_name: str = "NLLLoss"
    # FedLAW hyperparams (α, β from §3.2 / Table 1 of arXiv 2511.03529)
    alpha: float = 0.01   # server-side learning rate (model update step)
    beta: float = 1e-3    # weight update step size
    # Sparsity / cap (§3.2: s = (1 − fraction_byz) · n, t ≈ 1 / (n − f))
    sparsity: int = 18    # s — set per-run to n_total − n_byz
    cap: float = 0.0      # t — if 0.0, auto-set to 1/sparsity at init
    # Training
    nb_rounds: int = 100
    eval_every: int = 10
    # Byzantine
    nb_byz: int = 0
    attack_name: str = "SignFlipping"
    attack_params: dict[str, Any] = field(default_factory=dict)
    # Misc
    device: str = "cpu"
    seed: int = 0
    results_dir: str = "./results/fedlaw_run"


# ── FedLAW loop ────────────────────────────────────────────────────────────────

class FedLAWLoop:
    """Two-round-per-epoch FedLAW training loop.

    Each call to run() produces:
      results_dir/metrics.csv     — round, test_acc, test_loss
      results_dir/weights.npy     — (nb_rounds+1, n_total) weight trajectories
    """

    def __init__(self, cfg: FedLAWConfig) -> None:
        self.cfg = cfg
        set_global_seed(cfg.seed)

        n_total = cfg.nb_honest + cfg.nb_byz
        self.n_total = n_total

        # Weight vector w ∈ R^{n_total}, init uniform.
        self.w = np.ones(n_total, dtype=np.float64) / n_total

        # Resolve cap.
        self.sparsity = cfg.sparsity
        self.cap = cfg.cap if cfg.cap > 0.0 else 1.0 / cfg.sparsity

        if self.sparsity * self.cap < 1.0 - 1e-9:
            raise ValueError(
                f"Infeasible projection: s={self.sparsity}, t={self.cap:.4g}, "
                f"s·t={self.sparsity*self.cap:.4g} < 1."
            )

        os.makedirs(cfg.results_dir, exist_ok=True)

        # Build data loaders.
        client_loaders, test_loader = _build_loaders(
            cfg.dataset, cfg.data_folder, cfg.nb_honest,
            cfg.dist_name, cfg.dist_param, cfg.batch_size,
        )

        # Client objects (honest only; ByzantineClient generates attack vectors).
        client_base = {
            "model_name": cfg.model_name,
            "device": cfg.device,
            "loss_name": cfg.loss_name,
            "LabelFlipping": False,
            "nb_labels": cfg.nb_labels,
            "momentum": 0.0,       # FedLAW does server-side update; no local momentum
            "store_per_client_metrics": True,
            "learning_rate": cfg.alpha,
            "weight_decay": 0.0,
            "milestones": [],
            "learning_rate_decay": 1.0,
            "optimizer_name": "SGD",
            "optimizer_params": {},
        }
        self.clients: list[Client] = [
            Client({**client_base, "training_dataloader": client_loaders[i]})
            for i in range(cfg.nb_honest)
        ]

        # Server holds the global model and test evaluation.
        self.server = Server({
            "model_name": cfg.model_name,
            "device": cfg.device,
            "test_loader": test_loader,
            "optimizer_name": "SGD",
            "optimizer_params": {},
            "learning_rate": cfg.alpha,
            "weight_decay": 0.0,
            "milestones": [],
            "learning_rate_decay": 1.0,
            "aggregator_info": {"name": "Average", "parameters": {}},
            "pre_agg_list": [],
        })

        # Byzantine client (None when nb_byz == 0).
        if cfg.nb_byz > 0:
            self.byz: Optional[ByzantineClient] = ByzantineClient({
                "name": cfg.attack_name,
                "f": cfg.nb_byz,
                "parameters": dict(cfg.attack_params),
            })
        else:
            self.byz = None

    # ── internal helpers ───────────────────────────────────────────────────────

    def _get_global_flat(self) -> np.ndarray:
        """Current global parameters as a float64 numpy vector."""
        return self.server.get_flat_parameters().detach().cpu().numpy().astype(np.float64)

    def _set_global_flat(self, flat: np.ndarray) -> None:
        """Push a new flat parameter vector into the server's model."""
        self.server.set_parameters(torch.from_numpy(flat).float())

    def _push_to_clients(self, flat: np.ndarray) -> None:
        """Broadcast the server's global parameters to all honest clients."""
        t = torch.from_numpy(flat).float()
        for c in self.clients:
            c.set_parameters(t)

    def _eval(self) -> tuple[float, float]:
        """Test accuracy and NLLLoss on the global model."""
        acc = self.server.compute_test_accuracy()
        crit = torch.nn.NLLLoss(reduction="sum")
        total_loss, n = 0.0, 0
        self.server.model.eval()
        with torch.no_grad():
            for x, y in self.server.test_loader:
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                total_loss += float(crit(self.server.model(x), y))
                n += y.numel()
        return acc, total_loss / max(n, 1)

    def _gradients_and_losses(
        self, flat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """One gradient/loss round at `flat` for all clients (honest + byz).

        Steps:
          1. Push `flat` to all honest clients.
          2. Each client: compute_gradients() → batch loss at flat; get gradients.
          3. Apply Byzantine attack on honest gradients → byz gradient vectors.

        Returns:
          G_mat   — (n_total, d) float64 gradient matrix, rows = clients
          losses  — (n_total,) float64; byz entries = mean(honest losses)
          honest_grads — list of honest gradient arrays (for byz attack input)
        """
        self._push_to_clients(flat)

        honest_losses: list[float] = []
        honest_grads: list[np.ndarray] = []

        for c in self.clients:
            loss_val = float(c.compute_gradients())
            honest_losses.append(loss_val)
            honest_grads.append(
                c.get_flat_gradients().detach().cpu().numpy().astype(np.float64)
            )

        # Byzantine gradient vectors via ByzantineClient.
        if self.byz is not None:
            byz_grads = [
                np.asarray(v, dtype=np.float64)
                for v in self.byz.apply_attack(honest_grads)
            ]
        else:
            byz_grads = []

        all_grads = honest_grads + byz_grads

        # Impute mean honest loss for byz clients (server has no ground truth).
        mean_honest = float(np.mean(honest_losses)) if honest_losses else 0.0
        all_losses = np.array(
            honest_losses + [mean_honest] * len(byz_grads), dtype=np.float64
        )

        G_mat = np.stack(all_grads, axis=0)  # (n_total, d)
        return G_mat, all_losses, honest_grads

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cfg = self.cfg
        csv_path = os.path.join(cfg.results_dir, "metrics.csv")
        weights_path = os.path.join(cfg.results_dir, "weights.npy")

        weight_history = [self.w.copy()]

        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["round", "test_acc", "test_loss"])

            for k in range(cfg.nb_rounds):
                # ---- eval ----
                if k % cfg.eval_every == 0:
                    acc, loss = self._eval()
                    writer.writerow([k, f"{acc:.6f}", f"{loss:.6f}"])
                    fh.flush()
                    w_str = " ".join(f"{wi:.3f}" for wi in self.w)
                    print(
                        f"[round {k:4d}] acc={acc:.4f}  loss={loss:.4f}"
                        f"  w=[{w_str}]"
                    )

                # ---- round 1: gradients at θ_k ----
                theta_k = self._get_global_flat()
                G_k, f_k, honest_grads_k = self._gradients_and_losses(theta_k)

                # ---- server: tentative step θ̃ = θ_k − α · Gₖᵀ wₖ ----
                # Gₖᵀ wₖ in our layout: G_k.T @ w  (d,)
                weighted_grad_k = G_k.T @ self.w   # (d,)
                theta_tilde = theta_k - cfg.alpha * weighted_grad_k

                # ---- round 2: gradients at θ̃ ----
                G_tilde, f_tilde, _ = self._gradients_and_losses(theta_tilde)

                # ---- weight update (Algorithm 2, step 4) ----
                # cross = Gₖ_mat @ G̃_mat.T  ∈ R^{n×n}
                # In paper: [Gₖ]ᵀ [G̃ₖ] with Gₖ ∈ R^{d×n}
                #         = Gₖ_mat @ G̃_mat.T  (our row-client convention)
                cross = G_k @ G_tilde.T                          # (n, n)
                h_k = self.w + cfg.alpha * cfg.beta * (cross @ self.w) - cfg.beta * f_tilde

                # ---- project onto Δ(s, t) ----
                self.w = project_sparse_capped_simplex(h_k, s=self.sparsity, t=self.cap)

                # ---- final model update (Algorithm 2, step 6) ----
                weighted_grad_new = G_k.T @ self.w   # (d,)
                theta_new = theta_k - cfg.alpha * weighted_grad_new
                self._set_global_flat(theta_new)

                weight_history.append(self.w.copy())

            # ---- final eval ----
            acc, loss = self._eval()
            writer.writerow([cfg.nb_rounds, f"{acc:.6f}", f"{loss:.6f}"])
            w_str = " ".join(f"{wi:.3f}" for wi in self.w)
            print(
                f"[round {cfg.nb_rounds:4d}] acc={acc:.4f}  loss={loss:.4f}"
                f"  w=[{w_str}]"
            )

        np.save(weights_path, np.array(weight_history))
        print(f"\nMetrics → {csv_path}")
        print(f"Weights  → {weights_path}  shape={np.array(weight_history).shape}")
