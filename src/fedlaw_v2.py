"""FedLAW v2 — paper-faithful implementation (ICLR 2026).

All four algorithm fixes applied:
  Gap 1: pseudo-gradient g_i = (θ − ψ_i)/α from E local SGD epochs
  Gap 2: server-side ℓ2 clipping to C = max honest norm
  Gap 3: cap t = 1/(s − 10)  [s·t ≥ 1 enforced]
  w-freeze: weight vector w updated only for first 20 rounds

Attacks:
  Data-poison (flip-label, backdoor): Byzantine clients have poisoned DataLoaders,
    train normally, submit genuine pseudo-gradients.
  Gradient-manipulation (inverse-gradient, global-parameter, double, lie):
    Byzantine clients train normally; pseudo-gradients replaced post-collection.
"""
from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from byzfl import Client, Server
import src.models  # registers mlp3_mnist
from src.data_partition import cao_partition, select_malicious_indices
from src.attacks import (
    FlipLabelDataset, BackdoorDataset,
    InverseGradientAttack, GlobalParamAttack, DoubleAttack,
    LIEAttack, LIERawGradAttack,
)
from src.projections import project_sparse_capped_simplex


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class FedLAWV2Config:
    # Data
    dataset: str = "mnist"
    data_folder: str = "./data"
    n_clients: int = 20
    n_labels: int = 10
    q: float = 0.9
    frac_malicious: float = 0.4
    batch_size: int = 64
    # Model
    model_name: str = "mlp3_mnist"
    loss_name: str = "NLLLoss"
    # FedLAW hyperparams
    alpha: float = 0.01
    beta: float = 0.01
    E: int = 3
    w_freeze_rounds: int = 20
    T: int = 200
    eval_every: int = 10
    # Attack
    attack_name: str = "flipping_label"
    lie_tau: float = 1.5
    # Partial participation — Bernoulli-p sampling (Step 1: harness only)
    p: float = 1.0
    # Output
    seed: int = 0
    results_dir: str = "./results/v2"
    device: str = "cpu"


# ── Clip helper ────────────────────────────────────────────────────────────────

def _clip_gradients(
    G: np.ndarray,
    honest_indices: list[int],
) -> tuple[np.ndarray, float]:
    """Clip all rows of G to ℓ2 norm C = max honest norm (Gap 2 fix).

    Returns (G_clipped, C). G_clipped is a copy; original G is unchanged.
    """
    norms_h = np.linalg.norm(G[honest_indices], axis=1)
    C = float(norms_h.max())
    G_out = G.copy()
    for i in range(len(G_out)):
        n = float(np.linalg.norm(G_out[i]))
        if n > C + 1e-10:
            G_out[i] *= C / n
    return G_out, C


# ── Seed ───────────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


# ── Data loading ───────────────────────────────────────────────────────────────

_MNIST_TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def _load_mnist(data_folder: str):
    train = datasets.MNIST(data_folder, train=True,  download=True, transform=_MNIST_TFM)
    test  = datasets.MNIST(data_folder, train=False, download=True, transform=_MNIST_TFM)
    return train, test


# ── Client / Server builders ───────────────────────────────────────────────────

def _build_clients(loaders: list[DataLoader], cfg: FedLAWV2Config) -> list[Client]:
    base = {
        "model_name": cfg.model_name,
        "device": cfg.device,
        "loss_name": cfg.loss_name,
        "LabelFlipping": False,
        "nb_labels": cfg.n_labels,
        "momentum": 0.0,
        "store_per_client_metrics": True,
        "learning_rate": cfg.alpha,
        "weight_decay": 0.0,
        "milestones": [],
        "learning_rate_decay": 1.0,
        "optimizer_name": "SGD",
        "optimizer_params": {},
    }
    return [Client({**base, "training_dataloader": loader}) for loader in loaders]


def _build_server(test_loader: DataLoader, cfg: FedLAWV2Config) -> Server:
    return Server({
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


# ── Attack factory ─────────────────────────────────────────────────────────────

_DATA_POISON_ATTACKS = {"flipping_label", "backdoor"}


def _poison_loader(loader: DataLoader, attack_name: str,
                   cfg: FedLAWV2Config, client_idx: int) -> DataLoader:
    base_ds = loader.dataset
    if attack_name == "flipping_label":
        poisoned = FlipLabelDataset(base_ds, n_labels=cfg.n_labels)
    elif attack_name == "backdoor":
        poisoned = BackdoorDataset(base_ds, trigger_size=8,
                                   n_labels=cfg.n_labels,
                                   seed=cfg.seed + client_idx)
    else:
        raise ValueError(f"Unknown data-poison attack: {attack_name!r}")
    return DataLoader(poisoned, batch_size=cfg.batch_size,
                      shuffle=True, drop_last=False)


def _build_gradient_attack(cfg: FedLAWV2Config, n_byz: int):
    if cfg.attack_name == "inverse_gradient":
        return InverseGradientAttack()
    if cfg.attack_name == "global_parameter":
        return GlobalParamAttack(nu1=-5.0, nu2=1.5, alpha=cfg.alpha, seed=cfg.seed)
    if cfg.attack_name == "double":
        return DoubleAttack(alpha=cfg.alpha, seed=cfg.seed)
    if cfg.attack_name == "lie":
        return LIEAttack(n_byz=n_byz, tau=cfg.lie_tau)
    if cfg.attack_name == "lie_raw":
        return LIERawGradAttack(n_byz=n_byz, tau=cfg.lie_tau)
    raise ValueError(f"Unknown gradient attack: {cfg.attack_name!r}")


# ── Trainer ────────────────────────────────────────────────────────────────────

class FedLAWV2Trainer:
    """Unified FedLAW trainer — data-poison and gradient-attack Byzantine clients."""

    def __init__(self, cfg: FedLAWV2Config) -> None:
        _set_seed(cfg.seed)
        self.cfg = cfg

        n_per_group = cfg.n_clients // cfg.n_labels
        self.n_byz = round(cfg.frac_malicious * cfg.n_clients)
        self.n_honest = cfg.n_clients - self.n_byz

        # Data partition (Cao et al. q-parameter)
        train_set, test_set = _load_mnist(cfg.data_folder)
        loaders = cao_partition(train_set, cfg.n_clients, cfg.n_labels,
                                cfg.q, cfg.batch_size, cfg.seed)

        # Byzantine index selection (group-oriented)
        self.byz_indices: list[int] = select_malicious_indices(
            cfg.n_clients, self.n_byz, n_per_group, cfg.seed)
        self.honest_indices: list[int] = [
            i for i in range(cfg.n_clients) if i not in set(self.byz_indices)]

        # Apply data poisoning to Byzantine loaders (if needed)
        if cfg.attack_name in _DATA_POISON_ATTACKS:
            for idx in self.byz_indices:
                loaders[idx] = _poison_loader(loaders[idx], cfg.attack_name,
                                              cfg, client_idx=idx)
            self.gradient_attack = None
            self.byz_grad_indices: list[int] = []
        else:
            self.gradient_attack = _build_gradient_attack(cfg, self.n_byz)
            self.byz_grad_indices = list(self.byz_indices)

        # Build all n_clients ByzFL Client objects
        self.clients: list[Client] = _build_clients(loaders, cfg)

        # Server (global model + test evaluation)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
        self.server = _build_server(test_loader, cfg)

        # Weights: uniform init
        self.w = np.ones(cfg.n_clients, dtype=np.float64) / cfg.n_clients

        # Projection parameters: s = n_honest, t = 1/(s − slack)
        self.sparsity = self.n_honest
        slack = min(10, self.sparsity - 2)   # paper: s−10; guard small n
        self.cap = 1.0 / max(self.sparsity - slack, 1)
        if self.sparsity * self.cap < 1.0 - 1e-9:
            raise ValueError(
                f"Infeasible projection: s={self.sparsity}, t={self.cap:.4g}, "
                f"s·t={self.sparsity * self.cap:.4g} < 1")

        os.makedirs(cfg.results_dir, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_global_flat(self) -> np.ndarray:
        return (self.server.get_flat_parameters()
                .detach().cpu().numpy().astype(np.float64))

    def _set_global_flat(self, flat: np.ndarray) -> None:
        self.server.set_parameters(torch.from_numpy(flat).float())

    def _eval(self) -> tuple[float, float]:
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

    # ── pseudo-gradient collection ────────────────────────────────────────────

    def _collect(
        self, theta: np.ndarray, round_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect pseudo-gradients and losses at theta for all clients.

        Steps:
          1. Every client: set θ, record f_i(θ), run E local epochs → ψ_i.
          2. Pseudo-gradient: g_i = (θ − ψ_i) / α.
          3. Gradient-attack Byzantine clients: replace g_i post-collection.
          4. Impute Byzantine losses with mean honest loss.

        Returns:
          G: (n_clients, d) float64 — one pseudo-gradient row per client.
          f: (n_clients,) float64  — losses; Byzantine entries imputed.
        """
        cfg = self.cfg
        tv = torch.from_numpy(theta).float()
        pseudo_grads: list[np.ndarray] = []
        losses: list[float] = []

        for client in self.clients:
            client.set_parameters(tv)

            # Loss at theta (before local training) — matches eq. (151) p.60
            loss_i = float(client.compute_gradients())

            # E full local SGD epochs → psi_i
            steps = cfg.E * len(client.training_dataloader)
            client.compute_model_update(steps)

            # Pseudo-gradient: (theta − psi_i) / alpha
            psi = np.concatenate([
                p.detach().cpu().numpy().ravel()
                for p in client.model.parameters()
            ]).astype(np.float64)
            g_i = (theta - psi) / cfg.alpha

            pseudo_grads.append(g_i)
            losses.append(loss_i)

        # ── gradient-attack override ──────────────────────────────────────────
        if self.gradient_attack is not None and self.byz_grad_indices:
            byz_grads = [pseudo_grads[i] for i in self.byz_grad_indices]

            if isinstance(self.gradient_attack, LIERawGradAttack):
                # Extra backward pass at θ on honest clients — raw ∇f(θ; batch)
                raw_grads: list[np.ndarray] = []
                for i in self.honest_indices:
                    self.clients[i].set_parameters(tv)
                    self.clients[i].compute_gradients()
                    rg = np.concatenate([
                        p.grad.detach().cpu().numpy().ravel()
                        for p in self.clients[i].model.parameters()
                    ]).astype(np.float64)
                    raw_grads.append(rg)

                replaced, mu_raw, sigma_raw, b_raw = self.gradient_attack(
                    raw_grads, theta, round_k)

                if round_k == 0:
                    mu_pseudo = np.mean(
                        [pseudo_grads[i] for i in self.honest_indices], axis=0)
                    norm_mu_raw   = float(np.linalg.norm(mu_raw))
                    norm_sig_raw  = float(np.linalg.norm(sigma_raw))
                    norm_mu_ps    = float(np.linalg.norm(mu_pseudo))
                    cos_b_mups = float(
                        np.dot(b_raw, mu_pseudo)
                        / (np.linalg.norm(b_raw) * norm_mu_ps + 1e-30))
                    print("\n[lie_raw diagnostic @ round 0]")
                    print(f"  ||μ_raw||     = {norm_mu_raw:.4f}")
                    print(f"  ||σ_raw||     = {norm_sig_raw:.4f}")
                    print(f"  σ_raw/μ_raw   = {norm_sig_raw / (norm_mu_raw + 1e-30):.4f}")
                    print(f"  ||b_lie_raw|| = {float(np.linalg.norm(b_raw)):.4f}")
                    print(f"  ||μ_pseudo||  = {norm_mu_ps:.4f}")
                    print(f"  cos(b_lie_raw, μ_pseudo) = {cos_b_mups:.4f}  "
                          f"({'co-aligned ✓' if cos_b_mups > 0.5 else 'anti-aligned ✗' if cos_b_mups < -0.1 else 'orthogonal ~'})")
                    print()

            elif isinstance(self.gradient_attack, LIEAttack):
                honest_grads = [pseudo_grads[i] for i in self.honest_indices]
                replaced = self.gradient_attack(honest_grads, theta, round_k)
            else:
                replaced = self.gradient_attack(byz_grads, theta, round_k)

            for idx, new_g in zip(self.byz_grad_indices, replaced):
                pseudo_grads[idx] = new_g

        # ── impute Byzantine losses ───────────────────────────────────────────
        mean_f_honest = float(np.mean([losses[i] for i in self.honest_indices]))
        for i in self.byz_indices:
            losses[i] = mean_f_honest

        G = np.stack(pseudo_grads, axis=0)   # (n_clients, d)
        f = np.array(losses, dtype=np.float64)
        return G, f

    # ── training loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute FedLAW training for cfg.T rounds.

        Outputs:
          results_dir/metrics.csv   — round, test_acc, test_loss
          results_dir/weights.npy   — (T+1, n_clients) weight history
        """
        cfg = self.cfg
        csv_path     = os.path.join(cfg.results_dir, "metrics.csv")
        weights_path = os.path.join(cfg.results_dir, "weights.npy")

        weight_history = [self.w.copy()]
        # Bernoulli-p sampling RNG (separate stream from training RNG so that
        # p=1.0 does NOT consume any sampling randomness — guarantees the
        # fast path is byte-exact equivalent to the original trainer).
        sampling_rng = np.random.default_rng(cfg.seed + 0xBEEF)
        byz_set = set(self.byz_indices)

        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["round", "test_acc", "test_loss"])

            for k in range(cfg.T):
                if k % cfg.eval_every == 0:
                    acc, loss = self._eval()
                    writer.writerow([k, f"{acc:.6f}", f"{loss:.6f}"])
                    fh.flush()
                    sum_byz = float(sum(self.w[i] for i in self.byz_indices))
                    max_byz = float(max(self.w[i] for i in self.byz_indices))
                    max_hon = float(max(self.w[i] for i in self.honest_indices))
                    print(f"[round {k:4d}] acc={acc:.4f}  loss={loss:.4f}"
                          f"  sum_byz={sum_byz:.4f}  max_byz={max_byz:.4f}"
                          f"  max_hon={max_hon:.4f}")

                theta_k = self._get_global_flat()

                # ── Round A: pseudo-grads at θ_k ─────────────────────────────
                G, f = self._collect(theta_k, round_k=k)
                G, _ = _clip_gradients(G, self.honest_indices)

                if cfg.p >= 1.0:
                    # ── Fast path: full participation (byte-exact baseline) ──
                    if k < cfg.w_freeze_rounds:
                        theta_tilde = theta_k - cfg.alpha * (G.T @ self.w)
                        G_tilde, f_tilde = self._collect(theta_tilde, round_k=k)
                        G_tilde, _ = _clip_gradients(G_tilde, self.honest_indices)

                        cross = G @ G_tilde.T
                        h = (self.w
                             + cfg.alpha * cfg.beta * (cross @ self.w)
                             - cfg.beta * f_tilde)
                        self.w = project_sparse_capped_simplex(
                            h, s=self.sparsity, t=self.cap)

                    theta_new = theta_k - cfg.alpha * (G.T @ self.w)

                else:
                    # ── Bernoulli-p sampling path (Design A — naive) ─────────
                    mask = sampling_rng.random(cfg.n_clients) < cfg.p
                    S_t = np.where(mask)[0]
                    if k % cfg.eval_every == 0:
                        n_byz_St = int(sum(1 for i in S_t if int(i) in byz_set))
                        print(f"           |S_t|={len(S_t)}  byz_in_S_t={n_byz_St}  "
                              f"hon_in_S_t={len(S_t) - n_byz_St}")

                    if len(S_t) == 0:
                        theta_new = theta_k     # nobody participated; skip
                    else:
                        hon_St = np.array(
                            [j for j, idx in enumerate(S_t) if int(idx) not in byz_set],
                            dtype=int)
                        n_hon_St = len(hon_St)

                        # This round's projection geometry
                        s_t = max(n_hon_St, 1)
                        slack_t = min(10, max(s_t - 2, 0))
                        cap_t = 1.0 / max(s_t - slack_t, 1)
                        # If s·t < 1 (degenerate small subset), fall back to t=1/s_t
                        if s_t * cap_t < 1.0 - 1e-9:
                            cap_t = 1.0 / s_t

                        G_St = G[S_t]                                   # (|S_t|, d)

                        # Renormalize current weights over S_t → simplex
                        w_active = self.w[S_t].astype(np.float64).copy()
                        s_sum = float(w_active.sum())
                        if s_sum > 1e-12:
                            w_active /= s_sum
                        else:
                            w_active[:] = 1.0 / len(S_t)

                        if k < cfg.w_freeze_rounds and n_hon_St > 0:
                            theta_tilde = theta_k - cfg.alpha * (G_St.T @ w_active)
                            G_tilde, f_tilde = self._collect(theta_tilde, round_k=k)
                            G_tilde, _ = _clip_gradients(G_tilde, self.honest_indices)
                            G_tilde_St = G_tilde[S_t]
                            f_tilde_St = f_tilde[S_t]

                            cross = G_St @ G_tilde_St.T
                            h = (w_active
                                 + cfg.alpha * cfg.beta * (cross @ w_active)
                                 - cfg.beta * f_tilde_St)
                            w_active = project_sparse_capped_simplex(
                                h, s=s_t, t=cap_t)

                        # Store back into persistent self.w
                        self.w[S_t] = w_active
                        theta_new = theta_k - cfg.alpha * (G_St.T @ w_active)

                self._set_global_flat(theta_new)
                weight_history.append(self.w.copy())

            # ── final eval ───────────────────────────────────────────────────
            acc, loss = self._eval()
            writer.writerow([cfg.T, f"{acc:.6f}", f"{loss:.6f}"])
            print(f"[round {cfg.T:4d}] acc={acc:.4f}  loss={loss:.4f}  [FINAL]")

        W = np.array(weight_history)          # (T+1, n_clients)
        np.save(weights_path, W)
        print(f"\nMetrics → {csv_path}")
        print(f"Weights  → {weights_path}  shape={W.shape}")
