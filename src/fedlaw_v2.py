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
from dataclasses import dataclass, field

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
    # Partial participation — Bernoulli-p sampling (Step 1 harness).
    # p = 1.0 → full participation (fast path, byte-equivalent to baseline).
    # p < 1.0 → naive A or a cache_weight Design B variant; see participation_mode.
    p: float = 1.0
    # participation_mode controls what happens to absent clients (§2.4 of
    # PARTIAL_PARTICIPATION_DESIGN.md). Only consulted when p < 1.0.
    #   "naive_A" — Design A control. Each round, a FRESH uniform w_active
    #              over S_t. Absent client's w_i is NOT persisted; no
    #              carry-over between rounds. Loses weight continuity by
    #              construction. This is what should degrade — see §2.1.
    #   "cache_weight_B_i" — Design B, §2.4 Option (i). Cache WEIGHT only.
    #              Absent g_i := 0 (no contribution to cross-product or model
    #              update). Detector CANNOT re-score absent clients → dormancy
    #              is structurally possible. Empirically under-trains because
    #              absent clients hold weight mass with zero contribution
    #              (~10x effective step throttle at p=0.1).
    #   "cache_grad_B_ii" — Design B, §2.4 Option (ii). Cache GRADIENT and
    #              weight. Absent g_i := cached × (1-αp)^τ_i (DeMoA's staleness
    #              decay, §A.1). Absent clients contribute to BOTH the model
    #              update (no stranded weight) AND the cross-product detector
    #              (their cached gradient is re-scored every round) → dormancy
    #              MAY be defeated by re-scoring (the opposite of Option i).
    #              This is canonical Design B from §2.2 of the design doc.
    participation_mode: str = "naive_A"
    # Dormancy attack (§3 of PARTIAL_PARTICIPATION_DESIGN.md).
    # If dormancy_T_dark > 0 and dormancy_client_idx >= 0:
    #   rounds [0, T_dark) — dormant client is FORCED into S_t each round
    #     (builds trust by submitting honest pseudo-gradients).
    #   round  T_dark − 1  — dormant client's submitted G and G_tilde are
    #     replaced with  −mean(other_honest)  (the poison that gets cached).
    #   rounds [T_dark, T) — dormant client is FORCED out of S_t every round
    #     (it has "gone dark"; its cached poisoned gradient persists under
    #     cache_grad_B_ii with DeMoA decay; absent under naive_A → no payload).
    dormancy_T_dark: int = -1
    dormancy_client_idx: int = -1
    # For coordinated multi-client dormancy: an entire cohort goes dark
    # together after building trust. All cached gradients are set to the
    # SAME poison vector so they sum constructively (not cancelling like
    # the reproduction's group-oriented Byzantine gradients did at f=0.4).
    # If non-empty, this overrides dormancy_client_idx.
    dormancy_client_indices: list = field(default_factory=list)
    # Dormancy payload — what gets cached at round T_dark − 1.
    #   "inverse_mean" — −mean(other_honest). Anti-aligned (cos ≈ −0.99);
    #                     trivially caught by re-scoring. Use as control.
    #   "stealth_lie"  — mean(other_honest) + τ·std(other_honest). LIE-style:
    #                     mostly aligned (cos ≈ +0.5), positive cross-product,
    #                     evades sharp anti-alignment detection. The genuine
    #                     stealthy dormancy payload.
    #   "stealth_honest" — leave dormant's actual honest gradient unchanged.
    #                     Cached value is HONEST when stored; the "attack" is
    #                     the staleness — does the honest-when-cached gradient
    #                     get re-scored as the model moves past it?
    dormancy_payload: str = "stealth_lie"
    # LIE τ for stealth_lie payload (Baruch stealth bound; matches our LIE
    # reproduction work, results/paper_fixes/REPORT.md §"LIE Check 1+2").
    dormancy_lie_tau: float = 0.9346
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
        # Dormancy attack state — cohort-first. Backward-compat: if the
        # list is empty and the singleton idx is set, treat as a 1-element cohort.
        T_dark = cfg.dormancy_T_dark
        if cfg.dormancy_client_indices:
            dormant_cohort = list(cfg.dormancy_client_indices)
        elif cfg.dormancy_client_idx >= 0:
            dormant_cohort = [cfg.dormancy_client_idx]
        else:
            dormant_cohort = []
        dormant_set = set(dormant_cohort)
        dormancy_on = len(dormant_cohort) > 0 and T_dark > 0
        if dormancy_on:
            dormancy_csv = os.path.join(cfg.results_dir, "dormancy_diag.csv")
            dfh = open(dormancy_csv, "w", newline="")
            dwriter = csv.writer(dfh)
            dwriter.writerow(["round",
                              "cohort_size",
                              "sum_w_cohort",
                              "avg_w_cohort",
                              "avg_norm_cached_g",
                              "avg_cos_cached_vs_honest_mean",
                              "n_in_S_t",
                              "decay_factor_this_round"])

        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["round", "test_acc", "test_loss", "sum_byz"])

            for k in range(cfg.T):
                if k % cfg.eval_every == 0:
                    acc, loss = self._eval()
                    sum_byz = float(sum(self.w[i] for i in self.byz_indices)) if self.byz_indices else 0.0
                    max_byz = float(max(self.w[i] for i in self.byz_indices)) if self.byz_indices else 0.0
                    max_hon = float(max(self.w[i] for i in self.honest_indices))
                    writer.writerow([k, f"{acc:.6f}", f"{loss:.6f}", f"{sum_byz:.6f}"])
                    fh.flush()
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
                    # ── Bernoulli-p sampling path ────────────────────────────
                    # Behaviour for absent clients is dictated by
                    # cfg.participation_mode — see config docstring.
                    mask = sampling_rng.random(cfg.n_clients) < cfg.p
                    S_t = np.where(mask)[0]
                    # Dormancy attack override on S_t (§3). Cohort-wide:
                    # every dormant client is forced IN during build-trust and
                    # OUT after T_dark.
                    if dormancy_on:
                        S_t_set = set(int(i) for i in S_t)
                        if k < T_dark:
                            missing = [d for d in dormant_cohort if d not in S_t_set]
                            if missing:
                                S_t = np.sort(np.concatenate(
                                    [S_t, np.array(missing, dtype=S_t.dtype)]))
                        else:
                            S_t = np.array([i for i in S_t if int(i) not in dormant_set],
                                           dtype=S_t.dtype)
                    # Dormancy attack — poison the dormant COHORT's Round-A
                    # pseudo-gradient on the last build-trust round.
                    # Coordination: compute a SINGLE poison vector over the
                    # non-dormant honest clients and copy it into every
                    # dormant slot so the cached gradients sum constructively
                    # (contrast: the reproduction's group-oriented gradients
                    # at f=0.4 partially cancelled).
                    if dormancy_on and k == T_dark - 1:
                        non_dormant = np.array(
                            [i for i in range(cfg.n_clients) if i not in dormant_set],
                            dtype=int)
                        if cfg.dormancy_payload == "inverse_mean":
                            poison_g = -G[non_dormant].mean(axis=0)
                            for d in dormant_cohort:
                                G[d] = poison_g
                        elif cfg.dormancy_payload == "stealth_lie":
                            mu = G[non_dormant].mean(axis=0)
                            sigma = G[non_dormant].std(axis=0, ddof=1)
                            poison_g = mu + cfg.dormancy_lie_tau * sigma
                            for d in dormant_cohort:
                                G[d] = poison_g
                        elif cfg.dormancy_payload == "stealth_honest":
                            pass  # leave every dormant's honest g unchanged
                        else:
                            raise ValueError(
                                f"Unknown dormancy_payload: "
                                f"{cfg.dormancy_payload!r}")
                    if k % cfg.eval_every == 0:
                        n_byz_St = int(sum(1 for i in S_t if int(i) in byz_set))
                        print(f"           |S_t|={len(S_t)}  byz_in_S_t={n_byz_St}  "
                              f"hon_in_S_t={len(S_t) - n_byz_St}  "
                              f"mode={cfg.participation_mode}")

                    if len(S_t) == 0:
                        theta_new = theta_k     # nobody participated; skip
                    elif cfg.participation_mode == "naive_A":
                        # Design A — control. Project over S_t with fresh
                        # uniform w_active. No carry-over: self.w is wiped
                        # outside S_t at write-back so the next round must
                        # re-initialise from uniform regardless.
                        n_hon_St = int(sum(
                            1 for idx in S_t if int(idx) not in byz_set))
                        s_t = max(n_hon_St, 1)
                        slack_t = min(10, max(s_t - 2, 0))
                        cap_t = 1.0 / max(s_t - slack_t, 1)
                        if s_t * cap_t < 1.0 - 1e-9:
                            cap_t = 1.0 / s_t

                        G_St = G[S_t]
                        w_active = np.ones(len(S_t), dtype=np.float64) / len(S_t)

                        if k < cfg.w_freeze_rounds and n_hon_St > 0:
                            theta_tilde = theta_k - cfg.alpha * (G_St.T @ w_active)
                            G_tilde, f_tilde = self._collect(theta_tilde, round_k=k)
                            G_tilde, _ = _clip_gradients(G_tilde, self.honest_indices)
                            cross = G_St @ G_tilde[S_t].T
                            h = (w_active
                                 + cfg.alpha * cfg.beta * (cross @ w_active)
                                 - cfg.beta * f_tilde[S_t])
                            w_active = project_sparse_capped_simplex(
                                h, s=s_t, t=cap_t)

                        self.w[:] = 0.0
                        self.w[S_t] = w_active
                        theta_new = theta_k - cfg.alpha * (G_St.T @ w_active)

                    elif cfg.participation_mode == "cache_weight_B_i":
                        # Design B canonical (§2.2) + §2.4 Option (i):
                        # cache WEIGHT for absent clients (their w_i persists in
                        # self.w), do NOT cache gradient (absent g_i := 0 → no
                        # cross-product contribution, no re-scoring → dormancy
                        # possible). Critically: project over ALL n clients each
                        # round so self.w is a valid simplex (Σ = 1) at the
                        # full-n scale s=n_honest, t=1/(s-10). Absent clients
                        # appear in h as their previous w_i unchanged; active
                        # clients get the standard FedLAW update.
                        G_St = G[S_t]
                        s_full = self.sparsity
                        cap_full = self.cap

                        if k < cfg.w_freeze_rounds:
                            theta_tilde = theta_k - cfg.alpha * (G_St.T @ self.w[S_t])
                            G_tilde, f_tilde = self._collect(theta_tilde, round_k=k)
                            G_tilde, _ = _clip_gradients(G_tilde, self.honest_indices)

                            # cross over full n with absent g_i = 0 simplifies:
                            #   (cross_full @ self.w)[active] = cross_St @ self.w[S_t]
                            #   (cross_full @ self.w)[absent] = 0
                            cross_St = G_St @ G_tilde[S_t].T
                            cross_term = np.zeros(cfg.n_clients, dtype=np.float64)
                            cross_term[S_t] = cross_St @ self.w[S_t]
                            f_tilde_full = np.zeros(cfg.n_clients, dtype=np.float64)
                            f_tilde_full[S_t] = f_tilde[S_t]

                            h = (self.w
                                 + cfg.alpha * cfg.beta * cross_term
                                 - cfg.beta * f_tilde_full)
                            self.w = project_sparse_capped_simplex(
                                h, s=s_full, t=cap_full)

                        # Model update: absent contribute 0 (cached g_i=0), so
                        # θ - α G_full^T self.w  =  θ - α G_St^T self.w[S_t].
                        theta_new = theta_k - cfg.alpha * (G_St.T @ self.w[S_t])

                    elif cfg.participation_mode == "cache_grad_B_ii":
                        # Design B canonical (§2.2) + §2.4 Option (ii):
                        # cache GRADIENT and weight. Absent g_i := cached × decay
                        # (DeMoA staleness decay (1-αp)^τ_i, §A.1). Absent
                        # contributes to BOTH the cross-product detector and
                        # the model update — no stranded weight. Dormancy may
                        # be defeated because the detector re-scores absent
                        # clients via their (decaying) cached gradient.
                        s_full = self.sparsity
                        cap_full = self.cap
                        decay = 1.0 - cfg.alpha * cfg.p

                        # Lazy-init caches with the correct d.
                        if not hasattr(self, "_G_cache") or self._G_cache is None:
                            d = G.shape[1]
                            self._G_cache = np.zeros((cfg.n_clients, d), dtype=np.float64)
                            self._G_tilde_cache = np.zeros((cfg.n_clients, d), dtype=np.float64)
                            self._f_tilde_cache = np.zeros(cfg.n_clients, dtype=np.float64)

                        # Apply per-round decay to all cached entries. Active
                        # clients will overwrite below — so effective decay
                        # accumulates only for absent clients (=(1-αp)^τ_i).
                        self._G_cache *= decay
                        self._G_tilde_cache *= decay
                        self._f_tilde_cache *= decay

                        # Build effective full-n G: fresh for active, decayed
                        # cached for absent.
                        absent_mask = np.ones(cfg.n_clients, dtype=bool)
                        absent_mask[S_t] = False
                        G_full = G.copy()
                        G_full[absent_mask] = self._G_cache[absent_mask]

                        if k < cfg.w_freeze_rounds:
                            # Tentative step uses the full effective G.
                            theta_tilde = theta_k - cfg.alpha * (G_full.T @ self.w)
                            G_tilde, f_tilde = self._collect(theta_tilde, round_k=k)
                            G_tilde, _ = _clip_gradients(G_tilde, self.honest_indices)

                            # Dormancy Round-B poisoning — mirror Round-A's
                            # cohort-wide coordination.
                            if dormancy_on and k == T_dark - 1:
                                non_dormant = np.array(
                                    [i for i in range(cfg.n_clients)
                                     if i not in dormant_set],
                                    dtype=int)
                                if cfg.dormancy_payload == "inverse_mean":
                                    poison_gt = -G_tilde[non_dormant].mean(axis=0)
                                    for d in dormant_cohort:
                                        G_tilde[d] = poison_gt
                                elif cfg.dormancy_payload == "stealth_lie":
                                    mu = G_tilde[non_dormant].mean(axis=0)
                                    sigma = G_tilde[non_dormant].std(axis=0, ddof=1)
                                    poison_gt = mu + cfg.dormancy_lie_tau * sigma
                                    for d in dormant_cohort:
                                        G_tilde[d] = poison_gt
                                elif cfg.dormancy_payload == "stealth_honest":
                                    pass

                            G_tilde_full = G_tilde.copy()
                            G_tilde_full[absent_mask] = self._G_tilde_cache[absent_mask]
                            f_tilde_full = f_tilde.copy()
                            f_tilde_full[absent_mask] = self._f_tilde_cache[absent_mask]

                            # Cross-product over full n — absent clients are
                            # re-scored via their (decayed) cached pair.
                            cross = G_full @ G_tilde_full.T
                            h = (self.w
                                 + cfg.alpha * cfg.beta * (cross @ self.w)
                                 - cfg.beta * f_tilde_full)
                            self.w = project_sparse_capped_simplex(
                                h, s=s_full, t=cap_full)

                            # Refresh cache for active clients (overwrites the
                            # decayed values from this round's pre-decay step).
                            self._G_cache[S_t] = G[S_t]
                            self._G_tilde_cache[S_t] = G_tilde[S_t]
                            self._f_tilde_cache[S_t] = f_tilde[S_t]
                        else:
                            # w-freeze: weights stable, but still refresh
                            # gradient cache so model update uses fresh active.
                            self._G_cache[S_t] = G[S_t]

                        # Model update uses full effective G — absent now
                        # contribute (decayed cached g_i) × w_i. No stranded
                        # weight at any p.
                        theta_new = theta_k - cfg.alpha * (G_full.T @ self.w)

                    else:
                        raise ValueError(
                            f"Unknown participation_mode: "
                            f"{cfg.participation_mode!r}. "
                            f"Valid: 'naive_A', 'cache_weight_B_i', "
                            f"'cache_grad_B_ii'.")

                self._set_global_flat(theta_new)
                weight_history.append(self.w.copy())

                # ── Dormancy diagnostic (per round, cohort aggregate) ────────
                if dormancy_on:
                    S_t_now = set(int(i) for i in S_t) if cfg.p < 1.0 else set(range(cfg.n_clients))
                    n_in_St = sum(1 for d in dormant_cohort if d in S_t_now)
                    cohort_ws = self.w[dormant_cohort]
                    sum_w = float(cohort_ws.sum())
                    avg_w = float(cohort_ws.mean()) if len(cohort_ws) > 0 else 0.0
                    avg_norm = 0.0
                    avg_cos = 0.0
                    decay_this = 1.0 - cfg.alpha * cfg.p if cfg.p < 1.0 else 1.0
                    if (cfg.participation_mode == "cache_grad_B_ii"
                            and hasattr(self, "_G_cache") and self._G_cache is not None
                            and cfg.p < 1.0 and len(S_t) > 0):
                        hon_active = np.array(
                            [i for i in S_t if int(i) not in dormant_set
                             and int(i) not in byz_set], dtype=int)
                        if len(hon_active) > 0:
                            hon_mean = G[hon_active].mean(axis=0)
                            nm = float(np.linalg.norm(hon_mean))
                            per_norms, per_cos = [], []
                            for d in dormant_cohort:
                                g = self._G_cache[d]
                                ng = float(np.linalg.norm(g))
                                per_norms.append(ng)
                                if ng > 1e-12 and nm > 1e-12:
                                    per_cos.append(float(np.dot(g, hon_mean) / (ng * nm)))
                                else:
                                    per_cos.append(0.0)
                            avg_norm = float(np.mean(per_norms))
                            avg_cos = float(np.mean(per_cos))
                    dwriter.writerow([k,
                                      len(dormant_cohort),
                                      f"{sum_w:.6f}",
                                      f"{avg_w:.6f}",
                                      f"{avg_norm:.6f}",
                                      f"{avg_cos:+.6f}",
                                      n_in_St,
                                      f"{decay_this:.6f}"])
                    dfh.flush()

            # ── final eval ───────────────────────────────────────────────────
            acc, loss = self._eval()
            sum_byz = float(sum(self.w[i] for i in self.byz_indices)) if self.byz_indices else 0.0
            writer.writerow([cfg.T, f"{acc:.6f}", f"{loss:.6f}", f"{sum_byz:.6f}"])
            print(f"[round {cfg.T:4d}] acc={acc:.4f}  loss={loss:.4f}  "
                  f"sum_byz={sum_byz:.4f}  [FINAL]")

        W = np.array(weight_history)          # (T+1, n_clients)
        np.save(weights_path, W)
        print(f"\nMetrics → {csv_path}")
        print(f"Weights  → {weights_path}  shape={W.shape}")
        if dormancy_on:
            dfh.close()
            print(f"Dormancy → {dormancy_csv}")
