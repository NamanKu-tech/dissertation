"""Baseline trainers under partial participation — Krum, TrMean, DeMoA.

Shares data / client infrastructure with FedLAW v2. Simpler aggregator:
compute a robust average of client pseudo-gradients (no learnable weights).

  Krum(f), TrMean(f)  — ByzFL implementations.
  DeMoA-style caching — cache each client's pseudo-grad and momentum-decay
                        (1-αp)^τ across rounds; aggregate the fresh (S_t) +
                        cached (absent) combined set of n vectors per round.

Dormancy attack surface identical to FedLAWV2Trainer:
  dormancy_client_indices, dormancy_T_dark, dormancy_payload,
  dormancy_lie_tau.
"""
from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from byzfl import Client, Server
from byzfl.aggregators.aggregators import Krum, TrMean, Median, CenteredClipping
import src.models   # registers mlp3_mnist
from src.data_partition import cao_partition, select_malicious_indices
from src.fedlaw_v2 import _MNIST_TFM   # reuse the transform


@dataclass
class BaselineConfig:
    dataset: str = "mnist"
    data_folder: str = "./data"
    n_clients: int = 200
    n_labels: int = 10
    q: float = 0.9
    frac_malicious: float = 0.0
    batch_size: int = 64
    model_name: str = "mlp3_mnist"
    loss_name: str = "NLLLoss"
    alpha: float = 0.01
    E: int = 3
    T: int = 200
    eval_every: int = 10
    seed: int = 0
    device: str = "cpu"
    results_dir: str = "./results/baselines"

    # Aggregator
    aggregator: str = "krum"      # "krum" | "trmean" | "median" | "cclip"
    aggregator_f: int = 20        # number of Byzantine to tolerate (trmean/krum)
    cclip_tau: float = 100.0      # CCLIP clipping radius (Karimireddy 2021)
    cclip_L: int = 1              # CCLIP inner iterations

    # Partial participation
    p: float = 1.0
    use_demoa_cache: bool = False   # if True, DeMoA-style cache + decay

    # Dormancy attack (mirrors FedLAWV2Trainer)
    dormancy_T_dark: int = -1
    dormancy_client_indices: list = field(default_factory=list)
    dormancy_payload: str = "stealth_lie"
    dormancy_lie_tau: float = 0.9346


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _load_mnist(data_folder: str):
    from torchvision import datasets
    train = datasets.MNIST(data_folder, train=True,  download=True, transform=_MNIST_TFM)
    test  = datasets.MNIST(data_folder, train=False, download=True, transform=_MNIST_TFM)
    return train, test


class BaselineTrainer:
    """Krum / TrMean, with optional DeMoA-style cache + decay, plus dormancy."""

    def __init__(self, cfg: BaselineConfig) -> None:
        _set_seed(cfg.seed)
        self.cfg = cfg

        n_per_group = cfg.n_clients // cfg.n_labels
        train_set, test_set = _load_mnist(cfg.data_folder)
        loaders = cao_partition(train_set, cfg.n_clients, cfg.n_labels,
                                cfg.q, cfg.batch_size, cfg.seed)

        base = {"model_name": cfg.model_name, "device": cfg.device,
                "loss_name": cfg.loss_name, "LabelFlipping": False,
                "nb_labels": cfg.n_labels, "momentum": 0.0,
                "learning_rate": cfg.alpha, "weight_decay": 0.0,
                "milestones": [], "learning_rate_decay": 1.0,
                "optimizer_name": "SGD", "optimizer_params": {},
                "store_per_client_metrics": True}
        self.clients: list[Client] = [
            Client({**base, "training_dataloader": ld}) for ld in loaders]

        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
        self.server = Server({
            "model_name": cfg.model_name, "device": cfg.device,
            "test_loader": test_loader, "optimizer_name": "SGD",
            "optimizer_params": {}, "learning_rate": cfg.alpha,
            "weight_decay": 0.0, "milestones": [], "learning_rate_decay": 1.0,
            "aggregator_info": {"name": "Average", "parameters": {}},
            "pre_agg_list": []})

        if cfg.aggregator == "krum":
            self.agg = Krum(f=cfg.aggregator_f)
        elif cfg.aggregator == "trmean":
            self.agg = TrMean(f=cfg.aggregator_f)
        elif cfg.aggregator == "median":
            self.agg = Median()   # Median doesn't take f — always picks middle
        elif cfg.aggregator == "cclip":
            # CenteredClipping (Karimireddy et al. 2021) — the aggregator
            # DeMoA's headline results (Figure 1) use. Init center m is
            # updated STATEFULLY across rounds (previous round's aggregate
            # becomes next round's initial center) — set in run().
            self.agg = CenteredClipping(m=None, L=cfg.cclip_L, tau=cfg.cclip_tau)
        else:
            raise ValueError(f"Unknown aggregator: {cfg.aggregator!r}")

        os.makedirs(cfg.results_dir, exist_ok=True)

    def _get_flat(self) -> np.ndarray:
        return (self.server.get_flat_parameters()
                .detach().cpu().numpy().astype(np.float64))

    def _set_flat(self, flat: np.ndarray) -> None:
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

    def _collect_fresh(self, theta: np.ndarray, indices: list[int]) -> np.ndarray:
        """Run E local epochs for each client in `indices` and return their
        pseudo-gradients (θ − ψ_i)/α as a (len(indices), d) array."""
        tv = torch.from_numpy(theta).float()
        out = []
        for i in indices:
            client = self.clients[i]
            client.set_parameters(tv)
            client.compute_gradients()
            steps = self.cfg.E * len(client.training_dataloader)
            client.compute_model_update(steps)
            psi = np.concatenate([
                p.detach().cpu().numpy().ravel()
                for p in client.model.parameters()
            ]).astype(np.float64)
            g_i = (theta - psi) / self.cfg.alpha
            out.append(g_i)
        return np.stack(out, axis=0)

    def run(self) -> None:
        cfg = self.cfg
        csv_path = os.path.join(cfg.results_dir, "metrics.csv")
        sampling_rng = np.random.default_rng(cfg.seed + 0xBEEF)

        # Dormancy cohort
        dormant_cohort = list(cfg.dormancy_client_indices)
        dormant_set = set(dormant_cohort)
        dormancy_on = len(dormant_cohort) > 0 and cfg.dormancy_T_dark > 0

        # DeMoA-style cache
        cache_g = None   # (n, d) once initialised
        if dormancy_on:
            ddiag = open(os.path.join(cfg.results_dir, "dormancy_diag.csv"),
                         "w", newline="")
            dwriter = csv.writer(ddiag)
            dwriter.writerow(["round", "sum_w_cohort_proxy", "avg_cos_cached_vs_hon_mean",
                              "n_in_S_t", "decay_factor"])

        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["round", "test_acc", "test_loss"])

            for k in range(cfg.T):
                if k % cfg.eval_every == 0:
                    acc, loss = self._eval()
                    writer.writerow([k, f"{acc:.6f}", f"{loss:.6f}"])
                    fh.flush()
                    print(f"[round {k:4d}] acc={acc:.4f}  loss={loss:.4f}")

                theta_k = self._get_flat()

                # Sample S_t
                if cfg.p >= 1.0:
                    S_t = list(range(cfg.n_clients))
                else:
                    mask = sampling_rng.random(cfg.n_clients) < cfg.p
                    S_t = [int(i) for i in np.where(mask)[0]]
                # Dormancy sampling override
                if dormancy_on:
                    if k < cfg.dormancy_T_dark:
                        S_t = sorted(set(S_t) | dormant_set)
                    else:
                        S_t = sorted(set(S_t) - dormant_set)
                if len(S_t) == 0:
                    continue

                # Collect fresh pseudo-grads from S_t
                G_St = self._collect_fresh(theta_k, S_t)  # (|S_t|, d)

                # Lazy-init cache
                if cache_g is None:
                    d = G_St.shape[1]
                    cache_g = np.zeros((cfg.n_clients, d), dtype=np.float64)

                # Coordinated dormancy poison at boundary round
                if dormancy_on and k == cfg.dormancy_T_dark - 1:
                    # Map S_t index to row in G_St
                    idx_map = {c: r for r, c in enumerate(S_t)}
                    non_dormant_active = [c for c in S_t if c not in dormant_set]
                    non_dormant_rows = [idx_map[c] for c in non_dormant_active]
                    if cfg.dormancy_payload == "stealth_lie":
                        mu = G_St[non_dormant_rows].mean(axis=0)
                        sigma = G_St[non_dormant_rows].std(axis=0, ddof=1)
                        poison = mu + cfg.dormancy_lie_tau * sigma
                    elif cfg.dormancy_payload == "inverse_mean":
                        poison = -G_St[non_dormant_rows].mean(axis=0)
                    elif cfg.dormancy_payload == "stealth_honest":
                        poison = None   # leave dormants' honest g
                    else:
                        raise ValueError(f"Unknown payload: {cfg.dormancy_payload!r}")
                    if poison is not None:
                        for c in dormant_cohort:
                            if c in idx_map:
                                G_St[idx_map[c]] = poison

                # Assemble the set the aggregator will see
                if cfg.use_demoa_cache:
                    # Decay cache each round (active gets overwritten below)
                    decay = 1.0 - cfg.alpha * cfg.p if cfg.p < 1.0 else 1.0
                    cache_g *= decay
                    # Refresh cache for active
                    for r, c in enumerate(S_t):
                        cache_g[c] = G_St[r]
                    # Aggregate over ALL n clients — cache for absent, fresh for active
                    combined = [cache_g[i] for i in range(cfg.n_clients)]
                else:
                    # Naive PP: aggregate over S_t only
                    combined = [G_St[r] for r in range(len(S_t))]

                # Robust aggregation (Krum returns 1D array; TrMean too).
                # ByzFL expects list of 1D arrays.
                agg = np.asarray(self.agg(combined), dtype=np.float64)
                # Persist CCLIP's initial center = previous round's aggregate
                # (Karimireddy 2021 canonical stateful usage). Without this,
                # CCLIP resets to the zero vector each round and can't cluster.
                if cfg.aggregator == "cclip":
                    self.agg.m = agg.copy()

                # Model update
                theta_new = theta_k - cfg.alpha * agg
                self._set_flat(theta_new)

                # Dormancy diagnostic (cohort-level)
                if dormancy_on:
                    n_in = sum(1 for c in dormant_cohort if c in S_t)
                    avg_cos = 0.0
                    if cfg.use_demoa_cache and len(dormant_cohort) > 0:
                        # cohort mean cached gradient
                        cohort_g = cache_g[dormant_cohort]
                        # Fresh honest active mean = mean over non-dormant rows of G_St
                        non_dorm_rows = [r for r, c in enumerate(S_t) if c not in dormant_set]
                        if len(non_dorm_rows) > 0:
                            hon_mean = G_St[non_dorm_rows].mean(axis=0)
                            nm = float(np.linalg.norm(hon_mean))
                            cosines = []
                            for g in cohort_g:
                                ng = float(np.linalg.norm(g))
                                if ng > 1e-12 and nm > 1e-12:
                                    cosines.append(float(np.dot(g, hon_mean) / (ng * nm)))
                            if cosines:
                                avg_cos = float(np.mean(cosines))
                    decay_val = 1.0 - cfg.alpha * cfg.p if cfg.p < 1.0 else 1.0
                    # sum_w proxy: naive aggregator has no client weights; report
                    # the fraction of the aggregator's *input* set that is cohort.
                    if cfg.use_demoa_cache:
                        proxy = len(dormant_cohort) / cfg.n_clients
                    else:
                        proxy = n_in / max(len(S_t), 1)
                    dwriter.writerow([k, f"{proxy:.6f}", f"{avg_cos:+.6f}",
                                      n_in, f"{decay_val:.6f}"])
                    ddiag.flush()

            acc, loss = self._eval()
            writer.writerow([cfg.T, f"{acc:.6f}", f"{loss:.6f}"])
            print(f"[round {cfg.T:4d}] acc={acc:.4f}  loss={loss:.4f}  [FINAL]")

        if dormancy_on:
            ddiag.close()
        print(f"\nMetrics → {csv_path}")
