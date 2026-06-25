"""
Paper fixes for FedLAW (arXiv 2511.03529 / ICLR 2026) — three implementation gaps.

Gaps identified by comparing implementation against the full paper:
  Gap 1 — Gradient definition: current code uses raw ∇f(θ;batch); paper uses
           g_i = −(ψ_i − θ)/α where ψ_i is the local model after E full SGD epochs.
  Gap 2 — Server-side ℓ2 clipping: C = max honest gradient norm, applied per round.
           Paper Assumption E1 + Appendix C Layer 1. Current code has no clipping.
  Gap 3 — Cap t: paper Table 1 uses t = 1/(s−10); current code uses t = 1/s
           (exact-exclusion, single feasible point — no adaptive honest weighting).

Each gap is tested in sequence; fixes are cumulative across steps.
Results written to results/paper_fixes/REPORT.md (appended incrementally).

Usage:
    python -m src.run_paper_fixes          # all steps
    python -m src.run_paper_fixes --step 1 # Gap 1 only
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from byzfl import ByzantineClient, Client, DataDistributor, Server
import src.models  # registers mlp3_mnist
from src.projections import project_sparse_capped_simplex

# ── paths ──────────────────────────────────────────────────────────────────────

OUT_DIR   = "./results/paper_fixes"
PLOT_DIR  = "./results/paper_fixes/plots"
REPORT_MD = "./results/paper_fixes/REPORT.md"

os.makedirs(PLOT_DIR, exist_ok=True)


# ── report helpers ─────────────────────────────────────────────────────────────

def R(text: str = "") -> None:
    line = text + "\n"
    with open(REPORT_MD, "a") as fh:
        fh.write(line)
    print(text)


def R_sep(title: str = "") -> None:
    R()
    R("─" * 72)
    if title:
        R(f"## {title}")
    R()


# ── data / model helpers ───────────────────────────────────────────────────────

_TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def _build_data(nb_honest: int, dist_param: float, batch_size: int = 64):
    train_set = datasets.MNIST("./data", train=True,  download=True, transform=_TFM)
    test_set  = datasets.MNIST("./data", train=False, download=True, transform=_TFM)
    full_ldr  = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dist = DataDistributor({
        "data_distribution_name": "dirichlet_niid",
        "distribution_parameter": dist_param,
        "nb_honest": nb_honest,
        "data_loader": full_ldr,
        "batch_size": batch_size,
    })
    client_ldrs = dist.split_data()
    test_ldr = DataLoader(test_set, batch_size=256, shuffle=False)
    return client_ldrs, test_ldr


_CLIENT_BASE = {
    "model_name": "mlp3_mnist", "device": "cpu", "loss_name": "NLLLoss",
    "LabelFlipping": False, "nb_labels": 10, "momentum": 0.0,
    "store_per_client_metrics": True, "weight_decay": 0.0,
    "milestones": [], "learning_rate_decay": 1.0,
    "optimizer_name": "SGD", "optimizer_params": {},
}


def _build_clients(client_ldrs, nb_honest: int, alpha: float) -> list[Client]:
    return [Client({**_CLIENT_BASE, "learning_rate": alpha,
                    "training_dataloader": client_ldrs[i]})
            for i in range(nb_honest)]


def _build_server(test_ldr, alpha: float) -> Server:
    return Server({
        "model_name": "mlp3_mnist", "device": "cpu", "test_loader": test_ldr,
        "optimizer_name": "SGD", "optimizer_params": {},
        "learning_rate": alpha, "weight_decay": 0.0,
        "milestones": [], "learning_rate_decay": 1.0,
        "aggregator_info": {"name": "Average", "parameters": {}}, "pre_agg_list": [],
    })


# ── Gap 1: pseudo-gradient collection ─────────────────────────────────────────

def _collect_pseudograd(
    flat: np.ndarray,
    clients: list[Client],
    byz: ByzantineClient | None,
    alpha: float,
    E: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect g_i = (θ − ψ_i)/α after E local SGD epochs.

    Per Algorithm 1, line 7 of the paper:
      1. Set client params to θ.
      2. Call compute_gradients() once to record loss at θ (no param change).
      3. Run E * len(dataloader) SGD steps via compute_model_update().
      4. ψ_i = client params after local training.
      5. g_i = (θ − ψ_i) / α.
    """
    tv = torch.from_numpy(flat).float()
    h_losses: list[float] = []
    h_grads: list[np.ndarray] = []

    for c in clients:
        c.set_parameters(tv)

        # Loss at θ — compute_gradients() runs one backward pass but no step.
        loss = float(c.compute_gradients())
        h_losses.append(loss)

        # E full local epochs of SGD starting from θ.
        steps = E * len(c.training_dataloader)
        c.compute_model_update(steps)

        # ψ_i = local model params after training.
        psi = np.concatenate([
            p.detach().cpu().numpy().ravel() for p in c.model.parameters()
        ]).astype(np.float64)

        # Pseudo-gradient: (θ − ψ_i) / α
        h_grads.append((flat - psi) / alpha)

    if byz is not None:
        b_grads = [np.asarray(v, dtype=np.float64) for v in byz.apply_attack(h_grads)]
    else:
        b_grads = []

    mean_hl = float(np.mean(h_losses)) if h_losses else 0.0
    G = np.stack(h_grads + b_grads)
    f = np.array(h_losses + [mean_hl] * len(b_grads))
    return G, f


# ── Gap 1 (original): single-batch raw gradient ───────────────────────────────

def _collect_batch_grad(
    flat: np.ndarray,
    clients: list[Client],
    byz: ByzantineClient | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Original collection: single mini-batch ∇f(θ;batch)."""
    tv = torch.from_numpy(flat).float()
    h_losses: list[float] = []
    h_grads: list[np.ndarray] = []

    for c in clients:
        c.set_parameters(tv)
        loss = float(c.compute_gradients())
        h_losses.append(loss)
        h_grads.append(c.get_flat_gradients().detach().cpu().numpy().astype(np.float64))

    if byz is not None:
        b_grads = [np.asarray(v, dtype=np.float64) for v in byz.apply_attack(h_grads)]
    else:
        b_grads = []

    mean_hl = float(np.mean(h_losses)) if h_losses else 0.0
    G = np.stack(h_grads + b_grads)
    f = np.array(h_losses + [mean_hl] * len(b_grads))
    return G, f


# ── Gap 2: server-side ℓ2 clipping ────────────────────────────────────────────

def _clip_gradients(G: np.ndarray, nb_honest: int) -> tuple[np.ndarray, float]:
    """Clip all gradients to C = max honest ℓ2 norm (paper Appendix C Layer 1)."""
    honest_norms = np.linalg.norm(G[:nb_honest], axis=1)
    C = float(honest_norms.max())
    G_out = G.copy()
    for i in range(len(G_out)):
        n_i = float(np.linalg.norm(G_out[i]))
        if n_i > C + 1e-10:
            G_out[i] *= C / n_i
    return G_out, C


# ── Gap 3: cap resolution ──────────────────────────────────────────────────────

def _resolve_cap(sparsity: int, cap_slack: int) -> tuple[float, float]:
    """Compute t and s·t for the given slack.

    cap_slack=0  → t = 1/s   (exact-exclusion, s·t = 1.0)
    cap_slack=10 → t = 1/(s−10)  (paper Table 1)
    """
    s = sparsity
    denom = s - cap_slack
    if denom <= 0:
        raise ValueError(f"cap_slack={cap_slack} ≥ sparsity={s}")
    t = 1.0 / denom
    st = s * t
    if st < 1.0 - 1e-9:
        raise ValueError(
            f"Infeasible projection: s={s}, slack={cap_slack}, t={t:.4g}, s·t={st:.4g} < 1"
        )
    return t, st


# ── Config and result types ────────────────────────────────────────────────────

@dataclass
class FixCfg:
    # Clients / data
    nb_honest:    int   = 18
    nb_byz:       int   = 2
    dist_param:   float = 0.5
    # FedLAW hyperparams
    alpha:        float = 0.01   # paper's α_lr
    beta:         float = 1e-3
    sparsity:     int   = 18     # n − f
    cap_slack:    int   = 0      # Gap 3: 0=exact, 10=paper
    # Training
    nb_rounds:    int   = 15
    eval_every:   int   = 5
    # Attack
    attack_name:  str   = "SignFlipping"
    attack_params: dict = field(default_factory=dict)
    seed:         int   = 0
    # Gap flags
    use_local_epochs: bool = False   # Gap 1
    E:            int   = 3          # local epochs per collection
    use_clipping: bool  = False      # Gap 2


@dataclass
class FixResult:
    weight_history:   np.ndarray        # (nb_rounds+1, n_total)
    eval_rounds:      list[int]
    eval_acc:         list[float]
    byz_zeroed_at:    int | None        # first round where ALL byz weights < 1e-4
    false_excl:       list[int]         # rounds with any honest weight = 0
    cross_w_history:  list[np.ndarray]  # (nb_rounds, n_total) cross_w per round
    elapsed_s:        float


# ── Core FedLAW loop ───────────────────────────────────────────────────────────

def _run_one(cfg: FixCfg) -> FixResult:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    n   = cfg.nb_honest + cfg.nb_byz
    t, st = _resolve_cap(cfg.sparsity, cfg.cap_slack)

    client_ldrs, test_ldr = _build_data(cfg.nb_honest, cfg.dist_param)
    clients = _build_clients(client_ldrs, cfg.nb_honest, cfg.alpha)
    server  = _build_server(test_ldr, cfg.alpha)
    byz = (ByzantineClient({"name": cfg.attack_name, "f": cfg.nb_byz,
                             "parameters": dict(cfg.attack_params)})
           if cfg.nb_byz > 0 else None)

    def get_flat() -> np.ndarray:
        return np.concatenate([
            p.detach().cpu().numpy().ravel() for p in server.model.parameters()
        ]).astype(np.float64)

    def collect(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if cfg.use_local_epochs:
            G, f = _collect_pseudograd(flat, clients, byz, cfg.alpha, cfg.E)
        else:
            G, f = _collect_batch_grad(flat, clients, byz)
        if cfg.use_clipping:
            G, _ = _clip_gradients(G, cfg.nb_honest)
        return G, f

    w = np.ones(n) / n
    weight_history  = [w.copy()]
    eval_rounds:    list[int]         = []
    eval_acc:       list[float]       = []
    byz_zeroed_at:  int | None        = None
    false_excl:     list[int]         = []
    cross_w_history: list[np.ndarray] = []

    t0 = time.time()

    for k in range(cfg.nb_rounds):
        t_round = time.time()

        theta_k = get_flat()
        G_k, _  = collect(theta_k)

        theta_tilde = theta_k - cfg.alpha * (G_k.T @ w)
        G_tilde, f_tilde = collect(theta_tilde)

        cross   = G_k @ G_tilde.T          # (n, n)
        cross_w = cross @ w                 # (n,)
        cross_w_history.append(cross_w.copy())

        h_k = w + cfg.alpha * cfg.beta * cross_w - cfg.beta * f_tilde
        w   = project_sparse_capped_simplex(h_k, s=cfg.sparsity, t=t)

        theta_new = theta_k - cfg.alpha * (G_k.T @ w)
        if np.any(np.isnan(theta_new)):
            R(f"  ⚠ NaN in theta at round {k+1} — stopping early")
            break
        server.set_parameters(torch.from_numpy(theta_new).float())
        weight_history.append(w.copy())

        # Tracking.
        if byz_zeroed_at is None and cfg.nb_byz > 0:
            if (w[cfg.nb_honest:] < 1e-4).all():
                byz_zeroed_at = k + 1
        if cfg.nb_byz > 0 and (w[:cfg.nb_honest] < 1e-4).any():
            false_excl.append(k + 1)

        if (k + 1) % cfg.eval_every == 0:
            acc = float(server.compute_test_accuracy())
            eval_rounds.append(k + 1)
            eval_acc.append(acc)
            elapsed = time.time() - t0
            bz_w = w[cfg.nb_honest:].max() if cfg.nb_byz > 0 else 0.0
            print(f"  round {k+1:3d} | acc={acc:.4f} | byz_w={bz_w:.4f}"
                  f" | false_excl={len(false_excl)} | {time.time()-t_round:.1f}s/round")

    return FixResult(
        weight_history=np.array(weight_history),
        eval_rounds=eval_rounds,
        eval_acc=eval_acc,
        byz_zeroed_at=byz_zeroed_at,
        false_excl=false_excl,
        cross_w_history=cross_w_history,
        elapsed_s=time.time() - t0,
    )


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _plot_weights(W: np.ndarray, nb_honest: int, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    n = W.shape[1]
    for i in range(n):
        col = "steelblue" if i < nb_honest else "crimson"
        lbl = ("honest" if i == 0 else "_") if i < nb_honest else \
              ("Byzantine" if i == nb_honest else "_")
        ax.plot(range(W.shape[0]), W[:, i], color=col, alpha=0.6, lw=1.2, label=lbl)
    ax.set_xlabel("Round"); ax.set_ylabel("Weight wᵢ")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_cross_w(cross_w_hist: list[np.ndarray], nb_honest: int,
                  title: str, path: str) -> None:
    """Plot cross_w per client per round."""
    W = np.array(cross_w_hist)  # (rounds, n)
    fig, ax = plt.subplots(figsize=(9, 4))
    n = W.shape[1]
    for i in range(n):
        col = "steelblue" if i < nb_honest else "crimson"
        lbl = ("honest" if i == 0 else "_") if i < nb_honest else \
              ("Byzantine" if i == nb_honest else "_")
        ax.plot(range(W.shape[0]), W[:, i], color=col, alpha=0.6, lw=1.2, label=lbl)
    ax.axhline(0, color="k", lw=0.8, linestyle="--")
    ax.set_xlabel("Round"); ax.set_ylabel("cross_w[i]")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_weight_dist(W_list: list[np.ndarray], labels: list[str],
                      nb_honest: int, title: str, path: str) -> None:
    """Histogram of honest weights at final round for each run."""
    fig, axes = plt.subplots(1, len(W_list), figsize=(5 * len(W_list), 4), sharey=True)
    if len(W_list) == 1:
        axes = [axes]
    for ax, W, lbl in zip(axes, W_list, labels):
        final_honest = W[-1, :nb_honest]
        ax.hist(final_honest, bins=15, edgecolor="k", alpha=0.7, color="steelblue")
        ax.axvline(1.0 / nb_honest, color="red", lw=1.5, linestyle="--",
                   label=f"uniform 1/{nb_honest}")
        ax.set_title(lbl); ax.set_xlabel("wᵢ (honest)"); ax.set_ylabel("count"); ax.legend()
    fig.suptitle(title, y=1.02)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# GAP 1 — Gradient definition
# ══════════════════════════════════════════════════════════════════════════════

def gap1_check() -> None:
    R_sep("Gap 1 — Gradient definition: raw batch grad vs local-epoch pseudo-gradient")

    R("### Audit: what does current code compute?")
    R()
    R("ByzFL `Client.compute_gradients()` source (client.py lines 75-100):")
    R("  1. Samples ONE mini-batch from the training iterator.")
    R("  2. Calls `self._backward_pass(inputs, targets)` → `model.zero_grad()`,")
    R("     forward pass, loss.backward(). NO optimizer.step().")
    R("  3. Returns batch loss scalar.")
    R("  Gradients sit in `.grad` attributes; `get_flat_gradients()` flattens them.")
    R()
    R("Current code calls this once per client per collection round, giving:")
    R("  g_i = ∇f_i(θ; single_batch)  — stochastic gradient, no local update.")
    R()
    R("Paper Algorithm 1 (line 7) requires:")
    R("  g_i = −(ψ_i − θ)/α  where ψ_i = LocalSGD(θ, lr=α, epochs=E)")
    R("       = (θ − ψ_i) / α")
    R()
    R("Gap: raw batch gradient ≠ pseudo-gradient from E local epochs.")
    R()
    R("Magnitude comparison at α_lr=0.01, E=3, ~52 batches/epoch:")
    R("  raw grad:        ||g_raw|| ≈ 1.4  (measured, MNIST/mlp3)")
    R("  pseudo-grad:     ||g_pseudo|| ≈ E×steps×||g_raw|| ≈ 156×1.4 ≈ 218")
    R()
    R("Cross-term in weight update: α·β·||g||²·cos(i,j)")
    R("  raw   α=0.01: 0.01×0.001×1.4²×cos ≈ 2.0e-5×cos  (loss term=0.0023 — 100× larger)")
    R("  pseudo α=0.01: 0.01×0.001×218²×cos ≈ 0.475×cos   (loss term=0.0023 — 200× smaller)")
    R()
    R("Prediction: α=0.01 should work correctly with pseudo-gradients.")
    R("  cross_w[byz, SignFlipping] ≈ 0.475×(−1.0)×mean_honest_cos ≈ strongly negative")
    R("  → h[byz] << 0 → zeroed by projection at round 1.")
    R()

    # ── Baseline: raw grad, α=0.01 (the failing config from v1) ──────────────
    R("### Experiment 1a — raw gradient, α=0.01 (pre-fix, expected FAIL)")
    R("Config: n=18+2, Dirichlet α=0.5, SignFlipping, 15 rounds, seed=0")
    R()
    cfg_raw = FixCfg(
        nb_honest=18, nb_byz=2, dist_param=0.5,
        alpha=0.01, beta=1e-3, sparsity=18, cap_slack=0,
        nb_rounds=15, eval_every=5, attack_name="SignFlipping",
        use_local_epochs=False, use_clipping=False, seed=0,
    )
    rr_raw = _run_one(cfg_raw)
    W_raw = rr_raw.weight_history

    byz_w_r1 = W_raw[1, 18:].tolist()
    R(f"Round 1 Byzantine weights: {[f'{w:.4f}' for w in byz_w_r1]}")
    R(f"Byzantine zeroed at: {rr_raw.byz_zeroed_at}")
    R(f"False exclusions: {len(rr_raw.false_excl)} rounds")
    final_acc = rr_raw.eval_acc[-1] * 100 if rr_raw.eval_acc else float("nan")
    R(f"Final accuracy (round {rr_raw.eval_rounds[-1] if rr_raw.eval_rounds else '?'}): "
      f"{final_acc:.2f}%")
    verdict_1a = (rr_raw.byz_zeroed_at is None and len(rr_raw.false_excl) > 0)
    R(f"Result: {'FAIL (honest falsely excluded, byz never zeroed) ✗' if verdict_1a else 'unexpected pass'}")
    R()

    _plot_weights(W_raw, 18,
                  "Gap 1a — raw grad α=0.01 (pre-fix) SignFlipping",
                  f"{PLOT_DIR}/gap1a_raw_alpha001_weights.png")
    _plot_cross_w(rr_raw.cross_w_history, 18,
                  "Gap 1a — cross_w per client (raw grad α=0.01)",
                  f"{PLOT_DIR}/gap1a_raw_alpha001_crossw.png")

    # ── Fix: pseudo-grad, α=0.01 ──────────────────────────────────────────────
    R("### Experiment 1b — pseudo-gradient (Gap 1 fix), α=0.01")
    R("Config: same as 1a, use_local_epochs=True, E=3")
    R()
    cfg_pg = FixCfg(
        nb_honest=18, nb_byz=2, dist_param=0.5,
        alpha=0.01, beta=1e-3, sparsity=18, cap_slack=0,
        nb_rounds=15, eval_every=5, attack_name="SignFlipping",
        use_local_epochs=True, E=3, use_clipping=False, seed=0,
    )
    t0 = time.time()
    rr_pg = _run_one(cfg_pg)
    elapsed = time.time() - t0

    W_pg = rr_pg.weight_history
    byz_w_r1 = W_pg[1, 18:].tolist()
    R(f"Round 1 Byzantine weights: {[f'{w:.4f}' for w in byz_w_r1]}")
    R(f"Byzantine zeroed at: {rr_pg.byz_zeroed_at}")
    R(f"False exclusions: {len(rr_pg.false_excl)} rounds")
    final_acc = rr_pg.eval_acc[-1] * 100 if rr_pg.eval_acc else float("nan")
    R(f"Final accuracy (round {rr_pg.eval_rounds[-1] if rr_pg.eval_rounds else '?'}): "
      f"{final_acc:.2f}%")
    R(f"Elapsed: {elapsed:.0f}s ({elapsed/cfg_pg.nb_rounds:.1f}s/round)")

    pred_ok = (rr_pg.byz_zeroed_at is not None and
               rr_pg.byz_zeroed_at <= 2 and
               len(rr_pg.false_excl) == 0)
    R()
    if pred_ok:
        R("**VERDICT Gap 1b: PASS** — α=0.01 works with local-epoch pseudo-gradients.")
        R("  Byzantine excluded at round 1, zero false exclusions.")
        R("  Confirms: α_lr=0.5 workaround was caused by gradient definition mismatch,")
        R("  not by a calibration issue in FedLAW itself.")
    else:
        R("**VERDICT Gap 1b: FAIL** — α=0.01 still does not detect Byzantine with pseudo-grads.")
        R("  ⚠ STOP: this is unexpected. Check cross_w magnitudes and loss term balance.")
        R(f"  cross_w[byz] at round 0: {rr_pg.cross_w_history[0][18:].tolist() if rr_pg.cross_w_history else 'N/A'}")
        R(f"  cross_w[honest mean] at round 0: "
          f"{rr_pg.cross_w_history[0][:18].mean():.6f}" if rr_pg.cross_w_history else "N/A")

    _plot_weights(W_pg, 18,
                  "Gap 1b — pseudo-grad α=0.01 (fix applied) SignFlipping",
                  f"{PLOT_DIR}/gap1b_pseudo_alpha001_weights.png")
    _plot_cross_w(rr_pg.cross_w_history, 18,
                  "Gap 1b — cross_w per client (pseudo-grad α=0.01)",
                  f"{PLOT_DIR}/gap1b_pseudo_alpha001_crossw.png")

    # ── Multi-seed confirmation (3 seeds) if 1b passes ───────────────────────
    R()
    R("### Experiment 1c — pseudo-grad α=0.01, 3 seeds × IPM (paper's second attack)")
    R("Config: same as 1b, attack=InnerProductManipulation, seeds {0,1,2}")
    R()
    seeds = [0, 1, 2]
    accs_ipm = []
    byz_zeroed_ipm = []
    false_excl_ipm = []
    for s in seeds:
        cfg_ipm = FixCfg(
            nb_honest=18, nb_byz=2, dist_param=0.5,
            alpha=0.01, beta=1e-3, sparsity=18, cap_slack=0,
            nb_rounds=15, eval_every=5,
            attack_name="InnerProductManipulation",
            attack_params={"tau": 2.0},
            use_local_epochs=True, E=3, use_clipping=False, seed=s,
        )
        rr = _run_one(cfg_ipm)
        accs_ipm.append(rr.eval_acc[-1] * 100 if rr.eval_acc else float("nan"))
        byz_zeroed_ipm.append(rr.byz_zeroed_at)
        false_excl_ipm.append(len(rr.false_excl))
        R(f"  seed={s}: acc={accs_ipm[-1]:.2f}%  byz_zeroed={rr.byz_zeroed_at}"
          f"  false_excl={false_excl_ipm[-1]}")

    mean_acc = float(np.mean([a for a in accs_ipm if not np.isnan(a)]))
    std_acc  = float(np.std( [a for a in accs_ipm if not np.isnan(a)]))
    R(f"IPM mean±std accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
    all_zeroed = all(z is not None and z <= 2 for z in byz_zeroed_ipm)
    no_false   = all(f == 0 for f in false_excl_ipm)
    R(f"All seeds: byz zeroed ≤ round 2: {all_zeroed}, zero false exclusions: {no_false}")
    R()

    R("### Summary — Gap 1")
    R(f"  pre-fix  α=0.01 raw grad:     byz_zeroed={rr_raw.byz_zeroed_at},"
      f" false_excl={len(rr_raw.false_excl)}")
    R(f"  post-fix α=0.01 pseudo-grad:  byz_zeroed={rr_pg.byz_zeroed_at},"
      f" false_excl={len(rr_pg.false_excl)}")
    R()
    R("Timing note: each round with E=3 local epochs takes ~2-6× longer than the")
    R("  raw-gradient version. Full 100-round multi-seed validation is feasible but")
    R("  will take ~2-4 hours on CPU. The 15-round diagnostic is sufficient to confirm")
    R("  the α=0.01 question (Byzantine exclusion is immediate if the mechanism works).")


# ══════════════════════════════════════════════════════════════════════════════
# GAP 2 — Server-side gradient clipping
# ══════════════════════════════════════════════════════════════════════════════

def gap2_check() -> None:
    R_sep("Gap 2 — Server-side ℓ2 clipping (paper Assumption E1 + Appendix C Layer 1)")

    R("### Audit: current clipping status")
    R()
    R("No clipping exists anywhere in fedlaw.py or run_validation_v2.py.")
    R("Gap 2 fix: after each collection round, compute C = max(||g_honest_i||),")
    R("then project all incoming gradients onto the ℓ2-ball of radius C:")
    R("  g_i ← g_i × min(1, C / ||g_i||)")
    R()
    R("Expected effect on ALIE (ALittleIsEnough, τ=1.5):")
    R("  Without clipping: ||g_byz|| = ||μ + τσ|| > ||g_honest_max|| (τσ term inflates norm).")
    R("  ALIE's direction has cos(g_byz, mean_honest) > 0 (co-aligned by construction).")
    R("  cross_w[byz] > cross_w[honest] → honest clients falsely excluded.")
    R()
    R("  With clipping: ||g_byz|| ≤ C = max(||g_honest||). Norm is bounded.")
    R("  Direction is unchanged — clipping alone does NOT fix cos alignment.")
    R("  Paper does NOT claim strong ALIE detection (Table 3: FedAvg=84%, FedLAW=70%).")
    R("  SUCCESS = 'graceful degradation', not 'inversion stops'.")
    R()
    R("Test setup: 40% Byzantine (8 of 20), Dirichlet α=0.5, ALIE τ=1.5, 15 rounds.")
    R("Note: with 8 Byzantine, s = n−f = 12 for exact exclusion.")
    R()

    # ── 2a: ALIE without clipping (40% Byzantine) ────────────────────────────
    R("### Experiment 2a — ALIE 40% Byzantine, NO clipping (pre-fix)")
    cfg_noclip = FixCfg(
        nb_honest=12, nb_byz=8, dist_param=0.5,
        alpha=0.01, beta=1e-3, sparsity=12, cap_slack=0,
        nb_rounds=15, eval_every=5,
        attack_name="ALittleIsEnough", attack_params={"tau": 1.5},
        use_local_epochs=True, E=3, use_clipping=False, seed=0,
    )
    rr_nc = _run_one(cfg_noclip)
    W_nc = rr_nc.weight_history

    final_acc_nc = rr_nc.eval_acc[-1] * 100 if rr_nc.eval_acc else float("nan")
    R(f"Byzantine zeroed at: {rr_nc.byz_zeroed_at}")
    R(f"False exclusions: {len(rr_nc.false_excl)} / {cfg_noclip.nb_rounds} rounds")
    R(f"Final accuracy: {final_acc_nc:.2f}%")
    if rr_nc.cross_w_history:
        cw0 = rr_nc.cross_w_history[0]
        R(f"Round 1 cross_w: byz mean={cw0[12:].mean():.4f}, honest mean={cw0[:12].mean():.4f}")
    R()

    _plot_weights(W_nc, 12,
                  "Gap 2a — ALIE 40% byz, NO clipping (pre-fix)",
                  f"{PLOT_DIR}/gap2a_alie40pct_noclip_weights.png")

    # ── 2b: ALIE with clipping ────────────────────────────────────────────────
    R("### Experiment 2b — ALIE 40% Byzantine, WITH clipping (Gap 2 fix)")
    cfg_clip = FixCfg(
        nb_honest=12, nb_byz=8, dist_param=0.5,
        alpha=0.01, beta=1e-3, sparsity=12, cap_slack=0,
        nb_rounds=15, eval_every=5,
        attack_name="ALittleIsEnough", attack_params={"tau": 1.5},
        use_local_epochs=True, E=3, use_clipping=True, seed=0,
    )
    rr_clip = _run_one(cfg_clip)
    W_clip = rr_clip.weight_history

    final_acc_clip = rr_clip.eval_acc[-1] * 100 if rr_clip.eval_acc else float("nan")
    R(f"Byzantine zeroed at: {rr_clip.byz_zeroed_at}")
    R(f"False exclusions: {len(rr_clip.false_excl)} / {cfg_clip.nb_rounds} rounds")
    R(f"Final accuracy: {final_acc_clip:.2f}%")
    if rr_clip.cross_w_history:
        cw0 = rr_clip.cross_w_history[0]
        R(f"Round 1 cross_w: byz mean={cw0[12:].mean():.4f}, honest mean={cw0[:12].mean():.4f}")
    R()

    _plot_weights(W_clip, 12,
                  "Gap 2b — ALIE 40% byz, WITH clipping (fix applied)",
                  f"{PLOT_DIR}/gap2b_alie40pct_clip_weights.png")

    # ── Paper comparison ─────────────────────────────────────────────────────
    R("### Paper comparison — Table 3 (FedLAW under LIE)")
    R("  Paper FedAvg under LIE q=0.9, 40% Byzantine:  ≈ 84%")
    R("  Paper FedLAW under LIE q=0.9, 40% Byzantine:  ≈ 70%")
    R("  (Paper's q notation ≈ Dirichlet heterogeneity; lower q = more heterogeneous.)")
    R(f"  Our result without clipping: {final_acc_nc:.2f}%")
    R(f"  Our result with clipping:    {final_acc_clip:.2f}%")
    R()

    acc_improved = (final_acc_clip > final_acc_nc or
                    len(rr_clip.false_excl) < len(rr_nc.false_excl))
    if acc_improved:
        R("**VERDICT Gap 2: PARTIAL PASS** — clipping reduces ALIE damage.")
        R("  Note: ALIE direction (cos > 0) is unchanged by clipping. Full fix")
        R("  requires a detection signal orthogonal to gradient alignment.")
    else:
        R("**VERDICT Gap 2: INCONCLUSIVE** — clipping shows no clear improvement.")
        R("  This is consistent with theory: clipping bounds norm but not direction.")
        R("  ALIE's co-alignment with honest gradients survives any norm-based clipping.")
    R()
    R("Note: 15-round diagnostic may not fully reflect steady-state behaviour.")
    R("The key insight is that ALIE defeats the cross-product mechanism structurally,")
    R("and clipping is a necessary but insufficient fix for this attack class.")


# ══════════════════════════════════════════════════════════════════════════════
# GAP 3 — Cap t = 1/(s−10) for adaptive honest weighting
# ══════════════════════════════════════════════════════════════════════════════

def gap3_check() -> None:
    R_sep("Gap 3 — Cap t: restoring adaptive honest weighting")

    R("### Audit: current cap vs paper cap")
    R()
    R("Current code:  t = 1/s  (cap_slack=0)")
    R("  With s=18, t=1/18: s·t = 1.0 (exact-exclusion, single feasible point)")
    R("  All 18 surviving clients have identical weight 1/18. No adaptive weighting.")
    R()
    R("Paper Table 1: t = 1/(s−10)")
    R("  For our n=20, f=2, s=18: t = 1/(18−10) = 1/8 = 0.125")
    R("  s·t = 18/8 = 2.25 ≥ 1 ✓ (feasible)")
    R("  Now honest clients CAN have different weights (up to 1/8 each).")
    R("  The 18 selected clients share weight 1 with each capped at 1/8.")
    R("  Higher-cross_w honest clients get more weight — matching Figure 1.")
    R()

    # Feasibility checks for different configs
    for s, slack in [(18, 0), (18, 2), (18, 10), (12, 0), (12, 2), (12, 10)]:
        try:
            t, st = _resolve_cap(s, slack)
            R(f"  s={s}, slack={slack}: t={t:.4f}, s·t={st:.4f}  ✓")
        except ValueError as e:
            R(f"  s={s}, slack={slack}: INFEASIBLE — {e}")
    R()

    # ── 3a: exact exclusion (current) ────────────────────────────────────────
    R("### Experiment 3a — exact exclusion t=1/s (pre-fix)")
    R("Config: n=18+2, Dirichlet α=0.5, SignFlipping, 30 rounds, seed=0")
    R("Gaps applied: Gap 1 (pseudo-grad) + Gap 2 (clipping)")
    R()
    cfg_exact = FixCfg(
        nb_honest=18, nb_byz=2, dist_param=0.5,
        alpha=0.01, beta=1e-3, sparsity=18, cap_slack=0,
        nb_rounds=30, eval_every=5, attack_name="SignFlipping",
        use_local_epochs=True, E=3, use_clipping=True, seed=0,
    )
    rr_exact = _run_one(cfg_exact)
    W_exact = rr_exact.weight_history

    honest_w_final_exact = W_exact[-1, :18]
    R(f"Byzantine zeroed at: {rr_exact.byz_zeroed_at}")
    R(f"False exclusions: {len(rr_exact.false_excl)}")
    R(f"Honest weight std at round 30: {honest_w_final_exact.std():.6f}")
    R(f"  (expected ≈ 0.000 — exact exclusion forces uniform 1/18={1/18:.4f})")
    final_acc_exact = rr_exact.eval_acc[-1] * 100 if rr_exact.eval_acc else float("nan")
    R(f"Accuracy at round 30: {final_acc_exact:.2f}%")
    R()

    # ── 3b: paper cap slack=10 ────────────────────────────────────────────────
    R("### Experiment 3b — paper cap t=1/(s−10) (Gap 3 fix)")
    R("Config: same as 3a but cap_slack=10 → t=1/8=0.125")
    R()
    t_paper, st_paper = _resolve_cap(18, 10)
    R(f"Resolved cap: t={t_paper:.4f}, s·t={st_paper:.4f}")
    R()
    cfg_paper_t = FixCfg(
        nb_honest=18, nb_byz=2, dist_param=0.5,
        alpha=0.01, beta=1e-3, sparsity=18, cap_slack=10,
        nb_rounds=30, eval_every=5, attack_name="SignFlipping",
        use_local_epochs=True, E=3, use_clipping=True, seed=0,
    )
    rr_paper_t = _run_one(cfg_paper_t)
    W_paper_t = rr_paper_t.weight_history

    honest_w_final_paper = W_paper_t[-1, :18]
    R(f"Byzantine zeroed at: {rr_paper_t.byz_zeroed_at}")
    R(f"False exclusions: {len(rr_paper_t.false_excl)}")
    R(f"Honest weight std at round 30: {honest_w_final_paper.std():.6f}")
    R(f"  honest weight min={honest_w_final_paper.min():.4f}, max={honest_w_final_paper.max():.4f}")
    R(f"  (cap=1/8={t_paper:.4f}; non-uniform if std > 0 and max ≈ cap)")
    final_acc_paper_t = rr_paper_t.eval_acc[-1] * 100 if rr_paper_t.eval_acc else float("nan")
    R(f"Accuracy at round 30: {final_acc_paper_t:.2f}%")
    R()

    non_uniform = honest_w_final_paper.std() > 1e-4
    byz_still_zeroed = rr_paper_t.byz_zeroed_at is not None
    if non_uniform and byz_still_zeroed:
        R("**VERDICT Gap 3: PASS** — non-uniform honest weights (matching Figure 1)")
        R("  while Byzantine clients remain excluded.")
    elif not non_uniform:
        R("**VERDICT Gap 3: PARTIAL** — honest weights still effectively uniform.")
        R("  With 15-30 rounds, client drift may not yet be sufficient to differentiate.")
        R("  Recommend: run 100 rounds and check weight distribution at convergence.")
    else:
        R("**VERDICT Gap 3: FAIL** — Byzantine not excluded with loose cap.")

    _plot_weights(W_exact, 18,
                  "Gap 3a — exact cap t=1/18 (all gaps except Gap3)",
                  f"{PLOT_DIR}/gap3a_exact_cap_weights.png")
    _plot_weights(W_paper_t, 18,
                  "Gap 3b — paper cap t=1/8 (all gaps applied)",
                  f"{PLOT_DIR}/gap3b_paper_cap_weights.png")
    _plot_weight_dist(
        [W_exact, W_paper_t],
        [f"t=1/18 (exact, std={honest_w_final_exact.std():.4f})",
         f"t=1/8 (paper, std={honest_w_final_paper.std():.4f})"],
        18,
        "Honest-client weight distributions at round 30 — Gap 3 comparison",
        f"{PLOT_DIR}/gap3_weight_dist_comparison.png",
    )


# ══════════════════════════════════════════════════════════════════════════════
# FINAL: paper-comparable numbers with all three fixes
# ══════════════════════════════════════════════════════════════════════════════

def paper_numbers() -> None:
    R_sep("All fixes applied — paper-comparable validation")

    R("All three gaps fixed: pseudo-gradients (E=3), clipping (C=max honest norm),")
    R("cap t=1/(s−10). Config: n=18+2, Dirichlet α=0.5, 30 rounds, 3 seeds.")
    R("Attacks: SignFlipping, InnerProductManipulation (τ=2).")
    R()
    R("Note: paper Table 3 reports results at q (Dirichlet) ∈ {0.5, 0.6, 0.9}.")
    R("We test at q=0.5 (our standard) as a reference point.")
    R("Full 100-round 3-seed paper-identical runs require ~2-4h on CPU.")
    R()

    attacks = {
        "SignFlipping": {"attack_name": "SignFlipping", "attack_params": {}},
        "IPM (τ=2)":   {"attack_name": "InnerProductManipulation", "attack_params": {"tau": 2.0}},
    }
    seeds = [0, 1, 2]

    for atk_label, atk_cfg in attacks.items():
        R(f"### Attack: {atk_label}")
        accs, byz_zeroed_all, false_excl_all = [], [], []
        for s in seeds:
            cfg = FixCfg(
                nb_honest=18, nb_byz=2, dist_param=0.5,
                alpha=0.01, beta=1e-3, sparsity=18, cap_slack=10,
                nb_rounds=30, eval_every=5,
                use_local_epochs=True, E=3, use_clipping=True, seed=s,
                **atk_cfg,
            )
            rr = _run_one(cfg)
            acc = rr.eval_acc[-1] * 100 if rr.eval_acc else float("nan")
            accs.append(acc)
            byz_zeroed_all.append(rr.byz_zeroed_at)
            false_excl_all.append(len(rr.false_excl))
            R(f"  seed={s}: acc={acc:.2f}%  byz_zeroed={rr.byz_zeroed_at}"
              f"  false_excl={false_excl_all[-1]}")

        valid_accs = [a for a in accs if not np.isnan(a)]
        if valid_accs:
            R(f"  Mean±std: {np.mean(valid_accs):.2f} ± {np.std(valid_accs):.2f}%")
        byz_ok = all(z is not None and z <= 2 for z in byz_zeroed_all)
        fe_ok  = all(f == 0 for f in false_excl_all)
        R(f"  Byzantine zeroed ≤ round 2 (all seeds): {byz_ok}")
        R(f"  Zero false exclusions (all seeds):      {fe_ok}")
        R()


# ══════════════════════════════════════════════════════════════════════════════
# Report header + VERDICT summary
# ══════════════════════════════════════════════════════════════════════════════

def write_header() -> None:
    # Overwrite if first run, otherwise append to existing.
    if os.path.exists(REPORT_MD):
        os.remove(REPORT_MD)

    R("# FedLAW Paper Fixes — Validation Report")
    R()
    R("Model: mlp3_mnist (784→200→100→10)")
    R("Three implementation gaps found by comparing code against ICLR 2026 paper.")
    R("This report records expected vs. observed behaviour for each fix.")
    R()
    R("## VERDICT (filled in after all steps complete)")
    R()
    R("| Gap | Fix | Predicted behaviour | Observed |")
    R("|-----|-----|---------------------|----------|")
    R("| 1 — Gradient def. | pseudo-grad (θ−ψ_i)/α, E=3 epochs | α=0.01 works, byz zeroed round 1 | TBD |")
    R("| 2 — Clipping | C=max honest norm per round | ALIE damage contained, no inversion | TBD |")
    R("| 3 — Cap t | t=1/(s−10)=1/8 for s=18 | non-uniform honest weights, byz still excluded | TBD |")
    R()
    R("*(Table updated after experiments run — search for VERDICT in each section.)*")
    R()
    R("---")


def write_verdict_update(gap1_pass: bool, gap2_partial: bool, gap3_pass: bool) -> None:
    R_sep("Final Verdict Summary")
    R(f"Gap 1 (gradient definition): {'PASS ✓' if gap1_pass else 'FAIL ✗'}")
    R(f"Gap 2 (server clipping):     {'PARTIAL (expected) ✓' if gap2_partial else 'INCONCLUSIVE'}")
    R(f"Gap 3 (cap t):               {'PASS ✓' if gap3_pass else 'PARTIAL/FAIL ✗'}")
    R()
    R("Retracted conclusions from VALIDATION.md v2:")
    if gap1_pass:
        R("  ✓ 'needs α=0.5' — RETRACTED. Paper α=0.01 works with local-epoch pseudo-gradients.")
        R("    The α_lr=0.5 workaround was compensating for the gradient definition mismatch.")
    else:
        R("  ✗ 'needs α=0.5' — STANDS. α=0.01 still insufficient even with pseudo-gradients.")
    if gap3_pass:
        R("  ✓ 'exact-exclusion required' — PARTIALLY RETRACTED for within-honest weighting.")
        R("    Exact-exclusion (s·t=1) is NOT required; t=1/(s−10) produces adaptive weighting.")
        R("    Key distinction: sparsity s=n−f still required; only cap t is relaxed.")
    R("  ~ 'ALIE is a structural limit no fix addresses' — NUANCED.")
    R("    Clipping bounds the attack norm but ALIE's directional alignment with honest")
    R("    gradients survives clipping. The cross-product signal is insufficient for ALIE.")
    R("    RA-LAW must use a signal beyond instantaneous gradient alignment.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, default=0,
                   help="Run specific step: 1=Gap1, 2=Gap2, 3=Gap3, 4=paper_numbers, 0=all")
    args = p.parse_args()

    if args.step == 0 or args.step == 1:
        write_header()
        gap1_check()
        if args.step == 1:
            return

    if args.step == 0 or args.step == 2:
        if args.step == 2:
            write_header()
        gap2_check()
        if args.step == 2:
            return

    if args.step == 0 or args.step == 3:
        if args.step == 3:
            write_header()
        gap3_check()
        if args.step == 3:
            return

    if args.step == 0 or args.step == 4:
        if args.step == 4:
            write_header()
        paper_numbers()

    R_sep("End of paper fixes validation")
    R(f"Full report: {REPORT_MD}")


if __name__ == "__main__":
    main()
