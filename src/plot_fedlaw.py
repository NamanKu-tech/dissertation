"""Quick validation plots for FedLAW runs.

Usage:
    python -m src.plot_fedlaw                  # uses default result dirs
    python -m src.plot_fedlaw --out plots/     # write to custom dir
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


_DEFAULT_RUNS = {
    "clean (no attack)":  "./results/fedlaw_clean",
    "SignFlipping":        "./results/fedlaw_signflipping",
    "IPM":                 "./results/fedlaw_ipm",
}


def _load_metrics(results_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = os.path.join(results_dir, "metrics.csv")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]  # rounds, acc, loss


def _load_weights(results_dir: str) -> np.ndarray:
    """Returns (n_rounds+1, n_clients) weight matrix."""
    return np.load(os.path.join(results_dir, "weights.npy"))


def plot_accuracy(runs: dict[str, str], out: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, d in runs.items():
        try:
            rounds, acc, _ = _load_metrics(d)
            ax.plot(rounds, acc * 100, marker="o", markersize=3, label=label)
        except FileNotFoundError:
            print(f"  [skip] {d} not found")
    ax.set_xlabel("Round")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("FedLAW — accuracy under attack (MNIST, mlp3_mnist)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out, "fedlaw_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_weights(results_dir: str, nb_honest: int, out: str, tag: str = "") -> None:
    """Per-client weight trajectories; honest clients blue, Byzantine red."""
    try:
        W = _load_weights(results_dir)
    except FileNotFoundError:
        print(f"  [skip] weights not found in {results_dir}")
        return

    n_rounds, n_clients = W.shape
    rounds = np.arange(n_rounds)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(n_clients):
        colour = "steelblue" if i < nb_honest else "crimson"
        label = ("honest" if i == 0 else "_") if i < nb_honest else ("Byzantine" if i == nb_honest else "_")
        ax.plot(rounds, W[:, i], color=colour, alpha=0.6, lw=1.2, label=label)

    ax.set_xlabel("Round")
    ax.set_ylabel("Weight wᵢ")
    title = f"FedLAW — per-client weights ({tag})" if tag else "FedLAW — per-client weights"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_tag = tag.replace(" ", "_").replace("/", "_") if tag else "weights"
    path = os.path.join(out, f"fedlaw_{safe_tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./results/plots")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    plot_accuracy(_DEFAULT_RUNS, args.out)

    # Weight trajectories for attack runs (18 honest + 2 byz).
    plot_weights("./results/fedlaw_signflipping", nb_honest=18, out=args.out, tag="SignFlipping")
    plot_weights("./results/fedlaw_ipm",          nb_honest=18, out=args.out, tag="IPM")


if __name__ == "__main__":
    main()
