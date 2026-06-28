"""Generate DeMoA-protocol comparison plots.

Reads results/v2/demoa/<label>/metrics.csv and produces:
  results/v2/demoa/plots/comparison_p0.5.png — 2x2 grid (frac × {acc, sum_byz})
  results/v2/demoa/plots/comparison_p0.1.png — same layout

Each subplot has 2 lines: naive_A vs cache_weight_B_i.
Conforms to DeMoA's "fix p, compare methods within p" protocol.
"""
from __future__ import annotations
import csv, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results/v2/demoa")
OUT_DIR = ROOT / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_traj(p: float, frac: float, mode: str):
    """Return (rounds, acc, sum_byz) arrays for one run."""
    label = f"p{p}_f{frac}_{mode}"
    path = ROOT / label / "metrics.csv"
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    rounds, acc, sb = [], [], []
    with open(path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            rounds.append(int(row["round"]))
            acc.append(float(row["test_acc"]))
            sb.append(float(row.get("sum_byz", "nan")))
    return np.array(rounds), np.array(acc), np.array(sb)


def plot_p(p: float):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Partial participation, p = {p}: naive_A vs cache_weight_B_i\n"
                 f"n=200, q=0.9, backdoor, seed=0",
                 fontsize=13)

    for col, frac in enumerate([0.1, 0.4]):
        for mode, color, ls in [("naive_A",            "tab:blue",  "-"),
                                ("cache_weight_B_i",   "tab:orange", "-")]:
            tr = load_traj(p, frac, mode)
            if tr is None:
                continue
            rounds, acc, sb = tr
            axes[0, col].plot(rounds, acc, color=color, linestyle=ls,
                              linewidth=2, label=mode)
            axes[1, col].plot(rounds, sb, color=color, linestyle=ls,
                              linewidth=2, label=mode)

        axes[0, col].set_title(f"f = {frac}")
        axes[0, col].set_ylabel("test accuracy")
        axes[0, col].set_xlabel("round")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend()
        axes[0, col].set_ylim(0, 1.0)

        # Cap reference line on sum_byz axis
        n_byz = round(frac * 200)
        n_hon = 200 - n_byz
        s = n_hon
        slack = min(10, s - 2)
        cap = 1.0 / max(s - slack, 1)
        max_byz_mass = n_byz * cap
        axes[1, col].axhline(max_byz_mass, color="red", linestyle=":",
                             alpha=0.6, label=f"cap floor = {max_byz_mass:.3f}")
        axes[1, col].set_title(f"f = {frac}")
        axes[1, col].set_ylabel("sum_byz")
        axes[1, col].set_xlabel("round")
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend()

    plt.tight_layout()
    out = OUT_DIR / f"comparison_p{p}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  wrote {out}")


if __name__ == "__main__":
    for p in [0.5, 0.1]:
        print(f"plotting p={p}")
        plot_p(p)
    print("DONE")
