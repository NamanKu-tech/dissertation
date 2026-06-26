"""CLI entrypoint for FedLAW v2 experiments.

Usage:
    python -m src.run_fedlaw_v2 --config configs/fedlaw_v2_small.yaml
    python -m src.run_fedlaw_v2 --config configs/fedlaw_v2_mnist.yaml --seed 1
    python -m src.run_fedlaw_v2 --config configs/fedlaw_v2_mnist.yaml --attack backdoor
"""
from __future__ import annotations

import argparse
import os
import shutil

import yaml

from src.fedlaw_v2 import FedLAWV2Config, FedLAWV2Trainer


def _load_config(path: str, overrides: dict) -> FedLAWV2Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return FedLAWV2Config(**{k: v for k, v in raw.items()
                             if k in FedLAWV2Config.__dataclass_fields__})


def main() -> None:
    p = argparse.ArgumentParser(description="FedLAW v2 experiment runner")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--seed",   type=int, default=None, help="Override seed")
    p.add_argument("--attack", default=None,
                   choices=["flipping_label", "backdoor", "inverse_gradient",
                            "global_parameter", "double", "lie"],
                   help="Override attack_name")
    p.add_argument("--n-clients", type=int, default=None, dest="n_clients",
                   help="Override n_clients")
    p.add_argument("--q", type=float, default=None, help="Override q")
    p.add_argument("--frac-malicious", type=float, default=None,
                   dest="frac_malicious", help="Override frac_malicious")
    args = p.parse_args()

    overrides = {
        "seed": args.seed,
        "attack_name": args.attack,
        "n_clients": args.n_clients,
        "q": args.q,
        "frac_malicious": args.frac_malicious,
    }
    cfg = _load_config(args.config, overrides)

    # Stamp results dir with attack / q / seed
    cfg.results_dir = os.path.join(
        cfg.results_dir,
        cfg.attack_name,
        f"q{cfg.q}",
        f"frac{cfg.frac_malicious}",
        f"seed{cfg.seed}",
    )
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Save config copy next to results
    shutil.copy(args.config, os.path.join(cfg.results_dir, "config.yaml"))

    print(f"FedLAW v2  attack={cfg.attack_name}  n={cfg.n_clients}"
          f"  q={cfg.q}  frac_mal={cfg.frac_malicious}  seed={cfg.seed}")
    print(f"Results → {cfg.results_dir}")

    FedLAWV2Trainer(cfg).run()


if __name__ == "__main__":
    main()
