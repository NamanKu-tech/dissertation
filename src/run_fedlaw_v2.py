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
                            "global_parameter", "double", "lie", "lie_raw"],
                   help="Override attack_name")
    p.add_argument("--n-clients", type=int, default=None, dest="n_clients",
                   help="Override n_clients")
    p.add_argument("--q", type=float, default=None, help="Override q")
    p.add_argument("--frac-malicious", type=float, default=None,
                   dest="frac_malicious", help="Override frac_malicious")
    p.add_argument("--lie-tau", type=float, default=None,
                   dest="lie_tau", help="LIE attack tau (default 1.5; use Baruch z for theory-grounded value)")
    p.add_argument("--p", type=float, default=None,
                   dest="participation_p",
                   help="Bernoulli-p client participation (default 1.0 = full)")
    p.add_argument("--participation-mode", default=None,
                   dest="participation_mode",
                   choices=["naive_A", "cache_weight_B_i", "cache_grad_B_ii"],
                   help="Behaviour for absent clients (only used when p<1.0). "
                        "naive_A = no weight persistence (control). "
                        "cache_weight_B_i = persist w_i, absent g_i=0 "
                        "(§2.4 Option (i), under-trains, dormancy possible). "
                        "cache_grad_B_ii = persist w_i + cached g_i with "
                        "DeMoA decay (§2.4 Option (ii), canonical §2.2 B, "
                        "absent re-scored by detector, may defeat dormancy).")
    p.add_argument("--dormancy-T-dark", type=int, default=None,
                   dest="dormancy_T_dark",
                   help="Round at which the dormant client goes dark "
                        "(§3 attack). -1 = no dormancy. Combine with "
                        "--dormancy-client-idx.")
    p.add_argument("--dormancy-client-idx", type=int, default=None,
                   dest="dormancy_client_idx",
                   help="Index of the dormant client (§3 attack, singleton). "
                        "-1 = none. Ignored if --dormancy-client-indices given.")
    p.add_argument("--dormancy-client-indices", default=None,
                   dest="dormancy_client_indices",
                   help="Comma-separated list of dormant client indices "
                        "(§3 coordinated cohort attack).")
    p.add_argument("--dormancy-payload", default=None,
                   dest="dormancy_payload",
                   choices=["inverse_mean", "stealth_lie", "stealth_honest"],
                   help="Payload cached at T_dark-1. inverse_mean = anti-aligned "
                        "(control). stealth_lie = μ + τ·σ (default). "
                        "stealth_honest = leave honest unchanged.")
    p.add_argument("--dormancy-lie-tau", type=float, default=None,
                   dest="dormancy_lie_tau",
                   help="τ for stealth_lie payload (default 0.9346 = Baruch z).")
    args = p.parse_args()

    overrides = {
        "seed": args.seed,
        "attack_name": args.attack,
        "n_clients": args.n_clients,
        "q": args.q,
        "frac_malicious": args.frac_malicious,
        "lie_tau": args.lie_tau,
        "p": args.participation_p,
        "participation_mode": args.participation_mode,
        "dormancy_T_dark": args.dormancy_T_dark,
        "dormancy_client_idx": args.dormancy_client_idx,
        "dormancy_payload": args.dormancy_payload,
        "dormancy_lie_tau": args.dormancy_lie_tau,
        "dormancy_client_indices": (
            [int(x) for x in args.dormancy_client_indices.split(",")]
            if args.dormancy_client_indices else None),
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
