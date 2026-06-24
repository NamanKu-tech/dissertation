"""Entry point: python -m src.run_fedlaw --config configs/fedlaw_mnist.yaml"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import yaml

from .fedlaw import FedLAWConfig, FedLAWLoop


def _load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="FedLAW training runner.")
    p.add_argument("--config", required=True, help="Path to YAML FedLAWConfig.")
    args = p.parse_args(argv)

    raw = _load_yaml(args.config)

    # Coerce list milestones to tuple if present.
    if "attack_params" not in raw:
        raw["attack_params"] = {}

    cfg_fields = {f.name for f in dataclasses.fields(FedLAWConfig)}
    cfg_kwargs = {k: v for k, v in raw.items() if k in cfg_fields}
    cfg = FedLAWConfig(**cfg_kwargs)

    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(os.path.join(cfg.results_dir, "config.json"), "w") as fh:
        json.dump(raw, fh, indent=2)

    FedLAWLoop(cfg).run()


if __name__ == "__main__":
    main()
