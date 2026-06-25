"""Entry point: ``python -m src.run_loop --config configs/loop_mnist.yaml``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any

import yaml

from . import aggregators
from .loop import FederatedLoop, LoopConfig


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Custom FL loop runner.")
    p.add_argument("--config", required=True, help="Path to a YAML LoopConfig.")
    p.add_argument(
        "--aggregator",
        help="Override aggregator name (otherwise taken from the YAML).",
    )
    args = p.parse_args(argv)

    raw = _load_yaml(args.config)
    if args.aggregator:
        raw["aggregator_name"] = args.aggregator

    # Tuple-ify milestones if the YAML gave a list.
    if "milestones" in raw and isinstance(raw["milestones"], list):
        raw["milestones"] = tuple(raw["milestones"])

    cfg_fields = {f.name for f in dataclasses.fields(LoopConfig)}
    cfg_kwargs = {k: v for k, v in raw.items() if k in cfg_fields}
    cfg = LoopConfig(**cfg_kwargs)

    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(os.path.join(cfg.results_dir, "config.json"), "w") as fh:
        json.dump(raw, fh, indent=2)

    agg = aggregators.build(cfg.aggregator_name, **(cfg.aggregator_params or {}))
    FederatedLoop(cfg, agg).run()


if __name__ == "__main__":
    main()
