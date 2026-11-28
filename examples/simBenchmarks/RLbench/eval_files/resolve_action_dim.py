"""Resolve the native RLBench action dimension for an action-mode config."""

import argparse
import os
from importlib import import_module
from pathlib import Path

import yaml

from examples.simBenchmarks.RLbench.eval_files.rlbench_adapter import RLBenchAdapter, build_action_mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    env_config = config["env"]
    if env_config.get("name") != "rlbench":
        raise ValueError("env.name must be rlbench")

    action_mode = build_action_mode(env_config["action_mode"])
    environment = import_module("rlbench.environment").Environment(action_mode=action_mode, headless=True)
    try:
        environment.launch()
        print(f"RLBENCH_ACTION_DIM={RLBenchAdapter(environment, action_mode).spec.action_dim}")
    finally:
        environment.shutdown()


if __name__ == "__main__":
    if "COPPELIASIM_ROOT" not in os.environ:
        raise RuntimeError("Set COPPELIASIM_ROOT before resolving an RLBench action dimension")
    main()
