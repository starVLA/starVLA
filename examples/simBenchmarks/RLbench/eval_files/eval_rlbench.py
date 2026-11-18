"""Run StarVLA policy-server evaluation on an RLBench task."""

import argparse
import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.simBenchmarks.RLbench.eval_files.rlbench_adapter import (
    _DEFAULT_STATE_KEYS,
    RLBenchAdapter,
    StarVLAPolicyAdapter,
    build_action_mode,
)


def rollout_success(*, total_reward: float, terminate: bool) -> bool:
    """RLBench may terminate failed episodes; only a positive success reward wins."""
    return bool(total_reward > 0.0)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load an RLBench action-mode config") from exc
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get("env"), dict):
        raise ValueError("RLBench config must contain an env mapping")
    return config


def _task_class(task_name: str) -> type:
    tasks = import_module("rlbench.tasks")
    try:
        return getattr(tasks, task_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown RLBench task class: {task_name}") from exc


def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config containing env.action_mode")
    parser.add_argument("--task", default="ReachTarget", help="RLBench task class, e.g. ReachTarget")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10095)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if "COPPELIASIM_ROOT" not in os.environ:
        raise RuntimeError("Set COPPELIASIM_ROOT before running RLBench evaluation")

    config = _load_config(args.config)
    env_config = config["env"]
    if env_config.get("name") != "rlbench":
        raise ValueError("env.name must be rlbench")
    action_mode = build_action_mode(env_config["action_mode"])

    environment_module = import_module("rlbench.environment")
    environment = environment_module.Environment(action_mode=action_mode, headless=args.headless)
    environment.launch()
    adapter = RLBenchAdapter(
        environment,
        action_mode,
        env_config.get("camera_names", ("front_rgb",)),
        state_keys=_DEFAULT_STATE_KEYS if env_config.get("include_state", False) else (),
        gripper_type=env_config["action_mode"]["gripper"]["type"],
    )
    # The policy adapter wraps the websocket client policy to handle RLBench-specific action normalization and observation formatting.
    policy = StarVLAPolicyAdapter(
        WebsocketClientPolicy(args.host, args.port),
        action_mode,
        args.unnorm_key,
        action_specification=adapter.spec,
    )
    # Here is not the instance of the task, but the class itself. 
    # The RLBenchAdapter will handle creating an instance of the task when adapter.reset(task) is called.
    task = _task_class(args.task)

    try:
        for episode in range(args.episodes):
            descriptions, observation = adapter.reset(task)
            total_reward = 0.0
            for _ in range(args.max_steps):
                raw_action = policy.predict(adapter.policy_example(observation, descriptions))
                observation, reward, terminate = adapter.step(raw_action)
                total_reward += reward
                if terminate:
                    break
            print(json.dumps({"episode": episode, "reward": total_reward, "success": rollout_success(total_reward=total_reward, terminate=terminate)}))
    finally:
        environment.shutdown()


if __name__ == "__main__":
    main()
