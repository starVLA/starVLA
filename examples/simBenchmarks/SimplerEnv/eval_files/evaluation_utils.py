"""Small, dependency-light helpers for reproducible SimplerEnv evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from deployment.model_server.seed_utils import set_seed_everywhere


def validate_server_metadata(args: Any, server_metadata: Mapping[str, Any] | None) -> None:
    """Validate the policy server identity before running an evaluation."""
    if not server_metadata or not server_metadata.get("ckpt_path"):
        raise ValueError("policy server metadata must include the served ckpt_path")
    server_seed = server_metadata.get("seed")
    if server_seed is None:
        raise ValueError("policy server metadata must include the served seed")
    if server_seed != args.seed:
        raise ValueError(f"evaluation seed {args.seed} does not match policy server seed {server_seed}")


def build_evaluation_summary(
    args: Any,
    success_arr: Iterable[bool],
    server_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable summary for one evaluator invocation."""
    validate_server_metadata(args, server_metadata)

    successes = [bool(value) for value in success_arr]
    num_successes = sum(successes)
    num_episodes = len(successes)

    return {
        "seed": args.seed,
        "policy_model": args.policy_model,
        "policy_setup": args.policy_setup,
        "checkpoint": server_metadata["ckpt_path"],
        "requested_checkpoint": args.ckpt_path,
        "environment": args.env_name,
        "task": args.env_name,
        "scene": args.scene_name,
        "robot": args.robot,
        "object_variation_mode": args.obj_variation_mode,
        "object_episode_range": list(args.obj_episode_range),
        "max_episode_steps": args.max_episode_steps,
        "num_episodes": num_episodes,
        "num_successes": num_successes,
        "success_rate": num_successes / num_episodes if num_episodes else 0.0,
        "successes": successes,
    }


def write_evaluation_summary(path: str | Path, summary: dict[str, Any]) -> None:
    """Write an evaluation summary, creating its parent directory if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
