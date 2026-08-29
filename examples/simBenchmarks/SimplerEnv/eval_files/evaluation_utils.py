"""Small, dependency-light helpers for reproducible SimplerEnv evaluations."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed_everywhere(seed: int) -> None:
    """Seed the RNGs used by the evaluation process.

    SimplerEnv owns the environment construction, so the caller seeds the
    process before invoking its evaluator. This keeps the integration limited
    to StarVLA while covering the RNGs used by model and environment helpers.
    """
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError(f"seed must be between 0 and {np.iinfo(np.uint32).max}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_evaluation_summary(args: Any, success_arr: Iterable[bool]) -> dict[str, Any]:
    """Build a JSON-serializable summary for one evaluator invocation."""
    successes = [bool(value) for value in success_arr]
    num_successes = sum(successes)
    num_episodes = len(successes)

    return {
        "seed": args.seed,
        "policy_model": args.policy_model,
        "policy_setup": args.policy_setup,
        "checkpoint": args.ckpt_path,
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
