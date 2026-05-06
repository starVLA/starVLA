"""<<TODO_BENCH>> — Policy bridge between trained checkpoint and the env / server.

Copy to: examples/<<TODO_BENCH>>/eval_files/model2<<TODO_BENCH>>_interface.py

The contract evaluators expect:

    policy = <Bench>Policy(checkpoint_path)
    actions = policy.run_policy(obs, prompt)   # (n_action_steps, action_dim) un-normalized

Always exercise this in 3 steps (Skill Phase 7):
    1. local_self_test.py     — dummy obs, no env
    2. closed-loop on a recorded episode / mock server
    3. closed-loop on real env

Cribbing references:
    examples/Robocasa_365/eval_files/model2robocasa365_interface.py        (in-process sim)
    examples/RoboChallenge_table30v2/eval_files/model2robochallenge_interface.py  (HTTP server)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from starVLA.model.framework.base_framework import build_framework


@dataclass
class RobotSpec:
    image_keys: Sequence[str]            # cameras the model expects, in order
    state_dim: int
    action_dim: int
    norm_unnorm_key: str                 # key in dataset_statistics.json for state+action stats


# Replace with your robot(s). Mirror data_registry/data_config.py.
ROBOT_SPECS: dict[str, RobotSpec] = {
    "<<TODO_robot_type_string>>": RobotSpec(
        image_keys=["<<TODO_CAM_1>>", "<<TODO_CAM_2>>"],
        state_dim=<<TODO_STATE_DIM>>,
        action_dim=<<TODO_ACTION_DIM>>,
        norm_unnorm_key="<<TODO_robot_type_string>>",
    ),
}


def _load_norm_stats(checkpoint_path: str) -> dict:
    """Find dataset_statistics.json saved next to the checkpoint at training time."""
    ckpt = Path(checkpoint_path)
    # Trainer writes it at <run_root_dir>/<run_id>/dataset_statistics.json
    candidate = ckpt.parents[1] / "dataset_statistics.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"dataset_statistics.json not found at {candidate} — "
                                f"keep it paired with the checkpoint")
    return json.loads(candidate.read_text())


def _normalize(x: np.ndarray, stats: dict, keys: Sequence[str]) -> np.ndarray:
    """min_max → [-1, 1]. Mirror of starVLA's training-time normalization."""
    lo = np.array([stats[k]["min"] for k in keys], dtype=np.float32)
    hi = np.array([stats[k]["max"] for k in keys], dtype=np.float32)
    rng = np.maximum(hi - lo, 1e-8)
    return 2.0 * (x - lo) / rng - 1.0


def _unnormalize(y: np.ndarray, stats: dict, keys: Sequence[str], mask: np.ndarray | None = None) -> np.ndarray:
    lo = np.array([stats[k]["min"] for k in keys], dtype=np.float32)
    hi = np.array([stats[k]["max"] for k in keys], dtype=np.float32)
    out = (y + 1.0) / 2.0 * (hi - lo) + lo
    if mask is not None:
        out = np.where(mask, out, y)   # leave un-normalizable dims (e.g. quaternions) raw
    return out


class <<TODO_BENCH_PASCAL>>Policy:
    def __init__(self, checkpoint_path: str, robot_tag: str = "<<TODO_robot_type_string>>"):
        if robot_tag not in ROBOT_SPECS:
            raise KeyError(f"robot_tag={robot_tag} not in {list(ROBOT_SPECS)}")
        self.spec = ROBOT_SPECS[robot_tag]
        self.stats = _load_norm_stats(checkpoint_path)
        self.model = build_framework(checkpoint_path=checkpoint_path)
        self.model.eval()

    @property
    def n_action_steps(self) -> int:
        return self.model.cfg.framework.action_model.action_horizon

    def run_policy(self, obs: dict, prompt: str) -> np.ndarray:
        # 1. images: (H, W, 3) uint8 per camera, in self.spec.image_keys order
        images = [Image.fromarray(obs["images"][k]).convert("RGB") for k in self.spec.image_keys]

        # 2. normalize state
        raw_state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        norm_state = _normalize(raw_state, self.stats[self.spec.norm_unnorm_key]["state"],
                                keys=list(self.stats[self.spec.norm_unnorm_key]["state"].keys()))
        norm_state = norm_state[None, :]   # (1, S)

        # 3. forward
        sample = {"image": images, "lang": prompt, "state": norm_state}
        out = self.model.predict_action([sample])
        norm_action = out["normalized_actions"][0].cpu().numpy()  # (T, A)

        # 4. un-normalize
        action_keys = list(self.stats[self.spec.norm_unnorm_key]["action"].keys())
        return _unnormalize(norm_action, self.stats[self.spec.norm_unnorm_key]["action"], action_keys)
