"""Policy bridge for RoboChallenge_table30v2 evaluation.

Wraps a trained ``baseframework`` checkpoint and adapts it to the I/O contract
of ``https://github.com/RoboChallenge/RoboChallengeInference`` (cvpr branch):

State input from the RC server (pickled dict)::

    {
        "images":  {"<cam_name>": <PNG bytes>, ...},
        "action":  [...],            # current robot state in the requested action_type
        "pending_actions": int,
        "timestamp": float,
        "state":   "normal" | "abnormal" | "size_none",
    }

The wrapper reads the camera PNGs + a 7-d joint state (UR5: 6 joint + 1 gripper),
runs the policy, and returns an 8-d action chunk (UR5 ``leftpos``: 7 ee_pose
quat + 1 gripper) ready to be POSTed via ``InterfaceClient.post_actions``.

Robot-specific shapes are read from a small registry below — consistent with
``examples/RoboChallenge_table30v2/train_files/data_registry/data_config.py``.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robot registry — keep in sync with train_files/data_registry/data_config.py
# ---------------------------------------------------------------------------

@dataclass
class RobotSpec:
    robot_tag: str                    # RC server tag: "ur5" | "arx5" | "w1" | "aloha"
    image_types: List[str]            # cameras requested from /state.pkl in order
    state_action_type: str            # action_type used to fetch the *state* (we want joint+gripper)
    post_action_type: str             # action_type used to *post* actions (we send ee_pose+gripper)
    state_dim: int                    # policy state input dim
    action_dim: int                   # policy action output dim
    norm_unnorm_key: str              # key inside dataset_statistics.json (e.g. "new_embodiment")


# Single-arm tasks shipped by Table30v2.
ROBOT_SPECS: Dict[str, RobotSpec] = {
    "ur5": RobotSpec(
        robot_tag="ur5",
        image_types=["cam_global", "cam_arm"],
        state_action_type="leftjoint",   # state["action"] = joint(6)+gripper(1) = 7
        post_action_type="leftpos",      # outgoing actions = ee_pose(7 quat)+gripper(1) = 8
        state_dim=7,
        action_dim=8,
        norm_unnorm_key="new_embodiment",
    ),
    "arx5": RobotSpec(
        robot_tag="arx5",
        image_types=["cam_global", "cam_arm", "cam_side"],
        state_action_type="leftjoint",
        post_action_type="leftpos",
        state_dim=7,
        action_dim=8,
        norm_unnorm_key="new_embodiment",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_norm_stats(checkpoint_path: str | Path) -> dict:
    """Load ``dataset_statistics.json`` from the run dir of *checkpoint_path*.

    Layout::

        <run_dir>/dataset_statistics.json
        <run_dir>/checkpoints/steps_*_pytorch_model.pt
    """
    ckpt = Path(checkpoint_path)
    run_dir = ckpt.parents[1] if ckpt.parents[1].joinpath("dataset_statistics.json").exists() else ckpt.parent
    stats_json = run_dir / "dataset_statistics.json"
    if not stats_json.exists():
        raise FileNotFoundError(f"Missing dataset_statistics.json beside {ckpt} (looked in {run_dir})")
    with stats_json.open() as f:
        return json.load(f)


def _decode_png(buf: bytes) -> np.ndarray:
    """Decode PNG bytes returned by RC ``/state.pkl`` into an HxWx3 RGB ndarray."""
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    return np.asarray(img)


def _normalize_state(state: np.ndarray, stats: dict) -> np.ndarray:
    """Apply the same ``min_max`` normalization used during training (per-dim).

    Mirror of ``StateActionTransform`` (see
    ``starVLA/dataloader/gr00t_lerobot/transform/state_action.py``):
    ``y = 2 * (x - min) / (max - min) - 1`` with passthrough where ``min == max``.
    """
    s_min = np.asarray(stats["min"], dtype=np.float32)
    s_max = np.asarray(stats["max"], dtype=np.float32)
    out = state.astype(np.float32).copy()
    mask = s_max != s_min
    out[..., mask] = 2.0 * (out[..., mask] - s_min[mask]) / (s_max[mask] - s_min[mask]) - 1.0
    return out


def _unnormalize_action(norm_action: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of ``min_max`` for actions, respecting the ``mask`` field.

    ``mask[d] == True``  → (norm+1)/2 * (max-min) + min
    ``mask[d] == False`` → keep ``norm`` raw (the gripper is stored unnormalized).
    """
    a_min = np.asarray(stats["min"], dtype=np.float32)
    a_max = np.asarray(stats["max"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", [True] * len(a_min)), dtype=bool)
    norm = np.clip(norm_action.astype(np.float32), -1.0, 1.0)
    out = np.where(mask, (norm + 1.0) / 2.0 * (a_max - a_min) + a_min, norm_action.astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class RoboChallengePolicy:
    """Concrete ``DummyPolicy`` replacement compatible with upstream demo.py / test.py.

    Owns the model directly (no websocket) — keeps the hop count low for the
    self-test and for the production demo loop where latency matters.

    Usage::

        policy = RoboChallengePolicy(checkpoint_path, robot_tag="ur5")
        actions = policy.run_policy(state_dict, prompt="shred the paper")
        # actions: list[list[float]] with shape (n_action_steps, 8) for UR5.
    """

    def __init__(
        self,
        checkpoint_path: str,
        robot_tag: str = "ur5",
        n_action_steps: int = 8,
        image_size: Sequence[int] = (224, 224),
        device: str = "cuda",
        use_bf16: bool = True,
    ) -> None:
        if robot_tag not in ROBOT_SPECS:
            raise ValueError(f"Unsupported robot_tag={robot_tag}; options={list(ROBOT_SPECS)}")
        self.spec = ROBOT_SPECS[robot_tag]
        self.checkpoint_path = checkpoint_path
        self.n_action_steps = int(n_action_steps)
        self.image_size = tuple(image_size)
        self.device = torch.device(device)

        logger.info("[RC] Loading framework from %s", checkpoint_path)
        self.model = baseframework.from_pretrained(checkpoint_path)
        if use_bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(self.device).eval()

        norm_stats_full = _load_norm_stats(checkpoint_path)
        if self.spec.norm_unnorm_key not in norm_stats_full:
            available = list(norm_stats_full.keys())
            if len(available) == 1:
                self.spec.norm_unnorm_key = available[0]
                logger.warning("[RC] unnorm_key fallback to %s (only one available)", available[0])
            else:
                raise KeyError(f"unnorm_key {self.spec.norm_unnorm_key} not in {available}")
        self.state_stats = norm_stats_full[self.spec.norm_unnorm_key]["state"]
        self.action_stats = norm_stats_full[self.spec.norm_unnorm_key]["action"]

        # Cache last prompt for logging.
        self._last_prompt: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API expected by upstream GPUClient
    # ------------------------------------------------------------------

    def run_policy(self, input_data: dict, prompt: Optional[str] = None) -> List[List[float]]:
        """Single-call inference.

        Args:
            input_data: The unpickled response of ``GET /state.pkl`` — see module docstring.
            prompt: Free-form task instruction.

        Returns:
            list[list[float]]: ``n_action_steps`` actions, each of length ``action_dim``.
            The list is JSON-serialisable and ready for ``InterfaceClient.post_actions``.
        """
        if prompt is not None and prompt != self._last_prompt:
            logger.info("[RC] Prompt: %r", prompt)
            self._last_prompt = prompt
        instruction = prompt if prompt else (self._last_prompt or "perform the task")

        # ---- 1. Images ---------------------------------------------------
        images_dict = input_data.get("images") or {}
        pil_images: List[Image.Image] = []
        for cam in self.spec.image_types:
            if cam not in images_dict:
                raise KeyError(f"[RC] Missing camera {cam!r} in state.images keys={list(images_dict)}")
            arr = _decode_png(images_dict[cam])
            if arr.shape[:2] != self.image_size:
                arr = np.asarray(Image.fromarray(arr).resize(self.image_size, Image.BILINEAR))
            pil_images.append(Image.fromarray(arr))

        # ---- 2. State ----------------------------------------------------
        raw_state = np.asarray(input_data.get("action") or [], dtype=np.float32)
        if raw_state.size != self.spec.state_dim:
            raise ValueError(
                f"[RC] state vector length {raw_state.size} != expected {self.spec.state_dim} "
                f"(make sure get_state was called with action_type={self.spec.state_action_type!r})"
            )
        norm_state = _normalize_state(raw_state, self.state_stats)  # (D,)
        state_input = norm_state[None, :]  # (1, D)

        # ---- 3. Forward --------------------------------------------------
        sample = {
            "image": pil_images,
            "lang": instruction,
            "state": state_input,
        }
        out = self.model.predict_action([sample])
        normalized_actions = out["normalized_actions"]  # (1, T, D)

        # ---- 4. Unnormalize + slice -------------------------------------
        actions = _unnormalize_action(normalized_actions[0], self.action_stats)  # (T, D)
        actions = actions[: self.n_action_steps]
        return actions.astype(np.float32).tolist()

    # ------------------------------------------------------------------
    # Convenience accessors used by the launcher scripts
    # ------------------------------------------------------------------

    @property
    def image_type(self) -> List[str]:
        return list(self.spec.image_types)

    @property
    def state_action_type(self) -> str:
        return self.spec.state_action_type

    @property
    def post_action_type(self) -> str:
        return self.spec.post_action_type
