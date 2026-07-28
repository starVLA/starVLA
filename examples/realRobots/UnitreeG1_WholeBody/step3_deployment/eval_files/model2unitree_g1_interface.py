"""StarVLA -> Unitree G1 inference adapter.

This client mirrors the GR00T PolicyClient contract used by
`gear_sonic/scripts/run_vla_inference.py`, but it talks to the StarVLA
websocket policy server instead of Isaac-GR00T.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


_STATE_ORDER = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
    "left_wrist_pos",
    "left_wrist_abs_quat",
    "right_wrist_pos",
    "right_wrist_abs_quat",
    "root_orientation",
    "projected_gravity",
    "cpp_rotation_offset",
    "init_base_quat",
]


def _as_array(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _extract_prompt(observation: dict, fallback: str) -> str:
    lang = observation.get("language")
    if isinstance(lang, dict):
        for value in lang.values():
            if value is None:
                continue
            if isinstance(value, list) and value and isinstance(value[0], list):
                return str(value[0][0])
            if isinstance(value, list) and value:
                return str(value[0])
            return str(value)
    if "lang" in observation and observation["lang"] is not None:
        return str(observation["lang"])
    return fallback


def _extract_image(observation: dict) -> np.ndarray:
    if "image" in observation:
        return np.asarray(observation["image"])
    video = observation.get("video", {})
    if isinstance(video, dict):
        if "ego_view" in video:
            return np.asarray(video["ego_view"])[0, 0]
    raise KeyError("observation does not contain an image or video.ego_view")


def _extract_state(observation: dict) -> Optional[np.ndarray]:
    if "state" not in observation:
        return None

    state = observation["state"]
    if isinstance(state, np.ndarray):
        return state.reshape(-1).astype(np.float32)

    if isinstance(state, dict):
        pieces = []
        for key in _STATE_ORDER:
            if key not in state:
                continue
            pieces.append(_as_array(state[key]).reshape(-1))
        if pieces:
            return np.concatenate(pieces, axis=0).astype(np.float32)

    if "q" in observation:
        return np.asarray(observation["q"]).reshape(-1).astype(np.float32)

    return None


@dataclass
class PolicyOutput:
    motion_token: np.ndarray
    left_hand_joints: np.ndarray
    right_hand_joints: np.ndarray

    def as_action_dict(self) -> dict[str, np.ndarray]:
        return {
            "action.motion_token": self.motion_token,
            "action.left_hand_joints": self.left_hand_joints,
            "action.right_hand_joints": self.right_hand_joints,
        }


class ModelClient:
    """Drop-in replacement for the GR00T PolicyClient used in inference."""

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "unitree_g1_sonic",
        image_size: list[int] | None = None,
        host: str = "127.0.0.1",
        port: int = 10093,
    ) -> None:
        self.policy_ckpt_path = policy_ckpt_path
        self.policy_setup = policy_setup
        self.image_size = image_size or [224, 224]
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.server_meta = self.client.get_server_metadata()
        self.unnorm_key = (
            unnorm_key
            or self.server_meta.get("default_unnorm_key")
            or "new_embodiment"
        )

    def ping(self) -> bool:
        return self.server_meta is not None

    def get_server_metadata(self) -> Dict[str, Any]:
        return self.server_meta

    def close(self) -> None:
        self.client.close()

    def _build_starvla_example(self, observation: dict, fallback_prompt: str = "demo") -> dict:
        image = _extract_image(observation)
        prompt = _extract_prompt(observation, fallback_prompt)
        state = _extract_state(observation)

        example = {
            "image": [image],
            "lang": prompt,
        }
        if state is not None:
            example["state"] = state[np.newaxis, :]
        return example

    def _split_action(self, actions: np.ndarray) -> PolicyOutput:
        chunk = np.asarray(actions)
        if chunk.ndim == 3:
            chunk = chunk[0]
        if chunk.ndim != 2:
            raise ValueError(f"Expected actions with shape [T, D] or [1, T, D], got {chunk.shape}")

        if chunk.shape[-1] < 78:
            raise ValueError(f"Expected at least 78 action dims, got {chunk.shape[-1]}")

        return PolicyOutput(
            motion_token=chunk[:, :64].astype(np.float32),
            left_hand_joints=chunk[:, 64:71].astype(np.float32),
            right_hand_joints=chunk[:, 71:78].astype(np.float32),
        )

    def get_action(self, observation: dict):
        """Mirror the Isaac-GR00T PolicyClient API.

        Returns:
            (action_dict, info_dict)
        """
        example = self._build_starvla_example(observation)
        response = self.client.predict_action(
            {
                "examples": [example],
                "unnorm_key": self.unnorm_key,
            }
        )
        if response.get("status") != "ok":
            raise RuntimeError(f"StarVLA policy server error: {response}")

        actions = np.asarray(response["data"]["actions"])
        split = self._split_action(actions)
        return split.as_action_dict(), {"actions": actions, "server_meta": self.server_meta}


# Backward-compatible alias for GR00T import sites.
PolicyClient = ModelClient
