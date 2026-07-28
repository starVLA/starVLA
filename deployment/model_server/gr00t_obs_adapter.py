# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""GR00T observation/action adapter.

Converts between the Isaac-GR00T ``PolicyServer`` request contract (nested
modality dicts) and the starVLA ``examples`` contract (flat state / flat
action chunk). Used by ``server_policy_gr00t_zmq.py`` so that existing
GR00T clients — e.g. the GR00T-WBC-Bridge deploy script — can talk to a
starVLA checkpoint without any client-side changes.

GR00T observation (what clients like the WBC bridge send)::

    {
        "video":    {"<view>": np.uint8 (1, n_frames, H, W, 3)},
        "state":    {"left_arm": (1, n_hist, 7) f32, "right_arm": ..., ...},
        "language": {"annotation.human.task_description": [[str]]},
    }

GR00T action response (what those clients consume)::

    ({"left_arm": (1, T, 7) f32, "right_arm": ..., "waist": (1, T, 3), ...},
     info_dict)

The state flattening order and the action split are both driven by the
checkpoint's training-time DataConfig (via ``PolicyNormProcessor``), so the
adapter has no per-robot hardcoding: register the embodiment's DataConfig,
train, and the wire contract follows.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_GR00T_LANGUAGE_KEY = "annotation.human.task_description"


def _latest_frame(value: Any, dim: int, key: str) -> np.ndarray:
    """Pull the most recent (dim,) vector from a state entry of shape
    (1, n_hist, dim) / (n_hist, dim) / (dim,)."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[-1] != dim:
        raise ValueError(
            f"state key {key!r}: expected last dim {dim} (from the checkpoint's "
            f"DataConfig), got shape {tuple(arr.shape)}"
        )
    return arr.reshape(-1)[-dim:]


def _latest_image(video_value: Any) -> np.ndarray:
    """Pull the most recent (H, W, 3) frame from a video entry of shape
    (1, n_frames, H, W, 3) / (n_frames, H, W, 3) / (H, W, 3)."""
    arr = np.asarray(video_value)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim != 3:
        raise ValueError(f"video entry has unsupported shape {tuple(arr.shape)}")
    return arr


def _extract_language(observation: dict, fallback: str) -> str:
    lang = observation.get("language")
    if isinstance(lang, dict):
        value = lang.get(_GR00T_LANGUAGE_KEY)
        if value is None and lang:
            value = next(iter(lang.values()))
        while isinstance(value, (list, tuple)) and value:
            value = value[0]
        if value is not None:
            return str(value)
    if observation.get("lang") is not None:
        return str(observation["lang"])
    return fallback


class Gr00tCompatPolicy:
    """Presents a starVLA ``PolicyServerWrapper`` as a GR00T policy.

    Args:
        wrapper: A ``PolicyServerWrapper`` (or duck-typed stand-in) exposing
            ``predict_action(examples, unnorm_key)``, ``metadata`` and
            ``get_norm_processor(unnorm_key)``.
        unnorm_key: Dataset statistics key; ``None`` uses the wrapper default.
        send_state: Include flattened proprioception in the example. Set False
            for checkpoints trained without state input.
        fallback_instruction: Used when the observation carries no language.
    """

    def __init__(
        self,
        wrapper,
        unnorm_key: Optional[str] = None,
        send_state: bool = True,
        fallback_instruction: str = "",
    ) -> None:
        self._wrapper = wrapper
        self._send_state = send_state
        self._fallback_instruction = fallback_instruction

        proc = wrapper.get_norm_processor(unnorm_key)
        self._unnorm_key: str = proc.unnorm_key
        # Full keys keep DataConfig order ("state.left_arm", ...); subkeys are
        # what appears on the wire ("left_arm", ...).
        self._state_keys: List[str] = list(proc.state_keys)
        self._state_key_dims: Dict[str, int] = dict(proc.state_key_dims)
        self._action_keys: List[str] = list(proc.action_keys)
        self._action_key_dims: Dict[str, int] = dict(proc.action_key_dims)

        logger.info(
            "Gr00tCompatPolicy ready: unnorm_key=%s state order=%s action split=%s",
            self._unnorm_key,
            [(k.split(".", 1)[-1], self._state_key_dims.get(k, 1)) for k in self._state_keys],
            [(k.split(".", 1)[-1], self._action_key_dims.get(k, 1)) for k in self._action_keys],
        )

    # -- GR00T PolicyServer endpoints -----------------------------------------

    def get_action(
        self, observation: Dict[str, Any], options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        example = self.obs_to_example(observation)
        result = self._wrapper.predict_action(examples=[example], unnorm_key=self._unnorm_key)
        actions = np.asarray(result["actions"])  # (B, T, D)
        action_dict = self.split_actions(actions)
        info = {"unnorm_key": self._unnorm_key, "action_dim": int(actions.shape[-1])}
        return action_dict, info

    def reset(self, options: Optional[dict] = None) -> Dict[str, Any]:
        # starVLA inference is stateless per request; nothing to clear.
        return {"ok": True}

    def get_modality_config(self) -> Dict[str, Any]:
        """Not a real GR00T ModalityConfig — a plain-dict contract summary that
        lets clients sanity-check key order and dims at handshake time."""
        return {
            "state_keys": self._state_keys,
            "state_key_dims": self._state_key_dims,
            "action_keys": self._action_keys,
            "action_key_dims": self._action_key_dims,
            "unnorm_key": self._unnorm_key,
        }

    # -- Conversions -----------------------------------------------------------

    def obs_to_example(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        video = observation.get("video")
        if not isinstance(video, dict) or not video:
            raise KeyError("observation is missing the 'video' modality dict")
        images = [_latest_image(v) for v in video.values()]

        example: Dict[str, Any] = {
            "image": images,
            "lang": _extract_language(observation, self._fallback_instruction),
        }

        if self._send_state and self._state_keys:
            state_in = observation.get("state")
            if not isinstance(state_in, dict):
                raise KeyError(
                    "observation is missing the 'state' modality dict but the "
                    f"checkpoint was trained with state keys {self._state_keys}"
                )
            pieces = []
            for full_key in self._state_keys:
                subkey = full_key.split(".", 1)[-1]
                if subkey not in state_in:
                    raise KeyError(
                        f"observation state is missing {subkey!r} (required by the "
                        f"checkpoint's DataConfig). Received keys: {sorted(state_in)}"
                    )
                dim = self._state_key_dims.get(full_key, 1)
                pieces.append(_latest_frame(state_in[subkey], dim, subkey))
            flat = np.concatenate(pieces, axis=0).astype(np.float32)
            example["state"] = flat[np.newaxis, :]
        return example

    def split_actions(self, actions: np.ndarray) -> Dict[str, np.ndarray]:
        """Split an unnormalized (B, T, D) chunk into GR00T-named groups of
        shape (B, T, dim_k), ordered/sized by the training DataConfig."""
        if actions.ndim == 2:
            actions = actions[np.newaxis, ...]
        if actions.ndim != 3:
            raise ValueError(f"expected actions (B, T, D), got shape {tuple(actions.shape)}")

        out: Dict[str, np.ndarray] = {}
        cursor = 0
        for full_key in self._action_keys:
            subkey = full_key.split(".", 1)[-1]
            dim = self._action_key_dims.get(full_key, 1)
            out[subkey] = np.ascontiguousarray(
                actions[:, :, cursor : cursor + dim], dtype=np.float32
            )
            cursor += dim
        if cursor != actions.shape[-1]:
            raise ValueError(
                f"sum of per-key action dims ({cursor}) != action_dim "
                f"({actions.shape[-1]}); action_keys={self._action_keys}, "
                f"action_key_dims={self._action_key_dims}"
            )
        return out
