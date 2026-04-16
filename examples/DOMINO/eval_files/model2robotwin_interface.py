"""StarVLA ↔ DOMINO / RoboTwin evaluation interface.

This module implements the ``get_model`` / ``reset_model`` / ``eval`` API
expected by ``DOMINO/script/eval_policy.py`` (via ``eval_function_decorator``).

History support
---------------
By default the client sends only the **current frame** to the policy server.
Set ``history_k > 0`` (via ``deploy_policy.yml`` or ``get_model`` kwargs) to
enable an extensible *historical-context* pipeline:

* **optical-flow** (``history_mode="flow"``, default when history is on):
  Compute Farneback optical-flow RGB images between consecutive historical
  frames and pass them as ``example["history_images"]``.

* **raw-frames** (``history_mode="frames"``):
  Pass the last ``history_k`` raw RGB frames directly.

* **custom**: Subclass ``ModelClient`` and override ``_build_history_context``
  to implement any other historical representation.

The server-side model (``QwenOFT.predict_action``) currently ignores
``history_images`` — but the WebSocket transport passes it through
transparently.  When a model that consumes history is available (e.g.
PUMA), no client-side change is needed.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


def _load_history_utils():
    """Import history helpers without forcing a single import style."""
    try:
        from history_flow_utils import (
            compute_flow_rgb_farneback,
            parse_hw_size,
            sample_history_offsets,
        )
    except ImportError:
        from examples.DOMINO.eval_files.history_flow_utils import (
            compute_flow_rgb_farneback,
            parse_hw_size,
            sample_history_offsets,
        )
    return compute_flow_rgb_farneback, parse_hw_size, sample_history_offsets


# ---------------------------------------------------------------------------
# ModelClient
# ---------------------------------------------------------------------------

class ModelClient:
    """WebSocket client for StarVLA policy inference during DOMINO evaluation.

    Parameters
    ----------
    policy_ckpt_path : str
        Path to a StarVLA checkpoint directory.
    history_k : int
        Number of historical frames to include.  0 = current frame only.
    history_stride : int
        Temporal stride between sampled historical frames.
    history_mode : str
        ``"flow"`` → optical-flow RGB;  ``"frames"`` → raw RGB.
    history_image_size : tuple[int,int] | None
        Output resolution for history images.  Defaults to ``image_size``.
    history_flow_compute_size : tuple[int,int] | None
        Internal resolution for flow computation.  Defaults to ``(128, 128)``.
    """

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble: bool = False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] | None = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "127.0.0.1",
        port: int = 5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
        # --- history / optical-flow ---
        history_k: int = 0,
        history_stride: int = 1,
        history_mode: str = "flow",
        history_image_size: Optional[list[int]] = None,
        history_flow_compute_size: Optional[list[int]] = None,
    ) -> None:
        if image_size is None:
            image_size = [224, 224]

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode

        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # --- History configuration ---
        self.history_k = max(0, int(history_k))
        self.history_stride = max(1, int(history_stride))
        self.history_mode = self._normalize_history_mode(history_mode)
        _, parse_hw_size, _ = _load_history_utils()
        self.history_enabled = self.history_k > 0 and self.history_mode != "none"
        self.history_image_size: Tuple[int, int] = parse_hw_size(history_image_size, default_size=tuple(image_size))
        self.history_flow_compute_size: Tuple[int, int] = parse_hw_size(
            history_flow_compute_size, default_size=(128, 128)
        )
        # Frame buffer for history — stores resized RGB arrays at flow_compute_size
        if self.history_enabled:
            buf_len = self.history_k * self.history_stride
            self.history_frame_buffer: deque[np.ndarray] = deque(maxlen=buf_len)
            print(
                f"*** history enabled: k={self.history_k}, stride={self.history_stride}, "
                f"mode={self.history_mode}, out_size={self.history_image_size}, "
                f"flow_compute_size={self.history_flow_compute_size} ***"
            )
        else:
            self.history_frame_buffer = deque(maxlen=0)

        self.action_norm_stats = self.get_action_stats(
            self.unnorm_key, policy_ckpt_path=policy_ckpt_path, action_mode=action_mode
        )
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.raw_actions = None

    # ---- lifecycle -------------------------------------------------------

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.history_frame_buffer.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        self.initial_state = None
        self.prev_action = None

    # ---- history pipeline ------------------------------------------------

    @staticmethod
    def _normalize_history_mode(history_mode: Optional[str]) -> str:
        mode = "flow" if history_mode is None else str(history_mode).strip().lower()
        if mode in {"off", "none", "disabled"}:
            return "none"
        if mode not in {"flow", "frames"}:
            raise ValueError(
                f"Unknown history_mode: {history_mode!r}. Expected one of ['flow', 'frames', 'none']."
            )
        return mode

    def _push_history_frame(self, image: np.ndarray) -> None:
        """Push a frame into the history buffer (resized to flow_compute_size)."""
        if not self.history_enabled:
            return
        h, w = self.history_flow_compute_size
        resized = cv.resize(image, (w, h), interpolation=cv.INTER_AREA)
        self.history_frame_buffer.append(resized)

    def _build_history_context(self, current_image: np.ndarray) -> Optional[List[np.ndarray]]:
        """Build history representation from the frame buffer.

        Override this method to implement custom history representations.

        Returns ``None`` when history is disabled or the buffer is empty.
        Otherwise returns a list of RGB arrays (one per historical slot).
        """
        if not self.history_enabled:
            return None

        if self.history_mode == "flow":
            return self._build_history_flow(current_image)
        if self.history_mode == "frames":
            return self._build_history_frames(current_image)
        return None

    def _build_history_payload(self, current_image: np.ndarray) -> Dict[str, object]:
        """Build extra model inputs derived from historical context.

        Override this method when a future history-aware model needs a
        different payload schema. The default keeps the transport backward
        compatible by sending `history_images` when history is enabled.
        """
        history_images = self._build_history_context(current_image)
        if history_images is None:
            return {}
        return {"history_images": history_images}

    def _build_history_flow(self, current_image: np.ndarray) -> List[np.ndarray]:
        """Compute optical-flow images between consecutive sampled history frames."""
        compute_flow_rgb_farneback, _, sample_history_offsets = _load_history_utils()

        h, w = self.history_flow_compute_size
        current_small = cv.resize(current_image, (w, h), interpolation=cv.INTER_AREA)
        offsets = sample_history_offsets(self.history_k, self.history_stride)

        buf = self.history_frame_buffer
        buf_len = len(buf)

        # Build list of sampled frames (clamped to buffer bounds)
        sampled: List[np.ndarray] = []
        for off in offsets:
            idx = buf_len + off  # off is negative
            idx = max(0, min(idx, buf_len - 1)) if buf_len > 0 else -1
            sampled.append(buf[idx] if idx >= 0 else current_small)

        # Add current frame at the end for the last flow pair
        sampled.append(current_small)

        # Compute flow between consecutive pairs → history_k flow images
        out_h, out_w = self.history_image_size
        flow_images: List[np.ndarray] = []
        for i in range(len(sampled) - 1):
            flow_rgb = compute_flow_rgb_farneback(
                sampled[i], sampled[i + 1], compute_size=self.history_flow_compute_size,
            )
            flow_rgb = cv.resize(flow_rgb, (out_w, out_h), interpolation=cv.INTER_AREA)
            flow_images.append(flow_rgb)

        return flow_images

    def _build_history_frames(self, current_image: np.ndarray) -> List[np.ndarray]:
        """Return raw historical frames (resized to history_image_size)."""
        _, _, sample_history_offsets = _load_history_utils()

        h, w = self.history_flow_compute_size
        current_small = cv.resize(current_image, (w, h), interpolation=cv.INTER_AREA)
        offsets = sample_history_offsets(self.history_k, self.history_stride)

        buf = self.history_frame_buffer
        buf_len = len(buf)
        out_h, out_w = self.history_image_size

        frames: List[np.ndarray] = []
        for off in offsets:
            idx = buf_len + off
            idx = max(0, min(idx, buf_len - 1)) if buf_len > 0 else -1
            frame = buf[idx] if idx >= 0 else current_small
            frame = cv.resize(frame, (out_w, out_h), interpolation=cv.INTER_AREA)
            frames.append(frame)

        return frames

    # ---- main step -------------------------------------------------------

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        state = example.get("state", None)

        # Store initial state for delta/rel modes
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        images = example["image"]

        if task_description != self.task_description:
            self.reset(task_description)
            if self.action_mode in ["delta", "rel"] and state is not None:
                self.initial_state = np.array(state).copy()

        images = [self._resize_image(image) for image in images]
        example["image"] = images

        example_copy = example.copy()
        example_copy.pop("state", None)
        # Use the first camera (head camera) as the reference for history.
        example_copy.update(self._build_history_payload(images[0]))

        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        action_chunk_size = self.action_chunk_size

        if step % action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")

            normalized_actions = normalized_actions[0]
            raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
                normalization_mode=self.normalization_mode,
            )

            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions

        # --- Push current frame into history buffer (after inference) ---
        self._push_history_frame(images[0])

        action_idx = step % action_chunk_size
        if action_idx >= len(self.raw_actions):
            pass

        current_action = self.raw_actions[action_idx]

        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        current_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        return current_action

    # ---- normalization helpers -------------------------------------------

    @staticmethod
    def normalize_state(
        state: dict[str, np.ndarray],
        state_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> dict[str, np.ndarray]:
        continuous_mask = np.array(
            [True, True, True, True, True, True, True, True, True, True, True, True, False, False],
            dtype=bool,
        )
        state_high, state_low = ModelClient._get_normalization_bounds(
            state_norm_stats, normalization_mode=normalization_mode
        )
        valid_mask = continuous_mask & (state_high != state_low)
        normalized_state = np.where(
            valid_mask,
            (state - state_low) / (state_high - state_low) * 2 - 1,
            state,
        )
        normalized_state = np.where(
            ~continuous_mask,
            (normalized_state > 0.5).astype(normalized_state.dtype),
            normalized_state,
        )
        return normalized_state

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> np.ndarray:
        action_high, action_low = ModelClient._get_normalization_bounds(
            action_norm_stats, normalization_mode=normalization_mode
        )
        mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))
        normalized_actions = np.clip(normalized_actions, -1, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        abs_actions = np.zeros_like(delta_actions)
        mask = self.action_norm_stats.get("mask", np.ones(delta_actions.shape[-1], dtype=bool))
        base = self.prev_action if self.prev_action is not None else self.initial_state
        for i in range(len(delta_actions)):
            abs_actions[i] = np.where(mask, delta_actions[i] + base, delta_actions[i])
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        abs_actions = np.zeros_like(rel_actions)
        mask = self.action_norm_stats.get("mask", np.ones(rel_actions.shape[-1], dtype=bool))
        for i in range(len(rel_actions)):
            abs_actions[i] = np.where(mask, rel_actions[i] + self.initial_state, rel_actions[i])
        return abs_actions

    # ---- stats / config helpers ------------------------------------------

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path, action_mode: str = "abs") -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        stats = norm_stats[unnorm_key]

        if action_mode in stats:
            mode_stats = stats[action_mode]
            return mode_stats.get("action", mode_stats)
        if "action" in stats:
            if action_mode != "abs":
                raise ValueError(
                    f"Statistics key `{unnorm_key}` only provides `abs` action stats, "
                    f"but action_mode=`{action_mode}` was requested."
                )
            return stats["action"]
        raise ValueError(
            f"Invalid statistics file format for key `{unnorm_key}`. "
            f"Available top-level keys: {sorted(stats.keys())}"
        )

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        available_keys = sorted(norm_stats.keys())
        if unnorm_key is None:
            if len(available_keys) == 1:
                return available_keys[0]
            raise ValueError(
                "`unnorm_key` must be provided when multiple normalization statistics are available. "
                f"Available keys: {available_keys}"
            )
        if unnorm_key not in norm_stats:
            raise KeyError(f"Unknown `unnorm_key`: `{unnorm_key}`. Available keys: {available_keys}")
        return unnorm_key

    @staticmethod
    def _get_normalization_bounds(
        norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> tuple[np.ndarray, np.ndarray]:
        if normalization_mode == "q99":
            if "q01" not in norm_stats or "q99" not in norm_stats:
                raise KeyError("Normalization mode `q99` requires statistics keys `q01` and `q99`.")
            return np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
        if normalization_mode == "min_max":
            if "min" not in norm_stats or "max" not in norm_stats:
                raise KeyError("Normalization mode `min_max` requires statistics keys `min` and `max`.")
            return np.array(norm_stats["max"]), np.array(norm_stats["min"])
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode}. Expected one of ['min_max', 'q99'].")


# ---------------------------------------------------------------------------
# DOMINO eval_policy.py API
# ---------------------------------------------------------------------------

def get_model(usr_args: dict) -> ModelClient:
    """Construct a ``ModelClient`` from the merged DOMINO config dict.

    All keys in ``deploy_policy.yml`` plus ``--overrides`` appear in
    *usr_args*.  History settings are optional and default to disabled.
    """
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=usr_args.get("host", "127.0.0.1"),
        port=usr_args.get("port", 5694),
        unnorm_key=usr_args.get("unnorm_key", None),
        action_mode=usr_args.get("action_mode", "abs"),
        normalization_mode=usr_args.get(
            "action_normalization_mode",
            usr_args.get("normalization_mode", "min_max"),
        ),
        # History (default: disabled)
        history_k=int(usr_args.get("history_k", 0)),
        history_stride=int(usr_args.get("history_stride", 1)),
        history_mode=usr_args.get("history_mode", "flow"),
        history_image_size=usr_args.get("history_image_size", None),
        history_flow_compute_size=usr_args.get("history_flow_compute_size", None),
    )


def reset_model(model: ModelClient) -> None:
    model.reset(task_description="")


def eval(TASK_ENV, model: ModelClient, observation: dict) -> None:
    instruction = TASK_ENV.get_instruction()

    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    images = [head_img, left_img, right_img]  # [head, left_wrist, right_wrist]
    state = observation["joint_action"]["vector"]

    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)
    TASK_ENV.take_action(action)
