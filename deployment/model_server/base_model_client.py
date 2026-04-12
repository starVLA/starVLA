# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""BaseModelClient — shared inference client for all evaluation benchmarks.

This module extracts common logic that was previously duplicated across
``model2libero_interface.py``, ``model2simpler_interface.py``,
``model2robotwin_interface.py``, etc.  Benchmark-specific clients should
inherit from :class:`BaseModelClient` and override only the parts that
differ (e.g. action post-processing, step return format).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


class BaseModelClient:
    """Shared base class for all benchmark model clients.

    Provides:
        * WebSocket client setup and ``predict_action`` call
        * Checkpoint-based action stats and chunk size loading
        * Image resize
        * Flexible un-normalization (``min_max`` / ``q99``)
        * Action chunking with step-based cache
        * Optional adaptive action ensemble

    Subclasses typically only need to override :meth:`step` and, optionally,
    :meth:`_format_raw_action`.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        image_size: list[int] | None = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        normalization_mode: str = "min_max",
        action_ensemble: bool = False,
        action_ensemble_horizon: Optional[int] = None,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10093,
    ) -> None:
        if image_size is None:
            image_size = [224, 224]

        # WebSocket client
        self.client = WebsocketClientPolicy(host, port)

        # Inference config
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.normalization_mode = normalization_mode

        # Load normalization stats and chunk size from checkpoint
        self.unnorm_key = unnorm_key
        self.action_norm_stats = self.load_action_stats(unnorm_key, policy_ckpt_path)
        self.action_chunk_size = self.load_action_chunk_size(policy_ckpt_path)

        # Optional action ensemble
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        if self.action_ensemble and action_ensemble_horizon is not None:
            self.action_ensembler = AdaptiveEnsembler(
                action_ensemble_horizon, adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None

        # Runtime state
        self.task_description: Optional[str] = None
        self._cached_actions: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, task_description: str) -> None:
        """Reset internal state for a new episode."""
        self.task_description = task_description
        self._cached_actions = None
        if self.action_ensembler is not None:
            self.action_ensembler.reset()

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------
    def predict_normalized_actions(self, example: dict) -> np.ndarray:
        """Send *example* to the policy server and return normalised actions.

        Returns:
            np.ndarray: Shape ``(chunk_size, action_dim)``.
        """
        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        response = self.client.predict_action(vla_input)
        try:
            normalized_actions = response["data"]["normalized_actions"]  # (B, chunk, D)
        except KeyError:
            raise KeyError(
                f"Key 'normalized_actions' not found in response. "
                f"Available keys: {list(response.get('data', {}).keys())}"
            )
        return np.asarray(normalized_actions[0])  # drop batch dim → (chunk, D)

    def get_actions_for_step(self, example: dict, step: int) -> np.ndarray:
        """Action-chunking helper: fetch a new chunk or reuse a cached one.

        When ``step`` aligns with the chunk boundary the client issues a new
        server call; otherwise the previously cached chunk is reused.

        Returns:
            np.ndarray: A single action vector of shape ``(action_dim,)``.
        """
        if step % self.action_chunk_size == 0 or self._cached_actions is None:
            normalized_actions = self.predict_normalized_actions(example)
            self._cached_actions = self.unnormalize_actions(
                normalized_actions,
                self.action_norm_stats,
                normalization_mode=self.normalization_mode,
            )
        return self._cached_actions[step % self.action_chunk_size]

    # ------------------------------------------------------------------
    # Un-normalization
    # ------------------------------------------------------------------
    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
        gripper_indices: list[int] | None = None,
    ) -> np.ndarray:
        """Un-normalize actions with support for ``min_max`` and ``q99`` modes.

        Args:
            normalized_actions: Shape ``(T, D)``, values in ``[-1, 1]``.
            action_norm_stats: Dict containing bounds and optional ``"mask"``.
            normalization_mode: ``"min_max"`` uses ``min``/``max`` keys;
                ``"q99"`` uses ``q01``/``q99`` keys.
            gripper_indices: Column indices treated as binary gripper outputs.
                ``None`` (default) skips binarization entirely.  Pass an
                explicit list (e.g. ``[6]``) to binarize specific columns.

        Returns:
            np.ndarray: Un-normalized actions, same shape as input.
        """
        action_high, action_low = BaseModelClient.get_normalization_bounds(
            action_norm_stats, normalization_mode
        )
        mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))
        normalized_actions = np.clip(normalized_actions, -1, 1)

        if gripper_indices is not None:
            for idx in gripper_indices:
                normalized_actions[:, idx] = np.where(
                    normalized_actions[:, idx] < 0.5, 0, 1
                )

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    # ------------------------------------------------------------------
    # Image utilities
    # ------------------------------------------------------------------
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize *image* to :attr:`image_size` using INTER_AREA interpolation."""
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    # ------------------------------------------------------------------
    # Checkpoint loading utilities
    # ------------------------------------------------------------------
    @staticmethod
    def load_action_stats(
        unnorm_key: Optional[str],
        policy_ckpt_path: str,
        action_mode: str = "abs",
    ) -> dict:
        """Load action normalisation statistics from a checkpoint.

        Supports two checkpoint stat formats:

        * *New*: ``{key: {"abs": {…}, "delta": {…}, …}}``
        * *Legacy*: ``{key: {"action": {…}, "state": {…}}}``
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        _, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = BaseModelClient._resolve_unnorm_key(norm_stats, unnorm_key)
        stats = norm_stats[unnorm_key]

        # New format: per-action-mode stats
        if action_mode in stats:
            mode_stats = stats[action_mode]
            return mode_stats.get("action", mode_stats)
        # Legacy format
        if "action" in stats:
            if action_mode != "abs":
                raise ValueError(
                    f"Statistics for '{unnorm_key}' only provide 'abs' action stats, "
                    f"but action_mode='{action_mode}' was requested."
                )
            return stats["action"]
        raise ValueError(
            f"Invalid statistics format for key '{unnorm_key}'. "
            f"Available sub-keys: {sorted(stats.keys())}"
        )

    @staticmethod
    def load_state_stats(
        unnorm_key: Optional[str],
        policy_ckpt_path: str,
    ) -> dict:
        """Load state normalisation statistics from a checkpoint."""
        policy_ckpt_path = Path(policy_ckpt_path)
        _, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = BaseModelClient._resolve_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def load_action_chunk_size(policy_ckpt_path: str) -> int:
        """Read the action chunk size from the model config."""
        model_config, _ = read_mode_config(Path(policy_ckpt_path))
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    # ------------------------------------------------------------------
    # Normalization bound helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_normalization_bounds(
        norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(high, low)`` bounds for un-normalization.

        Args:
            normalization_mode: ``"min_max"`` → ``(max, min)`` keys;
                ``"q99"`` → ``(q99, q01)`` keys.
        """
        key_map = {
            "min_max": ("max", "min"),
            "q99": ("q99", "q01"),
        }
        if normalization_mode not in key_map:
            raise ValueError(
                f"Unsupported normalization_mode '{normalization_mode}'. "
                f"Expected one of {sorted(key_map.keys())}."
            )
        high_key, low_key = key_map[normalization_mode]
        if high_key not in norm_stats or low_key not in norm_stats:
            raise KeyError(
                f"Normalization mode '{normalization_mode}' requires keys "
                f"'{high_key}' and '{low_key}', but available keys are: "
                f"{sorted(norm_stats.keys())}"
            )
        return np.array(norm_stats[high_key]), np.array(norm_stats[low_key])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_unnorm_key(
        norm_stats: dict, unnorm_key: Optional[str]
    ) -> str:
        """Pick the correct dataset key inside *norm_stats*."""
        if unnorm_key is None:
            if len(norm_stats) == 1:
                return next(iter(norm_stats.keys()))
            raise ValueError(
                "Model was trained on multiple datasets; please supply "
                f"`unnorm_key` from: {sorted(norm_stats.keys())}"
            )
        if unnorm_key not in norm_stats:
            raise KeyError(
                f"Unknown unnorm_key '{unnorm_key}'. "
                f"Available: {sorted(norm_stats.keys())}"
            )
        return unnorm_key
