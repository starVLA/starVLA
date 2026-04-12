# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""RoboTwin benchmark model client.

Inherits shared inference logic from :class:`BaseModelClient` and adds
RoboTwin-specific features: delta/rel action modes, state normalisation,
and dual-arm index reordering.
"""

from typing import Dict, Optional

import numpy as np

from deployment.model_server.base_model_client import BaseModelClient


class ModelClient(BaseModelClient):
    """Model client for the RoboTwin benchmark.

    Supports three action modes:

    * ``"abs"``   – absolute joint positions (no conversion)
    * ``"delta"`` – delta actions accumulated from initial state
    * ``"rel"``   – actions relative to the initial state

    Expected *example* dict layout::

        {
            "image": [head_img, left_img, right_img],
            "lang":  "task description",
            "state": np.ndarray,  # required for delta/rel modes
        }
    """

    # Dual-arm index reorder: starVLA convention → RoboTwin convention
    _REORDER_INDICES = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        image_size: list[int] | None = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_ensemble: bool = False,
        action_ensemble_horizon: Optional[int] = 3,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "127.0.0.1",
        port: int = 5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
        # Legacy kwargs accepted for backward compatibility
        **kwargs,
    ) -> None:
        if image_size is None:
            image_size = [224, 224]
        super().__init__(
            policy_ckpt_path=policy_ckpt_path,
            unnorm_key=unnorm_key,
            image_size=image_size,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
            normalization_mode=normalization_mode,
            action_ensemble=action_ensemble,
            action_ensemble_horizon=action_ensemble_horizon,
            adaptive_ensemble_alpha=adaptive_ensemble_alpha,
            host=host,
            port=port,
        )
        self.policy_setup = policy_setup
        self.action_mode = action_mode

        # State tracking for delta/rel action modes
        self.initial_state: Optional[np.ndarray] = None
        self.prev_action: Optional[np.ndarray] = None

        # Load state stats (used by normalize_state)
        self.state_norm_stats = self.load_state_stats(unnorm_key, policy_ckpt_path)

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode} ***"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, task_description: str) -> None:
        super().reset(task_description)
        self.initial_state = None
        self.prev_action = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(self, example: dict, step: int = 0) -> np.ndarray:
        """Run one inference step and return a reordered absolute action.

        Returns:
            np.ndarray: Action vector of shape ``(14,)`` in RoboTwin order.
        """
        state = example.get("state", None)

        # Capture initial state for delta/rel modes
        if self.action_mode in ("delta", "rel") and self.initial_state is None:
            if state is None:
                raise ValueError(
                    f"action_mode='{self.action_mode}' requires 'state' in example"
                )
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)
            # Re-capture initial state after reset
            if self.action_mode in ("delta", "rel") and state is not None:
                self.initial_state = np.array(state).copy()

        # Resize images
        example["image"] = [self.resize_image(img) for img in example["image"]]

        # Strip state before sending to server (server doesn't use raw state)
        server_example = {k: v for k, v in example.items() if k != "state"}

        # Fetch (or reuse cached) action chunk
        current_action = self._get_action_for_step(server_example, step, state)

        # Update delta-mode tracking
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        # Reorder to RoboTwin convention
        return current_action[self._REORDER_INDICES]

    # ------------------------------------------------------------------
    # Action-mode-aware chunking
    # ------------------------------------------------------------------
    def _get_action_for_step(
        self, example: dict, step: int, state: Optional[np.ndarray]
    ) -> np.ndarray:
        if step % self.action_chunk_size == 0 or self._cached_actions is None:
            normalized = self.predict_normalized_actions(example)
            raw_actions = self.unnormalize_actions(
                normalized,
                self.action_norm_stats,
                normalization_mode=self.normalization_mode,
                gripper_indices=None,  # RoboTwin handles grippers via mask
            )
            # Convert to absolute coordinates if needed
            if self.action_mode == "delta":
                self._cached_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                self._cached_actions = self._rel_to_absolute(raw_actions)
            else:
                self._cached_actions = raw_actions

        action_idx = step % self.action_chunk_size
        return self._cached_actions[min(action_idx, len(self._cached_actions) - 1)]

    # ------------------------------------------------------------------
    # Delta / Relative → Absolute conversion
    # ------------------------------------------------------------------
    def _delta_to_absolute(
        self, delta_actions: np.ndarray, current_state: Optional[np.ndarray]
    ) -> np.ndarray:
        """Convert delta actions to absolute.

        Training convention: ``delta[0] = a[0] - s[0]``,
        ``delta[t] = a[t] - a[t-1]``.
        """
        mask = self.action_norm_stats.get(
            "mask", np.ones(delta_actions.shape[-1], dtype=bool)
        )
        base = self.prev_action if self.prev_action is not None else self.initial_state
        abs_actions = np.zeros_like(delta_actions)
        for i in range(len(delta_actions)):
            abs_actions[i] = np.where(mask, delta_actions[i] + base, delta_actions[i])
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        """Convert relative actions to absolute.

        Training convention: ``rel[t] = a[t] - s[0]``.
        """
        mask = self.action_norm_stats.get(
            "mask", np.ones(rel_actions.shape[-1], dtype=bool)
        )
        abs_actions = np.zeros_like(rel_actions)
        for i in range(len(rel_actions)):
            abs_actions[i] = np.where(
                mask, rel_actions[i] + self.initial_state, rel_actions[i]
            )
        return abs_actions

    # ------------------------------------------------------------------
    # State normalization
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_state(
        state: np.ndarray,
        state_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
        continuous_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Normalize proprioceptive state for model input.

        Continuous dimensions use linear normalization to ``[-1, 1]``;
        discrete (gripper) dimensions are binarized.
        """
        if continuous_mask is None:
            # Default for 14-DoF dual-arm: last 2 dims are binary grippers
            continuous_mask = np.array(
                [True] * 12 + [False, False], dtype=bool
            )
        state_high, state_low = BaseModelClient.get_normalization_bounds(
            state_norm_stats, normalization_mode
        )
        valid = continuous_mask & (state_high != state_low)
        normalized = np.where(
            valid,
            (state - state_low) / (state_high - state_low) * 2 - 1,
            state,
        )
        normalized = np.where(
            ~continuous_mask,
            (normalized > 0.5).astype(normalized.dtype),
            normalized,
        )
        return normalized

    # ------------------------------------------------------------------
    # Backward-compatible static helpers  (deprecated – prefer base class)
    # ------------------------------------------------------------------
    @staticmethod
    def get_action_stats(unnorm_key, policy_ckpt_path, action_mode="abs"):
        return BaseModelClient.load_action_stats(
            unnorm_key, policy_ckpt_path, action_mode=action_mode
        )

    @staticmethod
    def get_state_stats(unnorm_key, policy_ckpt_path):
        return BaseModelClient.load_state_stats(unnorm_key, policy_ckpt_path)

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        return BaseModelClient.load_action_chunk_size(policy_ckpt_path)


# ======================================================================
# Convenience factory (used by RoboTwin eval scripts)
# ======================================================================

def get_model(usr_args: dict) -> ModelClient:
    """Create a :class:`ModelClient` from a flat config dict."""
    return ModelClient(
        policy_ckpt_path=usr_args["policy_ckpt_path"],
        host=usr_args.get("host", "127.0.0.1"),
        port=usr_args.get("port", 5694),
        unnorm_key=usr_args.get("unnorm_key"),
        action_mode=usr_args.get("action_mode", "abs"),
        normalization_mode=usr_args.get(
            "action_normalization_mode",
            usr_args.get("normalization_mode", "min_max"),
        ),
    )


def reset_model(model: ModelClient) -> None:
    model.reset(task_description="")


def eval(TASK_ENV, model: ModelClient, observation: dict) -> None:
    """One-step eval loop helper for RoboTwin."""
    instruction = TASK_ENV.get_instruction()
    images = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
    ]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": observation["joint_action"]["vector"],
    }
    action = model.step(example, step=TASK_ENV.take_action_cnt)
    TASK_ENV.take_action(action)
