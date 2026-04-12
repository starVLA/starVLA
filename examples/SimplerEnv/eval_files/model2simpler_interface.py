# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""SimplerEnv benchmark model client.

Inherits shared inference logic from :class:`BaseModelClient` and adds
SimplerEnv-specific features: sticky gripper logic, euler-to-axis-angle
conversion, and per-robot-setup defaults (widowx_bridge / google_robot).
"""

from __future__ import annotations

import os
from collections import deque
from typing import Optional, Sequence

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle

from deployment.model_server.base_model_client import BaseModelClient


class ModelClient(BaseModelClient):
    """Model client for the SimplerEnv (ManiSkill2) benchmark.

    Unlike other benchmark clients, SimplerEnv's :meth:`step` takes a raw
    ``image`` and ``task_description`` rather than an ``example`` dict, and
    returns a ``(raw_action, action)`` tuple where ``action`` contains
    axis-angle rotations and processed gripper values.

    Supports two robot setups:

    * ``"widowx_bridge"`` – WidowX arm, binary gripper, default
      ``unnorm_key="oxe_bridge"``
    * ``"google_robot"`` – Google Robot, sticky gripper with repeat=10,
      default ``unnorm_key="oxe_rt1"``
    """

    # Default unnorm keys per policy setup
    _DEFAULT_UNNORM_KEYS = {
        "widowx_bridge": "oxe_bridge",
        "google_robot": "oxe_rt1",
    }

    # Default ensemble horizon per policy setup
    _DEFAULT_ENSEMBLE_HORIZONS = {
        "widowx_bridge": 7,
        "google_robot": 2,
    }

    # Sticky gripper repeat count per policy setup
    _STICKY_GRIPPER_REPEATS = {
        "widowx_bridge": 1,
        "google_robot": 10,
    }

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 0,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] | None = None,
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_ensemble: bool = True,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10093,
    ) -> None:
        if policy_setup not in self._DEFAULT_UNNORM_KEYS:
            raise NotImplementedError(
                f"Policy setup '{policy_setup}' not supported. "
                f"Choose from: {sorted(self._DEFAULT_UNNORM_KEYS.keys())}."
            )

        if image_size is None:
            image_size = [224, 224]
        if unnorm_key is None:
            unnorm_key = self._DEFAULT_UNNORM_KEYS[policy_setup]
        if action_ensemble_horizon is None:
            action_ensemble_horizon = self._DEFAULT_ENSEMBLE_HORIZONS[policy_setup]

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        super().__init__(
            policy_ckpt_path=policy_ckpt_path,
            unnorm_key=unnorm_key,
            image_size=image_size,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
            normalization_mode="q99",
            action_ensemble=action_ensemble,
            action_ensemble_horizon=action_ensemble_horizon,
            adaptive_ensemble_alpha=adaptive_ensemble_alpha,
            host=host,
            port=port,
        )

        self.policy_setup = policy_setup
        self.cfg_scale = cfg_scale
        self.action_scale = action_scale
        self.horizon = horizon
        self.sticky_gripper_num_repeat = self._STICKY_GRIPPER_REPEATS[policy_setup]

        # Image history (for potential multi-frame input)
        self.image_history: deque[np.ndarray] = deque(maxlen=self.horizon)
        self.num_image_history = 0

        # Sticky gripper state
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, task_description: str) -> None:
        super().reset(task_description)
        self.image_history.clear()
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(
        self,
        image: np.ndarray,
        task_description: Optional[str] = None,
        *args,
        **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Run one inference step.

        Args:
            image: Shape ``(H, W, 3)``, dtype ``uint8``.
            task_description: If different from previous, resets the client.

        Returns:
            ``(raw_action, action)`` where *raw_action* contains
            ``world_vector``, ``rotation_delta``, ``open_gripper`` and
            *action* contains ``world_vector``, ``rot_axangle``, ``gripper``,
            ``terminate_episode`` formatted for ManiSkill2.
        """
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        assert image.dtype == np.uint8
        resized = self.resize_image(image)
        self._add_image_to_history(resized)

        example = {
            "image": [resized],
            "lang": self.task_description,
        }

        # Get un-normalized action chunk (first action only, no chunking cache)
        normalized_actions = self.predict_normalized_actions(example)
        raw_actions = self.unnormalize_actions(
            normalized_actions,
            self.action_norm_stats,
            normalization_mode=self.normalization_mode,
            gripper_indices=[6],
        )

        if self.action_ensembler is not None:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = self._format_raw_action(raw_actions[0])

        # Convert to ManiSkill2 action format
        action = self._to_maniskill_action(raw_action)

        return raw_action, action

    # ------------------------------------------------------------------
    # Action formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format_raw_action(action: np.ndarray) -> dict[str, np.ndarray]:
        """Split flat 7-D action into named dict."""
        return {
            "world_vector": np.array(action[:3]),
            "rotation_delta": np.array(action[3:6]),
            "open_gripper": np.array(action[6:7]),  # [0, 1]; 1 = open, 0 = close
        }

    def _to_maniskill_action(
        self, raw_action: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert raw action to ManiSkill2 environment format."""
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale

        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        axes, angles = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = axes * angles * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = self._sticky_gripper_action(raw_action["open_gripper"])
        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])
        return action

    # ------------------------------------------------------------------
    # Sticky gripper logic (google_robot only)
    # ------------------------------------------------------------------
    def _sticky_gripper_action(
        self, current_gripper_action: np.ndarray
    ) -> np.ndarray:
        """Apply sticky gripper repeat logic for google_robot."""
        if self.previous_gripper_action is None:
            relative_gripper_action = np.array([0])
            self.previous_gripper_action = current_gripper_action
        else:
            relative_gripper_action = (
                self.previous_gripper_action - current_gripper_action
            )

        if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action
            self.previous_gripper_action = current_gripper_action

        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action

        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0

        return relative_gripper_action

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    # ------------------------------------------------------------------
    # Backward-compatible static helpers  (deprecated – prefer base class)
    # ------------------------------------------------------------------
    @staticmethod
    def get_action_stats(unnorm_key, policy_ckpt_path, **kwargs):
        return BaseModelClient.load_action_stats(unnorm_key, policy_ckpt_path)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [self.resize_image(img) for img in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate(
                    [a["world_vector"], a["rotation_delta"], a["open_gripper"]],
                    axis=-1,
                )
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
