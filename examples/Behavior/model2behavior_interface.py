# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""BEHAVIOR benchmark model client (R1Pro robot).

Inherits shared inference logic from :class:`BaseModelClient` and adds
BEHAVIOR-specific features: multi-camera observation extraction, 23-DoF
R1Pro action decomposition, and chunked adaptive action ensembling.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import BEHAVIOR-specific utilities
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES, ROBOT_CAMERA_NAMES

from deployment.model_server.base_model_client import BaseModelClient
from examples.Behavior.adaptive_ensemble import ChunkedAdaptiveEnsembler


class M1Inference(BaseModelClient):
    """Model client for the BEHAVIOR benchmark with R1Pro robot.

    23-DoF action space decomposition::

        base_pose        (3)  — indices  0:3
        torso_pose       (4)  — indices  3:7
        left_arm_pose    (7)  — indices  7:14
        left_gripper     (1)  — index   14
        right_arm_pose   (7)  — indices 15:22
        right_gripper    (1)  — index   22
    """

    def __init__(
        self,
        policy_ckpt_path: str,
        policy_setup: str = "R1Pro",
        horizon: int = 0,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] | None = None,
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_ensemble: bool = False,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10093,
        task_description: Optional[str] = None,
        use_state: bool = False,
    ) -> None:
        if policy_setup != "R1Pro":
            raise NotImplementedError(
                f"Policy setup '{policy_setup}' not supported for BEHAVIOR models."
            )

        if image_size is None:
            image_size = [224, 224]

        # R1Pro defaults
        unnorm_key = "R1Pro"
        if action_ensemble_horizon is None:
            action_ensemble_horizon = 5
        # Force ensemble on for BEHAVIOR
        action_ensemble = True

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        super().__init__(
            policy_ckpt_path=policy_ckpt_path,
            unnorm_key=unnorm_key,
            image_size=image_size,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
            normalization_mode="min_max",
            action_ensemble=False,  # We handle ensemble ourselves via ChunkedAdaptiveEnsembler
            host=host,
            port=port,
        )

        self.policy_setup = policy_setup
        self.policy_ckpt_path = policy_ckpt_path
        self.cfg_scale = cfg_scale
        self.action_scale = action_scale
        self.horizon = horizon
        self.use_state = use_state
        self.sticky_gripper_num_repeat = 3

        # BEHAVIOR uses 2-step action chunks with ChunkedAdaptiveEnsembler
        self.action_chunk_size = 2
        self.current_step = 0

        # Gripper state
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        # Override task description if provided at init
        if task_description is not None:
            self.task_description = task_description

        # Chunked ensemble (BEHAVIOR-specific, handles multi-step chunks)
        self.action_ensemble = action_ensemble
        if self.action_ensemble:
            self.chunked_ensembler = ChunkedAdaptiveEnsembler(
                action_ensemble_horizon, self.action_chunk_size, adaptive_ensemble_alpha
            )
        else:
            self.chunked_ensembler = None

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, task_description: Optional[str] = None) -> None:
        if task_description is not None:
            self.task_description = task_description
        if self.chunked_ensembler is not None:
            self.chunked_ensembler.reset()
        self._cached_actions = None

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.current_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for BEHAVIOR environment.

        Args:
            obs: Observation dict from BEHAVIOR/OmniGibson environment.

        Returns:
            torch.Tensor: Action tensor of shape ``(23,)`` for R1Pro.
        """
        processed_obs = self._process_behavior_obs(obs)

        # Build image inputs
        primary_image = processed_obs["full_image"]
        left_wrist_image = processed_obs["left_wrist_image"]
        right_wrist_image = processed_obs["right_wrist_image"]

        if "dual" in self.policy_ckpt_path.lower():
            image_input = [primary_image]
            wrist_image_input = [left_wrist_image, right_wrist_image]
        else:
            image_input = [primary_image, left_wrist_image, right_wrist_image]
            wrist_image_input = None

        if self.task_description is None:
            print("Warning: Could not get task description")
            self.task_description = "Turn on the radio receiver that's on the table in the living room."

        # Build example dict
        example = {
            "image": image_input,
            "wrist_views": wrist_image_input,
            "lang": self.task_description,
        }
        if self.use_state:
            raw_state = processed_obs["state"]  # (state_dim,)
            example["state"] = raw_state[None, :]  # → (1, state_dim)

        vla_input = {"examples": [example]}

        # Action chunking with ensemble
        if self.current_step % self.action_chunk_size == 0:
            if self.current_step % 100 == 0:
                print("Step:", self.current_step)

            response = self.client.predict_action(vla_input)

            if response.get("ok", True) is False or response.get("status") == "error":
                error_info = response.get("error", {})
                raise RuntimeError(f"Websocket server error: {error_info}")

            try:
                normalized_actions = response["data"]["normalized_actions"]  # (B, chunk, D)
            except KeyError:
                raise KeyError(
                    f"Key 'normalized_actions' not found. "
                    f"Available: {list(response.get('data', {}).keys())}"
                )

            normalized_actions = normalized_actions[0]  # drop batch → (T, D)
            self.raw_actions = self.unnormalize_actions(
                normalized_actions,
                self.action_norm_stats,
                normalization_mode=self.normalization_mode,
                gripper_indices=None,  # No gripper binarization for BEHAVIOR
            )

            # Feed chunk to chunked ensembler
            if self.action_ensemble and self.action_chunk_size > 1:
                self.chunked_ensembler.ensemble_action(self.raw_actions)

        # Get single action from ensemble or cache
        if self.action_ensemble:
            raw_actions = self.chunked_ensembler.step()[None]
        else:
            raw_actions = self.raw_actions[self.current_step % self.action_chunk_size][None]

        self.current_step += 1

        # Decompose into R1Pro body parts
        raw_action = {
            "base_pose": np.array(raw_actions[0, :3]),
            "torso_pose": np.array(raw_actions[0, 3:7]),
            "left_arm_pose": np.array(raw_actions[0, 7:14]),
            "left_gripper_pose": np.array(raw_actions[0, 14:15]),
            "right_arm_pose": np.array(raw_actions[0, 15:22]),
            "right_gripper_pose": np.array(raw_actions[0, 22:23]),
        }

        action = self._process_action_for_behavior(raw_action)
        return torch.from_numpy(action).float()

    # ------------------------------------------------------------------
    # BEHAVIOR-specific observation processing
    # ------------------------------------------------------------------
    def _process_behavior_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and resize images + proprioception from BEHAVIOR obs."""
        try:
            head_key = ROBOT_CAMERA_NAMES[self.policy_setup]["head"] + "::rgb"
            left_key = ROBOT_CAMERA_NAMES[self.policy_setup]["left_wrist"] + "::rgb"
            right_key = ROBOT_CAMERA_NAMES[self.policy_setup]["right_wrist"] + "::rgb"

            full_image = obs[head_key][:, :, :3]
            left_wrist_image = obs[left_key][:, :, :3]
            right_wrist_image = obs[right_key][:, :, :3]
            prop_state = self._generate_prop_state(obs["robot_r1::proprio"])
        except KeyError as e:
            print(f"Error extracting observations: {e}")
            print(f"Available keys in obs: {list(obs.keys())}")
            raise

        return {
            "full_image": self._resize_behavior_image(full_image),
            "left_wrist_image": self._resize_behavior_image(left_wrist_image),
            "right_wrist_image": self._resize_behavior_image(right_wrist_image),
            "state": prop_state,
        }

    def _generate_prop_state(self, proprio: np.ndarray) -> np.ndarray:
        """Generate proprioceptive state for R1Pro robot."""
        idx = PROPRIOCEPTION_INDICES[self.policy_setup]
        qpos_list = [
            proprio[idx["joint_qpos_sin"]][6:],  # Skip first 6 base joints (standard track)
            proprio[idx["joint_qpos_cos"]][6:],
        ]
        assert qpos_list[0].shape == (22,)
        assert qpos_list[1].shape == (22,)
        return np.concatenate(qpos_list, axis=0)

    def _resize_behavior_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image, handling both numpy arrays and torch tensors."""
        if hasattr(image, "numpy"):
            image = image.numpy()
        return self.resize_image(image)

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------
    @staticmethod
    def _process_action_for_behavior(raw_action: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine body-part actions into the 23-D vector expected by BEHAVIOR."""
        return np.concatenate([
            raw_action["base_pose"],           # 0:3   (3 dims)
            raw_action["torso_pose"],          # 3:7   (4 dims)
            raw_action["left_arm_pose"],       # 7:14  (7 dims)
            raw_action["left_gripper_pose"],   # 14:15 (1 dim)
            raw_action["right_arm_pose"],      # 15:22 (7 dims)
            raw_action["right_gripper_pose"],  # 22:23 (1 dim)
        ])

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
        images = [self._resize_behavior_image(img) for img in images]
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
