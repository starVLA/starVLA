# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""LIBERO benchmark model client.

Inherits shared inference logic from :class:`BaseModelClient` and adds
LIBERO-specific action formatting (7-DoF single-arm with binary gripper).
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.base_model_client import BaseModelClient


class ModelClient(BaseModelClient):
    """Model client for the LIBERO benchmark family.

    Expected *example* dict layout::

        {
            "image": [primary_img, wrist_img],  # list[np.ndarray], uint8
            "lang":  "task description",
        }
    """

    # Gripper is the last (7th) column for LIBERO's single-arm Franka
    _GRIPPER_INDICES = [6]

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        image_size: list[int] | None = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10095,
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
            normalization_mode="min_max",
            action_ensemble=action_ensemble,
            action_ensemble_horizon=action_ensemble_horizon,
            adaptive_ensemble_alpha=adaptive_ensemble_alpha,
            host=host,
            port=port,
        )
        self.policy_setup = policy_setup
        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(
        self,
        example: dict,
        step: int = 0,
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Run one inference step and return the raw 7-DoF action.

        Returns:
            ``{"raw_action": {"world_vector": (3,), "rotation_delta": (3,),
            "open_gripper": (1,)}}``
        """
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)

        # Resize images in-place
        example["image"] = [self.resize_image(img) for img in example["image"]]

        # Fetch (or reuse cached) un-normalized action
        raw_action_vec = self.get_actions_for_step(example, step)

        return {"raw_action": self._format_raw_action(raw_action_vec)}

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def get_actions_for_step(self, example: dict, step: int) -> np.ndarray:
        """Override to use LIBERO-specific gripper binarization."""
        if step % self.action_chunk_size == 0 or self._cached_actions is None:
            normalized = self.predict_normalized_actions(example)
            self._cached_actions = self.unnormalize_actions(
                normalized,
                self.action_norm_stats,
                normalization_mode=self.normalization_mode,
                gripper_indices=self._GRIPPER_INDICES,
            )
        return self._cached_actions[step % self.action_chunk_size]

    @staticmethod
    def _format_raw_action(action: np.ndarray) -> dict[str, np.ndarray]:
        """Split a flat 7-D action into the named dict expected by LIBERO."""
        return {
            "world_vector": action[:3],
            "rotation_delta": action[3:6],
            "open_gripper": action[6:7],  # range [0, 1]; 1 = open, 0 = close
        }

    # ------------------------------------------------------------------
    # Backward-compatible static helpers  (deprecated – prefer base class)
    # ------------------------------------------------------------------
    @staticmethod
    def get_action_stats(unnorm_key, policy_ckpt_path, **kwargs):
        return BaseModelClient.load_action_stats(unnorm_key, policy_ckpt_path)

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        return BaseModelClient.load_action_chunk_size(policy_ckpt_path)

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
