# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""Robocasa tabletop benchmark model client.

Inherits shared inference logic from :class:`BaseModelClient` and adds
Robocasa-specific features: batched multi-view inference, sin/cos state
encoding, and structured multi-body-part action outputs.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.base_model_client import BaseModelClient


class PolicyWarper(BaseModelClient):
    """Model client for the Robocasa tabletop benchmark.

    Key differences from other benchmark clients:

    * Supports **batched** inputs — multiple samples per ``step()`` call.
    * State is encoded via **sin/cos** transformation before being sent
      to the policy server.
    * Returns a structured action dict with separate keys for each
      body part: ``left_arm``, ``right_arm``, ``left_hand``,
      ``right_hand``, ``waist``.

    Expected *observations* dict layout::

        {
            "annotation.human.coarse_action": [task_str, ...],
            "video.ego_view": np.ndarray,     # (B, N_view, H, W, 3)
            "state.left_arm":  np.ndarray,    # (B, 1, 7)
            "state.right_arm": np.ndarray,    # (B, 1, 7)
            "state.left_hand": np.ndarray,    # (B, 1, 6)
            "state.right_hand": np.ndarray,   # (B, 1, 6)
            "state.waist":     np.ndarray,    # (B, 1, 3)
        }
    """

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] | None = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10095,
        n_action_steps: int = 2,
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
        self.n_action_steps = n_action_steps
        self.horizon = horizon

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(
        self,
        observations: dict,
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Run one batched inference step.

        Returns:
            ``{"actions": {"action.left_arm": (B, n, 7), ...}}``
        """
        task_description = observations["annotation.human.coarse_action"][0]
        images = observations["video.ego_view"]  # (B, N_view, H, W, 3)

        # Build per-body-part state dict and encode via sin/cos
        state = {
            "left_arm": observations["state.left_arm"],
            "right_arm": observations["state.right_arm"],
            "left_hand": observations["state.left_hand"],
            "right_hand": observations["state.right_hand"],
            "waist": observations["state.waist"],
        }
        state = self.normalize_state(state)
        input_state = np.concatenate(list(state.values()), axis=-1)

        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        # Resize all images: (B, N_view, H, W, 3)
        images = [
            [self.resize_image(img) for img in sample] for sample in images
        ]
        input_state = [s for s in input_state]  # list of (N_history, state_dim)

        # Build batched examples
        instructions = (
            [self.task_description]
            if isinstance(self.task_description, str)
            else self.task_description
        )
        batch_size = len(images)
        examples = []
        for b in range(batch_size):
            examples.append({
                "image": images[b],
                "lang": instructions[b] if isinstance(instructions, list) else instructions,
                "state": input_state[b],
            })

        vla_input = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        response = self.client.predict_action(vla_input)

        # Un-normalize actions in batch form: (B, chunk, D)
        normalized_actions = response["data"]["normalized_actions"]
        raw_actions = self.unnormalize_actions(
            normalized_actions,
            self.action_norm_stats,
            normalization_mode=self.normalization_mode,
            gripper_indices=None,  # Robocasa uses mask, no gripper binarization
        )

        # Apply ensemble per sample if enabled
        if self.action_ensembler is not None:
            ensembled = []
            for b in range(raw_actions.shape[0]):
                ensembled.append(
                    self.action_ensembler.ensemble_action(raw_actions[b])[None]
                )
            raw_actions = np.stack(ensembled, axis=0)  # (B, 1, D)

        # Slice to n_action_steps and split into body parts
        n = self.n_action_steps
        raw_action = {
            "action.left_arm": raw_actions[:, :n, :7],
            "action.right_arm": raw_actions[:, :n, 7:14],
            "action.left_hand": raw_actions[:, :n, 14:20],
            "action.right_hand": raw_actions[:, :n, 20:26],
            "action.waist": raw_actions[:, :n, 26:29],
        }

        return {"actions": raw_action}

    # ------------------------------------------------------------------
    # State normalization (sin/cos encoding)
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Encode each state component via sin/cos concatenation."""
        encoded = {}
        for key, val in state.items():
            encoded[key] = np.concatenate(
                [np.sin(val), np.cos(val)], axis=-1
            )
        return encoded

    # ------------------------------------------------------------------
    # Backward-compatible static helpers  (deprecated – prefer base class)
    # ------------------------------------------------------------------
    @staticmethod
    def get_action_stats(unnorm_key, policy_ckpt_path, **kwargs):
        return BaseModelClient.load_action_stats(unnorm_key, policy_ckpt_path)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        return BaseModelClient._resolve_unnorm_key(norm_stats, unnorm_key)

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
