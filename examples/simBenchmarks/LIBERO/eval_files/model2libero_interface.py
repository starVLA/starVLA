# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LIBERO env-side adapter (thin client).

After the server-side refactor (see `deployment/model_server/policy_wrapper.py`),
the websocket *server* now returns already-unnormalized actions and ships
model-invariant fields (`action_chunk_size`, `available_unnorm_keys`) at
handshake. This client therefore no longer needs to:
  - load `dataset_statistics.json`
  - know `future_action_window_size`
  - perform un-normalization

It only handles env-specific adaptation: image history bookkeeping, action
ensembling, gripper sticky logic, and chunk-cache scheduling.
"""

from collections import deque
import logging
import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler


class ModelClient:
    def __init__(
        self,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        constrain_to_action_tokens: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10095,
    ) -> None:
        # Connect & receive handshake metadata (action_chunk_size, etc.)
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        forced_chunk_size = os.getenv("LIBERO_FORCE_ACTION_CHUNK_SIZE")
        if forced_chunk_size:
            self.action_chunk_size = max(1, int(forced_chunk_size))
        self.action_index_offset = max(0, int(os.getenv("LIBERO_ACTION_INDEX_OFFSET", "0") or "0"))
        if self.action_index_offset >= self.action_chunk_size:
            raise ValueError(
                f"LIBERO_ACTION_INDEX_OFFSET={self.action_index_offset} must be smaller than "
                f"action_chunk_size={self.action_chunk_size}"
            )
        self.effective_action_chunk_size = max(1, self.action_chunk_size - self.action_index_offset)
        self._server_metadata = meta
        self.vla_obs = list(meta.get("vla_obs") or [])

        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_chunk_size: {self.action_chunk_size}, "
            f"action_index_offset: {self.action_index_offset}, "
            f"server_meta: {meta} ***"
        )

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.constrain_to_action_tokens = constrain_to_action_tokens
        self.max_new_tokens = max_new_tokens
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        # Gripper sticky state (kept for parity with the previous client; not
        # currently consumed by LIBERO but other policy_setup paths use it).
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Cached unnormalized chunk; refreshed every `action_chunk_size` steps.
        self.raw_actions: Optional[np.ndarray] = None

    @property
    def server_metadata(self) -> dict:
        return self._server_metadata

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.raw_actions = None

    def step(self, example: dict, step: int = 0, **kwargs) -> dict:
        """One env step.

        Args:
            example: dict with keys ``image`` (list of np.uint8 HWC arrays) and ``lang`` (str).
            step: env step counter; used for chunk caching.

        Returns:
            ``{"raw_action": {"world_vector": ..., "rotation_delta": ..., "open_gripper": ...}}``
        """
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)

        # Refresh chunk if needed.
        if step % self.effective_action_chunk_size == 0 or self.raw_actions is None:
            vla_input = {
                "examples": [example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            if self.constrain_to_action_tokens is not None:
                vla_input["constrain_to_action_tokens"] = self.constrain_to_action_tokens
            if self.max_new_tokens is not None:
                vla_input["max_new_tokens"] = self.max_new_tokens
            response = self.client.predict_action(vla_input)
            try:
                data = response["data"]
                actions_batch = data["actions"]  # (B, T, D), unnormalized server-side
            except KeyError:
                raise KeyError(
                    f"Key 'actions' not found in response data: keys={list(response.get('data', {}).keys())}, "
                    f"full response={response}"
                )
            diagnostics = data.get("generation_diagnostics")
            action_tokens = data.get("action_tokens")
            normalized_actions = data.get("normalized_actions")
            if diagnostics is not None or action_tokens is not None or normalized_actions is not None:
                token_preview = None
                if action_tokens is not None:
                    first_tokens = action_tokens[0] if isinstance(action_tokens, (list, tuple)) else np.asarray(action_tokens, dtype=object)[0]
                    if first_tokens is not None:
                        token_preview = np.asarray(first_tokens).reshape(-1)[:16].tolist()
                norm_summary = None
                if normalized_actions is not None:
                    norm_arr = np.asarray(normalized_actions)[0]
                    norm_summary = {
                        "shape": tuple(norm_arr.shape),
                        "min": float(np.nanmin(norm_arr)),
                        "max": float(np.nanmax(norm_arr)),
                        "first": np.array2string(norm_arr[0], precision=6, suppress_small=False),
                    }
                action_arr = np.asarray(actions_batch)[0]
                action_summary = {
                    "shape": tuple(action_arr.shape),
                    "min": float(np.nanmin(action_arr)),
                    "max": float(np.nanmax(action_arr)),
                    "first": np.array2string(action_arr[0], precision=6, suppress_small=False),
                }
                logging.info(
                    "Policy generation diagnostics: diagnostics=%s, first_action_tokens=%s, normalized=%s, unnormalized=%s",
                    diagnostics,
                    token_preview,
                    norm_summary,
                    action_summary,
                )
            self.raw_actions = np.asarray(actions_batch)[0]  # (T, D)

        action_index = (step % self.effective_action_chunk_size) + self.action_index_offset
        raw_actions = self.raw_actions[action_index][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # 1 = open; 0 = close
        }
        return {"raw_action": raw_action}

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
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
