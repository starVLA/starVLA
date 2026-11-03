# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""StarVLA WebSocket policy client for the LADDER long-horizon benchmark."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import cv2 as cv
import numpy as np
from transforms3d.euler import euler2axangle

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.simBenchmarks.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler


class ModelClient:
    """Batched WebSocket policy client for LADDER (SimplerEnv-Long, RobotArena-∞-Long, RoboLab-Long)."""

    def __init__(
        self,
        policy_setup: str = "widowx_bridge",
        unnorm_key: Optional[str] = None,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = None,
        adaptive_ensemble_alpha: float = 0.1,
        open_loop_horizon: int = 4,
        image_size: Sequence[int] = (224, 224),
        action_scale: float = 1.0,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        host: str = "127.0.0.1",
        port: int = 10093,
        n_envs: int = 1,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.host = str(host)
        self.port = int(port)
        self.client = WebsocketClientPolicy(self.host, self.port)
        self.server_metadata = dict(self.client.get_server_metadata())
        self.action_chunk_size = int(self.server_metadata.get("action_chunk_size", 1))

        if policy_setup == "widowx_bridge":
            if unnorm_key is None:
                avail = self.server_metadata.get("available_unnorm_keys") or []
                default_key = self.server_metadata.get("default_unnorm_key")
                if "oxe_bridge" in avail:
                    unnorm_key = "oxe_bridge"
                elif default_key:
                    unnorm_key = default_key
                elif avail:
                    unnorm_key = avail[0]
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 7
        elif policy_setup in ("franka_droid", "droid"):
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 1
        else:
            raise ValueError(f"Unsupported policy_setup={policy_setup!r}. Expected 'widowx_bridge' or 'franka_droid'.")

        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.action_ensemble = action_ensemble
        self.action_ensemble_horizon = int(action_ensemble_horizon)
        self.adaptive_ensemble_alpha = float(adaptive_ensemble_alpha)
        self.open_loop_horizon = max(1, min(int(open_loop_horizon), self.action_chunk_size))
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.action_scale = float(action_scale)
        self.use_ddim = bool(use_ddim)
        self.num_ddim_steps = int(num_ddim_steps)
        self.n_envs = int(n_envs)

        self._ensemblers: list[AdaptiveEnsembler] = []
        self._task_descriptions: list[Optional[str]] = []
        self._cached_chunks: Optional[np.ndarray] = None
        self._chunk_step: int = 0
        self.reset(n_envs=self.n_envs)

    def reset(self, n_envs: Optional[int] = None, env_ids: Optional[Sequence[int]] = None) -> None:
        if n_envs is not None and n_envs != self.n_envs:
            self.n_envs = int(n_envs)
            env_ids = None
        if env_ids is None or len(self._ensemblers) != self.n_envs:
            self._ensemblers = [
                AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha) for _ in range(self.n_envs)
            ]
            self._task_descriptions = [None] * self.n_envs
            self._cached_chunks = None
            self._chunk_step = 0
        else:
            for idx in env_ids:
                self._ensemblers[int(idx)].reset()
                self._task_descriptions[int(idx)] = None
            self._cached_chunks = None
            self._chunk_step = 0

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        assert image.dtype == np.uint8, f"Expected uint8 image, got {image.dtype}"
        if image.shape[0] == self.image_size[1] and image.shape[1] == self.image_size[0]:
            return image
        return cv.resize(image, self.image_size, interpolation=cv.INTER_AREA)

    def step_batch(
        self,
        images_per_env: Sequence[Sequence[np.ndarray]],
        instructions: Sequence[str],
        states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict one simulator step action of shape ``(B, D)`` across ``B`` parallel environments."""
        batch_size = len(images_per_env)
        if batch_size != self.n_envs:
            self.reset(n_envs=batch_size)

        examples = []
        instr_changed = False
        for idx in range(batch_size):
            instr = str(instructions[idx])
            if self._task_descriptions[idx] is not None and instr != self._task_descriptions[idx]:
                self._ensemblers[idx].reset()
                instr_changed = True
            self._task_descriptions[idx] = instr

            resized_views = [self._resize_image(img) for img in images_per_env[idx]]
            ex = {"image": resized_views, "lang": instr}
            if states is not None:
                ex["state"] = np.asarray(states[idx], dtype=np.float32)[None, :]
            examples.append(ex)

        need_query = (
            self.action_ensemble
            or self._cached_chunks is None
            or instr_changed
            or (self._chunk_step % self.open_loop_horizon == 0)
        )
        if need_query:
            vla_input = {
                "examples": examples,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            if self.unnorm_key is not None:
                vla_input["unnorm_key"] = self.unnorm_key

            try:
                response = self.client.predict_action(vla_input)
            except Exception:
                self.client = WebsocketClientPolicy(self.host, self.port)
                response = self.client.predict_action(vla_input)
            self._cached_chunks = np.asarray(response["data"]["actions"], dtype=np.float32)  # (B, T, D)
            self._chunk_step = 0

        raw_chunks = self._cached_chunks
        assert raw_chunks is not None
        step_offset = 0 if self.action_ensemble else (self._chunk_step % self.open_loop_horizon)
        self._chunk_step += 1

        env_actions = []
        for idx in range(batch_size):
            chunk = raw_chunks[idx]  # (T, D)
            if self.action_ensemble and self.action_ensemble_horizon > 1:
                step_raw = np.asarray(self._ensemblers[idx].ensemble_action(chunk), dtype=np.float32).reshape(-1)
            else:
                step_raw = np.asarray(chunk[step_offset], dtype=np.float32).reshape(-1)

            if self.policy_setup == "widowx_bridge":
                world_vector = step_raw[:3] * self.action_scale
                roll, pitch, yaw = np.asarray(step_raw[3:6], dtype=np.float64)
                axes, angle = euler2axangle(roll, pitch, yaw)
                rot_axangle = np.asarray(axes * angle, dtype=np.float32) * self.action_scale
                gripper = np.asarray([2.0 * (step_raw[6] > 0.5) - 1.0], dtype=np.float32)
                env_actions.append(np.concatenate([world_vector, rot_axangle, gripper], axis=0))
            else:
                env_actions.append(step_raw)

        return np.stack(env_actions, axis=0).astype(np.float32)
