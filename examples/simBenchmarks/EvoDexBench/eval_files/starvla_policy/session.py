"""StarVLA physical-action session for the public Evo policy API."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
from dex_benchmark.policy_api.model_input import ModelInput
from dex_benchmark.policy_api.observation import (
    as_finite_numpy,
    compact_runtime_state,
    required_value,
)
from dex_benchmark.policy_api.policy_features import decode_semantic_action
from dex_benchmark.policy_api.session import SemanticAction

from .client import StarVLAWebsocketClient
from .config import CONTROL_HZ, EMBODIMENT_IDS, StarVLAPolicyConfig


class StarVLAClient(Protocol):
    def get_server_metadata(self) -> Mapping[str, Any]: ...

    def predict_action(self, query: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class StarVLAPolicySession:
    """Keep each environment's chunk state separate at exactly 30 control Hz."""

    def __init__(self, config: StarVLAPolicyConfig, *, client: StarVLAClient | None = None) -> None:
        self.config = config
        self.client = client or StarVLAWebsocketClient(config.host, config.port)
        self._actions: deque[np.ndarray] = deque()
        self._action_spec: Mapping[str, Any] | None = None
        try:
            self._check_server_metadata(self.client.get_server_metadata())
        except Exception:
            self.client.close()
            raise

    def _check_server_metadata(self, metadata: Mapping[str, Any]) -> None:
        expected_mix = f"evodex_{self.config.embodiment}_h{self.config.chunk_size}_q99"
        if metadata.get("env") != "starvla_policy_server":
            raise ValueError("The endpoint is not a StarVLA policy server.")
        if metadata.get("training_data_mix") != expected_mix:
            raise ValueError(f"Server data mix must be {expected_mix!r}.")
        if metadata.get("action_chunk_size") != self.config.chunk_size:
            raise ValueError(f"Server action chunk must contain {self.config.chunk_size} frames.")
        if list(metadata.get("training_obs_image_size") or ()) != [224, 224]:
            raise ValueError("Server training images must be 224 x 224.")
        if (
            self.config.expected_checkpoint_path is not None
            and metadata.get("ckpt_path") != self.config.expected_checkpoint_path
        ):
            raise ValueError("Server checkpoint path differs from the requested model.")
        available = metadata.get("available_unnorm_keys")
        if not isinstance(available, (tuple, list)) or not available:
            raise ValueError("Server did not advertise unnormalization keys.")
        if self.config.unnorm_key is None:
            if len(available) != 1:
                raise ValueError("Specify unnorm_key for a multi-dataset checkpoint.")
            self.unnorm_key = str(available[0])
        elif self.config.unnorm_key not in available:
            raise ValueError("Requested unnormalization key is unavailable.")
        else:
            self.unnorm_key = self.config.unnorm_key
        expected_state = [
            feature for role in self.config.roles for feature in (f"state.{role}_ee_pose", f"state.{role}_hand_qpos")
        ]
        expected_action = [
            feature for role in self.config.roles for feature in (f"action.{role}_ee_delta", f"action.{role}_hand_qpos")
        ]
        if metadata.get("state_keys") != expected_state:
            raise ValueError("Server state key order differs from Evo-DexBench.")
        if metadata.get("action_keys") != expected_action:
            raise ValueError("Server action key order differs from Evo-DexBench.")

    def bind_environment(self, metadata: Mapping[str, Any]) -> None:
        if tuple(metadata.get("manipulator_roles") or ()) != self.config.roles:
            raise ValueError("Environment manipulator order differs from the checkpoint.")
        if not set(self.config.camera_roles).issubset(metadata.get("camera_roles") or ()):
            raise ValueError("Environment lacks a required policy camera.")
        if metadata.get("control_frequency_hz") != CONTROL_HZ:
            raise ValueError(f"Environment control frequency must be {CONTROL_HZ} Hz.")
        spec = metadata.get("benchmark_action_spec")
        if not isinstance(spec, Mapping):
            raise ValueError("Environment lacks a benchmark action specification.")
        if (
            spec.get("action_format") != "benchmark_physical_semantic"
            or spec.get("action_schema_version") != 1
            or spec.get("embodiment_id") != EMBODIMENT_IDS[self.config.embodiment]
        ):
            raise ValueError("Environment action schema or embodiment differs.")
        manipulators = spec.get("manipulators")
        if not isinstance(manipulators, Mapping) or tuple(manipulators) != self.config.roles:
            raise ValueError("Environment physical action roles differ.")
        for role in self.config.roles:
            role_spec = manipulators[role]
            if tuple(role_spec["end_effector_delta_pose"]["shape"]) != (6,) or tuple(
                role_spec["hand_joint_position"]["shape"]
            ) != (24,):
                raise ValueError(f"Environment physical action dimensions differ for {role}.")
        self._action_spec = spec

    def _example(self, model_input: ModelInput) -> dict[str, Any]:
        observation = model_input["observation"]
        cameras = required_value(observation, "cameras")
        manipulators = required_value(observation, "manipulators")
        images: list[np.ndarray] = []
        for camera in self.config.camera_roles:
            image = as_finite_numpy(required_value(cameras, camera, "rgb"), name=f"{camera}.rgb")
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"{camera} must be uint8 RGB [224,224,3], got {image.shape} {image.dtype}.")
            images.append(np.ascontiguousarray(image))
        state = compact_runtime_state(manipulators, self.config.roles)
        if state.shape != (self.config.state_dim,):
            raise ValueError(f"Runtime state has shape {state.shape}.")
        instruction = model_input["instruction"].strip()
        if not instruction:
            raise ValueError("Task instruction must be non-empty.")
        # [D] -> [1, D]; image order is the explicit framework camera contract.
        return {"image": images, "state": state[None, :], "lang": instruction}

    def _check_physical_action(self, action: np.ndarray) -> None:
        if self._action_spec is None:
            raise RuntimeError("Bind environment metadata before inference.")
        if action.shape != (self.config.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Predicted physical action has wrong shape or non-finite values.")
        for index, role in enumerate(self.config.roles):
            block = action[index * 30 : (index + 1) * 30]
            role_spec = self._action_spec["manipulators"][role]
            arm = role_spec["end_effector_delta_pose"]
            hand = role_spec["hand_joint_position"]
            translation_bound = np.maximum(
                np.abs(np.asarray(arm["translation_lower"])),
                np.abs(np.asarray(arm["translation_upper"])),
            )
            rotation = arm["rotation_constraint"]
            rotation_bound = np.asarray(rotation.get("scale_rad", rotation.get("upper", [1, 1, 1])))
            if np.any(np.abs(block[:3]) > np.maximum(1.0, 10 * translation_bound)):
                raise ValueError(f"{role} EE translation exceeds the action guard.")
            if np.any(np.abs(block[3:6]) > np.maximum(1.0, 10 * rotation_bound)):
                raise ValueError(f"{role} EE rotation exceeds the action guard.")
            if np.any(block[6:] < np.asarray(hand["lower"]) - 1.0) or np.any(
                block[6:] > np.asarray(hand["upper"]) + 1.0
            ):
                raise ValueError(f"{role} hand target exceeds the action guard.")

    def reset(self) -> None:
        self._actions.clear()

    def act(self, model_input: ModelInput) -> SemanticAction:
        if not self._actions:
            response = self.client.predict_action(
                {
                    "examples": [self._example(model_input)],
                    "unnorm_key": self.unnorm_key,
                    "normalize_state": True,
                    "do_sample": False,
                }
            )
            data = response.get("data")
            if not isinstance(data, Mapping) or "actions" not in data:
                raise ValueError(f"StarVLA response lacks data.actions: {response.get('error', response.get('status'))}")
            # [1, T, 30*roles] -> T independent physical actions at 30 Hz.
            chunk = np.asarray(data["actions"], dtype=np.float32)
            if chunk.shape != (1, self.config.chunk_size, self.config.action_dim) or not np.isfinite(chunk).all():
                raise ValueError(f"Invalid physical action chunk {chunk.shape}.")
            for action in chunk[0, : self.config.steps_per_request]:
                self._check_physical_action(action)
                self._actions.append(action.copy())
        physical = decode_semantic_action(self._actions.popleft(), self.config.embodiment)
        if self.config.max_hand_target_step_rad is not None:
            manipulators = required_value(model_input["observation"], "manipulators")
            for role in self.config.roles:
                qpos = as_finite_numpy(
                    required_value(manipulators, role, "joint_position"),
                    name=f"{role}.joint_position",
                ).reshape(-1)
                if qpos.shape != (31,):
                    raise ValueError(f"{role} joint_position must have 31 values.")
                target = physical[role]["hand_joint_position"]
                physical[role]["hand_joint_position"] = np.clip(
                    target,
                    qpos[7:] - self.config.max_hand_target_step_rad,
                    qpos[7:] + self.config.max_hand_target_step_rad,
                )
        return physical

    def capture_visual_observation_after_action(self) -> bool:
        return not self._actions

    def capture_lowdim_observation_after_action(self) -> bool:
        return self.config.max_hand_target_step_rad is not None or not self._actions

    def requires_control_period_tactile(self) -> bool:
        return False

    def close(self) -> None:
        self.client.close()
