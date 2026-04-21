from collections import deque
from typing import Optional
import cv2 as cv
import numpy as np
from typing import Dict
from pathlib import Path

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
from starVLA.model.tools import read_mode_config


class ModelClient:
    """
    StarVLA WebSocket policy client adapted for VLA-Arena environments.

    Connects to the starVLA deployment server and provides step-by-step
    inference for VLA-Arena simulation environments (robosuite-based,
    mounted Franka Panda, 7-DOF delta-EEF actions).
    """

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "127.0.0.1",
        port: int = 10093,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

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

        self.action_norm_stats = self.get_action_stats(
            self.unnorm_key, policy_ckpt_path=policy_ckpt_path
        )
        self.action_chunk_size = self.get_action_chunk_size(
            policy_ckpt_path=policy_ckpt_path
        )

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

    def step(
        self,
        example: dict,
        step: int = 0,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Perform one step of inference for VLA-Arena.

        :param example: dict with keys "image" (list of np.ndarray HxWxC uint8)
                        and "lang" (str instruction)
        :param step: current timestep (used for action chunking)
        :return: dict with "raw_action" containing world_vector, rotation_delta,
                 open_gripper
        """
        task_description = example.get("lang", None)
        images = example["image"]  # list of images

        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        example["image"] = images

        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(
                    f"Key 'normalized_actions' not found in response: {response['data'].keys()}"
                )

            normalized_actions = normalized_actions[0]
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
            )

        raw_actions = self.raw_actions[step % action_chunk_size][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # [0,1]; 1=open, 0=close
        }

        return {"raw_action": raw_action}

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
    ) -> np.ndarray:
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["min"], dtype=bool)
        )
        action_high = np.array(action_norm_stats["max"])
        action_low = np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        # Binarize gripper channel
        normalized_actions[:, 6] = np.where(
            normalized_actions[:, 6] < 0.5, 0, 1
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path) -> int:
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Model trained on multiple datasets; pass an unnorm_key from: "
                f"{list(norm_stats.keys())}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"unnorm_key '{unnorm_key}' not found; available: {list(norm_stats.keys())}"
        )
        return unnorm_key
