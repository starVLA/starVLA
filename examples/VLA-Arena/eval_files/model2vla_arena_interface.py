from collections import deque
from typing import Optional
import cv2 as cv
import numpy as np
from typing import Dict

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler


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

        server_meta = self.client.get_server_metadata()
        self.action_chunk_size = server_meta["action_chunk_size"]
        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, server_meta: {server_meta} ***")

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
        vla_input["unnorm_key"] = self.unnorm_key

        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0:
            # === TRAIN/TEST CONSISTENCY: keep the observation below aligned with training ===
            # Embodied policies degrade SILENTLY (no error) when the eval-time observation
            # differs from what the model saw during TRAINING. Verify these match the
            # training config used for this checkpoint:
            #   - state       : whether proprioceptive state is included (and its dim/order/normalization))
            #   - image size  : resize / crop resolution (e.g. 224x224)
            #   - image count : how many camera views are fed
            #   - image order : the ordering of those camera views
            #   - action normalization: unnorm_key must match the training dataset stats
            # ==============================================================================
            response = self.client.predict_action(vla_input)
            # server already un-normalized via training-time transform
            self.raw_actions = np.array(response["data"]["actions"][0])  # (chunk, D)

        raw_actions = self.raw_actions[step % action_chunk_size][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # [0,1]; 1=open, 0=close
        }

        return {"raw_action": raw_action}

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
