"""Policy-side bridge for RoboCasa365 (PandaOmron, single-arm) evaluation.

Modeled after ``examples/Robocasa_tabletop/eval_files/model2robocasa_interface.py``
but adapted to the single-arm 12-d action / 16-d state layout produced by
``robocasa.wrappers.gym_wrapper.PandaOmronKeyConverter``.
"""

from collections import deque
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.Robocasa_tabletop.eval_files.adaptive_ensemble import AdaptiveEnsembler


# Order MUST match the LeRobot dataset ``observation.state`` produced by
# ``robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py``:
#   base_position(3) + base_rotation(4) + eef_pos_rel(3) + eef_rot_rel(4) + gripper_qpos(2) = 16
STATE_KEY_ORDER = [
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
]

# Action splits in the trained 12-d output (see PandaOmronRoboCasa365DataConfig)
ACTION_SLICES = {
    "action.end_effector_position": (0, 3),
    "action.end_effector_rotation": (3, 6),
    "action.gripper_close": (6, 7),
    "action.base_motion": (7, 11),
    "action.control_mode": (11, 12),
}


class PolicyWarper:
    """Single-arm PandaOmron policy wrapper that talks to the websocket server."""

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 10095,
        image_size=(224, 224),
        n_action_steps: int = 8,
        action_ensemble: bool = False,
        action_ensemble_horizon: int = 3,
        adaptive_ensemble_alpha: float = 0.1,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.unnorm_key = unnorm_key
        self.image_size = tuple(image_size)
        self.n_action_steps = n_action_steps
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps

        self.task_description = None
        self.action_ensemble = action_ensemble
        self.action_ensembler = (
            AdaptiveEnsembler(action_ensemble_horizon, adaptive_ensemble_alpha)
            if action_ensemble
            else None
        )

        server_meta = self.client.get_server_metadata()
        print(f"*** unnorm_key: {unnorm_key}, server_meta: {server_meta} ***")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def reset(self, task_description) -> None:
        self.task_description = task_description
        if self.action_ensemble:
            self.action_ensembler.reset()

    def step(self, observations: Dict, **_) -> Dict:
        # 1) instruction
        task_descs = observations["annotation.human.task_description"]  # tuple of B strs
        if isinstance(task_descs, (tuple, list)):
            instructions = list(task_descs)
        else:
            instructions = [task_descs]
        if instructions[0] != self.task_description:
            self.reset(instructions[0])

        # 2) image — the tabletop multi-view env returns (B, n_obs, H, W, 3); we use the
        # left agentview (the same one used during training).
        view = observations["video.robot0_agentview_left"]  # (B, 1, H, W, 3)
        images = [[self._resize_image(img) for img in sample] for sample in view]

        # 3) state — concatenate parts in the same order as in training
        state_parts = [observations[k] for k in STATE_KEY_ORDER]  # each (B, 1, d)
        input_state = np.concatenate(state_parts, axis=-1)  # (B, 1, 16)
        input_state = self._sin_cos_state(input_state)

        examples = []
        for b in range(len(images)):
            examples.append(
                {
                    "image": images[b],
                    "lang": instructions[b] if b < len(instructions) else instructions[0],
                    "state": input_state[b],
                }
            )

        vla_input = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        vla_input["unnorm_key"] = self.unnorm_key
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
        raw_actions = np.array(response["data"]["actions"])  # (B, chunk, D)

        if self.action_ensemble:
            ensembled = []
            for b in range(raw_actions.shape[0]):
                ensembled.append(self.action_ensembler.ensemble_action(raw_actions[b])[None])
            raw_actions = np.stack(ensembled, axis=0)

        # Slice into the dict structure consumed by RoboCasaGymEnv.step().
        out = {}
        for key, (s, e) in ACTION_SLICES.items():
            out[key] = raw_actions[:, : self.n_action_steps, s:e]
        return {"actions": out}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def _sin_cos_state(state: np.ndarray) -> np.ndarray:
        """Match training-time StateActionSinCosTransform on the state."""
        return np.concatenate([np.sin(state), np.cos(state)], axis=-1)


