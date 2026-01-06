from collections import deque
from typing import Optional, Sequence, Dict, Any
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

from transforms3d.euler import euler2axangle
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from examples.Behavior.adaptive_ensemble import AdaptiveEnsembler, ChunkedAdaptiveEnsembler
from pathlib import Path

from starVLA.model.tools import read_mode_config 

# Import BEHAVIOR-specific utilities
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES, PROPRIOCEPTION_INDICES

def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

class M1Inference:
    def __init__(
        self,
        policy_ckpt_path,
        policy_setup: str = "R1Pro",
        horizon: int = 0,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_ensemble: bool = False,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10093,
        task_description: str = None,
        use_state: bool = False,
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)

        action_ensemble = True

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Robot type for BEHAVIOR
        if policy_setup == "R1Pro":
            unnorm_key = "R1Pro" 
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 5  
            self.sticky_gripper_num_repeat = 3
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for BEHAVIOR models."
            )
        
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.policy_ckpt_path = policy_ckpt_path

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.cfg_scale = cfg_scale
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        
        # Gripper control state
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        # initial
        self.action_chunk_size = 2
        self.current_step = 0

        # Task and image history
        self.task_description = task_description
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            if self.action_chunk_size > 1:
                self.action_ensembler = ChunkedAdaptiveEnsembler(self.action_ensemble_horizon, self.action_chunk_size, self.adaptive_ensemble_alpha)
            else:
                self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Load action normalization stats
        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        
        self.use_state = use_state

        # If want to debug, uncomment the following line
        # if os.getenv("DEBUG", False):
        #     start_debugpy_once()

    def reset(self, task_description: Optional[str] = None) -> None:
        if task_description is not None:
            self.task_description = task_description
        if self.action_ensemble:
            self.action_ensembler.reset()

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.current_step = 0

    def _generate_prop_state(self, proprio: np.ndarray) -> np.ndarray:
        """Generate proprioceptive state for R1Pro robot.""" 
        idx = PROPRIOCEPTION_INDICES[self.policy_setup]
        qpos_list = [
            proprio[idx["joint_qpos_sin"]][6:],  # First 6 are base joints, which is NOT allowed in standard track
            proprio[idx["joint_qpos_cos"]][6:],  # First 6 are base joints, which is NOT allowed in standard track
        ] 
        assert qpos_list[0].shape == (22,)
        assert qpos_list[1].shape == (22,)
        return np.concatenate(qpos_list, axis=0)

    def _process_behavior_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process BEHAVIOR observations to extract images and proprioception."""
        # Extract images from different camera views
        try:
            head_camera_key = ROBOT_CAMERA_NAMES[self.policy_setup]["head"] + "::rgb"
            left_wrist_camera_key = ROBOT_CAMERA_NAMES[self.policy_setup]["left_wrist"] + "::rgb"
            right_wrist_camera_key = ROBOT_CAMERA_NAMES[self.policy_setup]["right_wrist"] + "::rgb"
            
            full_image = obs[head_camera_key][:, :, :3]  # [224, 224, 3]
            left_wrist_image = obs[left_wrist_camera_key][:, :, :3]  # [224, 224, 3]
            right_wrist_image = obs[right_wrist_camera_key][:, :, :3]  # [224, 224, 3]
            prop_state = self._generate_prop_state(obs["robot_r1::proprio"])  
                        
        except KeyError as e:
            print(f"Error extracting observations: {e}")
            print(f"Available keys in obs: {list(obs.keys())}")
            raise
        
        # Resize images to policy input size
        full_image = self._resize_image(full_image)
        left_wrist_image = self._resize_image(left_wrist_image)
        right_wrist_image = self._resize_image(right_wrist_image)
        
        
        return {
            "full_image": full_image,
            "left_wrist_image": left_wrist_image,
            "right_wrist_image": right_wrist_image,
            "state": prop_state,
        }

    def forward(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass for BEHAVIOR environment.
        
        Args:
            obs: Dictionary containing observations from BEHAVIOR environment
            
        Returns:
            torch.Tensor: Action tensor for the robot
        """
        # Process observations to extract images and proprioception
        processed_obs = self._process_behavior_obs(obs)
        
        # Use the head camera image as the primary input 
        primary_image = processed_obs["full_image"]
        left_wrist_image = processed_obs["left_wrist_image"]
        right_wrist_image = processed_obs["right_wrist_image"]
        if "dual" in self.policy_ckpt_path.lower():
            image_input = [primary_image]
            wrist_image_input = [left_wrist_image, right_wrist_image]
        else:
            image_input = [primary_image, left_wrist_image, right_wrist_image]
            wrist_image_input = None
        
        # Get task description from environment if not already set
        if self.task_description is None:
            # Try to get task description from the environment
            # This assumes the environment has a task attribute with show_instruction method
            print(f"Warning: Could not get task description")
            self.task_description = "Turn on the radio receiver that's on the table in the living room."
        
        # Prepare proprioceptive state.
        # GR00T action header expects state tensor to have shape (B, T_s, state_dim)
        # so that after the MLP it becomes (B, T_s, hidden).  If we pass a simple
        # 1-D vector the downstream `torch.cat` fails because the ranks differ.
        # Therefore, when state is enabled we reshape it to (1, 1, state_dim).
        
        if self.use_state:
            raw_state = processed_obs["state"]  # shape (state_dim,)
            state = raw_state[None, :]     # → (1, state_dim)
            example = {
                "image": image_input,
                "wrist_views": wrist_image_input,
                "state": state,
                "lang": self.task_description, 
            }
        else:
            example = {
                "image": image_input,
                "wrist_views": wrist_image_input,
                "lang": self.task_description, 
            }
        
        vla_input = {
            "examples": [example],
        }
        
        # Get action from websocket server
        action_chunk_size = self.action_chunk_size
        if self.current_step % action_chunk_size == 0:
            if self.current_step % 100 == 0:
                print('Step:', self.current_step)
            response = self.client.infer(vla_input)
            
            # Check if the response indicates an error
            if response.get("ok", True) == False or response.get("status") == "error":
                error_info = response.get("error", {})
                print(f"Websocket server returned an error:")
                print(f"Status: {response.get('status')}")
                print(f"Error details: {error_info}")
                raise RuntimeError(f"Websocket server error: {error_info}")
            
            # Extract and unnormalize actions
            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError as e:
                print(f"KeyError accessing response: {e}")
                print(f"Available keys in response: {list(response.keys())}")
                if "data" in response:
                    print(f"Keys in 'data': {list(response['data'].keys())}")
                raise
            # Take the first batch element (B dimension) → shape (T, D)
            normalized_actions = normalized_actions[0]
            # Un-normalize to get real-valued actions. Still shape (T, D)
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions, 
                action_norm_stats=self.action_norm_stats
            )

            # Apply action ensembling if enabled
            if self.action_ensemble and action_chunk_size > 1:
                self.action_ensembler.ensemble_action(self.raw_actions)
            
        if self.action_ensemble:
            if action_chunk_size == 1:
                raw_actions = self.action_ensembler.ensemble_action(self.raw_actions)[None]
            else:
                raw_actions = self.action_ensembler.step()[None]
        else:
            raw_actions = self.raw_actions[self.current_step % action_chunk_size][None]
        self.current_step += 1

        # Process raw actions for BEHAVIOR environment
        raw_action = {
            "base_pose": np.array(raw_actions[0,:3]),
            "torso_pose": np.array(raw_actions[0,3:7]),
            "left_arm_pose": np.array(raw_actions[0,7:14]),
            "left_gripper_pose": np.array(raw_actions[0,14:15]),
            "right_arm_pose": np.array(raw_actions[0,15:22]),
            "right_gripper_pose": np.array(raw_actions[0,22:23]),
        } 
        
        # Convert to BEHAVIOR action format
        action = self._process_action_for_behavior(raw_action)
        
        return torch.from_numpy(action).float()

    def _process_action_for_behavior(self, raw_action: Dict[str, np.ndarray]) -> np.ndarray:
        """Process raw action to BEHAVIOR environment format."""
        # Process base, torso, left arm, right arm
        base_pose = raw_action["base_pose"]
        torso_pose = raw_action["torso_pose"]
        left_arm_pose = raw_action["left_arm_pose"]
        right_arm_pose = raw_action["right_arm_pose"]

        # Process gripper action
        left_gripper_pose = self._process_gripper_action(raw_action["left_gripper_pose"])
        right_gripper_pose = self._process_gripper_action(raw_action["right_gripper_pose"])
        
        # Combine all actions into a single array
        # BEHAVIOR expects "ACTION_DIM": 23
        # See the following files:
        # - ACTION_QPOS_INDICES in omnigibson/learning/utils/eval_utils.py
        # - action_keys in omnigibson/learning/configs/robot/r1pro.yaml
        action = np.concatenate([
            base_pose,                      # indices 0:3   (3 dims)
            torso_pose,                     # indices 3:7   (4 dims)
            left_arm_pose,                  # indices 7:14  (7 dims)
            np.array([left_gripper_pose]),  # index  14:15  (1 dim)
            right_arm_pose,                 # indices 15:22 (7 dims)
            np.array([right_gripper_pose]), # index  22:23  (1 dim)
        ])
        
        return action


    def _process_gripper_action(self, open_gripper: np.ndarray) -> float:
        """Process gripper action with sticky behavior for BEHAVIOR environment."""
        # We currently don't apply any other sticky behavior for gripper action.
        current_gripper_action = open_gripper[0]

        return current_gripper_action

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """Unnormalize actions using the provided statistics."""
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
                
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """Load action normalization statistics from checkpoint."""
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        
        return norm_stats[unnorm_key]["action"]

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to policy input size."""
        
        image = image.numpy()
                                        
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        """Visualize predicted actions and images."""
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)