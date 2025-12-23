from collections import deque
from typing import Optional, Sequence
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from examples.Robocasa_tabletop.eval_files.adaptive_ensemble import AdaptiveEnsembler
from typing import Dict
import numpy as np
from pathlib import Path



from starVLA.model.framework.share_tools import read_mode_config



class PolicyWarper:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble = False, # @Jinhui
        action_ensemble_horizon: Optional[int] = 3, # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
        n_action_steps=2,
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.n_action_steps = n_action_steps

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str or tuple) -> None:
       
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
        observations,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        执行一步推理
        :param image: 输入图像 (H, W, 3) uint8格式
        :param task_description: 任务描述文本
        :return: (原始动作, 处理后的动作)
        """

        task_description = observations['annotation.human.coarse_action'][0] # tuple       
        ego_view = observations['video.ego_view']  # (N, 1, H, W, 3)
        images = ego_view
        state = {}
        state['left_arm'] = observations['state.left_arm']     
        state['right_arm'] = observations['state.right_arm']             # (N, 1, 7)
        state['left_hand'] = observations['state.left_hand']              # (N, 1, 6)
        state['right_hand'] = observations['state.right_hand']            # (N, 1, 6)
        state['waist'] = observations['state.waist']                      # (N, 1, 3)
        

        state = self.normalize_state(state)
        input_state = []
        for key in state.keys():
            input_state.append(state[key])
        input_state = np.concatenate(input_state, axis=-1)

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        # image: Image.Image = Image.fromarray(image)

        images = [[self._resize_image(img) for img in sample] for sample in images] # (B, N_view, H, W, 3)
        input_state = [input_s for input_s in input_state] # B, state_dim*(sin, cos)

        # prepare vla input
        examples = []
        batch_size = len(images)
        instructions = [self.task_description] if isinstance(self.task_description, str) else self.task_description
        for b in range(batch_size):
            example = {
                "image": images[b],  # A list of multi-view images for a single sample
                "lang": instructions[b] if isinstance(instructions, list) else instructions,
                "state": input_state[b],  # N_history, 58
            }
            examples.append(example)
        
        vla_input = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        
        response = self.client.predict_action(vla_input)
        
        
        # unnormalize the action
        normalized_actions = response["data"]["normalized_actions"] # B, chunk, D        
        
        # unnormalize actions in batch form
        raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)
        
        # raw_actions shape: (B, chunk, D)
        if self.action_ensemble:
            # 对batch中的每个样本进行ensemble
            batch_size = raw_actions.shape[0]
            ensembled_actions = []
            for b in range(batch_size):
                ensembled = self.action_ensembler.ensemble_action(raw_actions[b])[None]  # (1, D)
                ensembled_actions.append(ensembled)
            raw_actions = np.stack(ensembled_actions, axis=0)  # (B, 1, D)

        raw_action = {
            "action.left_arm": raw_actions[:, :self.n_action_steps, :7],      # (B, n_action_steps, 7)
            "action.right_arm": raw_actions[:, :self.n_action_steps, 7:14],   # (B, n_action_steps, 7)
            "action.left_hand": raw_actions[:, :self.n_action_steps, 14:20],  # (B, n_action_steps, 6)
            "action.right_hand": raw_actions[:, :self.n_action_steps, 20:26], # (B, n_action_steps, 6)
            "action.waist": raw_actions[:, :self.n_action_steps, 26:29],      # (B, n_action_steps, 3)
        }

        return {"actions": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Args:
            normalized_actions: shape (B, chunk, D) (chunk, D)
            action_norm_stats:
        Returns:
            actions
        """
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        
        normalized_actions = np.clip(normalized_actions, -1, 1)
        
        actions = np.where(
            mask,
            (normalized_actions + 1) / 2 * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = PolicyWarper._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]



    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
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
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
    
    def normalize_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalize the state
        """
        for key in state.keys():
            sin_state = np.sin(state[key])
            cos_state = np.cos(state[key])
            state[key] = np.concatenate([sin_state, cos_state], axis=-1)
        return state
    
    