"""StarVLA data configuration for native-action RLBench datasets."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RLBenchJointVelocityDataConfig:
    """Fixed schema for JointVelocity + Discrete Panda demonstrations."""

    def __init__(self, action_horizon: int):
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        # The loader applies offsets relative to the input observation. RLBench's
        # official imitation-learning example uses each observation's own
        # joint_velocities as the JointVelocity supervision target.
        self.action_indices = list(range(action_horizon))

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.front_rgb",
        "video.left_shoulder_rgb",
        "video.right_shoulder_rgb",
        "video.wrist_rgb",
        "video.overhead_rgb",
    ]
    state_keys = [
        "state.joint_positions",
        "state.joint_velocities",
        "state.joint_forces",
        "state.gripper_open",
        "state.gripper_joint_positions",
        "state.gripper_touch_forces",
        "state.gripper_pose",
        "state.gripper_matrix",
    ]
    state_key_dims = {
        "state.joint_positions": 7,
        "state.joint_velocities": 7,
        "state.joint_forces": 7,
        "state.gripper_open": 1,
        "state.gripper_joint_positions": 2,
        "state.gripper_touch_forces": 6,
        "state.gripper_pose": 7,
        "state.gripper_matrix": 16,
    }
    action_key_dims = {"action.native": 8}
    action_keys = ["action.native"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={
                        key: "min_max" for key in self.state_keys
                    },
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={"action.native": "min_max"},
                ),
            ]
        )

