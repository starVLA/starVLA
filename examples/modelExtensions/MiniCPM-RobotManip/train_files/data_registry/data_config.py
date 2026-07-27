"""MiniCPM-RobotManip LIBERO fine-tune — data config (80-D EE6D recipe).

The LIBERO LeRobot-v3 migration stores X-VLA-aligned single-arm **absolute**
EE6D targets in ``observation.xvla_abs_ee6d`` (10-D = xyz(3) + rot6d(6) +
gripper(1)). We read that column for both state (offset 0) and the 30-step
action chunk (offsets 1..30), keeping the raw metric values (no normalization),
matching the recipe that produced the released checkpoint. The 10-D vector is
lifted to the model's 80-D unified layout inside the ``MiniCPMRobotManip``
framework (left-arm eef slot [7:17]); the remaining channels are masked out.
"""

from typing import ClassVar

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor

ACTION_HORIZON = 30


class MiniCPMRobotManipLiberoEE6DConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys: ClassVar = ["video.primary_image", "video.wrist_image"]
    state_keys: ClassVar = ["state.eef_position", "state.eef_rotation", "state.gripper"]
    action_keys: ClassVar = ["action.eef_position", "action.eef_rotation", "action.gripper"]
    language_keys: ClassVar = ["annotation.human.action.task_description"]
    observation_indices: ClassVar = [0]
    state_indices: ClassVar = [0]
    # Absolute EE6D targets for the next ACTION_HORIZON steps (offsets 1..30).
    action_indices: ClassVar = list(range(1, ACTION_HORIZON + 1))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # Raw absolute EE6D targets — no normalization (matches released recipe).
        return ComposedModalityTransform(transforms=[StateActionToTensor(apply_to=self.state_keys + self.action_keys)])


ROBOT_TYPE_CONFIG_MAP = {
    "minicpm_robotmanip_libero_ee6d": MiniCPMRobotManipLiberoEE6DConfig(),
}

_TASKS = (
    "libero_10_agentview_rot180_wrist_raw_lerobot_v30_xvla_abs6d",
    "libero_goal_agentview_rot180_wrist_raw_lerobot_v30_xvla_abs6d",
    "libero_object_agentview_rot180_wrist_raw_lerobot_v30_xvla_abs6d",
    "libero_spatial_agentview_rot180_wrist_raw_lerobot_v30_xvla_abs6d",
)

DATASET_NAMED_MIXTURES = {
    "minicpm_robotmanip_libero_ee6d": [(t, 1.0, "minicpm_robotmanip_libero_ee6d") for t in _TASKS],
}
