"""MetaWorld MT50 benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class MetaWorldRobotDataConfig:
    """MetaWorld MT50: 4-DOF delta EEF (Δx, Δy, Δz, gripper), single corner2 camera."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.primary_image",
    ]
    state_keys = [
        "state.joint_pos_x",
        "state.joint_pos_y",
        "state.joint_pos_z",
        "state.gripper",
    ]
    action_keys = [
        "action.delta_x",
        "action.delta_y",
        "action.delta_z",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_x": "min_max",
                    "action.delta_y": "min_max",
                    "action.delta_z": "min_max",
                    "action.gripper": "min_max",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "metaworld_robot": MetaWorldRobotDataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "metaworld_mt50": [
        ("metaworld_mt50_lerobot", 1.0, "metaworld_robot"),
    ],
}
