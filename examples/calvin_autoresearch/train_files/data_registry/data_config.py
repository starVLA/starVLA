"""CALVIN AutoResearch data registry."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class CalvinFrankaDataConfig:
    """CALVIN Franka layout with explicit 8-D proprioception.

    CALVIN LeRobot stores state as:
    [x, y, z, roll, pitch, yaw, pad, gripper].
    The pad dimension is kept because it is present in the dataset statistics
    and must line up exactly between training and closed-loop evaluation.
    """

    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={
                        "state.x": "min_max",
                        "state.y": "min_max",
                        "state.z": "min_max",
                        "state.roll": "min_max",
                        "state.pitch": "min_max",
                        "state.yaw": "min_max",
                        "state.pad": "min_max",
                        "state.gripper": "min_max",
                    },
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.x": "min_max",
                        "action.y": "min_max",
                        "action.z": "min_max",
                        "action.roll": "min_max",
                        "action.pitch": "min_max",
                        "action.yaw": "min_max",
                    },
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "calvin_franka": CalvinFrankaDataConfig(),
}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "calvin_franka": EmbodimentTag.FRANKA,
}

DATASET_NAMED_MIXTURES = {
    # ABC demonstrations for imitation-learning smoke training.
    "calvin_abc_train_v3.0": [
        ("calvin_abc_train_v3.0", 1.0, "libero_franka"),
    ],
    # Same ABC demonstrations, but with CALVIN's 8-D robot proprioception.
    "calvin_abc_train_state_v3.0": [
        ("calvin_abc_train_v3.0", 1.0, "calvin_franka"),
    ],
    # Task D validation/test dataset used by the closed-loop evaluator.
    "calvin_task_D_D_v3.0": [
        ("calvin_task_D_D_v3.0", 1.0, "libero_franka"),
    ],
    "calvin_task_D_D_state_v3.0": [
        ("calvin_task_D_D_v3.0", 1.0, "calvin_franka"),
    ],
}
