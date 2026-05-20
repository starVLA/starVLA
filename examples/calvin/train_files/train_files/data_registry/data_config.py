"""CALVIN ABC LeRobot registry for Cosmos-Predict2 experiments."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class CalvinABCSingleViewCosmosDataConfig:
    embodiment_tag = EmbodimentTag.FRANKA

    video_keys = ["video.primary_image"]
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
    action_key_dims = {key: 1 for key in action_keys}
    state_key_dims = {key: 1 for key in state_keys}
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = list(range(-16, 0))

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
                        "action.gripper": "binary",
                    },
                ),
            ]
        )


class CalvinABCMultiviewCosmosDataConfig(CalvinABCSingleViewCosmosDataConfig):
    video_keys = ["video.primary_image", "video.wrist_image"]


ROBOT_TYPE_CONFIG_MAP = {
    "calvin_abc_cosmos_franka_single_view": CalvinABCSingleViewCosmosDataConfig(),
    "calvin_abc_cosmos_franka_multiview": CalvinABCMultiviewCosmosDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "calvin_abc_cosmos_franka_single_view": EmbodimentTag.FRANKA,
    "calvin_abc_cosmos_franka_multiview": EmbodimentTag.FRANKA,
}

DATASET_NAMED_MIXTURES = {
    "calvin_task_ABC_D": [
        ("calvin_task_ABC_D", 1.0, "calvin_abc_cosmos_franka_single_view"),
    ],
    "calvin_task_ABC_D_single_view": [
        ("calvin_task_ABC_D", 1.0, "calvin_abc_cosmos_franka_single_view"),
    ],
    "calvin_task_ABC_D_multiview": [
        ("calvin_task_ABC_D", 1.0, "calvin_abc_cosmos_franka_multiview"),
    ],
}
