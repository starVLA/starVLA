from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class CalvinABCCosmosDataConfig:
    # Cosmos-Predict2 当前代码会把 sample["image"] 当成视频帧序列。
    # 先只用 primary_image，避免把 primary/wrist 两个相机误当成时间帧。
    video_keys = ["video.primary_image"]

    state_keys = [
        "state.x", "state.y", "state.z",
        "state.roll", "state.pitch", "state.yaw",
        "state.pad", "state.gripper",
    ]

    action_keys = [
        "action.x", "action.y", "action.z",
        "action.roll", "action.pitch", "action.yaw",
        "action.gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = list(range(-16, 0))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
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
    "calvin_abc_cosmos_franka": CalvinABCCosmosDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "calvin_abc_cosmos_franka": EmbodimentTag.FRANKA,
}

DATASET_NAMED_MIXTURES = {
    "calvin_task_ABC_D": [
        ("calvin_task_ABC_D", 1.0, "calvin_abc_cosmos_franka"),
    ],
}