# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


from abc import ABC, abstractmethod

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)

# from gr00t.model.transforms import GR00TTransform


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


###########################################################################################

class Libero4in1DataConfig:
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
    action_indices = list(range(8))
    state_indices = list(range(-16, 0))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
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

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################

ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    # Other robot configs are auto-discovered from examples/*/train_files/data_registry/
}
