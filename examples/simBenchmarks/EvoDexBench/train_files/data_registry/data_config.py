"""Evo-DexBench benchmark: data configs, embodiment tags and 15-task mixtures.

Picked up by ``starVLA.dataloader.gr00t_lerobot.registry``. Each mixture expects one LeRobot v3
dataset per task (``m01``-``m15`` or ``b01``-``b15``) under ``datasets.vla_data.data_root_dir``.
"""

from typing import ClassVar

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class _EvoDexConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    observation_indices: ClassVar[list[int]] = [0]
    action_indices: ClassVar[list[int]] = list(range(50))
    roles: ClassVar[tuple[str, ...]] = ()
    video_keys: ClassVar[list[str]] = []

    @property
    def state_keys(self) -> list[str]:
        return [key for role in self.roles for key in (f"state.{role}_ee_pose", f"state.{role}_hand_qpos")]

    @property
    def action_keys(self) -> list[str]:
        return [key for role in self.roles for key in (f"action.{role}_ee_delta", f"action.{role}_hand_qpos")]

    @property
    def state_key_dims(self) -> dict[str, int]:
        return {key: 7 if key.endswith("ee_pose") else 24 for key in self.state_keys}

    @property
    def action_key_dims(self) -> dict[str, int]:
        return {key: 6 if key.endswith("ee_delta") else 24 for key in self.action_keys}

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=["annotation.human.action.task_description"],
            ),
        }

    def transform(self) -> ComposedModalityTransform:
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "q99" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )


class EvoDexSingleConfig(_EvoDexConfig):
    roles = ("right",)
    video_keys: ClassVar[list[str]] = [
        "video.context",
        "video.interaction",
        "video.wrist_right",
    ]


class EvoDexDualConfig(_EvoDexConfig):
    roles = ("left", "right")
    video_keys: ClassVar[list[str]] = [
        "video.context",
        "video.wrist_left",
        "video.wrist_right",
    ]


class EvoDexSingleH8Config(EvoDexSingleConfig):
    action_indices: ClassVar[list[int]] = list(range(8))


class EvoDexDualH8Config(EvoDexDualConfig):
    action_indices: ClassVar[list[int]] = list(range(8))


ROBOT_TYPE_CONFIG_MAP = {
    "evodex_single_h50_q99": EvoDexSingleConfig(),
    "evodex_dual_h50_q99": EvoDexDualConfig(),
    "evodex_single_h8_q99": EvoDexSingleH8Config(),
    "evodex_dual_h8_q99": EvoDexDualH8Config(),
}

SINGLE_ARM_TASKS: tuple[str, ...] = tuple(f"m{index:02d}" for index in range(1, 16))
DUAL_ARM_TASKS: tuple[str, ...] = tuple(f"b{index:02d}" for index in range(1, 16))

DATASET_NAMED_MIXTURES = {
    # Single-dataset mixtures, one per robot type. The benchmark client requires the server to report one of these;
    # serve a 15-task checkpoint with `datasets.vla_data.data_mix=<name>` (same robot type, saved statistics unchanged).
    "evodex_single_h50_q99": [("evodex_single", 1.0, "evodex_single_h50_q99")],
    "evodex_dual_h50_q99": [("evodex_dual", 1.0, "evodex_dual_h50_q99")],
    "evodex_single_h8_q99": [("evodex_single", 1.0, "evodex_single_h8_q99")],
    "evodex_dual_h8_q99": [("evodex_dual", 1.0, "evodex_dual_h8_q99")],
    # One LeRobot dataset per task under the data root, sampled with equal weight.
    "evodex_single15_h50_q99": [(task, 1.0, "evodex_single_h50_q99") for task in SINGLE_ARM_TASKS],
    "evodex_dual15_h50_q99": [(task, 1.0, "evodex_dual_h50_q99") for task in DUAL_ARM_TASKS],
    "evodex_single15_h8_q99": [(task, 1.0, "evodex_single_h8_q99") for task in SINGLE_ARM_TASKS],
    "evodex_dual15_h8_q99": [(task, 1.0, "evodex_dual_h8_q99") for task in DUAL_ARM_TASKS],
}
