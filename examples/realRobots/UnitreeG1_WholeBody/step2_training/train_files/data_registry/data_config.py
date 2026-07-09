"""Unitree G1 WholeBody - SONIC / Dex3 data registry for QwenOFT."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class UnitreeG1SonicDex3QwenOFTDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.left_hand",
        "state.right_arm",
        "state.right_hand",
        "state.left_wrist_pos",
        "state.left_wrist_abs_quat",
        "state.right_wrist_pos",
        "state.right_wrist_abs_quat",
        "state.root_orientation",
        "state.projected_gravity",
        "state.cpp_rotation_offset",
        "state.init_base_quat",
    ]
    action_keys = [
        "action.motion_token",
        "action.left_hand_joints",
        "action.right_hand_joints",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    state_key_dims = {
        "state.left_leg": 6,
        "state.right_leg": 6,
        "state.waist": 3,
        "state.left_arm": 7,
        "state.left_hand": 7,
        "state.right_arm": 7,
        "state.right_hand": 7,
        "state.left_wrist_pos": 3,
        "state.left_wrist_abs_quat": 4,
        "state.right_wrist_pos": 3,
        "state.right_wrist_abs_quat": 4,
        "state.root_orientation": 4,
        "state.projected_gravity": 3,
        "state.cpp_rotation_offset": 4,
        "state.init_base_quat": 4,
    }
    action_key_dims = {
        "action.motion_token": 64,
        "action.left_hand_joints": 7,
        "action.right_hand_joints": 7,
    }

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
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


ROBOT_TYPE_CONFIG_MAP = {
    "unitree_g1_sonic_dex3": UnitreeG1SonicDex3QwenOFTDataConfig(),
}

DATASET_NAMED_MIXTURES = {
    "unitree_g1_test_sonic": [("test_sonic", 1.0, "unitree_g1_sonic_dex3")],
}
