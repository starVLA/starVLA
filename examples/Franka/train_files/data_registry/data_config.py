"""Franka benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig — Franka Delta EEF
# ---------------------------------------------------------------------------
class SingleFrankaRobotiqDeltaEefDataConfig:
    video_keys = ["video.base_view", "video.ego_view"]
    state_keys = ["state.eef_position", "state.eef_rotation"]
    action_keys = ["action.delta_eef_position", "action.delta_eef_rotation", "action.gripper_close"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={"state.eef_position": "min_max", "state.eef_rotation": "min_max"},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_eef_position": "min_max",
                    "action.delta_eef_rotation": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Franka Delta Joints (sim)
# ---------------------------------------------------------------------------
class SingleFrankaRobotiqDeltaJointsDataConfig:
    video_keys = ["video.base_view", "video.ego_view"]
    state_keys = ["state.joints"]
    action_keys = ["action.delta_joints", "action.gripper_close"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={"state.joints": "min_max"}),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={"action.delta_joints": "min_max", "action.gripper_close": "binary"},
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — SO101
# ---------------------------------------------------------------------------
class SO101Config:
    video_keys = ["video.primary_image", "video.wrist_image"]
    state_keys = [
        "state.shoulder_pan.pos", "state.shoulder_lift.pos", "state.elbow_flex.pos",
        "state.wrist_flex.pos", "state.wrist_roll.pos", "state.gripper.pos",
    ]
    action_keys = [
        "action.shoulder_pan.pos", "action.shoulder_lift.pos", "action.elbow_flex.pos",
        "action.wrist_flex.pos", "action.wrist_roll.pos", "action.gripper.pos",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={k: "min_max" for k in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={k: "min_max" for k in self.action_keys},
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "custom_robot_config": SingleFrankaRobotiqDeltaEefDataConfig(),
    "demo_sim_franka_delta_joints": SingleFrankaRobotiqDeltaJointsDataConfig(),
    "SO101": SO101Config(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "demo_sim_franka_delta_joints": EmbodimentTag.FRANKA,
    "custom_robot_config": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "custom_dataset": [("custom_dataset_name", 1.0, "custom_robot_config")],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],
    "demo_sim_pick_place": [("sim_pick_place", 1.0, "demo_sim_franka_delta_joints")],
    "SO101_pick": [("pick_dataset_name", 1.0, "SO101")],
}
