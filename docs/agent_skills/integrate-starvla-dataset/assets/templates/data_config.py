"""<<TODO_BENCH>> — data_config / embodiments / mixtures.

Auto-discovered by ``starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge``.
Drop this file at::

    examples/<<TODO_BENCH>>/train_files/data_registry/data_config.py

It MUST export the three module-level dicts at the bottom — search for
``ROBOT_TYPE_CONFIG_MAP``, ``ROBOT_TYPE_TO_EMBODIMENT_TAG``,
``DATASET_NAMED_MIXTURES``. The names are what the registry looks for.
"""
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# Per-robot data layout — replace <<TODO_*>> placeholders.
# Keep camera / state / action key names IDENTICAL to what your converter
# wrote into the LeRobot dataset.
# ---------------------------------------------------------------------------
class _MyRobotDataConfig:
    video_keys = [
        "video.<<TODO_CAM_1>>",
        "video.<<TODO_CAM_2>>",
    ]
    state_keys = [
        "state.<<TODO_STATE_GROUP_1>>",        # e.g. "state.joint_positions"
        "state.<<TODO_STATE_GROUP_2>>",        # e.g. "state.gripper_width"
    ]
    action_keys = [
        "action.<<TODO_ACTION_GROUP_1>>",      # e.g. "action.ee_positions"
        "action.<<TODO_ACTION_GROUP_2>>",      # e.g. "action.gripper_width"
    ]
    language_keys = ["annotation.human.task_description"]

    observation_indices = [0]                  # current frame only
    action_indices = list(range(8))            # predict t .. t+7  (== action_horizon)

    def modality_config(self):
        return {
            "video":    ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state":    ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action":   ModalityConfig(delta_indices=self.action_indices,      modality_keys=self.action_keys),
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


# ---------------------------------------------------------------------------
# REQUIRED top-level exports — names matter, do not rename.
# ---------------------------------------------------------------------------
ROBOT_TYPE_CONFIG_MAP = {
    "<<TODO_robot_type_string>>": _MyRobotDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # NEW_EMBODIMENT is the safe default. Use FRANKA / UR5 / etc only if your
    # action space matches an existing tag's embedding head.
    "<<TODO_robot_type_string>>": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    # mixture_name : list of (dataset_subdir, weight, robot_type_string)
    # dataset_subdir is RELATIVE to yaml.datasets.vla_data.data_root_dir
    "<<TODO_mixture_name>>": [
        ("<<TODO_dataset_subdir_1>>", 1.0, "<<TODO_robot_type_string>>"),
        # ("<<TODO_dataset_subdir_2>>", 1.0, "<<TODO_robot_type_string>>"),
    ],
}
