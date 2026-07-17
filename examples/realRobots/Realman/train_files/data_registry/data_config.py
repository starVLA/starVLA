# ruff: noqa: RUF012
"""Realman RM-75 data config for a 7-DoF arm and parallel gripper.

This example expects a LeRobot dataset whose first eight state/action dimensions are seven arm
joints plus one gripper value. Additional state or action dimensions may exist in a user's dataset,
but this recipe does not map them. The model-facing fields are:

- ``state.joints``: seven joint angles
- ``state.gripper``: one normalized gripper opening
- ``action.delta_joints``: seven joint targets converted to sample-time deltas
- ``action.gripper_close``: one absolute gripper target

The YAML applies delta conversion only to ``action.delta_joints``. The gripper is deliberately
excluded and therefore remains absolute. Replace ``<your_dataset>`` in
``DATASET_NAMED_MIXTURES`` with the directory name of a compatible LeRobot dataset.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RealmanRM75DataConfig:
    """Seven joint deltas plus an absolute gripper target from two cameras."""

    # Realman is not yet a first-class tokenizer embodiment, so use the standard fallback.
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    # Camera order must match observation.images.<camera> fields in the LeRobot dataset.
    video_keys = ["video.cam0_rgb", "video.cam1_rgb"]

    state_keys = ["state.joints", "state.gripper"]
    action_keys = ["action.delta_joints", "action.gripper_close"]

    # Per-key dimensions let PolicyNormProcessor split combined statistics arrays.
    state_key_dims = {"state.joints": 7, "state.gripper": 1}
    action_key_dims = {"action.delta_joints": 7, "action.gripper_close": 1}

    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    state_indices = [0]
    # Action window for ACT: matches its chunk_size so the policy trains on the immediate
    # 50-step chunk. Diffusion Policy uses the dedicated config below instead — its window
    # must equal framework.horizon so training targets start at the sampled observation.
    action_indices = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices, modality_keys=self.video_keys
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices, modality_keys=self.state_keys
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices, modality_keys=self.action_keys
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices, modality_keys=self.language_keys
            ),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={
                        "state.joints": "mean_std",
                        "state.gripper": "min_max",
                    },
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.delta_joints": "mean_std",
                        "action.gripper_close": "min_max",
                    },
                ),
            ]
        )


class RealmanRM75DPDataConfig(RealmanRM75DataConfig):
    """Realman config for Diffusion Policy: the action window equals the DP horizon.

    Diffusion Policy models a fixed ``framework.horizon`` (16 in ``train_realman_dp.yaml``)
    starting at the sampled observation. Requesting exactly that window keeps the training
    targets immediate (t .. t+15) and avoids loading unused far-future actions.
    """

    action_indices = list(range(16))


ROBOT_TYPE_CONFIG_MAP = {
    "realman_rm75_delta_joints": RealmanRM75DataConfig(),
    "realman_rm75_delta_joints_dp": RealmanRM75DPDataConfig(),
}

# The embodiment tag is read from the DataConfig class variable.
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "realman_example": [
        (
            "<your_dataset>",
            1.0,
            "realman_rm75_delta_joints",
        ),
    ],
    # Diffusion Policy variant of the same dataset: identical schema/transforms, but the
    # action window matches framework.horizon (see RealmanRM75DPDataConfig).
    "realman_example_dp": [
        (
            "<your_dataset>",
            1.0,
            "realman_rm75_delta_joints_dp",
        ),
    ],
}
