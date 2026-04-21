"""VLA-Arena benchmark — data config, embodiment tags, and mixtures.

VLA-Arena uses a mounted Franka Panda with 7-DOF delta-EEF actions
(Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper).  The action layout is
identical to LIBERO, so it reuses EmbodimentTag.FRANKA.

HuggingFace repos (downloaded by data_preparation.sh):
    VLA-Arena/VLA_Arena_L0_S_lerobot_openpi  (Small)
    VLA-Arena/VLA_Arena_L0_M_lerobot_openpi  (Medium)
    VLA-Arena/VLA_Arena_L0_L_lerobot_openpi  (Large)
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class VLAArenaFrankaDataConfig:
    """
    Data config for VLA-Arena environments (robosuite, mounted Panda).

    Action space  : 7-DOF delta EEF  (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper)
    Observation   : primary agentview image (256×256 rendered, resized to 224×224)
    State         : EEF pos (3) + EEF axis-angle (3) + gripper qpos (1) = 7
    """

    video_keys = [
        "video.primary_image",   # agentview camera
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
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
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "vla_arena_franka": VLAArenaFrankaDataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "vla_arena_franka": EmbodimentTag.FRANKA,
}


# ---------------------------------------------------------------------------
# Mixtures
# Dataset root: playground/Datasets/VLA_ARENA_LEROBOT_DATA/
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "vla_arena_L0_S": [
        ("VLA_Arena_L0_S_lerobot_openpi", 1.0, "vla_arena_franka"),
    ],
    "vla_arena_L0_M": [
        ("VLA_Arena_L0_M_lerobot_openpi", 1.0, "vla_arena_franka"),
    ],
    "vla_arena_L0_L": [
        ("VLA_Arena_L0_L_lerobot_openpi", 1.0, "vla_arena_franka"),
    ],
}
