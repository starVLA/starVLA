"""IndoorUAV benchmark — data config, embodiment tags, and mixtures.

The IndoorUAV dataset contains short VLA segments extracted from VLN trajectories
collected in Habitat-Sim across replica / hm3d / gibson / mp3d scenes.

State  (4-dim):  [x, y, z, yaw_deg]  -- absolute drone pose in world frame
Action (4-dim):  [dx, dy, dz, dyaw_deg]  -- pose delta to the next frame
Image:           1280x720 monocular front-view RGB, encoded at 10 FPS
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class IndoorUAVDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.front",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.yaw",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.yaw",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))   # predict next 8 frames of action chunk
    state_indices = [0]

    def modality_config(self):
        return {
            "video":    ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state":    ModalityConfig(delta_indices=self.state_indices,       modality_keys=self.state_keys),
            "action":   ModalityConfig(delta_indices=self.action_indices,      modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x":   "min_max",
                    "action.y":   "min_max",
                    "action.z":   "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "indoor_uav": IndoorUAVDataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "indoor_uav": EmbodimentTag.NEW_EMBODIMENT,
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "indoor_uav_replica": [
        ("indoor_uav_replica_vla_lerobot", 1.0, "indoor_uav"),
    ],
}
