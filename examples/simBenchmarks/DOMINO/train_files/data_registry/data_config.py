"""DOMINO benchmark — data config, embodiment tags, and mixtures.

DOMINO shares its data format with RoboTwin 2.0 (Aloha-AgileX 14-D state/action,
three cameras), so we reuse the same `AgilexDataConfig`. The only DOMINO-specific
piece is the task set: 35 dynamic tasks collected under `demo_clean_dynamic` and
`demo_random_dynamic`.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig — Agilex (DOMINO, action_indices=16)
# ---------------------------------------------------------------------------
class AgilexDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.left_joints", "state.right_joints", "state.left_gripper", "state.right_gripper"]
    action_keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
    # Per-key dims for PolicyNormProcessor (Agilex 6-DOF arms + binary gripper = 14-D total)
    action_key_dims = {"action.left_joints": 6, "action.right_joints": 6, "action.left_gripper": 1, "action.right_gripper": 1}
    state_key_dims  = {"state.left_joints": 6, "state.right_joints": 6, "state.left_gripper": 1, "state.right_gripper": 1}
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
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "robotwin": AgilexDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}


# ---------------------------------------------------------------------------
# Task list — the 35 DOMINO benchmark tasks
# (matches DOMINO/script/extract_dynamic_gt.py :: DEFAULT_35_TASKS)
# ---------------------------------------------------------------------------
DOMINO_35_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_can_basket",
    "place_container_plate",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stamp_seal",
]


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
# Directory layout expected under `data_root_dir` (set in the train YAML):
#
#   <data_root_dir>/
#     Clean_Dynamic/<task>/...
#     Random_Dynamic/<task>/...
#     Clean/<task>/...          # optional, for DOMINO + RoboTwin co-training
#     Randomized/<task>/...     # optional, for DOMINO + RoboTwin co-training
#
# To rename the splits, adjust the prefixes below to match your local HDF5 /
# LeRobot conversion output.
# ---------------------------------------------------------------------------
_CLEAN_DYNAMIC = [(f"Clean_Dynamic/{t}", 1.0, "robotwin") for t in DOMINO_35_TASKS]
_RANDOM_DYNAMIC = [(f"Random_Dynamic/{t}", 1.0, "robotwin") for t in DOMINO_35_TASKS]
_CLEAN_STATIC = [(f"Clean/{t}", 1.0, "robotwin") for t in DOMINO_35_TASKS]
_RANDOM_STATIC = [(f"Randomized/{t}", 1.0, "robotwin") for t in DOMINO_35_TASKS]


DATASET_NAMED_MIXTURES = {
    # 35 × (Clean_Dynamic + Random_Dynamic)
    "domino": _CLEAN_DYNAMIC + _RANDOM_DYNAMIC,

    # Clean-only / random-only variants for ablations
    "domino_clean": _CLEAN_DYNAMIC,
    "domino_random": _RANDOM_DYNAMIC,

    # DOMINO + RoboTwin-static co-training (matches tab. "exp_mix" in the paper)
    "domino_cotrain": _CLEAN_DYNAMIC + _RANDOM_DYNAMIC + _CLEAN_STATIC + _RANDOM_STATIC,
}
