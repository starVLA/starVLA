"""RoboCasa365 (PandaOmron, single-arm Franka) — data config, embodiment tags, and mixtures.

Loaded automatically by ``starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge``.

Schema follows the official robocasa LeRobot conversion (see
``robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py``):

* observation.state (16d): base_position(3) + base_rotation(4) + eef_pos_rel(3) + eef_rot_rel(4) + gripper_qpos(2)
* action (12d): eef_pos(3) + eef_rot(3) + gripper_close(1) + base_motion(4) + control_mode(1)
* 3 cameras at 256x256, 20 fps. We only use ``robot0_agentview_left`` for training.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class PandaOmronRoboCasa365DataConfig:
    """Single-arm Franka PandaOmron used by upstream RoboCasa365."""

    video_keys = [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand"]
    state_keys = [
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]
    language_keys = ["annotation.human.task_description"]

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
            StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "panda_omron_robocasa365": PandaOmronRoboCasa365DataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Use NEW_EMBODIMENT since PandaOmron is not in the canonical EmbodimentTag enum
    "panda_omron_robocasa365": EmbodimentTag.NEW_EMBODIMENT,
}

# Each task lives at
#   ``playground/Datasets/robocasa365/<RELATIVE_PATH>/lerobot``
# where RELATIVE_PATH below is exactly what robocasa's upstream
# ``ATOMIC_TASK_DATASETS[name]['target']['human_path']`` (or the composite
# equivalent) returns. Lists below are a one-shot snapshot of those tables for
# every task that ships a ``target/human`` LeRobot bundle (50 tasks total: 18
# atomic + 32 composite). Re-generate via:
#
#   python -m robocasa.utils.dataset_registry  # has constants
#   # or run the helper at examples/Robocasa_365/train_files/dump_target_human_paths.py
_ROBOT_TAG = "panda_omron_robocasa365"

# Atomic single-skill tasks (target/human split, 18 tasks).
_TARGET_HUMAN_ATOMIC = {
    "CloseBlenderLid":           "v1.0/target/atomic/CloseBlenderLid/20250822",
    "CloseFridge":               "v1.0/target/atomic/CloseFridge/20250816",
    "CloseToasterOvenDoor":      "v1.0/target/atomic/CloseToasterOvenDoor/20250818",
    "CoffeeSetupMug":            "v1.0/target/atomic/CoffeeSetupMug/20250813",
    "NavigateKitchen":           "v1.0/target/atomic/NavigateKitchen/20250821",
    "OpenCabinet":               "v1.0/target/atomic/OpenCabinet/20250813",
    "OpenDrawer":                "v1.0/target/atomic/OpenDrawer/20250816",
    "OpenStandMixerHead":        "v1.0/target/atomic/OpenStandMixerHead/20250818",
    "PickPlaceCounterToCabinet": "v1.0/target/atomic/PickPlaceCounterToCabinet/20250811",
    "PickPlaceCounterToStove":   "v1.0/target/atomic/PickPlaceCounterToStove/20250818",
    "PickPlaceDrawerToCounter":  "v1.0/target/atomic/PickPlaceDrawerToCounter/20250820",
    "PickPlaceSinkToCounter":    "v1.0/target/atomic/PickPlaceSinkToCounter/20250813",
    "PickPlaceToasterToCounter": "v1.0/target/atomic/PickPlaceToasterToCounter/20250817",
    "SlideDishwasherRack":       "v1.0/target/atomic/SlideDishwasherRack/20250820",
    "TurnOffStove":              "v1.0/target/atomic/TurnOffStove/20250812",
    "TurnOnElectricKettle":      "v1.0/target/atomic/TurnOnElectricKettle/20250817",
    "TurnOnMicrowave":           "v1.0/target/atomic/TurnOnMicrowave/20250813",
    "TurnOnSinkFaucet":          "v1.0/target/atomic/TurnOnSinkFaucet/20250812",
}

# Composite multi-step tasks (target/human split, 32 tasks).
# Includes both ``composite_seen`` (16) and ``composite_unseen`` (16) — these
# are *training* data; ``unseen`` only refers to the eval task list.
_TARGET_HUMAN_COMPOSITE = {
    "ArrangeBreadBasket":   "v1.0/target/composite/ArrangeBreadBasket/20250809",
    "ArrangeTea":           "v1.0/target/composite/ArrangeTea/20250812",
    "BreadSelection":       "v1.0/target/composite/BreadSelection/20250815",
    "CategorizeCondiments": "v1.0/target/composite/CategorizeCondiments/20250814",
    "CuttingToolSelection": "v1.0/target/composite/CuttingToolSelection/20250814",
    "DeliverStraw":         "v1.0/target/composite/DeliverStraw/20250813",
    "GarnishPancake":       "v1.0/target/composite/GarnishPancake/20250815",
    "GatherTableware":      "v1.0/target/composite/GatherTableware/20250815",
    "GetToastedBread":      "v1.0/target/composite/GetToastedBread/20250812",
    "HeatKebabSandwich":    "v1.0/target/composite/HeatKebabSandwich/20250813",
    "KettleBoiling":        "v1.0/target/composite/KettleBoiling/20250814",
    "LoadDishwasher":       "v1.0/target/composite/LoadDishwasher/20250811",
    "MakeIceLemonade":      "v1.0/target/composite/MakeIceLemonade/20250813",
    "PackIdenticalLunches": "v1.0/target/composite/PackIdenticalLunches/20250815",
    "PanTransfer":          "v1.0/target/composite/PanTransfer/20250817",
    "PortionHotDogs":       "v1.0/target/composite/PortionHotDogs/20250816",
    "PreSoakPan":           "v1.0/target/composite/PreSoakPan/20250809",
    "PrepareCoffee":        "v1.0/target/composite/PrepareCoffee/20250812",
    "RecycleBottlesByType": "v1.0/target/composite/RecycleBottlesByType/20250812",
    "RinseSinkBasin":       "v1.0/target/composite/RinseSinkBasin/20250816",
    "ScrubCuttingBoard":    "v1.0/target/composite/ScrubCuttingBoard/20250816",
    "SearingMeat":          "v1.0/target/composite/SearingMeat/20250812",
    "SeparateFreezerRack":  "v1.0/target/composite/SeparateFreezerRack/20250815",
    "SetUpCuttingStation":  "v1.0/target/composite/SetUpCuttingStation/20250817",
    "StackBowlsCabinet":    "v1.0/target/composite/StackBowlsCabinet/20250815",
    "SteamInMicrowave":     "v1.0/target/composite/SteamInMicrowave/20250814",
    "StirVegetables":       "v1.0/target/composite/StirVegetables/20250814",
    "StoreLeftoversInBowl": "v1.0/target/composite/StoreLeftoversInBowl/20250813",
    "WaffleReheat":         "v1.0/target/composite/WaffleReheat/20250817",
    "WashFruitColander":    "v1.0/target/composite/WashFruitColander/20250811",
    "WashLettuce":          "v1.0/target/composite/WashLettuce/20250814",
    "WeighIngredients":     "v1.0/target/composite/WeighIngredients/20250812",
}


def _entries(path_dict):
    """Build mixture entries (relpath/lerobot, weight=1.0, robot_tag) from a path dict."""
    return [(f"{p}/lerobot", 1.0, _ROBOT_TAG) for p in path_dict.values()]


DATASET_NAMED_MIXTURES = {
    # ------- minimal walk-through mixture (1 atomic task) -------
    "robocasa365_open_drawer_target_human": [
        ("v1.0/target/atomic/OpenDrawer/20250816/lerobot", 1.0, _ROBOT_TAG),
    ],
    # ------- full mixtures (each task weighted 1.0; equal sampling per task) -------
    "robocasa365_atomic_target_human_all":    _entries(_TARGET_HUMAN_ATOMIC),
    "robocasa365_composite_target_human_all": _entries(_TARGET_HUMAN_COMPOSITE),
    "robocasa365_target_human_all":           _entries({**_TARGET_HUMAN_ATOMIC,
                                                       **_TARGET_HUMAN_COMPOSITE}),
}
