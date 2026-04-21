"""Fourier GR1 / Robocasa — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class FourierGr1ArmsWaistDataConfig:
    video_keys = ["video.ego_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand", "action.waist"]
    language_keys = ["annotation.human.coarse_action"]
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
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "fourier_gr1_arms_waist": EmbodimentTag.GR1,
}

DATASET_NAMED_MIXTURES = {
    "fourier_gr1_unified_1000": [
        ("gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
    ],
}
