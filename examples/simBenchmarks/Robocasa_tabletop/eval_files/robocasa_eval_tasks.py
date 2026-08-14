"""Task presets shared by RoboCasa eval launchers."""

from __future__ import annotations

from pathlib import Path


GR1_5_TASKS = [
    "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
]

GR1_24_TASKS = [
    *GR1_5_TASKS,
    "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
]

TASK_PRESETS = {
    "gr1_5": GR1_5_TASKS,
    "gr1_24": GR1_24_TASKS,
}


def task_slug(task: str) -> str:
    return task.replace("/", "_").removesuffix("_Env")


def load_tasks(*, preset: str, tasks_file: Path | None = None) -> list[str]:
    if tasks_file is not None:
        tasks = []
        for line in tasks_file.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item and not item.startswith("#"):
                tasks.append(item)
        if not tasks:
            raise ValueError(f"No tasks found in {tasks_file}")
        return tasks

    if preset not in TASK_PRESETS:
        raise ValueError(f"Unknown task preset {preset!r}; choose from {sorted(TASK_PRESETS)}")
    return list(TASK_PRESETS[preset])
