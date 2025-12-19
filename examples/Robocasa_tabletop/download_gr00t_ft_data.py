from huggingface_hub import list_repo_files, hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import random


REPO_ID = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
REPO_TYPE = "dataset"

LOCAL_DIR = "./playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"

FOLDERS = [
    "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
]


def download_with_retry(
    filename: str,
    max_retries: int = 50,
) -> bool:
    """
    Download a single file from Hugging Face Hub with retry logic.

    Args:
        filename: Path of the file in the repository.
        max_retries: Maximum number of retry attempts.

    Returns:
        True if download succeeds, False otherwise.
    """
    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=filename,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
            )
            return True
        except Exception as exc:
            wait_time = random.uniform(1.0, 5.0) * attempt / 10.0
            print(
                f"[{attempt}/{max_retries}] "
                f"Download failed: {filename} ({exc}); "
                f"retrying in {wait_time:.1f}s"
            )
            time.sleep(wait_time)

    print(f"Giving up after {max_retries} retries: {filename}")
    return False


def main() -> None:
    print("Listing all files in the repository...")
    all_files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

    target_files = [
        f
        for f in all_files
        if any(f.startswith(folder + "/") for folder in FOLDERS)
    ]

    print(f"Found {len(target_files)} matching files to download.\n")

    max_workers = 32
    print(f"Starting parallel download with {max_workers} workers...\n")

    failed_files: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_with_retry, filename): filename
            for filename in target_files
        }

        for idx, future in enumerate(as_completed(futures), start=1):
            filename = futures[future]
            try:
                success = future.result()
                if success:
                    print(f"[{idx}/{len(target_files)}] Downloaded: {filename}")
                else:
                    failed_files.append(filename)
            except Exception as exc:
                print(f"Unexpected error for {filename}: {exc}")
                failed_files.append(filename)

    print("\nAll download attempts finished.")

    if failed_files:
        print(f"{len(failed_files)} files failed to download:")
        for filename in failed_files:
            print(f"  - {filename}")
    else:
        print("All files downloaded successfully.")


if __name__ == "__main__":
    main()
