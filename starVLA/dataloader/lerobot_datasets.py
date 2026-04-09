# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025].
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [suport topdowm processing, suport param from config].

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.registry import (
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    DATASET_NAMED_MIXTURES,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


def collate_fn(batch):
    return batch


def make_padding_collate_fn(action_dim: int, action_horizon: int, state_dim: int | None = None):
    """Create a collate_fn that pads action (and optionally state) to uniform dimensions.

    Pads with zeros on the dim axis (right) and chunk/time axis (end).
    Raises ValueError if the source dimensions exceed the target dimensions.

    Args:
        action_dim: Target action dimension (second axis).
        action_horizon: Target action chunk length (first axis).
        state_dim: Target state dimension. If None, state is not padded.
    """

    def _pad_array(arr: np.ndarray, target_time: int, target_dim: int, name: str) -> np.ndarray:
        """Pad a [T, D] array to [target_time, target_dim] with zeros."""
        t, d = arr.shape
        if d > target_dim:
            raise ValueError(
                f"{name} dim ({d}) exceeds target dim ({target_dim}). "
                f"Check your config or dataset — source data should not be wider than the target."
            )
        if t > target_time:
            raise ValueError(
                f"{name} chunk length ({t}) exceeds target chunk length ({target_time}). "
                f"Check your config or dataset — source data should not be longer than the target."
            )
        if t == target_time and d == target_dim:
            return arr
        padded = np.zeros((target_time, target_dim), dtype=arr.dtype)
        padded[:t, :d] = arr
        return padded

    def padding_collate_fn(batch):
        for sample in batch:
            if "action" in sample:
                sample["action"] = _pad_array(sample["action"], action_horizon, action_dim, "action")
            if state_dim is not None and "state" in sample:
                state_time = sample["state"].shape[0]  # keep original time dim for state
                sample["state"] = _pad_array(sample["state"], state_time, state_dim, "state")
        return batch

    return padding_collate_fn


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
    lerobot_version: str | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param lerobot_version: Explicit lerobot version override ("v2.0" or "v3.0"). If None, auto-detected from dataset file structure.
    :return: A LeRobotSingleDataset object.
    """

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(
            f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default"
        )
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]

    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,  # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
        lerobot_version=lerobot_version,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append(
            (
                make_LeRobotSingleDataset(
                    Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg
                ),
                d_weight,
            )
        )

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )


if __name__ == "__main__":

    # import debugpy
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--data_mix",
        type=str,
        default=None,
        help="Override data_mix in config (e.g. libero_goal)",
    )
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()
    args.config_yaml = args.config_yaml  # use CLI arg or default
    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    if hasattr(args, 'data_mix') and args.data_mix:
        vla_dataset_cfg.data_mix = args.data_mix
    vla_dataset_cfg.task_id = "all"
    print(f"Config: {args.config_yaml}")
    print(f"Data mix: {vla_dataset_cfg.data_mix}")
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        # dataset
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm

    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch)
        # print(1)
        if count > 100:
            break
        count += 1
        pass
