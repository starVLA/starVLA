import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from omegaconf import OmegaConf
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
        )
        num_workers = int(vla_dataset_cfg.get("num_workers", 4))
        dataloader_kwargs = {
            "batch_size": cfg.datasets.vla_data.per_device_batch_size,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": bool(vla_dataset_cfg.get("pin_memory", True)),
            # shuffle=True
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = bool(vla_dataset_cfg.get("persistent_workers", True))
            dataloader_kwargs["prefetch_factor"] = int(vla_dataset_cfg.get("prefetch_factor", 2))

        vla_train_dataloader = DataLoader(
            vla_dataset,
            **dataloader_kwargs,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "var_stage2_token_dataset":
        from starVLA.training.train_var_stage1 import load_starvla_base_config

        stage1_cfg = OmegaConf.load(cfg.framework.stage1_tokenizer.stage1_config)
        stage1_path = cfg.framework.stage1_tokenizer.get("artifact", None) or cfg.framework.stage1_tokenizer.get("checkpoint", None)
        if stage1_path is None:
            raise ValueError("QwenVAR requires framework.stage1_tokenizer.artifact or .checkpoint.")
        token_cache_path = cfg.framework.stage1_tokenizer.get("token_cache", None)
        if str(stage1_cfg.data.get("dataset_format", "starvla_lerobot")) == "robotwin_raw_zip":
            from starVLA.dataloader.robotwin_raw_stage2_token_dataset import RoboTwinRawStage2TokenDataset
            from starVLA.dataloader.var_stage2_token_dataset import collate_var_stage2_token_batch

            vla_dataset = RoboTwinRawStage2TokenDataset(
                stage1_cfg,
                stage1_artifact_path=stage1_path,
                token_cache_path=token_cache_path,
                mode=cfg.datasets.vla_data.get("mode", "train"),
                max_samples=cfg.datasets.vla_data.get("max_samples", None),
                sample_indices=cfg.datasets.vla_data.get("sample_indices", None),
            )
        else:
            from starVLA.dataloader.var_stage2_token_dataset import (
                VARStage2TokenDataset,
                collate_var_stage2_token_batch,
            )

            base_cfg = load_starvla_base_config(stage1_cfg)
            for optional_key in (
                "include_state",
                "obs_image_size",
                "data_root_dir",
                "data_mix",
                "video_backend",
                "load_all_data_for_training",
                "delete_pause_frame",
            ):
                if cfg.datasets.vla_data.get(optional_key, None) is not None:
                    base_cfg.datasets.vla_data[optional_key] = cfg.datasets.vla_data[optional_key]
            vla_dataset = VARStage2TokenDataset(
                base_cfg,
                stage1_artifact_path=stage1_path,
                token_cache_path=token_cache_path,
                mode=cfg.datasets.vla_data.get("mode", "train"),
                balance_dataset_weights=bool(cfg.datasets.vla_data.get("balance_dataset_weights", False)),
                balance_trajectory_weights=bool(cfg.datasets.vla_data.get("balance_trajectory_weights", False)),
                seed=int(cfg.get("seed", 42)),
                window_mode=str(cfg.datasets.vla_data.get("window_mode", "full")),
                max_samples=cfg.datasets.vla_data.get("max_samples", None),
                sample_indices=cfg.datasets.vla_data.get("sample_indices", None),
                skip_bad_samples=bool(cfg.datasets.vla_data.get("skip_bad_samples", False)),
                max_read_retries=int(cfg.datasets.vla_data.get("max_read_retries", 8)),
            )
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            if hasattr(vla_dataset, "dataset_statistics"):
                save_dataset_statistics(vla_dataset.dataset_statistics(), output_dir)
            else:
                vla_dataset.stage1_dataset.source_dataset.save_dataset_statistics(
                    output_dir / "dataset_statistics.json"
                )
        return DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_var_stage2_token_batch,
            num_workers=int(cfg.datasets.vla_data.get("num_workers", 4)),
            shuffle=bool(cfg.datasets.vla_data.get("shuffle", True)),
        )
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
