#!/usr/bin/env python
"""Save before/after examples for CALVIN image augmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.dataloader.gr00t_lerobot.datasets import (
    _parse_image_aug_cfg,
    _resolve_image_aug_profile,
    canonicalize_calvin_task,
)


def _set_aug_enabled(cfg, enabled: bool):
    cfg = cfg.copy()
    if "image_augmentation" not in cfg.datasets.vla_data:
        cfg.datasets.vla_data.image_augmentation = {"enabled": enabled}
    else:
        cfg.datasets.vla_data.image_augmentation.enabled = enabled
    return cfg


def _save_image(image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image).astype(np.uint8))
    image.save(path)


def _mean_abs_diff(left, right) -> float:
    left_arr = np.asarray(left).astype(np.float32)
    right_arr = np.asarray(right).astype(np.float32)
    return float(np.abs(left_arr - right_arr).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tasks", nargs="*", default=["turn_off_lightbulb", "close_drawer", "move_slider_left", "turn_off_led"])
    parser.add_argument("--max-per-task", type=int, default=3)
    parser.add_argument("--max-scan", type=int, default=5000)
    parser.add_argument("--probability", type=float, default=None, help="Override augmentation probability for preview only.")
    args = parser.parse_args()

    cfg_aug = OmegaConf.load(args.config)
    if args.probability is not None:
        cfg_aug.datasets.vla_data.image_augmentation.probability = args.probability
    cfg_base = _set_aug_enabled(OmegaConf.load(args.config), False)

    ds_base = get_vla_dataset(cfg_base.datasets.vla_data)
    ds_aug = get_vla_dataset(cfg_aug.datasets.vla_data)
    image_aug_cfg = _parse_image_aug_cfg(cfg_aug.datasets.vla_data)
    video_keys = ds_base.datasets[0].modality_keys["video"] if hasattr(ds_base, "datasets") else []

    wanted = set(args.tasks)
    saved = {task: 0 for task in wanted}
    records = []
    for index in range(args.max_scan):
        base = ds_base[index]
        task = canonicalize_calvin_task(base["lang"])
        if task not in wanted or saved[task] >= args.max_per_task:
            continue
        aug = ds_aug[index]
        item = saved[task]
        diffs = []
        profiles = {}
        for cam_idx, (base_img, aug_img) in enumerate(zip(base["image"], aug["image"])):
            _save_image(base_img, args.output / task / f"{item:02d}_cam{cam_idx}_base.jpg")
            _save_image(aug_img, args.output / task / f"{item:02d}_cam{cam_idx}_aug.jpg")
            diffs.append(_mean_abs_diff(base_img, aug_img))
            if cam_idx < len(video_keys):
                key = video_keys[cam_idx]
                profile = _resolve_image_aug_profile(image_aug_cfg, task, key)
                profiles[key] = {
                    "photometric": bool(profile.get("photometric", True)),
                    "crop_translate": bool(profile.get("crop_translate", True)),
                    "max_translate_ratio": float(profile.get("max_translate_ratio", 0.0)),
                    "scale_range": list(profile.get("scale_range", [])),
                }
        records.append(
            {
                "index": index,
                "task": task,
                "language": base["lang"],
                "mean_abs_diff": diffs,
                "profiles": profiles,
            }
        )
        saved[task] += 1
        if all(count >= args.max_per_task for count in saved.values()):
            break

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"saved preview to {args.output}")
    print(saved)


if __name__ == "__main__":
    main()
