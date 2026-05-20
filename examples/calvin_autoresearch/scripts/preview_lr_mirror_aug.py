#!/usr/bin/env python
"""Save CALVIN left/right mirror preview images and language records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from starVLA.dataloader.lerobot_datasets import get_vla_dataset


TASK_SWAP = {
    "move_slider_left": "move_slider_right",
    "move_slider_right": "move_slider_left",
    "push_red_block_left": "push_red_block_right",
    "push_red_block_right": "push_red_block_left",
    "push_blue_block_left": "push_blue_block_right",
    "push_blue_block_right": "push_blue_block_left",
    "push_pink_block_left": "push_pink_block_right",
    "push_pink_block_right": "push_pink_block_left",
}


def swap_left_right_text(text: str) -> str:
    """Swap standalone left/right tokens while preserving common casing."""

    placeholder_left = "__LRMIRROR_LEFT__"
    placeholder_right = "__LRMIRROR_RIGHT__"

    def left_repl(match: re.Match) -> str:
        value = match.group(0)
        if value.isupper():
            return placeholder_left.upper()
        if value.istitle():
            return placeholder_left.title()
        return placeholder_left

    def right_repl(match: re.Match) -> str:
        value = match.group(0)
        if value.isupper():
            return placeholder_right.upper()
        if value.istitle():
            return placeholder_right.title()
        return placeholder_right

    out = re.sub(r"\bleft\b", left_repl, text, flags=re.IGNORECASE)
    out = re.sub(r"\bright\b", right_repl, out, flags=re.IGNORECASE)
    out = out.replace(placeholder_left, "right")
    out = out.replace(placeholder_right, "left")
    out = out.replace(placeholder_left.upper(), "RIGHT")
    out = out.replace(placeholder_right.upper(), "LEFT")
    out = out.replace(placeholder_left.title(), "Right")
    out = out.replace(placeholder_right.title(), "Left")
    return out


def _set_runtime_aug_disabled(cfg):
    cfg = cfg.copy()
    if "language_augmentation" in cfg.datasets.vla_data:
        cfg.datasets.vla_data.language_augmentation.enabled = False
    if "image_augmentation" in cfg.datasets.vla_data:
        cfg.datasets.vla_data.image_augmentation.enabled = False
    return cfg


def _to_image(frame) -> Image.Image:
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if np.issubdtype(frame.dtype, np.floating):
        if float(np.nanmax(frame)) <= 1.5:
            frame = frame * 255.0
    return Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).convert("RGB")


def _save_image(image: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _mirror_image(image: Image.Image) -> Image.Image:
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def _candidate_trajectory_indices(dataset, wanted: set[str]):
    for trajectory_index, canonical_task in enumerate(dataset.trajectory_canonical_tasks):
        task = str(canonical_task)
        if task in wanted:
            yield trajectory_index, task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tasks", nargs="*", default=sorted(TASK_SWAP.keys()))
    parser.add_argument("--max-per-task", type=int, default=3)
    parser.add_argument("--flip-wrist", action="store_true")
    parser.add_argument("--frame-position", choices=["start", "middle"], default="middle")
    parser.add_argument("--max-scan", type=int, default=20000)
    args = parser.parse_args()

    cfg = _set_runtime_aug_disabled(OmegaConf.load(args.config))
    mixture = get_vla_dataset(cfg.datasets.vla_data)
    dataset = mixture.datasets[0]
    video_keys = list(dataset.modality_keys["video"])
    language_key = dataset.modality_keys["language"][0]

    wanted = set(args.tasks)
    saved = {task: 0 for task in wanted}
    records = []
    scanned = 0
    for trajectory_index, task in _candidate_trajectory_indices(dataset, wanted):
        scanned += 1
        if scanned > args.max_scan:
            break
        if saved[task] >= args.max_per_task:
            continue

        trajectory_id = int(dataset.trajectory_ids[trajectory_index])
        trajectory_len = int(dataset.trajectory_lengths[trajectory_index])
        base_index = 0 if args.frame_position == "start" else max(0, trajectory_len // 2)
        raw = dataset.get_step_data(trajectory_id, base_index)
        original_language = str(raw[language_key][0])
        mirrored_task = TASK_SWAP[task]
        mirrored_language = swap_left_right_text(original_language)

        item = saved[task]
        task_dir = args.output / task
        record = {
            "trajectory_id": trajectory_id,
            "base_index": base_index,
            "task": task,
            "mirrored_task": mirrored_task,
            "original_language": original_language,
            "mirrored_language": mirrored_language,
            "flip_wrist": bool(args.flip_wrist),
            "images": {},
        }

        for key in video_keys:
            image = _to_image(raw[key])
            should_flip = key == "video.primary_image" or (key == "video.wrist_image" and args.flip_wrist)
            mirrored = _mirror_image(image) if should_flip else image.copy()
            safe_key = key.replace("video.", "")
            before_path = task_dir / f"{item:02d}_{trajectory_id}_{safe_key}_before.jpg"
            after_path = task_dir / f"{item:02d}_{trajectory_id}_{safe_key}_after.jpg"
            _save_image(image, before_path)
            _save_image(mirrored, after_path)
            record["images"][key] = {
                "before": str(before_path),
                "after": str(after_path),
                "flipped": bool(should_flip),
            }

        records.append(record)
        saved[task] += 1
        if all(count >= args.max_per_task for count in saved.values()):
            break

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"saved {len(records)} mirror preview records to {args.output}")
    print(saved)
    print(json.dumps(records[: min(8, len(records))], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
