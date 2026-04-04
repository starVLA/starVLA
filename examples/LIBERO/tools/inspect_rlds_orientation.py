#!/usr/bin/env python3
"""
Sample modified LIBERO RLDS frames and save side-by-side:
- left: original
- right: rotated 180 degrees
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
from PIL import Image


def _resolve_builder_dir(data_root_dir: Path, dataset_name: str) -> Path:
    version_dir = data_root_dir / dataset_name / "1.0.0"
    if version_dir.exists():
        return version_dir
    suite_dir = data_root_dir / dataset_name
    if suite_dir.exists():
        return suite_dir
    raise FileNotFoundError(f"Dataset not found: {dataset_name} under {data_root_dir}")


def _as_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict):
        return record
    if hasattr(record, "keys"):
        return {key: record[key] for key in record.keys()}
    if isinstance(record, np.void) and record.dtype.names:
        return {key: record[key] for key in record.dtype.names}
    return {}


def _iter_steps(episode_steps: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(episode_steps, dict):
        keys = list(episode_steps.keys())
        if not keys:
            return
        length = len(episode_steps[keys[0]])
        for idx in range(length):
            yield {key: episode_steps[key][idx] for key in keys}
        return
    for step in episode_steps:
        yield _as_dict(step)


def _to_rgb_uint8(image: Any) -> np.ndarray | None:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3:
        return None
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="libero_spatial_no_noops")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--frames-per-episode", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/LIBERO/rlds_inspection/orientation",
    )
    args = parser.parse_args()

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This script requires tensorflow-datasets. Install with: "
            "`pip install tensorflow-cpu tensorflow-datasets`"
        ) from exc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builder_dir = _resolve_builder_dir(Path(args.data_root_dir), args.dataset_name)
    builder = tfds.builder_from_directory(str(builder_dir))
    dataset = builder.as_dataset(split=args.split, shuffle_files=False)

    saved = 0
    for ep_idx, episode in enumerate(tfds.as_numpy(dataset)):
        if ep_idx >= args.num_episodes:
            break
        episode = _as_dict(episode)
        steps = list(_iter_steps(episode.get("steps", [])))
        if not steps:
            continue

        stride = max(1, len(steps) // max(1, args.frames_per_episode))
        for step_idx in range(0, len(steps), stride):
            step = _as_dict(steps[step_idx])
            observation = _as_dict(step.get("observation", {}))
            for image_key in ("image", "wrist_image"):
                if image_key not in observation:
                    continue
                image = _to_rgb_uint8(observation[image_key])
                if image is None:
                    continue
                rotated = np.rot90(image, 2)
                stacked = np.concatenate([image, rotated], axis=1)
                save_path = out_dir / f"{args.dataset_name}__ep{ep_idx:03d}__step{step_idx:04d}__{image_key}.png"
                Image.fromarray(stacked).save(save_path)
                saved += 1

                if saved >= args.num_episodes * args.frames_per_episode * 2:
                    break
            if saved >= args.num_episodes * args.frames_per_episode * 2:
                break

    print(f"Saved {saved} comparison images to: {out_dir}")


if __name__ == "__main__":
    main()
