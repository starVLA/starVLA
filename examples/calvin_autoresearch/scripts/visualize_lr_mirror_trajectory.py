#!/usr/bin/env python
"""Visualize CALVIN left/right mirror augmentation with trajectory overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LR_MIRROR_TASK_SWAP,
    _parse_lr_mirror_cfg,
    apply_calvin_lr_mirror,
)
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


DEFAULT_CONFIG = "examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang_taskaug_lrmirror.yaml"
DEFAULT_TASKS = tuple(sorted(LR_MIRROR_TASK_SWAP.keys()))
VIDEO_PRIMARY = "video.primary_image"
VIDEO_WRIST = "video.wrist_image"
LANGUAGE_KEY = "annotation.human.action.task_description"
ACTION_KEYS = ("action.x", "action.y", "action.z", "action.roll", "action.pitch", "action.yaw", "action.gripper")
STATE_KEYS = ("state.x", "state.y")


def _load_font(size: int = 14):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT = _load_font(14)
SMALL_FONT = _load_font(11)


def _to_image(frame) -> Image.Image:
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if np.issubdtype(frame.dtype, np.floating) and float(np.nanmax(frame)) <= 1.5:
        frame = frame * 255.0
    return Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).convert("RGB")


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _series(raw: dict, key: str) -> np.ndarray:
    if key not in raw:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(raw[key], dtype=np.float32).reshape(-1)


def _extract_target(task: str) -> tuple[str, str | None]:
    for color in ("red", "blue", "pink"):
        if f"push_{color}_block" in task:
            return f"{color} block", color
    if "slider" in task:
        return "slider door/handle", None
    return task, None


def _target_mask(image: Image.Image, color: str) -> np.ndarray:
    arr = np.asarray(image).astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    if color == "red":
        mask = (r > 135) & (g < 95) & (b < 95) & ((r - g) > 55) & ((r - b) > 55)
    elif color == "blue":
        mask = (b > 95) & ((b - r) > 35) & ((b - g) > 35)
    elif color == "pink":
        mask = (r > 125) & (b > 95) & (g < 125) & ((r - g) > 45) & ((b - g) > 25) & (np.abs(r - b) < 90)
    else:
        mask = np.zeros(arr.shape[:2], dtype=bool)
    return mask


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) < 20:
        return None
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = None
    best_area = 0
    for y0, x0 in zip(*np.where(mask & ~visited)):
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        visited[y0, x0] = True
        xs = []
        ys = []
        while stack:
            y, x = stack.pop()
            xs.append(x)
            ys.append(y)
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        area = len(xs)
        if area < 20:
            continue
        bbox = (min(xs), min(ys), max(xs), max(ys))
        bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        if bbox_area > width * height * 0.35:
            continue
        if area > best_area:
            best = bbox
            best_area = area
    return best


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(255, 255, 255), bg=(0, 0, 0)):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=SMALL_FONT)
    draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=bg)
    draw.text((x, y), text, font=SMALL_FONT, fill=fill)


def _annotate_image(image: Image.Image, task: str, language: str, is_mirror: bool) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    target, color = _extract_target(task)
    prefix = "MIRROR" if is_mirror else "ORIG"
    _draw_label(draw, (5, 5), f"{prefix} | {task}")
    _draw_label(draw, (5, 23), f"target: {target}")
    _draw_label(draw, (5, out.height - 17), language[:70])
    if color is not None:
        bbox = _bbox_from_mask(_target_mask(out, color))
        if bbox is not None:
            draw.rectangle(bbox, outline=(255, 240, 0), width=3)
            _draw_label(draw, (bbox[0], max(0, bbox[1] - 15)), target, fill=(0, 0, 0), bg=(255, 240, 0))
        else:
            _draw_label(draw, (5, 41), "color bbox: not found", fill=(255, 220, 220), bg=(80, 0, 0))
    return out


def _plot_xy(points: np.ndarray, title: str, size=(260, 200), line_color=(40, 120, 255)) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, (250, 250, 250))
    draw = ImageDraw.Draw(image)
    margin = 24
    draw.rectangle((margin, margin, width - margin, height - margin), outline=(190, 190, 190))
    draw.text((6, 5), title, font=SMALL_FONT, fill=(0, 0, 0))
    if points.shape[0] < 2:
        draw.text((35, height // 2), "not enough points", font=SMALL_FONT, fill=(120, 0, 0))
        return image
    x = points[:, 0]
    y = points[:, 1]
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    if abs(x_max - x_min) < 1e-6:
        x_min -= 1.0
        x_max += 1.0
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0
    pad_x = (x_max - x_min) * 0.08
    pad_y = (y_max - y_min) * 0.08
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    def to_px(point):
        px = margin + (float(point[0]) - x_min) / (x_max - x_min) * (width - 2 * margin)
        py = height - margin - (float(point[1]) - y_min) / (y_max - y_min) * (height - 2 * margin)
        return int(round(px)), int(round(py))

    pixels = [to_px(point) for point in points]
    draw.line(pixels, fill=line_color, width=3)
    r = 4
    draw.ellipse((pixels[0][0] - r, pixels[0][1] - r, pixels[0][0] + r, pixels[0][1] + r), fill=(20, 170, 60))
    draw.ellipse((pixels[-1][0] - r, pixels[-1][1] - r, pixels[-1][0] + r, pixels[-1][1] + r), fill=(220, 40, 40))
    draw.text((6, height - 17), f"x [{x_min:.2f},{x_max:.2f}] y [{y_min:.2f},{y_max:.2f}]", font=SMALL_FONT, fill=(80, 80, 80))
    return image


def _action_points(raw: dict) -> np.ndarray:
    dx = _series(raw, "action.x")
    dy = _series(raw, "action.y")
    if dx.size == 0 or dy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    n = min(dx.size, dy.size)
    points = np.stack([dx[:n], dy[:n]], axis=1)
    return np.concatenate([np.zeros((1, 2), dtype=np.float32), np.cumsum(points, axis=0)], axis=0)


def _episode_state_points(dataset, trajectory_id: int, trajectory_len: int, mirror_cfg: dict | None = None) -> np.ndarray:
    xs = []
    ys = []
    for index in range(trajectory_len):
        raw = dataset.get_step_data(trajectory_id, index)
        x = _scalar(raw["state.x"])
        y = _scalar(raw["state.y"])
        if mirror_cfg is not None:
            center = float(mirror_cfg.get("state_transform", {}).get("x_center", 0.03991219401359558))
            x = 2.0 * center - x
        xs.append(x)
        ys.append(y)
    return np.stack([np.asarray(xs), np.asarray(ys)], axis=1)


def _make_sample_sheet(
    original_raw: dict,
    mirrored_raw: dict,
    task: str,
    mirrored_task: str,
    original_state_points: np.ndarray,
    mirrored_state_points: np.ndarray,
    trajectory_id: int,
    base_index: int,
) -> Image.Image:
    original_language = str(original_raw[LANGUAGE_KEY][0])
    mirrored_language = str(mirrored_raw[LANGUAGE_KEY][0])

    primary_before = _annotate_image(_to_image(original_raw[VIDEO_PRIMARY]), task, original_language, False)
    primary_after = _annotate_image(_to_image(mirrored_raw[VIDEO_PRIMARY]), mirrored_task, mirrored_language, True)
    wrist_before = _annotate_image(_to_image(original_raw[VIDEO_WRIST]).resize((200, 200)), task, original_language, False)
    wrist_after = _annotate_image(_to_image(mirrored_raw[VIDEO_WRIST]).resize((200, 200)), mirrored_task, mirrored_language, True)

    action_before = _plot_xy(_action_points(original_raw), "action chunk cumulative x/y", size=(260, 200), line_color=(30, 120, 255))
    action_after = _plot_xy(_action_points(mirrored_raw), "mirrored action chunk x/y", size=(260, 200), line_color=(255, 120, 30))
    state_before = _plot_xy(original_state_points, "episode state x/y", size=(260, 200), line_color=(30, 120, 255))
    state_after = _plot_xy(mirrored_state_points, "mirrored state x/y", size=(260, 200), line_color=(255, 120, 30))

    title_h = 54
    gutter = 12
    col_w = 200 + gutter + 200 + gutter + 260
    width = col_w * 2 + gutter
    height = title_h + 200 + gutter + 200
    sheet = Image.new("RGB", (width, height), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (10, 8),
        f"trajectory_id={trajectory_id} base_index={base_index} | {task} -> {mirrored_task}",
        font=FONT,
        fill=(0, 0, 0),
    )
    draw.text((10, 30), f"language: {original_language} -> {mirrored_language}", font=SMALL_FONT, fill=(40, 40, 40))
    draw.text((10, title_h - 14), "green dot=start, red dot=end. Action/state plots use robot-frame x/y, not camera projection.", font=SMALL_FONT, fill=(80, 80, 80))

    x0 = 0
    y0 = title_h
    for image, dx, dy in (
        (primary_before, x0, y0),
        (wrist_before, x0 + 200 + gutter, y0),
        (action_before, x0 + 2 * (200 + gutter), y0),
        (primary_after, col_w + gutter, y0),
        (wrist_after, col_w + gutter + 200 + gutter, y0),
        (action_after, col_w + gutter + 2 * (200 + gutter), y0),
        (state_before, x0 + 2 * (200 + gutter), y0 + 200 + gutter),
        (state_after, col_w + gutter + 2 * (200 + gutter), y0 + 200 + gutter),
    ):
        sheet.paste(image, (dx, dy))
    return sheet


def _candidate_trajectory_indices(dataset, wanted: set[str]):
    for trajectory_index, canonical_task in enumerate(dataset.trajectory_canonical_tasks):
        task = str(canonical_task)
        if task in wanted:
            yield trajectory_index, task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    parser.add_argument("--max-per-task", type=int, default=1)
    parser.add_argument("--frame-position", choices=["start", "middle"], default="middle")
    parser.add_argument("--max-scan", type=int, default=20000)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if "language_augmentation" in cfg.datasets.vla_data:
        cfg.datasets.vla_data.language_augmentation.enabled = False
    if "image_augmentation" in cfg.datasets.vla_data:
        cfg.datasets.vla_data.image_augmentation.enabled = False

    mixture = get_vla_dataset(cfg.datasets.vla_data)
    dataset = mixture.datasets[0]
    video_keys = list(dataset.modality_keys["video"])
    language_keys = list(dataset.modality_keys["language"])
    mirror_cfg = _parse_lr_mirror_cfg(cfg.datasets.vla_data)
    mirror_cfg["probability"] = 1.0

    wanted = set(args.tasks)
    saved = {task: 0 for task in wanted}
    records = []
    sheets = []
    scanned = 0
    args.output.mkdir(parents=True, exist_ok=True)
    for trajectory_index, task in _candidate_trajectory_indices(dataset, wanted):
        scanned += 1
        if scanned > args.max_scan:
            break
        if saved[task] >= args.max_per_task:
            continue

        trajectory_id = int(dataset.trajectory_ids[trajectory_index])
        trajectory_len = int(dataset.trajectory_lengths[trajectory_index])
        base_index = 0 if args.frame_position == "start" else max(0, trajectory_len // 2)
        original_raw = dataset.get_step_data(trajectory_id, base_index)
        mirrored_raw, mirrored_task = apply_calvin_lr_mirror(
            original_raw,
            video_keys=video_keys,
            language_keys=language_keys,
            canonical_task=task,
            rng=np.random.default_rng(0),
            cfg=mirror_cfg,
        )
        original_state_points = _episode_state_points(dataset, trajectory_id, trajectory_len)
        mirrored_state_points = _episode_state_points(dataset, trajectory_id, trajectory_len, mirror_cfg=mirror_cfg)

        sheet = _make_sample_sheet(
            original_raw,
            mirrored_raw,
            task,
            mirrored_task,
            original_state_points,
            mirrored_state_points,
            trajectory_id,
            base_index,
        )
        safe_name = f"{task}_{saved[task]:02d}_tid{trajectory_id}.jpg"
        sheet_path = args.output / safe_name
        sheet.save(sheet_path, quality=92)
        sheets.append(sheet)

        record = {
            "trajectory_index": trajectory_index,
            "trajectory_id": trajectory_id,
            "trajectory_len": trajectory_len,
            "base_index": base_index,
            "task": task,
            "mirrored_task": mirrored_task,
            "original_language": str(original_raw[LANGUAGE_KEY][0]),
            "mirrored_language": str(mirrored_raw[LANGUAGE_KEY][0]),
            "sample_sheet": str(sheet_path),
            "action_x_before": _series(original_raw, "action.x").tolist(),
            "action_x_after": _series(mirrored_raw, "action.x").tolist(),
            "action_y_before": _series(original_raw, "action.y").tolist(),
            "action_y_after": _series(mirrored_raw, "action.y").tolist(),
        }
        records.append(record)
        saved[task] += 1
        if all(count >= args.max_per_task for count in saved.values()):
            break

    if sheets:
        width = max(sheet.width for sheet in sheets)
        height = sum(sheet.height for sheet in sheets) + 12 * (len(sheets) - 1)
        contact = Image.new("RGB", (width, height), (225, 225, 225))
        y = 0
        for sheet in sheets:
            contact.paste(sheet, (0, y))
            y += sheet.height + 12
        contact.save(args.output / "contact_sheet.jpg", quality=90)

    (args.output / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"saved {len(records)} samples to {args.output}")
    print("contact_sheet:", args.output / "contact_sheet.jpg")
    print(json.dumps(records[: min(4, len(records))], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
