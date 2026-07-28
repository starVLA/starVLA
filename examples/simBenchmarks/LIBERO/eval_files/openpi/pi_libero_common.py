#!/usr/bin/env python3
"""Shared utilities for StarVLA PI0/PI05 LIBERO eval."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deployment.model_server.tools import image_tools

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


DEFAULT_LIBERO_HOME = _env_path("LIBERO_HOME", "third_party/libero")
DEFAULT_ASSETS_ROOT = _env_path("OPENPI_ASSETS_ROOT", "openpi_libero/torch")
DEFAULT_CONVERTED_ROOT = _env_path("OPENPI_CONVERTED_ROOT", "openpi_converted_protocol")
DEFAULT_TOKENIZER = _env_path("PALIGEMMA_TOKENIZER", "paligemma_tokenizer.model")
OPENPI_MODEL_SOURCE = "openpi"
STARVLA_MODEL_SOURCE = "starvla"


def configure_torch() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def get_max_steps(task_suite: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[task_suite]


def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def apply_legacy_eval_gripper_binarize(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"Expected action chunk with shape [T, >=7], got {actions.shape}")
    for i in range(actions.shape[0]):
        actions[i, 6:7] = binarize_gripper_open(actions[i, 6:7])
    return actions


def load_norm_stats(assets_checkpoint: Path) -> dict[str, dict[str, np.ndarray]]:
    stats_path = assets_checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing LIBERO norm stats: {stats_path}")
    raw = json.loads(stats_path.read_text(encoding="utf-8"))["norm_stats"]
    return {
        key: {stat_name: np.asarray(value, dtype=np.float32) for stat_name, value in stats.items()}
        for key, stats in raw.items()
    }


def canonicalize_model_name(model_name: str) -> str:
    model_name = model_name.upper()
    if model_name == "PI5":
        model_name = "PI05"
    if model_name not in {"PI0", "PI05"}:
        raise ValueError(f"Unknown model: {model_name}")
    return model_name


def _libero_stats_key(model_name: str) -> str:
    return f"openpi_{model_name.lower()}_libero_franka"


def load_dataset_statistics(stats_path: Path, model_name: str) -> dict[str, dict[str, np.ndarray]]:
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing dataset statistics: {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    key = _libero_stats_key(model_name)
    if key not in raw:
        if len(raw) != 1:
            raise KeyError(f"Expected key {key!r} in {stats_path}, found {list(raw.keys())}")
        key = next(iter(raw.keys()))
    item = raw[key]
    return {
        "state": {stat_name: np.asarray(value, dtype=np.float32) for stat_name, value in item["state"].items()},
        "actions": {stat_name: np.asarray(value, dtype=np.float32) for stat_name, value in item["action"].items()},
    }


def normalize_zscore(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    mean = stats["mean"][..., : x.shape[-1]]
    std = stats["std"][..., : x.shape[-1]]
    return (x - mean) / (std + 1e-6)


def unnormalize_zscore(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    dims = stats["mean"].shape[-1]
    x_main = x[..., :dims] * (stats["std"] + 1e-6) + stats["mean"]
    if dims < x.shape[-1]:
        return np.concatenate([x_main, x[..., dims:]], axis=-1)
    return x_main


def normalize_quantile(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    q01 = stats["q01"][..., : x.shape[-1]]
    q99 = stats["q99"][..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    q01 = stats["q01"]
    q99 = stats["q99"]
    dims = q01.shape[-1]
    x_main = (x[..., :dims] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    if dims < x.shape[-1]:
        return np.concatenate([x_main, x[..., dims:]], axis=-1)
    return x_main


def normalize_openpi_value(x: np.ndarray, stats: dict[str, np.ndarray], use_quantile_norm: bool) -> np.ndarray:
    if use_quantile_norm:
        return normalize_quantile(x, stats)
    return normalize_zscore(x, stats)


def unnormalize_openpi_value(x: np.ndarray, stats: dict[str, np.ndarray], use_quantile_norm: bool) -> np.ndarray:
    if use_quantile_norm:
        return unnormalize_quantile(x, stats)
    return unnormalize_zscore(x, stats)


def default_converted_checkpoint(model_name: str, precision: str = "float32") -> Path:
    model_name = canonicalize_model_name(model_name)
    variant = "bfloat16" if precision == "bfloat16" else "fp32"
    stem = "pi05_libero_starvla" if model_name == "PI05" else "pi0_libero_starvla"
    return DEFAULT_CONVERTED_ROOT / stem / variant / "model.safetensors"


def default_assets_checkpoint(model_name: str) -> Path:
    model_name = canonicalize_model_name(model_name)
    converted_root = DEFAULT_CONVERTED_ROOT / ("pi05_libero_starvla" if model_name == "PI05" else "pi0_libero_starvla")
    converted_norm = converted_root / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    if converted_norm.exists():
        return converted_root
    return DEFAULT_ASSETS_ROOT / ("pi05_libero" if model_name == "PI05" else "pi0_libero")


def canonicalize_model_source(model_source: str) -> str:
    model_source = str(model_source).strip().lower()
    if model_source not in {OPENPI_MODEL_SOURCE, STARVLA_MODEL_SOURCE}:
        raise ValueError(f"Unknown model source: {model_source}")
    return model_source


def resolve_norm_stats_source(
    model_name: str,
    checkpoint: Path | None = None,
    assets_checkpoint: Path | None = None,
):
    if assets_checkpoint is not None:
        dataset_stats = assets_checkpoint / "dataset_statistics.json"
        if dataset_stats.exists():
            return load_dataset_statistics(dataset_stats, model_name)
        return load_norm_stats(assets_checkpoint)

    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        run_dir = ckpt_path.parents[1] if ckpt_path.is_file() else ckpt_path
        dataset_stats = run_dir / "dataset_statistics.json"
        if dataset_stats.exists():
            return load_dataset_statistics(dataset_stats, model_name)
        assets_norm = run_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
        if assets_norm.exists():
            return load_norm_stats(run_dir)

    return load_norm_stats(default_assets_checkpoint(model_name))


def build_model(checkpoint: Path, device: torch.device, precision: str = "float32"):
    from starVLA.model.framework.base_framework import baseframework

    model = baseframework.from_pretrained(str(checkpoint))
    if hasattr(model, "to_precision"):
        model.to_precision("bfloat16" if precision == "bfloat16" else "float32")
    else:
        model = model.to(dtype=torch.bfloat16 if precision == "bfloat16" else torch.float32)
    return model.to(device).eval()


def postprocess_openpi_actions(
    normalized_actions: np.ndarray,
    raw_state: np.ndarray,
    norm_stats: dict[str, dict[str, np.ndarray]],
    model_name: str,
    model_source: str,
) -> np.ndarray:
    model_name = canonicalize_model_name(model_name)
    model_source = canonicalize_model_source(model_source)
    use_quantile_norm = model_name != "PI0"
    actions = unnormalize_openpi_value(
        np.asarray(normalized_actions, dtype=np.float32),
        norm_stats["actions"],
        use_quantile_norm,
    )[:, :7]
    if model_name == "PI0":
        actions[:, :6] += np.asarray(raw_state, dtype=np.float32)[:6]
    if model_source == STARVLA_MODEL_SOURCE:
        actions = apply_legacy_eval_gripper_binarize(actions)
    return actions.astype(np.float32)


def preprocess_env_obs(obs: dict[str, Any], task_description: str, resize_size: int):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img[None], resize_size, resize_size)[0])
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img[None], resize_size, resize_size)[0])
    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            quat2axisangle(obs["robot0_eef_quat"]).reshape(-1),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        )
    ).astype(np.float32)
    if state.shape != (8,):
        raise ValueError(f"Expected LIBERO state shape (8,), got {state.shape}")
    return {"image": [img, wrist_img], "lang": str(task_description), "raw_state": state}, img
