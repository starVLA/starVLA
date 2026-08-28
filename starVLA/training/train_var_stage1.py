"""Train a VAR-style Stage 1 action tokenizer on StarVLA action chunks."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.modules.action_tokenizer import VARActionTokenizer, default_scales


def collate_action_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        "actions": torch.stack([item["actions"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
    }
    if "actions_raw" in batch[0]:
        output["actions_raw"] = torch.stack([item["actions_raw"] for item in batch], dim=0)
    return output


def load_starvla_base_config(cfg: Any) -> Any:
    base_path = cfg.data.starvla_config_yaml
    base_cfg = OmegaConf.load(base_path)
    base_cfg = apply_config_compat(base_cfg)

    if cfg.data.get("data_root_dir", None) is not None:
        base_cfg.datasets.vla_data.data_root_dir = cfg.data.data_root_dir
    if cfg.data.get("data_mix", None) is not None:
        base_cfg.datasets.vla_data.data_mix = cfg.data.data_mix
    if cfg.data.get("video_backend", None) is not None:
        base_cfg.datasets.vla_data.video_backend = cfg.data.video_backend
    for optional_key in (
        "action_mode",
        "action_mode_apply_keys",
        "action_mode_state_map",
        "include_state",
        "load_all_data_for_training",
        "delete_pause_frame",
        "obs_image_size",
    ):
        if cfg.data.get(optional_key, None) is not None:
            base_cfg.datasets.vla_data[optional_key] = cfg.data[optional_key]

    if cfg.data.get("expected_action_horizon", None) is not None:
        base_cfg.framework.action_model.action_horizon = int(cfg.data.expected_action_horizon)
    if cfg.data.get("expected_action_dim", None) is not None:
        base_cfg.framework.action_model.action_dim = int(cfg.data.expected_action_dim)

    # Stage 1 does not use image/language, but the underlying StarVLA dataset
    # currently expects those modalities. Keep the config faithful and discard
    # non-action fields in VARStage1ActionDataset.
    return base_cfg


def resolve_scales(scales_cfg: Any, seq_len: int) -> list[int]:
    if scales_cfg in (None, "auto"):
        return default_scales(seq_len)
    return [int(item) for item in scales_cfg]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_checkpoint(
    path: Path,
    *,
    model: VARActionTokenizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    cfg: Any,
    action_spec: Any,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.get_config(),
        "action_spec": action_spec.to_dict(),
        "token_order": action_spec.token_order,
        "history": history,
        "stage1_config": OmegaConf.to_container(cfg, resolve=True),
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_init_checkpoint(path: str, *, model: VARActionTokenizer, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load init checkpoint {checkpoint_path}: "
            f"missing keys={missing}, unexpected keys={unexpected}"
        )
    return checkpoint


def load_resume_checkpoint(
    path: str,
    *,
    model: VARActionTokenizer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load resume checkpoint {checkpoint_path}: "
            f"missing keys={missing}, unexpected keys={unexpected}"
        )
    if "optimizer_state_dict" not in checkpoint:
        raise KeyError(f"Resume checkpoint {checkpoint_path} does not contain optimizer_state_dict.")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_encoder_decoder_init(path: str, *, model: VARActionTokenizer, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    prefixes = (
        "encoder_",
        "decoder_",
    )
    selected = {
        key: value
        for key, value in source_state.items()
        if key.startswith(prefixes) and key in target_state and target_state[key].shape == value.shape
    }
    missing = sorted(key for key in target_state if key.startswith(prefixes) and key not in selected)
    model.load_state_dict(selected, strict=False)
    if missing:
        print(f"Skipped {len(missing)} encoder/decoder init keys due to missing or shape mismatch.")
    return checkpoint


@torch.no_grad()
def initialize_codebook_from_data(
    *,
    model: VARActionTokenizer,
    loader: DataLoader,
    device: torch.device,
    batches: int,
    noise_scale: float = 0.0,
) -> None:
    if batches <= 0:
        return

    model.eval()
    samples: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        actions = batch["actions"].to(device=device, dtype=torch.float32, non_blocking=True)
        latent_full = model.encode_features(actions)
        for scale in model.scales:
            z_scale = F.interpolate(latent_full, size=scale, mode="linear", align_corners=False)
            samples.append(z_scale.permute(0, 2, 1).reshape(-1, model.embed_dim).detach().cpu())
        if batch_index + 1 >= batches:
            break

    if not samples:
        raise RuntimeError("No latent samples were collected for codebook initialization.")

    latent_samples = torch.cat(samples, dim=0)
    if latent_samples.shape[0] < model.codebook_size:
        repeat_factor = (model.codebook_size + latent_samples.shape[0] - 1) // latent_samples.shape[0]
        latent_samples = latent_samples.repeat(repeat_factor, 1)

    if model.quantization_mode == "product_vq":
        group_dim = model.embed_dim // model.product_codebook_groups
        for group_idx, codebook_module in enumerate(model.product_codebooks):
            group_samples = latent_samples[:, group_idx * group_dim : (group_idx + 1) * group_dim]
            perm = torch.randperm(group_samples.shape[0])[: model.codebook_size]
            codebook = group_samples.index_select(0, perm).to(device=device, dtype=codebook_module.weight.dtype)
            if noise_scale > 0.0:
                codebook = codebook + float(noise_scale) * torch.randn_like(codebook)
            if model.normalize_codebook_for_lookup:
                codebook = F.normalize(codebook, dim=-1, eps=1e-6)
            codebook_module.weight.copy_(codebook)
    else:
        perm = torch.randperm(latent_samples.shape[0])[: model.codebook_size]
        codebook = latent_samples.index_select(0, perm).to(device=device, dtype=model.shared_codebook.weight.dtype)
        if noise_scale > 0.0:
            codebook = codebook + float(noise_scale) * torch.randn_like(codebook)
        if model.normalize_codebook_for_lookup:
            codebook = F.normalize(codebook, dim=-1, eps=1e-6)
        model.shared_codebook.weight.copy_(codebook)
    print(f"Initialized codebook from {latent_samples.shape[0]} latent samples over {batches} batches.")


def set_encoder_decoder_trainable(model: VARActionTokenizer, trainable: bool) -> None:
    prefixes = ("encoder_", "decoder_")
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = trainable


def weighted_dim_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dim_groups: dict[str, list[int]],
    gripper_weight: float = 1.0,
    group_weights: dict[str, float] | None = None,
    sample_weights: torch.Tensor | None = None,
    time_weights: torch.Tensor | None = None,
    weight_normalization: str = "mean",
) -> torch.Tensor:
    group_weights = dict(group_weights or {})
    if gripper_weight != 1.0 and "gripper" in dim_groups:
        group_weights.setdefault("gripper", float(gripper_weight))

    if not group_weights and sample_weights is None and time_weights is None:
        return F.mse_loss(pred, target)

    if weight_normalization not in {"mean", "none"}:
        raise ValueError(f"weight_normalization must be 'mean' or 'none', got {weight_normalization!r}.")

    dim_weights = torch.ones(pred.shape[-1], dtype=pred.dtype, device=pred.device)
    for group_name, group_weight in group_weights.items():
        if group_name not in dim_groups:
            raise ValueError(f"Unknown action dim group {group_name!r}. Available groups: {sorted(dim_groups)}")
        dim_weights[dim_groups[group_name]] = float(group_weight)

    weight = dim_weights.view(1, 1, -1)
    if sample_weights is not None:
        if sample_weights.ndim != 1 or sample_weights.shape[0] != pred.shape[0]:
            raise ValueError(
                f"Expected sample_weights with shape [{pred.shape[0]}], got {tuple(sample_weights.shape)}."
            )
        weight = weight * sample_weights.to(device=pred.device, dtype=pred.dtype).view(-1, 1, 1)
    if time_weights is not None:
        if time_weights.ndim != 1 or time_weights.shape[0] != pred.shape[1]:
            raise ValueError(f"Expected time_weights with shape [{pred.shape[1]}], got {tuple(time_weights.shape)}.")
        weight = weight * time_weights.to(device=pred.device, dtype=pred.dtype).view(1, -1, 1)

    err = (pred - target).pow(2)
    if weight_normalization == "none":
        denom = torch.as_tensor(err.numel(), device=pred.device, dtype=pred.dtype)
    else:
        denom = weight.expand_as(err).sum()
    return (err * weight).sum() / denom.clamp_min(1e-6)


def loss_group_weights(cfg: Any, prefix: str) -> dict[str, float]:
    """Read optional per-action-group loss weights from config.

    Supports keys such as ``position_recon_weight`` and
    ``position_vel_weight`` while preserving the older gripper-specific keys.
    """

    weights: dict[str, float] = {}
    for group_name in ("position", "rotation", "gripper"):
        key = f"{group_name}_{prefix}_weight"
        if cfg.loss.get(key, None) is not None:
            weights[group_name] = float(cfg.loss[key])
    return weights


def compute_vq_weight(epoch: int, *, target_weight: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return float(target_weight)
    return float(target_weight) * min(1.0, float(epoch + 1) / float(warmup_epochs))


def _simplify_dataset_name(dataset_name: str) -> str:
    name = dataset_name.split(".", 1)[-1]
    for suffix in ("_GR1ArmsAndWaistFourierHands_1000", "_GR1ArmsAndWaistFourierHands"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.startswith("Posttrain"):
        name = name[len("Posttrain") :]
    return name


def _is_robocasa_close_task(dataset_name: str) -> bool:
    task_name = _simplify_dataset_name(dataset_name)
    return task_name.startswith("PnP") and task_name.endswith("Close")


def _task_name_from_metadata(metadata: dict[str, Any]) -> str:
    explicit_task_name = str(metadata.get("task_name", "")).strip()
    if explicit_task_name:
        return explicit_task_name
    dataset_name = str(metadata.get("dataset_name", "unknown"))
    task_name = _simplify_dataset_name(dataset_name)
    return task_name or "unknown"


def _task_category_from_task_name(task_name: str) -> str:
    if "::task_" in task_name:
        return task_name.split("::", 1)[0]
    if task_name.startswith("PnP") and task_name.endswith("Close"):
        return "robocasa_close"
    match = re.search(r"From(.+?)To", task_name)
    if match:
        return f"from_{match.group(1)}"
    return "other"


def _new_reconstruction_stats() -> dict[str, dict[str, dict[str, Any]]]:
    return {"tasks": {}, "categories": {}}


def _new_reconstruction_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "mse_sum": 0.0,
        "mae_sum": 0.0,
        "max_mse": 0.0,
        "max_mae": 0.0,
        "group_mse_sum": {},
        "group_mae_sum": {},
    }


def _accumulate_reconstruction_bucket(
    bucket: dict[str, Any],
    *,
    mse: float,
    mae: float,
    group_mse: dict[str, float],
    group_mae: dict[str, float],
) -> None:
    bucket["count"] += 1
    bucket["mse_sum"] += float(mse)
    bucket["mae_sum"] += float(mae)
    bucket["max_mse"] = max(float(bucket["max_mse"]), float(mse))
    bucket["max_mae"] = max(float(bucket["max_mae"]), float(mae))
    for group_name, value in group_mse.items():
        bucket["group_mse_sum"][group_name] = float(bucket["group_mse_sum"].get(group_name, 0.0)) + float(value)
    for group_name, value in group_mae.items():
        bucket["group_mae_sum"][group_name] = float(bucket["group_mae_sum"].get(group_name, 0.0)) + float(value)


def update_reconstruction_stats(
    stats: dict[str, dict[str, dict[str, Any]]],
    *,
    metadata: list[dict[str, Any]],
    recon: torch.Tensor,
    target: torch.Tensor,
    dim_groups: dict[str, list[int]],
) -> None:
    with torch.no_grad():
        error = recon.detach().float() - target.detach().float()
        sample_mse = error.pow(2).mean(dim=(1, 2)).cpu().tolist()
        sample_mae = error.abs().mean(dim=(1, 2)).cpu().tolist()
        group_mse_values: dict[str, list[float]] = {}
        group_mae_values: dict[str, list[float]] = {}
        for group_name, dims in dim_groups.items():
            if not dims:
                continue
            group_error = error[:, :, dims]
            group_mse_values[group_name] = group_error.pow(2).mean(dim=(1, 2)).cpu().tolist()
            group_mae_values[group_name] = group_error.abs().mean(dim=(1, 2)).cpu().tolist()

    for idx, item in enumerate(metadata):
        task_name = _task_name_from_metadata(item)
        category_name = _task_category_from_task_name(task_name)
        group_mse = {name: values[idx] for name, values in group_mse_values.items()}
        group_mae = {name: values[idx] for name, values in group_mae_values.items()}
        for scope, key in (("tasks", task_name), ("categories", category_name)):
            bucket = stats[scope].setdefault(key, _new_reconstruction_bucket())
            _accumulate_reconstruction_bucket(
                bucket,
                mse=sample_mse[idx],
                mae=sample_mae[idx],
                group_mse=group_mse,
                group_mae=group_mae,
            )


def _finalize_reconstruction_scope(scope_stats: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    records: dict[str, Any] = {}
    for key, bucket in sorted(scope_stats.items()):
        count = max(int(bucket["count"]), 1)
        record: dict[str, Any] = {
            "count": int(bucket["count"]),
            "mse": float(bucket["mse_sum"]) / count,
            "mae": float(bucket["mae_sum"]) / count,
            "max_mse": float(bucket["max_mse"]),
            "max_mae": float(bucket["max_mae"]),
        }
        if bucket["group_mse_sum"]:
            record["group_mse"] = {
                name: float(value) / count for name, value in sorted(bucket["group_mse_sum"].items())
            }
        if bucket["group_mae_sum"]:
            record["group_mae"] = {
                name: float(value) / count for name, value in sorted(bucket["group_mae_sum"].items())
            }
        records[key] = record

    if not records:
        return records, {}

    mae_values = [float(record["mae"]) for record in records.values()]
    mse_values = [float(record["mse"]) for record in records.values()]
    worst_mae_key = max(records, key=lambda key: float(records[key]["mae"]))
    worst_mse_key = max(records, key=lambda key: float(records[key]["mse"]))
    summary = {
        "count": len(records),
        "mae_mean": float(sum(mae_values) / len(mae_values)),
        "mae_worst": float(records[worst_mae_key]["mae"]),
        "mae_worst_key": worst_mae_key,
        "mse_mean": float(sum(mse_values) / len(mse_values)),
        "mse_worst": float(records[worst_mse_key]["mse"]),
        "mse_worst_key": worst_mse_key,
    }
    return records, summary


def finalize_reconstruction_stats(stats: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    tasks, task_summary = _finalize_reconstruction_scope(stats["tasks"])
    categories, category_summary = _finalize_reconstruction_scope(stats["categories"])
    return {
        "tasks": tasks,
        "task_summary": task_summary,
        "categories": categories,
        "category_summary": category_summary,
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float(0.5 * (ordered[midpoint - 1] + ordered[midpoint]))


def _normalize_weights_to_mean_one(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        return {}
    mean_weight = _mean(list(weights.values()))
    if mean_weight <= 0.0:
        return weights
    return {key: float(value / mean_weight) for key, value in weights.items()}


def build_static_task_balance_weights(dataset: Any, cfg: Any) -> dict[str, float]:
    weighting_cfg = cfg.loss.get("task_balance_weighting", None)
    if not weighting_cfg or not bool(weighting_cfg.get("enabled", False)):
        return {}

    counts: dict[str, int] = {}
    source_dataset = getattr(dataset, "source_dataset", None)
    sources = getattr(source_dataset, "datasets", None)
    if sources:
        horizon = int(getattr(dataset, "action_spec").horizon)
        for dataset_index, source in enumerate(sources):
            dataset_name = _simplify_dataset_name(str(getattr(source, "dataset_name", dataset_index)))
            for trajectory_id, trajectory_length in zip(
                getattr(source, "trajectory_ids", []),
                getattr(source, "trajectory_lengths", []),
                strict=False,
            ):
                task_name = dataset_name
                try:
                    trajectory_data = source.get_trajectory_data(int(trajectory_id))
                    if "task_index" in trajectory_data.columns:
                        task_index = int(trajectory_data["task_index"].iloc[0])
                        suite_name = str(getattr(source, "dataset_name", dataset_index)).rsplit("/", 1)[-1]
                        for suffix in ("_no_noops_1.0.0_lerobot", "_1.0.0_lerobot", "_lerobot"):
                            if suite_name.endswith(suffix):
                                suite_name = suite_name[: -len(suffix)]
                                break
                        task_name = f"{suite_name}::task_{task_index:02d}"
                except Exception:
                    pass
                count = max(int(trajectory_length) - horizon + 1, 0)
                counts[task_name] = counts.get(task_name, 0) + count

    if not counts:
        return {}

    mean_count = _mean([float(value) for value in counts.values()])
    power = float(weighting_cfg.get("power", 1.0))
    min_weight = float(weighting_cfg.get("min_weight", 0.25))
    max_weight = float(weighting_cfg.get("max_weight", 4.0))
    weights = {}
    for task_name, count in counts.items():
        weight = (mean_count / max(float(count), 1.0)) ** power
        weight = max(min_weight, min(max_weight, weight))
        weights[task_name] = float(weight)
    if bool(weighting_cfg.get("normalize_mean", True)):
        weights = _normalize_weights_to_mean_one(weights)
    return weights


def build_adaptive_task_weights(reconstruction_summary: dict[str, Any] | None, cfg: Any) -> dict[str, float]:
    weighting_cfg = cfg.loss.get("adaptive_task_weighting", None)
    if not weighting_cfg or not bool(weighting_cfg.get("enabled", False)) or not reconstruction_summary:
        return {}

    scope = str(weighting_cfg.get("scope", "task"))
    scope_key = "categories" if scope in {"category", "class"} else "tasks"
    metric = str(weighting_cfg.get("metric", "mae"))
    if metric not in {"mae", "mse"}:
        raise ValueError(f"adaptive_task_weighting.metric must be 'mae' or 'mse', got {metric!r}.")

    min_count = int(weighting_cfg.get("min_count", 1))
    entries = reconstruction_summary.get(scope_key, {})
    values = {
        key: float(record[metric])
        for key, record in entries.items()
        if int(record.get("count", 0)) >= min_count and math.isfinite(float(record.get(metric, 0.0)))
    }
    if not values:
        return {}

    baseline_mode = str(weighting_cfg.get("baseline", "mean"))
    metric_values = list(values.values())
    if baseline_mode == "median":
        baseline = _median(metric_values)
    elif baseline_mode == "target":
        baseline = float(weighting_cfg.get("target", 0.0))
    elif baseline_mode == "mean":
        baseline = _mean(metric_values)
    else:
        raise ValueError(
            "adaptive_task_weighting.baseline must be one of 'mean', 'median', or 'target', "
            f"got {baseline_mode!r}."
        )

    eps = float(weighting_cfg.get("eps", 1e-8))
    power = float(weighting_cfg.get("power", 1.0))
    min_weight = float(weighting_cfg.get("min_weight", 0.5))
    max_weight = float(weighting_cfg.get("max_weight", 4.0))
    weights = {}
    for key, value in values.items():
        ratio = (float(value) + eps) / (float(baseline) + eps)
        weight = ratio**power
        weight = max(min_weight, min(max_weight, weight))
        weights[key] = float(weight)

    if bool(weighting_cfg.get("normalize_mean", True)) and weights:
        mean_weight = _mean(list(weights.values()))
        if mean_weight > 0.0:
            weights = {key: float(value / mean_weight) for key, value in weights.items()}
    return weights


def build_trajectory_length_lookup(dataset: Any) -> dict[int, dict[int, int]]:
    source_dataset = getattr(dataset, "source_dataset", None)
    sources = getattr(source_dataset, "datasets", None)
    if sources is None:
        return {}
    lookup: dict[int, dict[int, int]] = {}
    for dataset_index, source in enumerate(sources):
        lookup[dataset_index] = {
            int(traj_id): int(length)
            for traj_id, length in zip(source.trajectory_ids, source.trajectory_lengths, strict=False)
        }
    return lookup


def _phase_ratio(metadata: dict[str, Any], *, trajectory_lengths: dict[int, dict[int, int]], horizon: int) -> float | None:
    try:
        dataset_index = int(metadata["dataset_index"])
        trajectory_id = int(metadata["trajectory_id"])
        base_index = int(metadata["base_index"])
        trajectory_length = int(trajectory_lengths[dataset_index][trajectory_id])
    except Exception:
        return None
    max_start = max(trajectory_length - int(horizon), 1)
    return float(base_index) / float(max_start)


def build_sample_weights(
    metadata: list[dict[str, Any]],
    *,
    cfg: Any,
    trajectory_lengths: dict[int, dict[int, int]],
    horizon: int,
    device: torch.device,
    dtype: torch.dtype,
    task_error_weights: dict[str, float] | None = None,
    static_task_weights: dict[str, float] | None = None,
) -> torch.Tensor | None:
    weighting_cfg = cfg.loss.get("sample_weighting", None)
    static_weighting_enabled = bool(weighting_cfg.get("enabled", False)) if weighting_cfg else False
    adaptive_cfg = cfg.loss.get("adaptive_task_weighting", None)
    adaptive_weighting_enabled = bool(task_error_weights) and bool(adaptive_cfg.get("enabled", False)) if adaptive_cfg else False
    task_balance_enabled = bool(static_task_weights)
    if not static_weighting_enabled and not adaptive_weighting_enabled and not task_balance_enabled:
        return None

    close_task_weight = float(weighting_cfg.get("close_task_weight", 1.0)) if weighting_cfg else 1.0
    late_phase_weight = float(weighting_cfg.get("late_phase_weight", 1.0)) if weighting_cfg else 1.0
    close_late_weight = float(weighting_cfg.get("close_late_weight", 1.0)) if weighting_cfg else 1.0
    late_phase_start_ratio = float(weighting_cfg.get("late_phase_start_ratio", 0.5)) if weighting_cfg else 0.5
    static_max_weight = float(weighting_cfg.get("max_weight", 0.0)) if weighting_cfg else 0.0
    adaptive_scope = str(adaptive_cfg.get("scope", "task")) if adaptive_cfg else "task"
    configured_task_weights = dict(weighting_cfg.get("task_weights", {})) if weighting_cfg else {}
    phase_cfg = cfg.loss.get("trajectory_phase_weighting", None)
    phase_enabled = bool(phase_cfg.get("enabled", False)) if phase_cfg else False
    phase_suites = {str(value) for value in phase_cfg.get("suites", [])} if phase_cfg else set()
    phase_tasks = {str(value) for value in phase_cfg.get("tasks", [])} if phase_cfg else set()
    phase_start_ratio = float(phase_cfg.get("start_ratio", 0.5)) if phase_cfg else 0.5
    phase_weight = float(phase_cfg.get("weight", 1.0)) if phase_cfg else 1.0

    weights = []
    for item in metadata:
        weight = 1.0
        if static_weighting_enabled:
            dataset_name = str(item.get("dataset_name", ""))
            is_close = _is_robocasa_close_task(dataset_name)
            ratio = _phase_ratio(item, trajectory_lengths=trajectory_lengths, horizon=horizon)
            is_late = ratio is not None and ratio >= late_phase_start_ratio

            if is_close:
                weight *= close_task_weight
            if is_late:
                weight *= late_phase_weight
            if is_close and is_late:
                weight *= close_late_weight
            if static_max_weight > 0.0:
                weight = min(weight, static_max_weight)

        task_name = _task_name_from_metadata(item)
        weight *= float(configured_task_weights.get(task_name, 1.0))
        if phase_enabled:
            suite_name = str(item.get("suite_name", ""))
            ratio = _phase_ratio(item, trajectory_lengths=trajectory_lengths, horizon=horizon)
            selected = (not phase_suites and not phase_tasks) or suite_name in phase_suites or task_name in phase_tasks
            if selected and ratio is not None and ratio >= phase_start_ratio:
                weight *= phase_weight
        if task_balance_enabled and static_task_weights:
            weight *= float(static_task_weights.get(task_name, 1.0))

        if adaptive_weighting_enabled and task_error_weights:
            if adaptive_scope in {"category", "class"}:
                adaptive_key = _task_category_from_task_name(task_name)
            else:
                adaptive_key = task_name
            weight *= float(task_error_weights.get(adaptive_key, 1.0))

        weights.append(weight)

    return torch.tensor(weights, device=device, dtype=dtype)


def build_time_weights(seq_len: int, *, cfg: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    weighting_cfg = cfg.loss.get("time_weighting", None)
    if not weighting_cfg or not bool(weighting_cfg.get("enabled", False)):
        return None

    weights = torch.ones(seq_len, device=device, dtype=dtype)
    late_timestep_start = int(weighting_cfg.get("late_timestep_start", seq_len))
    late_timestep_weight = float(weighting_cfg.get("late_timestep_weight", 1.0))
    if late_timestep_start < seq_len:
        weights[max(late_timestep_start, 0) :] *= late_timestep_weight
    return weights


def train(cfg: Any) -> None:
    torch.manual_seed(int(cfg.experiment.get("seed", 42)))
    device = torch.device(cfg.train.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_starvla_base_config(cfg)
    if str(cfg.data.get("dataset_format", "starvla_lerobot")) == "robotwin_raw_zip":
        from starVLA.dataloader.robotwin_raw_stage1_action_dataset import RoboTwinRawStage1ActionDataset

        dataset = RoboTwinRawStage1ActionDataset(
            cfg.data.data_root_dir,
            splits=list(cfg.data.get("splits", ["clean"])),
            embodiment=str(cfg.data.get("embodiment", "aloha-agilex")),
            task_names=cfg.data.get("task_names", "all"),
            action_key=str(cfg.data.get("raw_action_key", "/joint_action/vector")),
            horizon=int(cfg.data.get("expected_action_horizon", 50)),
            action_dim=int(cfg.data.get("expected_action_dim", 14)),
            max_episodes_per_zip=cfg.data.get("max_episodes_per_zip", None),
            binary_threshold=float(cfg.data.get("binary_threshold", 0.49)),
        )
    else:
        from starVLA.dataloader.var_stage1_action_dataset import VARStage1ActionDataset

        dataset = VARStage1ActionDataset(
            base_cfg,
            mode="train",
            balance_dataset_weights=bool(cfg.data.get("balance_dataset_weights", False)),
            balance_trajectory_weights=bool(cfg.data.get("balance_trajectory_weights", False)),
            seed=int(cfg.experiment.get("seed", 42)),
            return_raw_actions=True,
            window_mode=str(cfg.data.get("window_mode", "full")),
        )
    action_spec = dataset.action_spec
    trajectory_lengths = build_trajectory_length_lookup(dataset)
    static_task_weights = build_static_task_balance_weights(dataset, cfg)
    if static_task_weights:
        save_json(output_dir / "static_task_balance_weights.json", static_task_weights)
    save_json(output_dir / "action_spec.json", action_spec.to_dict())
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    OmegaConf.save(base_cfg, output_dir / "starvla_base_config.yaml", resolve=True)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=bool(cfg.train.get("shuffle", True)),
        num_workers=int(cfg.train.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_action_batch,
        persistent_workers=int(cfg.train.get("num_workers", 4)) > 0,
    )

    model_seq_len = action_spec.horizon if cfg.model.get("seq_len", "auto") == "auto" else int(cfg.model.seq_len)
    model_action_dim = action_spec.action_dim if cfg.model.get("action_dim", "auto") == "auto" else int(cfg.model.action_dim)
    scales = resolve_scales(cfg.model.get("scales", "auto"), model_seq_len)

    model = VARActionTokenizer(
        action_dim=model_action_dim,
        seq_len=model_seq_len,
        scales=scales,
        embed_dim=int(cfg.model.get("embed_dim", 128)),
        codebook_size=int(cfg.model.get("codebook_size", 512)),
        use_dilated=bool(cfg.model.get("use_dilated", True)),
        quant_resi=float(cfg.model.get("quant_resi", 0.5)),
        commitment_cost=float(cfg.model.get("commitment_cost", 0.25)),
        normalize_codebook_for_lookup=bool(cfg.model.get("normalize_codebook_for_lookup", True)),
        decoder_head_type=str(cfg.model.get("decoder_head_type", "plain")),
        quantization_mode=str(cfg.model.get("quantization_mode", "vq")),
        product_codebook_groups=int(cfg.model.get("product_codebook_groups", 1)),
        dim_groups=action_spec.dim_groups,
        use_time_embedding=bool(cfg.model.get("use_time_embedding", False)),
        use_action_type_embedding=bool(cfg.model.get("use_action_type_embedding", False)),
        input_embedding_scale=float(cfg.model.get("input_embedding_scale", 1.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.train.get("weight_decay", 1e-5)),
    )

    history: list[dict[str, Any]] = []
    start_epoch = 0

    resume_checkpoint_path = cfg.train.get("resume_checkpoint", None)
    if resume_checkpoint_path:
        resume_checkpoint = load_resume_checkpoint(
            str(resume_checkpoint_path),
            model=model,
            optimizer=optimizer,
            device=device,
        )
        history = list(resume_checkpoint.get("history", []))
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        print(f"Resumed checkpoint {resume_checkpoint_path} from epoch {resume_checkpoint.get('epoch', 'unknown')}.")
    else:
        init_checkpoint_path = cfg.train.get("init_checkpoint", None)
        if init_checkpoint_path:
            init_checkpoint_mode = str(cfg.train.get("init_checkpoint_mode", "full"))
            if init_checkpoint_mode == "full":
                init_checkpoint = load_init_checkpoint(str(init_checkpoint_path), model=model, device=device)
            elif init_checkpoint_mode == "encoder_decoder":
                init_checkpoint = load_encoder_decoder_init(str(init_checkpoint_path), model=model, device=device)
            else:
                raise ValueError(f"Unsupported init_checkpoint_mode={init_checkpoint_mode!r}.")
            print(
                f"Loaded {init_checkpoint_mode} init checkpoint {init_checkpoint_path} "
                f"(source epoch {init_checkpoint.get('epoch', 'unknown')})"
            )

        initialize_codebook_from_data(
            model=model,
            loader=loader,
            device=device,
            batches=int(cfg.train.get("init_codebook_from_data_batches", 0)),
            noise_scale=float(cfg.train.get("init_codebook_noise_scale", 0.0)),
        )

    if history:
        best_recon = min(float(record["recon_loss"]) for record in history)
    else:
        best_recon = float("inf")
    min_usage_for_balanced = float(cfg.train.get("min_codebook_usage_ratio_for_balanced", 0.02))
    balanced_records = [
        float(record["recon_loss"])
        for record in history
        if float(record.get("codebook_usage_ratio", 0.0)) >= min_usage_for_balanced
    ]
    best_balanced_recon = min(balanced_records) if balanced_records else float("inf")
    worst_task_records = [
        float(record["task_mae_worst"])
        for record in history
        if record.get("task_mae_worst", None) is not None
    ]
    best_worst_task_mae = min(worst_task_records) if worst_task_records else float("inf")

    adaptive_task_weights_path = output_dir / "adaptive_task_weights.json"
    task_error_weights: dict[str, float] = {}
    if adaptive_task_weights_path.exists():
        try:
            with adaptive_task_weights_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            task_error_weights = {str(key): float(value) for key, value in payload.get("weights", {}).items()}
            if task_error_weights:
                print(f"Loaded adaptive task weights from {adaptive_task_weights_path}.")
        except Exception as exc:
            print(f"Failed to load adaptive task weights from {adaptive_task_weights_path}: {exc}")

    if start_epoch >= int(cfg.train.epochs):
        raise ValueError(
            f"resume checkpoint starts at epoch {start_epoch}, but cfg.train.epochs={int(cfg.train.epochs)}. "
            "Increase train.epochs to continue training."
        )

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        freeze_encoder_decoder_epochs = int(cfg.train.get("freeze_encoder_decoder_epochs", 0))
        set_encoder_decoder_trainable(model, epoch >= freeze_encoder_decoder_epochs)
        model.train()
        totals = {
            "recon_loss": 0.0,
            "vel_loss": 0.0,
            "vq_loss": 0.0,
            "total_loss": 0.0,
            "sample_weight_mean": 0.0,
        }
        batches = 0
        code_counts = torch.zeros(model.codebook_size, dtype=torch.long)
        vq_weight = compute_vq_weight(
            epoch,
            target_weight=float(cfg.loss.get("vq_weight", 0.1)),
            warmup_epochs=int(cfg.loss.get("vq_warmup_epochs", 20)),
        )
        adaptive_weight_cfg = cfg.loss.get("adaptive_task_weighting", None)
        track_reconstruction_by_task = bool(cfg.train.get("track_reconstruction_by_task", False)) or bool(
            adaptive_weight_cfg.get("enabled", False) if adaptive_weight_cfg else False
        )
        reconstruction_stats = _new_reconstruction_stats() if track_reconstruction_by_task else None

        progress = tqdm(loader, desc=f"epoch {epoch:03d}")
        sample_weight_cfg = cfg.loss.get("sample_weighting", None)
        weight_normalization = str(cfg.loss.get("weight_normalization", ""))
        if not weight_normalization:
            weight_normalization = str(sample_weight_cfg.get("normalization", "mean")) if sample_weight_cfg else "mean"
        for batch in progress:
            actions = batch["actions"].to(device=device, dtype=torch.float32, non_blocking=True)
            sample_weights = build_sample_weights(
                batch["metadata"],
                cfg=cfg,
                trajectory_lengths=trajectory_lengths,
                horizon=int(action_spec.horizon),
                device=device,
                dtype=actions.dtype,
                task_error_weights=task_error_weights,
                static_task_weights=static_task_weights,
            )
            time_weights = build_time_weights(actions.shape[1], cfg=cfg, device=device, dtype=actions.dtype)

            out = model(actions)
            recon = out["recon"]
            recon_loss = weighted_dim_mse(
                recon,
                actions,
                dim_groups=action_spec.dim_groups,
                gripper_weight=float(cfg.loss.get("gripper_recon_weight", 1.0)),
                group_weights=loss_group_weights(cfg, "recon"),
                sample_weights=sample_weights,
                time_weights=time_weights,
                weight_normalization=weight_normalization,
            )
            if reconstruction_stats is not None:
                update_reconstruction_stats(
                    reconstruction_stats,
                    metadata=batch["metadata"],
                    recon=recon,
                    target=actions,
                    dim_groups=action_spec.dim_groups,
                )

            if actions.shape[1] > 1:
                vel_recon = recon[:, 1:] - recon[:, :-1]
                vel_target = actions[:, 1:] - actions[:, :-1]
                vel_time_weights = time_weights[1:] if time_weights is not None else None
                vel_loss = weighted_dim_mse(
                    vel_recon,
                    vel_target,
                    dim_groups=action_spec.dim_groups,
                    gripper_weight=float(cfg.loss.get("gripper_vel_weight", 0.1)),
                    group_weights=loss_group_weights(cfg, "vel"),
                    sample_weights=sample_weights,
                    time_weights=vel_time_weights,
                    weight_normalization=weight_normalization,
                )
            else:
                vel_loss = torch.zeros((), device=device, dtype=actions.dtype)

            if actions.shape[1] > 2 and float(cfg.loss.get("jerk_weight", 0.0)) > 0.0:
                jerk_recon = recon[:, 2:] - 2.0 * recon[:, 1:-1] + recon[:, :-2]
                jerk_target = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]
                jerk_loss = F.mse_loss(jerk_recon, jerk_target)
            else:
                jerk_loss = torch.zeros((), device=device, dtype=actions.dtype)

            total_loss = (
                float(cfg.loss.get("recon_weight", 1.0)) * recon_loss
                + float(cfg.loss.get("vel_weight", 0.5)) * vel_loss
                + float(cfg.loss.get("jerk_weight", 0.0)) * jerk_loss
                + vq_weight * out["vq_loss"]
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_clip = cfg.train.get("grad_clip", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            totals["recon_loss"] += float(recon_loss.detach().cpu())
            totals["vel_loss"] += float(vel_loss.detach().cpu())
            totals["vq_loss"] += float(out["vq_loss"].detach().cpu())
            totals["total_loss"] += float(total_loss.detach().cpu())
            if sample_weights is not None:
                totals["sample_weight_mean"] += float(sample_weights.detach().mean().cpu())
            else:
                totals["sample_weight_mean"] += 1.0
            batches += 1

            flat_tokens = out["flat_token_ids"].detach().reshape(-1).cpu()
            if flat_tokens.numel() > 0:
                code_counts += torch.bincount(flat_tokens, minlength=model.codebook_size)

            progress.set_postfix(
                recon=f"{totals['recon_loss'] / batches:.5f}",
                vq=f"{totals['vq_loss'] / batches:.5f}",
            )

            max_batches = int(cfg.train.get("max_batches_per_epoch", 0))
            if max_batches > 0 and batches >= max_batches:
                break

        if batches == 0:
            raise RuntimeError("No batches were produced by the Stage 1 dataloader.")

        epoch_record = {key: value / batches for key, value in totals.items()}
        used_codes = int((code_counts > 0).sum().item())
        usage_ratio = used_codes / float(model.codebook_size)
        epoch_record.update(
            {
                "epoch": epoch,
                "vq_weight": vq_weight,
                "codebook_used": used_codes,
                "codebook_usage_ratio": usage_ratio,
            }
        )

        reconstruction_summary = None
        if reconstruction_stats is not None:
            reconstruction_summary = finalize_reconstruction_stats(reconstruction_stats)
            task_summary = reconstruction_summary.get("task_summary", {})
            category_summary = reconstruction_summary.get("category_summary", {})
            if task_summary:
                epoch_record.update(
                    {
                        "task_mae_mean": float(task_summary["mae_mean"]),
                        "task_mae_worst": float(task_summary["mae_worst"]),
                        "task_mae_worst_name": str(task_summary["mae_worst_key"]),
                        "task_mse_mean": float(task_summary["mse_mean"]),
                        "task_mse_worst": float(task_summary["mse_worst"]),
                        "task_mse_worst_name": str(task_summary["mse_worst_key"]),
                    }
                )
            if category_summary:
                epoch_record.update(
                    {
                        "category_mae_mean": float(category_summary["mae_mean"]),
                        "category_mae_worst": float(category_summary["mae_worst"]),
                        "category_mae_worst_name": str(category_summary["mae_worst_key"]),
                    }
                )

            save_json(output_dir / "reconstruction_by_task.json", reconstruction_summary)
            if bool(cfg.train.get("save_reconstruction_by_task_every_epoch", True)):
                save_json(output_dir / f"reconstruction_by_task_epoch_{epoch:03d}.json", reconstruction_summary)

            next_task_error_weights = build_adaptive_task_weights(reconstruction_summary, cfg)
            if next_task_error_weights:
                task_error_weights = next_task_error_weights
                save_json(
                    adaptive_task_weights_path,
                    {
                        "epoch": epoch,
                        "scope": str(cfg.loss.get("adaptive_task_weighting", {}).get("scope", "task")),
                        "metric": str(cfg.loss.get("adaptive_task_weighting", {}).get("metric", "mae")),
                        "weights": task_error_weights,
                    },
                )

        history.append(epoch_record)
        save_json(output_dir / "history.json", history)
        save_checkpoint(output_dir / "latest.ckpt", model=model, optimizer=optimizer, epoch=epoch, history=history, cfg=cfg, action_spec=action_spec)

        save_every_epochs = int(cfg.train.get("save_every_epochs", 0))
        if save_every_epochs > 0 and (epoch + 1) % save_every_epochs == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.ckpt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                cfg=cfg,
                action_spec=action_spec,
            )

        if epoch_record["recon_loss"] < best_recon:
            best_recon = epoch_record["recon_loss"]
            save_checkpoint(output_dir / "best_recon.ckpt", model=model, optimizer=optimizer, epoch=epoch, history=history, cfg=cfg, action_spec=action_spec)

        if usage_ratio >= min_usage_for_balanced and epoch_record["recon_loss"] < best_balanced_recon:
            best_balanced_recon = epoch_record["recon_loss"]
            save_checkpoint(output_dir / "best_balanced.ckpt", model=model, optimizer=optimizer, epoch=epoch, history=history, cfg=cfg, action_spec=action_spec)

        task_mae_worst = epoch_record.get("task_mae_worst", None)
        if task_mae_worst is not None and float(task_mae_worst) < best_worst_task_mae:
            best_worst_task_mae = float(task_mae_worst)
            save_checkpoint(
                output_dir / "best_worst_task_mae.ckpt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                cfg=cfg,
                action_spec=action_spec,
            )

    save_checkpoint(output_dir / "final.ckpt", model=model, optimizer=optimizer, epoch=history[-1]["epoch"], history=history, cfg=cfg, action_spec=action_spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAR Stage 1 action tokenizer.")
    parser.add_argument("--config_yaml", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_yaml)
    train(cfg)


if __name__ == "__main__":
    main()
