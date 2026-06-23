"""Metrics for VAR Stage 2 token-policy evaluation."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def scale_slices(scales: Sequence[int], *, product_codebook_groups: int = 1) -> dict[str, slice]:
    """Return scale-name to slice mapping for scale-major token sequences."""

    result: dict[str, slice] = {}
    offset = 0
    groups = int(product_codebook_groups)
    if groups <= 0:
        raise ValueError(f"product_codebook_groups must be positive, got {groups}.")
    for scale in scales:
        width = int(scale) * groups
        next_offset = offset + width
        result[f"scale_{int(scale)}"] = slice(offset, next_offset)
        offset = next_offset
    return result


def token_accuracy_by_scale(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    scales: Sequence[int],
    product_codebook_groups: int = 1,
) -> dict[str, float]:
    """Compute overall and per-scale token accuracy."""

    if predicted_tokens.shape != target_tokens.shape:
        raise ValueError(f"Token shape mismatch: predicted={tuple(predicted_tokens.shape)}, target={tuple(target_tokens.shape)}")
    if predicted_tokens.ndim != 2:
        raise ValueError(f"Expected tokens with shape [B, L], got {tuple(predicted_tokens.shape)}.")
    groups = int(product_codebook_groups)
    if groups <= 0:
        raise ValueError(f"product_codebook_groups must be positive, got {groups}.")
    expected_len = int(sum(scales)) * groups
    if predicted_tokens.shape[1] != expected_len:
        raise ValueError(f"Expected token length {expected_len}, got {predicted_tokens.shape[1]}.")

    correct = predicted_tokens.eq(target_tokens)
    metrics = {"overall": float(correct.float().mean().item())}
    for name, token_slice in scale_slices(scales, product_codebook_groups=groups).items():
        metrics[name] = float(correct[:, token_slice].float().mean().item())
    if groups > 1:
        for group_idx in range(groups):
            metrics[f"product_group_{group_idx}"] = float(correct[:, group_idx::groups].float().mean().item())
    return metrics


def decoded_action_metrics(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    dim_groups: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Compute decoded action error metrics for [B, T, D] actions."""

    if predicted_actions.shape != target_actions.shape:
        raise ValueError(
            f"Action shape mismatch: predicted={tuple(predicted_actions.shape)}, target={tuple(target_actions.shape)}"
        )
    if predicted_actions.ndim != 3:
        raise ValueError(f"Expected actions with shape [B, T, D], got {tuple(predicted_actions.shape)}.")

    error = predicted_actions.float() - target_actions.float()
    squared = error.pow(2)
    absolute = error.abs()
    metrics: dict[str, Any] = {
        "mse": float(squared.mean().item()),
        "mae": float(absolute.mean().item()),
        "rmse": float(squared.mean().sqrt().item()),
        "per_dim_mse": squared.mean(dim=(0, 1)).detach().cpu().tolist(),
        "per_dim_mae": absolute.mean(dim=(0, 1)).detach().cpu().tolist(),
    }
    if predicted_actions.shape[1] > 1:
        pred_vel = predicted_actions[:, 1:] - predicted_actions[:, :-1]
        target_vel = target_actions[:, 1:] - target_actions[:, :-1]
        metrics["vel_mse"] = float((pred_vel.float() - target_vel.float()).pow(2).mean().item())
    else:
        metrics["vel_mse"] = 0.0

    if dim_groups:
        group_mse = {}
        for name, dims in dim_groups.items():
            if dims:
                group_mse[name] = float(squared[:, :, dims].mean().item())
        metrics["group_mse"] = group_mse
    return metrics
