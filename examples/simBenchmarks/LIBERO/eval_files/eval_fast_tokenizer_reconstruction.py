"""Offline encode/decode MSE for the FAST action tokenizer on Stage 1 actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from starVLA.dataloader.var_stage1_action_dataset import VARStage1ActionDataset
from starVLA.training.train_var_stage1 import collate_action_batch, load_starvla_base_config


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _update_group_sums(
    group_sums: dict[str, float],
    group_counts: dict[str, int],
    squared_error: torch.Tensor,
    dim_groups: dict[str, list[int]],
) -> None:
    for name, dims in dim_groups.items():
        if not dims:
            continue
        group_sums[name] = group_sums.get(name, 0.0) + float(squared_error[:, :, dims].sum().item())
        group_counts[name] = group_counts.get(name, 0) + int(squared_error[:, :, dims].numel())


def _decode_fast_batch(fast_tokenizer: Any, actions: torch.Tensor) -> torch.Tensor:
    actions_np = actions.detach().cpu().numpy().astype(np.float32)
    tokens = fast_tokenizer(actions_np)
    decoded = fast_tokenizer.decode(
        tokens,
        time_horizon=int(actions.shape[1]),
        action_dim=int(actions.shape[2]),
    )
    decoded_np = np.asarray(decoded, dtype=np.float32)
    if decoded_np.shape != actions_np.shape:
        raise ValueError(f"FAST decoded shape {decoded_np.shape} != target shape {actions_np.shape}")
    return torch.as_tensor(decoded_np, dtype=torch.float32)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _safe_torch_load(args.stage1_checkpoint)
    train_cfg = OmegaConf.create(checkpoint["stage1_config"])
    base_cfg = load_starvla_base_config(train_cfg)
    dataset = VARStage1ActionDataset(
        base_cfg,
        mode="train",
        balance_dataset_weights=bool(train_cfg.data.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(train_cfg.data.get("balance_trajectory_weights", False)),
        seed=int(train_cfg.experiment.get("seed", 42)),
        return_raw_actions=False,
        window_mode=str(train_cfg.data.get("window_mode", "full")),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_action_batch,
    )
    fast_tokenizer = AutoProcessor.from_pretrained(args.fast_tokenizer_name, trust_remote_code=True)

    total_sse = 0.0
    total_abs = 0.0
    total_count = 0
    vel_sse = 0.0
    vel_count = 0
    dim_sse = torch.zeros(dataset.action_spec.action_dim, dtype=torch.float64)
    dim_abs = torch.zeros(dataset.action_spec.action_dim, dtype=torch.float64)
    dim_count = 0
    group_sums: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    samples = 0
    batches = 0

    for batch in tqdm(loader, desc="eval fast tokenizer recon"):
        actions = batch["actions"].float().cpu()
        decoded = _decode_fast_batch(fast_tokenizer, actions)
        error = decoded - actions
        squared_error = error.pow(2)
        abs_error = error.abs()

        total_sse += float(squared_error.sum().item())
        total_abs += float(abs_error.sum().item())
        total_count += int(squared_error.numel())
        dim_sse += squared_error.sum(dim=(0, 1)).double()
        dim_abs += abs_error.sum(dim=(0, 1)).double()
        dim_count += int(actions.shape[0] * actions.shape[1])
        if actions.shape[1] > 1:
            vel_error = (decoded[:, 1:] - decoded[:, :-1]) - (actions[:, 1:] - actions[:, :-1])
            vel_sse += float(vel_error.pow(2).sum().item())
            vel_count += int(vel_error.numel())
        _update_group_sums(group_sums, group_counts, squared_error, dataset.action_spec.dim_groups)

        samples += int(actions.shape[0])
        batches += 1
        if args.max_batches > 0 and batches >= args.max_batches:
            break
        if args.max_samples > 0 and samples >= args.max_samples:
            break

    report = {
        "fast_tokenizer_name": str(args.fast_tokenizer_name),
        "stage1_checkpoint_for_dataset_config": str(args.stage1_checkpoint),
        "num_samples": samples,
        "num_batches": batches,
        "dataset_len": len(dataset),
        "action_spec": dataset.action_spec.to_dict(),
        "metrics": {
            "mse": total_sse / max(total_count, 1),
            "mae": total_abs / max(total_count, 1),
            "rmse": (total_sse / max(total_count, 1)) ** 0.5,
            "vel_mse": vel_sse / max(vel_count, 1),
            "per_dim_mse": (dim_sse / max(dim_count, 1)).tolist(),
            "per_dim_mae": (dim_abs / max(dim_count, 1)).tolist(),
            "group_mse": {
                name: group_sums[name] / max(group_counts[name], 1)
                for name in sorted(group_sums)
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FAST tokenizer encode/decode MSE on Stage 1 actions.")
    parser.add_argument("--stage1_checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fast_tokenizer_name", type=str, default="physical-intelligence/fast")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps(report["metrics"], indent=2))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
