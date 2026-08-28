"""Offline reconstruction evaluation for VAR Stage 1 action tokenizers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from starVLA.dataloader.var_stage1_action_dataset import VARStage1ActionDataset
from starVLA.model.modules.action_tokenizer import VARActionTokenizer, VQVLARVQActionTokenizer
from starVLA.training.train_var_stage1 import collate_action_batch, load_starvla_base_config


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[VARActionTokenizer, dict[str, Any]]:
    checkpoint = _safe_torch_load(checkpoint_path)
    model_config = dict(checkpoint["model_config"])
    model_type = str(model_config.pop("model_type", "var_action_tokenizer"))
    if model_type == "vqvla_rvq_action_tokenizer":
        model = VQVLARVQActionTokenizer(**model_config).to(device)
    else:
        model = VARActionTokenizer(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _update_group_sums(
    group_sums: dict[str, float],
    group_counts: dict[str, int],
    squared_error: torch.Tensor,
    dim_groups: dict[str, list[int]],
) -> None:
    for name, dims in dim_groups.items():
        if not dims:
            continue
        value = squared_error[:, :, dims].sum().item()
        group_sums[name] = group_sums.get(name, 0.0) + float(value)
        group_counts[name] = group_counts.get(name, 0) + int(squared_error[:, :, dims].numel())


def evaluate(cfg: Any, checkpoint_path: Path, output_path: Path) -> dict[str, Any]:
    device = torch.device(cfg.eval.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    model, checkpoint = _load_model(checkpoint_path, device)

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
        batch_size=int(cfg.eval.get("batch_size", 512)),
        shuffle=False,
        num_workers=int(cfg.eval.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_action_batch,
        persistent_workers=int(cfg.eval.get("num_workers", 4)) > 0,
    )

    max_batches = int(cfg.eval.get("max_batches", 0))
    dim_sse = torch.zeros(model.action_dim, dtype=torch.float64)
    dim_abs = torch.zeros(model.action_dim, dtype=torch.float64)
    dim_count = 0
    total_sse = 0.0
    total_abs = 0.0
    total_count = 0
    vel_sse = 0.0
    vel_count = 0
    group_sums: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    code_counts = torch.zeros(model.codebook_size, dtype=torch.long)
    scale_code_counts = [torch.zeros(model.codebook_size, dtype=torch.long) for _ in model.scales]
    samples = 0
    batches = 0

    with torch.no_grad():
        progress = tqdm(loader, desc="eval var stage1 recon")
        for batch in progress:
            actions = batch["actions"].to(device=device, dtype=torch.float32, non_blocking=True)
            out = model(actions)
            recon = out["recon"]
            error = recon - actions
            squared_error = error.pow(2)
            abs_error = error.abs()

            total_sse += float(squared_error.sum().detach().cpu())
            total_abs += float(abs_error.sum().detach().cpu())
            total_count += int(squared_error.numel())
            dim_sse += squared_error.sum(dim=(0, 1)).detach().cpu().double()
            dim_abs += abs_error.sum(dim=(0, 1)).detach().cpu().double()
            dim_count += int(actions.shape[0] * actions.shape[1])

            if actions.shape[1] > 1:
                vel_error = (recon[:, 1:] - recon[:, :-1]) - (actions[:, 1:] - actions[:, :-1])
                vel_sse += float(vel_error.pow(2).sum().detach().cpu())
                vel_count += int(vel_error.numel())

            _update_group_sums(group_sums, group_counts, squared_error.detach().cpu(), dataset.action_spec.dim_groups)

            flat_tokens = out["flat_token_ids"].detach().reshape(-1).cpu()
            code_counts += torch.bincount(flat_tokens, minlength=model.codebook_size)
            for idx, token_ids in enumerate(out["token_ids"]):
                scale_code_counts[idx] += torch.bincount(
                    token_ids.detach().reshape(-1).cpu(),
                    minlength=model.codebook_size,
                )

            samples += int(actions.shape[0])
            batches += 1
            progress.set_postfix(mse=f"{total_sse / max(total_count, 1):.6f}")
            if max_batches > 0 and batches >= max_batches:
                break

    used_codes = int((code_counts > 0).sum().item())
    total_tokens = int(code_counts.sum().item())
    probs = code_counts[code_counts > 0].double() / max(total_tokens, 1)
    perplexity = float(torch.exp(-(probs * probs.log()).sum()).item()) if probs.numel() else 0.0

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "num_samples": samples,
        "num_batches": batches,
        "dataset_len": len(dataset),
        "action_spec": dataset.action_spec.to_dict(),
        "model_config": model.get_config(),
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
        "codebook": {
            "codebook_size": model.codebook_size,
            "used_codes": used_codes,
            "usage_ratio": used_codes / float(model.codebook_size),
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "scale_usage": [
                {
                    "scale": int(scale),
                    "used_codes": int((counts > 0).sum().item()),
                    "usage_ratio": int((counts > 0).sum().item()) / float(model.codebook_size),
                    "total_tokens": int(counts.sum().item()),
                }
                for scale, counts in zip(model.scales, scale_code_counts, strict=True)
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VAR Stage 1 action tokenizer reconstruction.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("playground/Checkpoints/var_stage1_pi05_libero/best_recon.ckpt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("playground/Checkpoints/var_stage1_pi05_libero/reconstruction_eval.json"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.create(
        {
            "eval": {
                "device": args.device,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "max_batches": args.max_batches,
            }
        }
    )
    report = evaluate(cfg, args.checkpoint, args.output)
    print(json.dumps(report["metrics"], indent=2))
    print(json.dumps(report["codebook"], indent=2))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
