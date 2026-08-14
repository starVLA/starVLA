"""Offline normalized-action MSE evaluation for StarVLA baseline checkpoints."""

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

from starVLA.dataloader.var_stage2_token_dataset import VARStage2TokenDataset, collate_var_stage2_token_batch
from starVLA.model.framework.base_framework import baseframework
from starVLA.training.train_var_stage1 import load_starvla_base_config
from starVLA.utils.var_stage2_metrics import decoded_action_metrics


def _build_stage2_dataset(stage2_config_yaml: Path, device: torch.device) -> VARStage2TokenDataset:
    cfg = OmegaConf.load(stage2_config_yaml)
    stage1_cfg = OmegaConf.load(cfg.framework.stage1_tokenizer.stage1_config)
    base_cfg = load_starvla_base_config(stage1_cfg)
    stage1_path = cfg.framework.stage1_tokenizer.get("artifact", None) or cfg.framework.stage1_tokenizer.get("checkpoint", None)
    if stage1_path is None:
        raise ValueError("Stage 2 config requires framework.stage1_tokenizer.artifact or .checkpoint.")
    return VARStage2TokenDataset(
        base_cfg,
        stage1_artifact_path=stage1_path,
        token_cache_path=cfg.framework.stage1_tokenizer.get("token_cache", None),
        mode=cfg.datasets.vla_data.get("mode", "train"),
        seed=int(cfg.get("seed", 42)),
        window_mode=str(cfg.datasets.vla_data.get("window_mode", "full")),
        skip_bad_samples=bool(cfg.datasets.vla_data.get("skip_bad_samples", False)),
        max_read_retries=int(cfg.datasets.vla_data.get("max_read_retries", 16)),
        device=device,
    )


def _update_mean(total: dict[str, float], metrics: dict[str, Any], batch_size: int) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0.0) + float(value) * batch_size


def _as_action_tensor(predicted: Any, target: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(predicted, dtype=np.float32), dtype=torch.float32)
    if tensor.shape != target.shape:
        raise ValueError(f"predicted action shape {tuple(tensor.shape)} != target shape {tuple(target.shape)}")
    return tensor


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model = baseframework.from_pretrained(str(args.checkpoint))
    model.to(device)
    model.eval()

    dataset = _build_stage2_dataset(args.stage2_config_yaml, device=torch.device("cpu"))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_var_stage2_token_batch,
    )

    metric_sums: dict[str, float] = {}
    valid_samples = 0
    invalid_samples = 0
    debug_examples = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="eval starvla baseline offline mse")):
            target_actions = torch.stack([torch.as_tensor(item["action"], dtype=torch.float32) for item in batch], dim=0)
            try:
                output = model.predict_action(batch)
                pred_actions = _as_action_tensor(output["normalized_actions"], target_actions)
            except Exception as exc:
                invalid_samples += len(batch)
                if len(debug_examples) < args.num_debug_examples:
                    debug_examples.append(
                        {
                            "batch_idx": batch_idx,
                            "error": f"{type(exc).__name__}: {exc}",
                            "metadata": [item.get("metadata", {}) for item in batch],
                        }
                    )
                if args.fail_fast:
                    raise
                continue

            metrics = decoded_action_metrics(pred_actions, target_actions, dim_groups=dataset.action_spec.dim_groups)
            _update_mean(metric_sums, metrics, len(batch))
            valid_samples += len(batch)

            if len(debug_examples) < args.num_debug_examples:
                for row, item in enumerate(batch):
                    debug_examples.append(
                        {
                            "metadata": item.get("metadata", {}),
                            "target_action_first": target_actions[row, 0].tolist(),
                            "pred_action_first": pred_actions[row, 0].tolist(),
                        }
                    )
                    if len(debug_examples) >= args.num_debug_examples:
                        break

            if args.max_batches > 0 and batch_idx + 1 >= args.max_batches:
                break
            if args.max_samples > 0 and valid_samples + invalid_samples >= args.max_samples:
                break

    report = {
        "checkpoint": str(args.checkpoint),
        "stage2_config_yaml": str(args.stage2_config_yaml),
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "total_attempted": valid_samples + invalid_samples,
        "decoded_action_metrics": {key: value / max(valid_samples, 1) for key, value in metric_sums.items()},
        "invalid_rate": invalid_samples / max(valid_samples + invalid_samples, 1),
        "debug_examples": debug_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StarVLA baseline normalized-action MSE on Stage 2 samples.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage2_config_yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_debug_examples", type=int, default=4)
    parser.add_argument("--fail_fast", action="store_true")
    args = parser.parse_args()

    report = evaluate(args)
    print(
        json.dumps(
            {
                "valid_samples": report["valid_samples"],
                "invalid_samples": report["invalid_samples"],
                "decoded_action_metrics": report["decoded_action_metrics"],
                "invalid_rate": report["invalid_rate"],
            },
            indent=2,
        )
    )
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
