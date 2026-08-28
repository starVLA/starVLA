"""Oracle reconstruction metrics for a frozen VAR Stage 1 action tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from starVLA.dataloader.var_stage2_token_dataset import VARStage2TokenDataset, collate_var_stage2_token_batch
from starVLA.training.train_var_stage1 import load_starvla_base_config
from starVLA.utils.var_stage2_metrics import decoded_action_metrics, token_accuracy_by_scale


def _build_dataset(stage2_config_yaml: Path, token_cache_path: Path | None, device: torch.device) -> VARStage2TokenDataset:
    cfg = OmegaConf.load(stage2_config_yaml)
    stage1_cfg = OmegaConf.load(cfg.framework.stage1_tokenizer.stage1_config)
    base_cfg = load_starvla_base_config(stage1_cfg)
    stage1_path = cfg.framework.stage1_tokenizer.get("artifact", None) or cfg.framework.stage1_tokenizer.get("checkpoint", None)
    if stage1_path is None:
        raise ValueError("Stage 2 config requires framework.stage1_tokenizer.artifact or .checkpoint.")
    return VARStage2TokenDataset(
        base_cfg,
        stage1_artifact_path=stage1_path,
        token_cache_path=token_cache_path or cfg.framework.stage1_tokenizer.get("token_cache", None),
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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dataset = _build_dataset(args.stage2_config_yaml, args.token_cache, device=device)
    tokenizer = dataset.stage1_artifact.tokenizer.to(device)
    tokenizer.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_var_stage2_token_batch,
    )

    action_metric_sums: dict[str, float] = {}
    token_metric_sums: dict[str, float] = {}
    samples = 0
    examples_out = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="eval var stage1 oracle mse")):
            target_actions = torch.stack([torch.as_tensor(item["action"], dtype=torch.float32) for item in batch], dim=0).to(device)
            cached_tokens = torch.stack([item["action_tokens"].long() for item in batch], dim=0).to(device)
            encoded_tokens = tokenizer.encode(target_actions)
            reconstructed = tokenizer.decode(encoded_tokens)

            batch_size = len(batch)
            action_metrics = decoded_action_metrics(reconstructed.cpu(), target_actions.cpu(), dim_groups=dataset.action_spec.dim_groups)
            token_metrics = token_accuracy_by_scale(
                encoded_tokens.cpu(),
                cached_tokens.cpu(),
                scales=tokenizer.scales,
                product_codebook_groups=int(getattr(tokenizer, "product_codebook_groups", 1)),
            )
            _update_mean(action_metric_sums, action_metrics, batch_size)
            _update_mean(token_metric_sums, token_metrics, batch_size)
            samples += batch_size

            if len(examples_out) < args.num_debug_examples:
                for row, item in enumerate(batch):
                    examples_out.append(
                        {
                            "metadata": item.get("metadata", {}),
                            "target_action_first": target_actions[row, 0].detach().cpu().tolist(),
                            "recon_action_first": reconstructed[row, 0].detach().cpu().tolist(),
                        }
                    )
                    if len(examples_out) >= args.num_debug_examples:
                        break

            if args.max_batches > 0 and batch_idx + 1 >= args.max_batches:
                break
            if args.max_samples > 0 and samples >= args.max_samples:
                break

    report = {
        "stage2_config_yaml": str(args.stage2_config_yaml),
        "token_cache": str(args.token_cache) if args.token_cache is not None else None,
        "num_samples": samples,
        "stage1_artifact_id": dataset.stage1_artifact.artifact_id,
        "decoded_action_metrics": {key: value / max(samples, 1) for key, value in action_metric_sums.items()},
        "token_cache_match_metrics": {key: value / max(samples, 1) for key, value in token_metric_sums.items()},
        "debug_examples": examples_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen Stage 1 oracle encode/decode MSE.")
    parser.add_argument("--stage2_config_yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token_cache", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_debug_examples", type=int, default=4)
    args = parser.parse_args()

    report = evaluate(args)
    print(
        json.dumps(
            {
                "num_samples": report["num_samples"],
                "decoded_action_metrics": report["decoded_action_metrics"],
                "token_cache_match_metrics": report["token_cache_match_metrics"],
            },
            indent=2,
        )
    )
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
