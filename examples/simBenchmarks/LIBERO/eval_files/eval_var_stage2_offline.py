"""Offline evaluation for QwenVAR / VAR Stage 2 token policies."""

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
from starVLA.model.framework.base_framework import baseframework, build_framework
from starVLA.training.train_var_stage1 import load_starvla_base_config
from starVLA.utils.var_stage2_metrics import decoded_action_metrics, token_accuracy_by_scale


def _resolve_checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {path}")
    candidates = [
        path / "model.safetensors",
        path / "pytorch_model.pt",
    ]
    candidates.extend(sorted(path.glob("*_model.safetensors"), reverse=True))
    candidates.extend(sorted(path.glob("*_pytorch_model.pt"), reverse=True))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported model checkpoint found under {path}")


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    ckpt_file = _resolve_checkpoint_file(path)
    if ckpt_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(ckpt_file), device="cpu")
    try:
        return torch.load(ckpt_file, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(ckpt_file, map_location="cpu")


def _load_model_from_config(config_yaml: Path, device: torch.device, checkpoint: Path | None = None):
    cfg = OmegaConf.load(config_yaml)
    model = build_framework(cfg)
    if checkpoint is not None:
        state_dict = _load_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys while loading Stage 2 checkpoint: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys while loading Stage 2 checkpoint: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model, cfg


def _load_model_from_pretrained(config_yaml: Path, device: torch.device, checkpoint: Path):
    cfg = OmegaConf.load(config_yaml)
    model = baseframework.from_pretrained(str(checkpoint))
    model.to(device)
    model.eval()
    return model, cfg


def _parse_sample_indices(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _build_dataset(
    cfg: Any,
    *,
    token_cache_path: Path | None,
    device: torch.device,
    sample_indices: list[int] | None = None,
) -> VARStage2TokenDataset:
    stage1_cfg = OmegaConf.load(cfg.framework.stage1_tokenizer.stage1_config)
    base_cfg = load_starvla_base_config(stage1_cfg)
    stage1_path = cfg.framework.stage1_tokenizer.get("artifact", None) or cfg.framework.stage1_tokenizer.get("checkpoint", None)
    if stage1_path is None:
        raise ValueError("Stage 2 config requires framework.stage1_tokenizer.artifact or .checkpoint.")
    return VARStage2TokenDataset(
        base_cfg,
        stage1_artifact_path=stage1_path,
        token_cache_path=token_cache_path,
        mode=cfg.datasets.vla_data.get("mode", "train"),
        seed=int(cfg.get("seed", 42)),
        window_mode=str(cfg.datasets.vla_data.get("window_mode", "full")),
        device=device,
        max_samples=cfg.datasets.vla_data.get("max_samples", None),
        sample_indices=sample_indices if sample_indices is not None else cfg.datasets.vla_data.get("sample_indices", None),
        skip_bad_samples=bool(cfg.datasets.vla_data.get("skip_bad_samples", False)),
        max_read_retries=int(cfg.datasets.vla_data.get("max_read_retries", 8)),
    )


def _update_mean(total: dict[str, float], count: int, metrics: dict[str, Any], batch_size: int) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0.0) + float(value) * batch_size


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if args.use_from_pretrained:
        if args.checkpoint is None:
            raise ValueError("--use_from_pretrained requires --checkpoint")
        model, cfg = _load_model_from_pretrained(args.config_yaml, device, args.checkpoint)
    else:
        model, cfg = _load_model_from_config(args.config_yaml, device, checkpoint=args.checkpoint)
    dataset = _build_dataset(
        cfg,
        token_cache_path=args.token_cache,
        device=device,
        sample_indices=_parse_sample_indices(args.sample_indices),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_var_stage2_token_batch,
    )

    token_metric_sums: dict[str, float] = {}
    action_metric_sums: dict[str, float] = {}
    invalid_generations = 0
    samples = 0
    examples_out = []
    scales = list(model.stage1_tokenizer.scales)
    product_codebook_groups = int(getattr(model.stage1_tokenizer, "product_codebook_groups", 1))

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="eval var stage2 offline")):
            output = model.predict_action(
                batch,
                max_new_tokens=args.max_new_tokens,
                constrain_to_action_tokens=args.constrain_to_action_tokens,
            )
            pred_tokens = torch.as_tensor(output["action_tokens"], dtype=torch.long)
            target_tokens = torch.stack([item["action_tokens"] for item in batch], dim=0).long()
            pred_actions = torch.as_tensor(output["normalized_actions"], dtype=torch.float32)
            target_actions = torch.stack([torch.as_tensor(item["action"], dtype=torch.float32) for item in batch], dim=0)

            batch_size = len(batch)
            token_metrics = token_accuracy_by_scale(
                pred_tokens,
                target_tokens,
                scales=scales,
                product_codebook_groups=product_codebook_groups,
            )
            action_metrics = decoded_action_metrics(pred_actions, target_actions, dim_groups=dataset.action_spec.dim_groups)
            _update_mean(token_metric_sums, samples, token_metrics, batch_size)
            _update_mean(action_metric_sums, samples, action_metrics, batch_size)

            diagnostics = output.get("generation_diagnostics", [])
            invalid_generations += sum(1 for item in diagnostics if item.get("valid_token_count", 0) < model.token_dim)
            samples += batch_size

            if len(examples_out) < args.num_debug_examples:
                for row, item in enumerate(batch):
                    examples_out.append(
                        {
                            "metadata": item.get("metadata", {}),
                            "target_tokens": target_tokens[row].tolist(),
                            "predicted_tokens": pred_tokens[row].tolist(),
                            "diagnostics": diagnostics[row] if row < len(diagnostics) else {},
                        }
                    )
                    if len(examples_out) >= args.num_debug_examples:
                        break

            if args.max_batches > 0 and batch_idx + 1 >= args.max_batches:
                break

    report = {
        "config_yaml": str(args.config_yaml),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "num_samples": samples,
        "stage1_artifact_id": model.stage1_artifact_id,
        "token_metrics": {key: value / max(samples, 1) for key, value in token_metric_sums.items()},
        "decoded_action_metrics": {key: value / max(samples, 1) for key, value in action_metric_sums.items()},
        "invalid_generation_rate": invalid_generations / max(samples, 1),
        "debug_examples": examples_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline eval for VAR Stage 2 token policy.")
    parser.add_argument("--config_yaml", type=Path, default=Path("examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_libero.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional Stage 2 checkpoint file or directory.")
    parser.add_argument("--use_from_pretrained", action="store_true", help="Load with baseframework.from_pretrained, matching the online server path.")
    parser.add_argument("--output", type=Path, default=Path("playground/Checkpoints/qwen_var_libero_stage2/offline_eval.json"))
    parser.add_argument("--token_cache", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--constrain_to_action_tokens", action="store_true")
    parser.add_argument("--num_debug_examples", type=int, default=8)
    parser.add_argument("--sample_indices", type=str, default=None, help="Comma-separated source dataset indices to evaluate.")
    args = parser.parse_args()

    report = evaluate(args)
    print(json.dumps({k: report[k] for k in ("num_samples", "token_metrics", "decoded_action_metrics", "invalid_generation_rate")}, indent=2))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
