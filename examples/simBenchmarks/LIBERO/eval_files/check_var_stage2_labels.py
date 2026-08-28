"""Sanity checks for VAR Stage 2 token labels and Stage 1 decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from starVLA.dataloader.var_stage2_token_dataset import VARStage2TokenDataset
from starVLA.model.modules.action_tokenizer import VARTokenTextCodec
from starVLA.training.train_var_stage1 import load_starvla_base_config
from starVLA.utils.var_stage2_metrics import decoded_action_metrics


def _build_dataset(args: argparse.Namespace) -> VARStage2TokenDataset:
    cfg = OmegaConf.load(args.config_yaml)
    stage1_cfg = OmegaConf.load(cfg.framework.stage1_tokenizer.stage1_config)
    base_cfg = load_starvla_base_config(stage1_cfg)
    stage1_path = cfg.framework.stage1_tokenizer.get("artifact", None) or cfg.framework.stage1_tokenizer.get("checkpoint", None)
    if stage1_path is None:
        raise ValueError("Stage 2 config requires framework.stage1_tokenizer.artifact or .checkpoint.")
    token_cache = args.token_cache or cfg.framework.stage1_tokenizer.get("token_cache", None)
    return VARStage2TokenDataset(
        base_cfg,
        stage1_artifact_path=stage1_path,
        token_cache_path=token_cache,
        validate_cache_online=False,
        mode=cfg.datasets.vla_data.get("mode", "train"),
        seed=int(cfg.get("seed", 42)),
        window_mode=str(cfg.datasets.vla_data.get("window_mode", "full")),
        device=args.device,
        max_samples=args.max_samples,
        sample_indices=args.sample_indices,
        skip_bad_samples=bool(cfg.datasets.vla_data.get("skip_bad_samples", False)),
        max_read_retries=int(cfg.datasets.vla_data.get("max_read_retries", 8)),
    )


def _check_qwen_tokenizer(model_path: str, codec: VARTokenTextCodec) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in codec.all_token_strings()]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    missing = []
    for token, token_id in zip(codec.all_token_strings(), token_ids, strict=True):
        if token_id < 0 or (unk_id is not None and token_id == unk_id):
            missing.append(token)
    return {
        "model_path": model_path,
        "missing_count": len(missing),
        "first_missing": missing[:10],
        "min_tokenizer_id": min(token_ids) if token_ids else None,
        "max_tokenizer_id": max(token_ids) if token_ids else None,
        "is_contiguous": sorted(token_ids) == list(range(min(token_ids), max(token_ids) + 1)) if token_ids else False,
    }


@torch.no_grad()
def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    cfg = OmegaConf.load(args.config_yaml)
    dataset = _build_dataset(args)
    tokenizer = dataset.stage1_artifact.tokenizer
    tokenizer.eval()
    codec_cfg = cfg.framework.get("action_token_text", {})
    codec = VARTokenTextCodec(
        codebook_size=int(tokenizer.codebook_size),
        prefix=str(codec_cfg.get("prefix", "<var_action_")),
        suffix=str(codec_cfg.get("suffix", ">")),
    )

    action_metric_sums: dict[str, float] = {}
    token_text_roundtrip_failures = []
    online_cache_mismatches = []
    online_cache_equal_count = 0
    checked = 0
    examples = []
    for index in range(min(len(dataset), args.num_samples)):
        item = dataset[index]
        target_actions = torch.as_tensor(item["action"], dtype=torch.float32).unsqueeze(0)
        target_tokens = item["action_tokens"].long().unsqueeze(0)
        if args.validate_cache_online:
            online_tokens = dataset._encode_actions(item["action"])
            equal = torch.equal(item["action_tokens"].long(), online_tokens)
            online_cache_equal_count += int(equal)
            if not equal and len(online_cache_mismatches) < args.num_debug_examples:
                mismatch_positions = torch.nonzero(item["action_tokens"].long() != online_tokens, as_tuple=False).flatten()
                online_cache_mismatches.append(
                    {
                        "index": index,
                        "metadata": item.get("metadata", {}),
                        "num_mismatched_tokens": int(mismatch_positions.numel()),
                        "first_mismatch_positions": mismatch_positions[:16].tolist(),
                        "cached_first_tokens": item["action_tokens"][:16].long().tolist(),
                        "online_first_tokens": online_tokens[:16].long().tolist(),
                    }
                )
        decoded_actions = tokenizer.decode(target_tokens.to(next(tokenizer.parameters()).device)).detach().cpu().float()
        metrics = decoded_action_metrics(decoded_actions, target_actions, dim_groups=dataset.action_spec.dim_groups)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                action_metric_sums[key] = action_metric_sums.get(key, 0.0) + float(value)

        token_text = codec.ids_to_text(target_tokens[0].tolist())
        roundtrip_tokens = codec.ids_from_text(token_text, expected_len=dataset.token_dim, strict=True)
        if roundtrip_tokens != target_tokens[0].tolist():
            token_text_roundtrip_failures.append(index)

        if len(examples) < args.num_debug_examples:
            examples.append(
                {
                    "index": index,
                    "metadata": item.get("metadata", {}),
                    "first_tokens": target_tokens[0, : min(16, dataset.token_dim)].tolist(),
                    "decoded_mse": metrics["mse"],
                    "decoded_mae": metrics["mae"],
                    "token_text_prefix": token_text[:160],
                }
            )
        checked += 1

    report = {
        "config_yaml": str(args.config_yaml),
        "token_cache": str(args.token_cache) if args.token_cache is not None else cfg.framework.stage1_tokenizer.get("token_cache", None),
        "checked_samples": checked,
        "dataset_len": len(dataset),
        "stage1_artifact_id": dataset.stage1_artifact.artifact_id,
        "token_dim": dataset.token_dim,
        "codebook_size": int(tokenizer.codebook_size),
        "decode_metrics": {key: value / max(checked, 1) for key, value in action_metric_sums.items()},
        "token_text_roundtrip_failures": token_text_roundtrip_failures,
        "online_cache_check": {
            "enabled": bool(args.validate_cache_online),
            "equal_count": online_cache_equal_count if args.validate_cache_online else None,
            "mismatch_count": checked - online_cache_equal_count if args.validate_cache_online else None,
            "mismatch_examples": online_cache_mismatches,
        },
        "debug_examples": examples,
    }
    if args.check_qwen_tokenizer:
        report["qwen_tokenizer"] = _check_qwen_tokenizer(str(cfg.framework.qwenvl.base_vlm), codec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check VAR Stage 2 label/cache/tokenizer alignment.")
    parser.add_argument("--config_yaml", type=Path, default=Path("examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_productvq_g8_last8_20k.yaml"))
    parser.add_argument("--token_cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("playground/Checkpoints/var_stage2_label_sanity.json"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_indices", type=int, nargs="*", default=None)
    parser.add_argument("--num_debug_examples", type=int, default=4)
    parser.add_argument("--validate_cache_online", action="store_true")
    parser.add_argument("--check_qwen_tokenizer", action="store_true")
    args = parser.parse_args()

    report = run_checks(args)
    print(
        json.dumps(
            {k: report[k] for k in ("checked_samples", "dataset_len", "decode_metrics", "token_text_roundtrip_failures", "online_cache_check")},
            indent=2,
        )
    )
    if "qwen_tokenizer" in report:
        print(json.dumps({"qwen_tokenizer": report["qwen_tokenizer"]}, indent=2))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
