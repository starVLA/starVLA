#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch
from omegaconf import OmegaConf


def build_cfg(
    model_name: str,
    *,
    tokenizer_path: str,
    action_dim: int,
    action_horizon: int,
    max_state_dim: int,
    max_token_len: int,
    discrete_state_input: bool,
    num_inference_steps: int,
) -> OmegaConf:
    return OmegaConf.create(
        {
            "framework": {
                "name": model_name,
                "precision": "float32",
                "action_dim": action_dim,
                "action_horizon": action_horizon,
                "max_state_dim": max_state_dim,
                "max_token_len": max_token_len,
                "discrete_state_input": discrete_state_input,
                "num_inference_steps": num_inference_steps,
                "tokenizer": {
                    "model_path": tokenizer_path,
                    "pad_state_value": -2.0,
                },
                "paligemma": {
                    "model_name_or_path": "",
                },
            },
            "trainer": {},
            "datasets": {},
        }
    )


def build_model(args: argparse.Namespace):
    cfg = build_cfg(
        args.model.upper(),
        tokenizer_path=args.tokenizer,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_state_dim=args.max_state_dim,
        max_token_len=args.max_token_len,
        discrete_state_input=args.discrete_state_input,
        num_inference_steps=args.num_inference_steps,
    )
    model_name = args.model.upper()
    if model_name == "PI0":
        from starVLA.model.framework.VLM4A.PI0 import PI0

        return PI0(cfg)
    if model_name == "PI05":
        from starVLA.model.framework.VLM4A.PI05 import PI05

        return PI05(cfg)
    raise ValueError(f"Unsupported model: {model_name}")


def save_state_dict(state_dict: dict[str, torch.Tensor], output_dir: Path, save_format: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_format == "safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, str(output_dir / "model.safetensors"))
    elif save_format == "pt":
        torch.save(state_dict, output_dir / "pytorch_model.pt")
    else:
        raise ValueError(f"Unsupported save format: {save_format}")


def copy_assets_if_present(source_checkpoint: Path, output_dir: Path) -> None:
    source_assets = source_checkpoint / "assets"
    if not source_assets.exists():
        return
    target_assets = output_dir / "assets"
    if target_assets.exists():
        shutil.rmtree(target_assets)
    shutil.copytree(source_assets, target_assets)


def save_config_yaml(output_root: Path, cfg: OmegaConf) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_root / "config.yaml")


def _libero_stats_key(model_name: str) -> str:
    return f"openpi_{model_name.lower()}_libero_franka"


def save_dataset_statistics_if_present(source_checkpoint: Path, output_root: Path, model_name: str) -> None:
    norm_stats_path = source_checkpoint / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    if not norm_stats_path.exists():
        return
    with open(norm_stats_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    norm_stats = raw.get("norm_stats")
    if norm_stats is None:
        raise KeyError(f"`norm_stats` missing in {norm_stats_path}")
    dataset_statistics = {
        _libero_stats_key(model_name): {
            "state": norm_stats["state"],
            "action": norm_stats["actions"],
        }
    }
    (output_root / "dataset_statistics.json").write_text(
        json.dumps(dataset_statistics, indent=2),
        encoding="utf-8",
    )


def export_variant(
    model_name: str,
    model,
    output_root: Path,
    dtype_name: str,
    save_format: str,
    source_checkpoint: Path,
) -> None:
    if dtype_name == "fp32":
        model.to_precision("float32")
        variant_dir = output_root / "fp32"
    elif dtype_name == "bfloat16":
        model.to_precision("bfloat16")
        variant_dir = output_root / "bfloat16"
    else:
        raise ValueError(f"Unsupported dtype variant: {dtype_name}")

    state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    save_state_dict(state_dict, variant_dir, save_format)
    metadata = {
        "model": model_name,
        "source_checkpoint": str(source_checkpoint),
        "variant": dtype_name,
        "save_format": save_format,
    }
    (variant_dir / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenPI PI0/PI05 checkpoints into StarVLA-native checkpoints."
    )
    parser.add_argument("--model", required=True, choices=["PI0", "PI05", "pi0", "pi05"])
    parser.add_argument("--checkpoint", type=Path, required=True, help="OpenPI checkpoint directory or file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for converted StarVLA checkpoints.")
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Optional tokenizer.model path. Only needed if you want the saved config to carry a tokenizer path.",
    )
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--max-state-dim", type=int, default=32)
    parser.add_argument("--max-token-len", type=int, default=None)
    parser.add_argument("--discrete-state-input", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["fp32", "bfloat16"],
        choices=["fp32", "bfloat16"],
        help="Checkpoint variants to export.",
    )
    parser.add_argument(
        "--save-format",
        default="safetensors",
        choices=["safetensors", "pt"],
        help="Output checkpoint format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model.upper()
    source_checkpoint = args.checkpoint
    output_root = args.output_dir
    if args.action_horizon is None:
        args.action_horizon = 10 if model_name == "PI05" else 50
    if args.max_token_len is None:
        args.max_token_len = 200 if model_name == "PI05" else 48

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model = build_model(args)
    model.load_pretrained_checkpoint(source_checkpoint)
    cfg = build_cfg(
        model_name,
        tokenizer_path=args.tokenizer,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_state_dim=args.max_state_dim,
        max_token_len=args.max_token_len,
        discrete_state_input=args.discrete_state_input,
        num_inference_steps=args.num_inference_steps,
    )
    save_config_yaml(output_root, cfg)
    source_root = source_checkpoint if source_checkpoint.is_dir() else source_checkpoint.parent
    copy_assets_if_present(source_root, output_root)
    save_dataset_statistics_if_present(source_root, output_root, model_name)

    for variant in args.variants:
        export_variant(
            model_name=model_name,
            model=model,
            output_root=output_root,
            dtype_name=variant,
            save_format=args.save_format,
            source_checkpoint=source_root,
        )
        print(f"saved {model_name} {variant} checkpoint to {output_root / variant}")


if __name__ == "__main__":
    main()
