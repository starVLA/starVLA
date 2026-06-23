"""Build a token-label cache for LIBERO VAR Stage 2 training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from starVLA.model.modules.action_tokenizer import load_frozen_var_action_tokenizer
from starVLA.training.train_var_stage1 import load_starvla_base_config


def collate_var_stage2_token_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def _metadata_from_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item["metadata"]) for item in batch]


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    cfg = OmegaConf.load(args.config_yaml)
    stage1_artifact = load_frozen_var_action_tokenizer(args.stage1_artifact, device=args.device)
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

        base_cfg = load_starvla_base_config(cfg)
        dataset = VARStage1ActionDataset(
            base_cfg,
            mode=args.mode,
            balance_dataset_weights=bool(cfg.data.get("balance_dataset_weights", False)),
            balance_trajectory_weights=bool(cfg.data.get("balance_trajectory_weights", False)),
            seed=int(cfg.experiment.get("seed", 42)),
            return_raw_actions=False,
            window_mode=str(cfg.data.get("window_mode", "full")),
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_var_stage2_token_batch,
        persistent_workers=args.num_workers > 0,
    )

    token_rows = []
    metadata_rows: list[dict[str, Any]] = []
    max_batches = int(args.max_batches)
    tokenizer = stage1_artifact.tokenizer
    tokenizer.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="build var stage2 token cache")):
            actions = torch.stack([item["actions"] for item in batch], dim=0).to(
                device=next(tokenizer.parameters()).device,
                dtype=torch.float32,
            )
            token_rows.append(tokenizer.encode(actions).detach().cpu().long())
            batch_metadata = _metadata_from_batch(batch)
            for item in batch_metadata:
                item["stage1_artifact_id"] = stage1_artifact.artifact_id
            metadata_rows.extend(batch_metadata)
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break

    tokens = torch.cat(token_rows, dim=0) if token_rows else torch.empty(0, tokenizer.token_dim, dtype=torch.long)
    payload = {
        "metadata": {
            "stage1_artifact_id": stage1_artifact.artifact_id,
            "stage1_checkpoint": str(stage1_artifact.checkpoint_path),
            "stage1_checkpoint_sha256": stage1_artifact.checkpoint_sha256,
            "action_spec": dataset.action_spec.to_dict(),
            "token_dim": tokenizer.token_dim,
            "codebook_size": stage1_artifact.codebook_size,
            "source_dataset_len": len(dataset),
            "cached_len": int(tokens.shape[0]),
            "window_mode": dataset.window_mode,
            "config_yaml": str(args.config_yaml),
            "mode": str(args.mode),
        },
        "tokens": tokens,
        "sample_metadata": metadata_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_name(f"{args.output.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(args.output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LIBERO VAR Stage 2 action-token label cache.")
    parser.add_argument("--config_yaml", type=Path, default=Path("examples/LIBERO/train_files/train_var_stage1_pi05_libero.yaml"))
    parser.add_argument("--stage1_artifact", type=Path, default=Path("playground/Checkpoints/var_stage1_pi05_libero/best_recon.ckpt"))
    parser.add_argument("--output", type=Path, default=Path("playground/Checkpoints/var_stage1_pi05_libero/stage2_token_cache.pt"))
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0, help="Use >0 for smoke tests.")
    args = parser.parse_args()

    payload = build_cache(args)
    print(
        {
            "output": str(args.output),
            "tokens_shape": tuple(payload["tokens"].shape),
            "stage1_artifact_id": payload["metadata"]["stage1_artifact_id"],
        }
    )


if __name__ == "__main__":
    main()
