"""Train a standalone VQ-VLA-style residual VQ-VAE action tokenizer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.var_stage1_action_dataset import VARStage1ActionDataset
from starVLA.model.modules.action_tokenizer import VQVLARVQActionTokenizer
from starVLA.training.train_var_stage1 import (
    collate_action_batch,
    compute_vq_weight,
    load_starvla_base_config,
    loss_group_weights,
    save_json,
    weighted_dim_mse,
)


def override_action_indices_for_horizon(data_mix: str, horizon: int) -> None:
    """Use the paper's action chunk length without editing shared data configs."""

    for _, _, robot_type in DATASET_NAMED_MIXTURES[str(data_mix)]:
        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        if hasattr(data_config, "action_indices"):
            data_config.action_indices = list(range(int(horizon)))


def save_checkpoint(
    path: Path,
    *,
    model: VQVLARVQActionTokenizer,
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


def train(cfg: Any) -> None:
    torch.manual_seed(int(cfg.experiment.get("seed", 42)))
    device = torch.device(cfg.train.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_starvla_base_config(cfg)
    if cfg.data.get("expected_action_horizon", None) is not None:
        override_action_indices_for_horizon(str(base_cfg.datasets.vla_data.data_mix), int(cfg.data.expected_action_horizon))
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
    save_json(output_dir / "action_spec.json", action_spec.to_dict())
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    OmegaConf.save(base_cfg, output_dir / "starvla_base_config.yaml")

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
    model = VQVLARVQActionTokenizer(
        action_dim=model_action_dim,
        seq_len=model_seq_len,
        embed_dim=int(cfg.model.get("embed_dim", 32)),
        codebook_size=int(cfg.model.get("codebook_size", 512)),
        residual_vq_layers=int(cfg.model.get("residual_vq_layers", 4)),
        commitment_cost=float(cfg.model.get("commitment_cost", 0.25)),
        codebook_loss_weight=float(cfg.model.get("codebook_loss_weight", 1.0)),
        normalize_codebook_for_lookup=bool(cfg.model.get("normalize_codebook_for_lookup", False)),
        use_dilated=bool(cfg.model.get("use_dilated", True)),
        dim_groups=action_spec.dim_groups,
        use_time_embedding=bool(cfg.model.get("use_time_embedding", True)),
        use_action_type_embedding=bool(cfg.model.get("use_action_type_embedding", True)),
        input_embedding_scale=float(cfg.model.get("input_embedding_scale", 1.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.train.get("weight_decay", 1e-5)),
    )

    history: list[dict[str, Any]] = []
    best_recon = float("inf")
    best_balanced_recon = float("inf")
    min_usage_for_balanced = float(cfg.train.get("min_codebook_usage_ratio_for_balanced", 0.02))

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        totals = {"recon_loss": 0.0, "vel_loss": 0.0, "vq_loss": 0.0, "total_loss": 0.0}
        batches = 0
        code_counts = torch.zeros(model.codebook_size, dtype=torch.long)
        vq_weight = compute_vq_weight(
            epoch,
            target_weight=float(cfg.loss.get("vq_weight", 0.1)),
            warmup_epochs=int(cfg.loss.get("vq_warmup_epochs", 20)),
        )

        progress = tqdm(loader, desc=f"vqvla epoch {epoch:03d}")
        for batch in progress:
            actions = batch["actions"].to(device=device, dtype=torch.float32, non_blocking=True)
            out = model(actions)
            recon = out["recon"]
            recon_loss = weighted_dim_mse(
                recon,
                actions,
                dim_groups=action_spec.dim_groups,
                gripper_weight=float(cfg.loss.get("gripper_recon_weight", 1.0)),
                group_weights=loss_group_weights(cfg, "recon"),
            )
            if actions.shape[1] > 1:
                vel_loss = weighted_dim_mse(
                    recon[:, 1:] - recon[:, :-1],
                    actions[:, 1:] - actions[:, :-1],
                    dim_groups=action_spec.dim_groups,
                    gripper_weight=float(cfg.loss.get("gripper_vel_weight", 0.1)),
                    group_weights=loss_group_weights(cfg, "vel"),
                )
            else:
                vel_loss = torch.zeros((), device=device, dtype=actions.dtype)
            if actions.shape[1] > 2 and float(cfg.loss.get("jerk_weight", 0.0)) > 0.0:
                jerk_loss = F.mse_loss(
                    recon[:, 2:] - 2.0 * recon[:, 1:-1] + recon[:, :-2],
                    actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2],
                )
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
            batches += 1
            flat_tokens = out["flat_token_ids"].detach().reshape(-1).cpu()
            if flat_tokens.numel() > 0:
                code_counts += torch.bincount(flat_tokens, minlength=model.codebook_size)
            progress.set_postfix(recon=f"{totals['recon_loss'] / batches:.5f}", vq=f"{totals['vq_loss'] / batches:.5f}")

            max_batches = int(cfg.train.get("max_batches_per_epoch", 0))
            if max_batches > 0 and batches >= max_batches:
                break

        if batches == 0:
            raise RuntimeError("No batches were produced by the VQ-VLA Stage 1 dataloader.")

        epoch_record = {key: value / batches for key, value in totals.items()}
        used_codes = int((code_counts > 0).sum().item())
        usage_ratio = used_codes / float(model.codebook_size)
        epoch_record.update({"epoch": epoch, "vq_weight": vq_weight, "codebook_used": used_codes, "codebook_usage_ratio": usage_ratio})
        history.append(epoch_record)
        save_json(output_dir / "history.json", history)
        save_checkpoint(output_dir / "latest.ckpt", model=model, optimizer=optimizer, epoch=epoch, history=history, cfg=cfg, action_spec=action_spec)

        save_every_epochs = int(cfg.train.get("save_every_epochs", 0))
        if save_every_epochs > 0 and (epoch + 1) % save_every_epochs == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.ckpt", model=model, optimizer=optimizer, epoch=epoch, history=history, cfg=cfg, action_spec=action_spec)

        if epoch_record["recon_loss"] < best_recon:
            best_recon = epoch_record["recon_loss"]
            save_checkpoint(output_dir / "best_recon.ckpt", model=model, optimizer=optimizer, epoch=epoch, history=history, cfg=cfg, action_spec=action_spec)

        if usage_ratio >= min_usage_for_balanced and epoch_record["recon_loss"] < best_balanced_recon:
            best_balanced_recon = epoch_record["recon_loss"]
            save_checkpoint(output_dir / "best_balanced.ckpt", model=model, optimizer=optimizer, epoch=epoch, history=history, cfg=cfg, action_spec=action_spec)

    save_checkpoint(output_dir / "final.ckpt", model=model, optimizer=optimizer, epoch=history[-1]["epoch"], history=history, cfg=cfg, action_spec=action_spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train standalone VQ-VLA-style RVQ Stage 1 action tokenizer.")
    parser.add_argument("--config_yaml", type=str, required=True)
    args = parser.parse_args()
    train(OmegaConf.load(args.config_yaml))


if __name__ == "__main__":
    main()
