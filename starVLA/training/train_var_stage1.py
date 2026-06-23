"""Train a VAR-style Stage 1 action tokenizer on StarVLA action chunks."""

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

from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.modules.action_tokenizer import VARActionTokenizer, default_scales


def collate_action_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        "actions": torch.stack([item["actions"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
    }
    if "actions_raw" in batch[0]:
        output["actions_raw"] = torch.stack([item["actions_raw"] for item in batch], dim=0)
    return output


def load_starvla_base_config(cfg: Any) -> Any:
    base_path = cfg.data.starvla_config_yaml
    base_cfg = OmegaConf.load(base_path)
    base_cfg = apply_config_compat(base_cfg)

    if cfg.data.get("data_root_dir", None) is not None:
        base_cfg.datasets.vla_data.data_root_dir = cfg.data.data_root_dir
    if cfg.data.get("data_mix", None) is not None:
        base_cfg.datasets.vla_data.data_mix = cfg.data.data_mix
    if cfg.data.get("video_backend", None) is not None:
        base_cfg.datasets.vla_data.video_backend = cfg.data.video_backend
    for optional_key in (
        "action_mode",
        "action_mode_apply_keys",
        "action_mode_state_map",
        "include_state",
        "load_all_data_for_training",
        "delete_pause_frame",
        "obs_image_size",
    ):
        if cfg.data.get(optional_key, None) is not None:
            base_cfg.datasets.vla_data[optional_key] = cfg.data[optional_key]

    if cfg.data.get("expected_action_horizon", None) is not None:
        base_cfg.framework.action_model.action_horizon = int(cfg.data.expected_action_horizon)
    if cfg.data.get("expected_action_dim", None) is not None:
        base_cfg.framework.action_model.action_dim = int(cfg.data.expected_action_dim)

    # Stage 1 does not use image/language, but the underlying StarVLA dataset
    # currently expects those modalities. Keep the config faithful and discard
    # non-action fields in VARStage1ActionDataset.
    return base_cfg


def resolve_scales(scales_cfg: Any, seq_len: int) -> list[int]:
    if scales_cfg in (None, "auto"):
        return default_scales(seq_len)
    return [int(item) for item in scales_cfg]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_checkpoint(
    path: Path,
    *,
    model: VARActionTokenizer,
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


def load_init_checkpoint(path: str, *, model: VARActionTokenizer, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load init checkpoint {checkpoint_path}: "
            f"missing keys={missing}, unexpected keys={unexpected}"
        )
    return checkpoint


def load_resume_checkpoint(
    path: str,
    *,
    model: VARActionTokenizer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load resume checkpoint {checkpoint_path}: "
            f"missing keys={missing}, unexpected keys={unexpected}"
        )
    if "optimizer_state_dict" not in checkpoint:
        raise KeyError(f"Resume checkpoint {checkpoint_path} does not contain optimizer_state_dict.")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_encoder_decoder_init(path: str, *, model: VARActionTokenizer, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    prefixes = (
        "encoder_",
        "decoder_",
    )
    selected = {
        key: value
        for key, value in source_state.items()
        if key.startswith(prefixes) and key in target_state and target_state[key].shape == value.shape
    }
    missing = sorted(key for key in target_state if key.startswith(prefixes) and key not in selected)
    model.load_state_dict(selected, strict=False)
    if missing:
        print(f"Skipped {len(missing)} encoder/decoder init keys due to missing or shape mismatch.")
    return checkpoint


@torch.no_grad()
def initialize_codebook_from_data(
    *,
    model: VARActionTokenizer,
    loader: DataLoader,
    device: torch.device,
    batches: int,
    noise_scale: float = 0.0,
) -> None:
    if batches <= 0:
        return

    model.eval()
    samples: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        actions = batch["actions"].to(device=device, dtype=torch.float32, non_blocking=True)
        latent_full = model.encode_features(actions)
        for scale in model.scales:
            z_scale = F.interpolate(latent_full, size=scale, mode="linear", align_corners=False)
            samples.append(z_scale.permute(0, 2, 1).reshape(-1, model.embed_dim).detach().cpu())
        if batch_index + 1 >= batches:
            break

    if not samples:
        raise RuntimeError("No latent samples were collected for codebook initialization.")

    latent_samples = torch.cat(samples, dim=0)
    if latent_samples.shape[0] < model.codebook_size:
        repeat_factor = (model.codebook_size + latent_samples.shape[0] - 1) // latent_samples.shape[0]
        latent_samples = latent_samples.repeat(repeat_factor, 1)

    if model.quantization_mode == "product_vq":
        group_dim = model.embed_dim // model.product_codebook_groups
        for group_idx, codebook_module in enumerate(model.product_codebooks):
            group_samples = latent_samples[:, group_idx * group_dim : (group_idx + 1) * group_dim]
            perm = torch.randperm(group_samples.shape[0])[: model.codebook_size]
            codebook = group_samples.index_select(0, perm).to(device=device, dtype=codebook_module.weight.dtype)
            if noise_scale > 0.0:
                codebook = codebook + float(noise_scale) * torch.randn_like(codebook)
            if model.normalize_codebook_for_lookup:
                codebook = F.normalize(codebook, dim=-1, eps=1e-6)
            codebook_module.weight.copy_(codebook)
    else:
        perm = torch.randperm(latent_samples.shape[0])[: model.codebook_size]
        codebook = latent_samples.index_select(0, perm).to(device=device, dtype=model.shared_codebook.weight.dtype)
        if noise_scale > 0.0:
            codebook = codebook + float(noise_scale) * torch.randn_like(codebook)
        if model.normalize_codebook_for_lookup:
            codebook = F.normalize(codebook, dim=-1, eps=1e-6)
        model.shared_codebook.weight.copy_(codebook)
    print(f"Initialized codebook from {latent_samples.shape[0]} latent samples over {batches} batches.")


def set_encoder_decoder_trainable(model: VARActionTokenizer, trainable: bool) -> None:
    prefixes = ("encoder_", "decoder_")
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = trainable


def weighted_dim_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dim_groups: dict[str, list[int]],
    gripper_weight: float = 1.0,
    group_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    group_weights = dict(group_weights or {})
    if gripper_weight != 1.0 and "gripper" in dim_groups:
        group_weights.setdefault("gripper", float(gripper_weight))

    if not group_weights:
        return F.mse_loss(pred, target)

    weights = torch.ones(pred.shape[-1], dtype=pred.dtype, device=pred.device)
    for group_name, group_weight in group_weights.items():
        if group_name not in dim_groups:
            raise ValueError(f"Unknown action dim group {group_name!r}. Available groups: {sorted(dim_groups)}")
        weights[dim_groups[group_name]] = float(group_weight)
    per_dim = (pred - target).pow(2).mean(dim=(0, 1))
    return (per_dim * weights).sum() / weights.sum().clamp_min(1e-6)


def loss_group_weights(cfg: Any, prefix: str) -> dict[str, float]:
    """Read optional per-action-group loss weights from config.

    Supports keys such as ``position_recon_weight`` and
    ``position_vel_weight`` while preserving the older gripper-specific keys.
    """

    weights: dict[str, float] = {}
    for group_name in ("position", "rotation", "gripper"):
        key = f"{group_name}_{prefix}_weight"
        if cfg.loss.get(key, None) is not None:
            weights[group_name] = float(cfg.loss[key])
    return weights


def compute_vq_weight(epoch: int, *, target_weight: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return float(target_weight)
    return float(target_weight) * min(1.0, float(epoch + 1) / float(warmup_epochs))


def train(cfg: Any) -> None:
    torch.manual_seed(int(cfg.experiment.get("seed", 42)))
    device = torch.device(cfg.train.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_starvla_base_config(cfg)
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
    OmegaConf.save(base_cfg, output_dir / "starvla_base_config.yaml", resolve=True)

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
    scales = resolve_scales(cfg.model.get("scales", "auto"), model_seq_len)

    model = VARActionTokenizer(
        action_dim=model_action_dim,
        seq_len=model_seq_len,
        scales=scales,
        embed_dim=int(cfg.model.get("embed_dim", 128)),
        codebook_size=int(cfg.model.get("codebook_size", 512)),
        use_dilated=bool(cfg.model.get("use_dilated", True)),
        quant_resi=float(cfg.model.get("quant_resi", 0.5)),
        commitment_cost=float(cfg.model.get("commitment_cost", 0.25)),
        normalize_codebook_for_lookup=bool(cfg.model.get("normalize_codebook_for_lookup", True)),
        decoder_head_type=str(cfg.model.get("decoder_head_type", "plain")),
        quantization_mode=str(cfg.model.get("quantization_mode", "vq")),
        product_codebook_groups=int(cfg.model.get("product_codebook_groups", 1)),
        dim_groups=action_spec.dim_groups,
        use_time_embedding=bool(cfg.model.get("use_time_embedding", False)),
        use_action_type_embedding=bool(cfg.model.get("use_action_type_embedding", False)),
        input_embedding_scale=float(cfg.model.get("input_embedding_scale", 1.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.train.get("weight_decay", 1e-5)),
    )

    history: list[dict[str, Any]] = []
    start_epoch = 0

    resume_checkpoint_path = cfg.train.get("resume_checkpoint", None)
    if resume_checkpoint_path:
        resume_checkpoint = load_resume_checkpoint(
            str(resume_checkpoint_path),
            model=model,
            optimizer=optimizer,
            device=device,
        )
        history = list(resume_checkpoint.get("history", []))
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        print(f"Resumed checkpoint {resume_checkpoint_path} from epoch {resume_checkpoint.get('epoch', 'unknown')}.")
    else:
        init_checkpoint_path = cfg.train.get("init_checkpoint", None)
        if init_checkpoint_path:
            init_checkpoint_mode = str(cfg.train.get("init_checkpoint_mode", "full"))
            if init_checkpoint_mode == "full":
                init_checkpoint = load_init_checkpoint(str(init_checkpoint_path), model=model, device=device)
            elif init_checkpoint_mode == "encoder_decoder":
                init_checkpoint = load_encoder_decoder_init(str(init_checkpoint_path), model=model, device=device)
            else:
                raise ValueError(f"Unsupported init_checkpoint_mode={init_checkpoint_mode!r}.")
            print(
                f"Loaded {init_checkpoint_mode} init checkpoint {init_checkpoint_path} "
                f"(source epoch {init_checkpoint.get('epoch', 'unknown')})"
            )

        initialize_codebook_from_data(
            model=model,
            loader=loader,
            device=device,
            batches=int(cfg.train.get("init_codebook_from_data_batches", 0)),
            noise_scale=float(cfg.train.get("init_codebook_noise_scale", 0.0)),
        )

    if history:
        best_recon = min(float(record["recon_loss"]) for record in history)
    else:
        best_recon = float("inf")
    min_usage_for_balanced = float(cfg.train.get("min_codebook_usage_ratio_for_balanced", 0.02))
    balanced_records = [
        float(record["recon_loss"])
        for record in history
        if float(record.get("codebook_usage_ratio", 0.0)) >= min_usage_for_balanced
    ]
    best_balanced_recon = min(balanced_records) if balanced_records else float("inf")

    if start_epoch >= int(cfg.train.epochs):
        raise ValueError(
            f"resume checkpoint starts at epoch {start_epoch}, but cfg.train.epochs={int(cfg.train.epochs)}. "
            "Increase train.epochs to continue training."
        )

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        freeze_encoder_decoder_epochs = int(cfg.train.get("freeze_encoder_decoder_epochs", 0))
        set_encoder_decoder_trainable(model, epoch >= freeze_encoder_decoder_epochs)
        model.train()
        totals = {
            "recon_loss": 0.0,
            "vel_loss": 0.0,
            "vq_loss": 0.0,
            "total_loss": 0.0,
        }
        batches = 0
        code_counts = torch.zeros(model.codebook_size, dtype=torch.long)
        vq_weight = compute_vq_weight(
            epoch,
            target_weight=float(cfg.loss.get("vq_weight", 0.1)),
            warmup_epochs=int(cfg.loss.get("vq_warmup_epochs", 20)),
        )

        progress = tqdm(loader, desc=f"epoch {epoch:03d}")
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
                vel_recon = recon[:, 1:] - recon[:, :-1]
                vel_target = actions[:, 1:] - actions[:, :-1]
                vel_loss = weighted_dim_mse(
                    vel_recon,
                    vel_target,
                    dim_groups=action_spec.dim_groups,
                    gripper_weight=float(cfg.loss.get("gripper_vel_weight", 0.1)),
                    group_weights=loss_group_weights(cfg, "vel"),
                )
            else:
                vel_loss = torch.zeros((), device=device, dtype=actions.dtype)

            if actions.shape[1] > 2 and float(cfg.loss.get("jerk_weight", 0.0)) > 0.0:
                jerk_recon = recon[:, 2:] - 2.0 * recon[:, 1:-1] + recon[:, :-2]
                jerk_target = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]
                jerk_loss = F.mse_loss(jerk_recon, jerk_target)
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

            progress.set_postfix(
                recon=f"{totals['recon_loss'] / batches:.5f}",
                vq=f"{totals['vq_loss'] / batches:.5f}",
            )

            max_batches = int(cfg.train.get("max_batches_per_epoch", 0))
            if max_batches > 0 and batches >= max_batches:
                break

        if batches == 0:
            raise RuntimeError("No batches were produced by the Stage 1 dataloader.")

        epoch_record = {key: value / batches for key, value in totals.items()}
        used_codes = int((code_counts > 0).sum().item())
        usage_ratio = used_codes / float(model.codebook_size)
        epoch_record.update(
            {
                "epoch": epoch,
                "vq_weight": vq_weight,
                "codebook_used": used_codes,
                "codebook_usage_ratio": usage_ratio,
            }
        )
        history.append(epoch_record)
        save_json(output_dir / "history.json", history)
        save_checkpoint(output_dir / "latest.ckpt", model=model, optimizer=optimizer, epoch=epoch, history=history, cfg=cfg, action_spec=action_spec)

        save_every_epochs = int(cfg.train.get("save_every_epochs", 0))
        if save_every_epochs > 0 and (epoch + 1) % save_every_epochs == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.ckpt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                cfg=cfg,
                action_spec=action_spec,
            )

        if epoch_record["recon_loss"] < best_recon:
            best_recon = epoch_record["recon_loss"]
            save_checkpoint(output_dir / "best_recon.ckpt", model=model, optimizer=optimizer, epoch=epoch, history=history, cfg=cfg, action_spec=action_spec)

        if usage_ratio >= min_usage_for_balanced and epoch_record["recon_loss"] < best_balanced_recon:
            best_balanced_recon = epoch_record["recon_loss"]
            save_checkpoint(output_dir / "best_balanced.ckpt", model=model, optimizer=optimizer, epoch=epoch, history=history, cfg=cfg, action_spec=action_spec)

    save_checkpoint(output_dir / "final.ckpt", model=model, optimizer=optimizer, epoch=history[-1]["epoch"], history=history, cfg=cfg, action_spec=action_spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAR Stage 1 action tokenizer.")
    parser.add_argument("--config_yaml", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_yaml)
    train(cfg)


if __name__ == "__main__":
    main()
