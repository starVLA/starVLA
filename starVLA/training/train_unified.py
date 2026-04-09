# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Unified Trainer: single entry-point that handles VLA-only, VLM-only, and
co-training through DataLoaderManager.

Training mode is determined entirely by which ``*_data`` sections are
present in the YAML config under ``datasets``:

- Only ``vla_data`` → VLA-only
- Only ``vlm_data`` → VLM-only
- Both → Co-training (same behaviour as the legacy ``train_starvla_cotrain.py``)
"""

# Standard Library
import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader_manager
from starVLA.dataloader.dataloader_manager import DataLoaderManager
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # Save full config (all parameters) immediately
        if isinstance(cfg, AccessTrackedConfig):
            cfg.save_full_config(output_dir / "config.full.yaml")
            logger.info(f"📋 Full configuration saved to {output_dir / 'config.full.yaml'}")

    return output_dir


def prepare_data(cfg) -> DataLoaderManager:
    """Build a DataLoaderManager from the config.

    The manager auto-discovers all ``*_data`` entries under ``cfg.datasets``
    that contain a ``dataset_py`` field and builds the corresponding DataLoaders.
    """
    manager = build_dataloader_manager(cfg)
    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return manager


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and learning rate scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


# ---------------------------------------------------------------------------
# Unified Trainer
# ---------------------------------------------------------------------------


class UnifiedTrainer(TrainerUtils):
    """Single trainer that handles VLA-only, VLM-only, and co-training via DataLoaderManager."""

    def __init__(self, cfg, model, data_manager: DataLoaderManager, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.data_manager = data_manager
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.supported_tags = [name for name in self.data_manager.names if self.model.supports_training_tag(name)]
        self.skipped_tags = [name for name in self.data_manager.names if name not in self.supported_tags]

        if not self.supported_tags:
            raise ValueError(
                f"{type(self.model).__name__} does not support any registered dataloader tags: "
                f"{self.data_manager.names}"
            )

        self.total_batch_size = self._calculate_total_batch_size()
        # @JinhuiYE The unified trainer is not as straightforward as expected. TODO: rethink encapsulation boundaries
        # Convenience flags derived from compatible dataloaders only
        self.has_vla = "vla" in self.supported_tags
        self.has_vlm = "vlm" in self.supported_tags

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        # Distributed preparation: model + optimizer + dataloaders in ONE call
        # DeepSpeed requires at least one dataloader in prepare() to infer train_micro_batch_size_per_gpu
        dl_names = list(self.data_manager.dataloaders.keys())
        all_dls = [self.data_manager.dataloaders[n] for n in dl_names]
        prepared = self.accelerator.prepare(self.model, self.optimizer, *all_dls)
        self.model, self.optimizer = prepared[0], prepared[1]
        for i, name in enumerate(dl_names):
            self.data_manager.dataloaders[name] = prepared[2 + i]

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size from the first compatible dataset."""
        for tag in self.supported_tags:
            ds_cfg = getattr(self.config.datasets, f"{tag}_data", None)
            if ds_cfg is not None:
                per_device_bs = getattr(ds_cfg, "per_device_batch_size", 1)
                return per_device_bs * self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps
        return self.accelerator.num_processes  # fallback

    # ------------------------------------------------------------------
    # Checkpointing (adopted from train_starvla.py – the most complete)
    # ------------------------------------------------------------------

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading / resume."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint via accelerator."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            self._save_config_snapshot()

        self.accelerator.wait_for_everyone()

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            mode_tag = "+".join(self.data_manager.names)  # e.g. "vla+vlm"
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group=f"unified-{mode_tag}",
            )

    # ------------------------------------------------------------------
    # Metrics / Logging
    # ------------------------------------------------------------------

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                # Use first available dataloader for epoch count
                for name in self.supported_tags:
                    dl = self.data_manager.dataloaders[name]
                    if hasattr(dl, "__len__") and len(dl):
                        metrics["epoch"] = round(self.completed_steps / len(dl), 2)
                        break
                wandb.log(metrics, step=self.completed_steps)
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Datasets = {self.data_manager.names}")
            logger.info(f"  Active training tags = {self.supported_tags}")
            if self.skipped_tags:
                logger.warning(f"  Skipping unsupported dataloader tags = {self.skipped_tags}")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")

    def _save_config_snapshot(self):
        """Save accessed config snapshot. Called at train start and each checkpoint."""
        if self.accelerator.is_main_process and isinstance(self.config, AccessTrackedConfig):
            output_dir = Path(self.config.output_dir)
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved to {output_dir / 'config.yaml'}")

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def _train_step(self):
        """Execute single training step – delegates dispatch to model.compute_loss()."""
        log_dict = {}
        loss_scale = OmegaConf.to_container(self.config.trainer.loss_scale, resolve=True)
        processed_batches = 0

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            batches = self.data_manager.get_next_batches(allowed_names=set(self.supported_tags))
            for tag, batch in batches:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss_dict = self.model.compute_loss(tag, batch, loss_scale=loss_scale)
                if not loss_dict:
                    continue
                processed_batches += 1
                for loss_val in loss_dict.values():
                    self.accelerator.backward(loss_val)
                log_dict.update({k: v.item() for k, v in loss_dict.items()})

            if processed_batches == 0:
                return log_dict, False

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            self.lr_scheduler.step()

        return log_dict, True

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        """Run simple action-eval on current batch.  No-op when VLA data is absent."""
        if not self.has_vla:
            return step_metrics or {}

        examples = self.data_manager._get_single_batch("vla")
        actions = [example["action"] for example in examples]
        output_dict = self.model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def train(self):
        """Execute the main training loop."""
        self._log_training_config()
        self._save_config_snapshot()
        self.data_manager.reset()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start = time.perf_counter()
            step_metrics, did_update = self._train_step()
            t_end = time.perf_counter()

            if not did_update:
                continue

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix({"step_time": f"{t_end - t_start:.3f}"})

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["step_time"] = t_end - t_start
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# Entry‑point
# ---------------------------------------------------------------------------


def main(cfg) -> None:
    logger.info("Unified Training :: Warming Up")
    logger.info(f"  Detected datasets: {list(cfg.datasets.keys())}")

    cfg = wrap_config(cfg)  # Idempotent — no-op if already wrapped
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    model = build_framework(cfg)
    data_manager = prepare_data(cfg=cfg)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)

    trainer = UnifiedTrainer(
        cfg=cfg,
        model=model,
        data_manager=data_manager,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Wrap immediately so ALL subsequent accesses (including is_debug) are tracked.
    cfg = wrap_config(cfg, cli_overrides=dotlist)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
