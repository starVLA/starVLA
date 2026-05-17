"""Single-GPU training smoke test for IndoorUAV × QwenPI.

Verifies the full closed loop: dataset -> model.forward -> backward ->
gradient clipping -> optimizer.step. Uses bitsandbytes 8-bit AdamW to fit
3.3B trainable parameters into a single ~50GB-free GPU.

Usage:
    CUDA_VISIBLE_DEVICES=<id> python scripts/smoke_train_indoor_uav.py [--steps 5]

Pre-requisites:
    pip install bitsandbytes
    Dataset already converted to:
        playground/Datasets/IndoorUAV/indoor_uav_replica_vla_lerobot/
"""

from __future__ import annotations
import argparse
import os

import torch
import bitsandbytes as bnb
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="examples/IndoorUAV/train_files/starvla_train_indoor_uav.yaml")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cfg = OmegaConf.load(args.config)
    print(f"--- Loading dataset (mix={cfg.datasets.vla_data.data_mix}) ---")
    ds = get_vla_dataset(cfg.datasets.vla_data, mode="train")
    print(f"Dataset size: {len(ds)}")

    print("--- Building QwenPI ---")
    model = Qwen_PI(cfg).cuda()

    # Honor `freeze_modules: qwen_vl_interface` from the yaml
    freeze_prefix = str(cfg.trainer.get("freeze_modules", ""))
    if freeze_prefix:
        for name, p in model.named_parameters():
            if name.startswith(freeze_prefix):
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable) / 1e6
    n_total = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Trainable: {n_trainable:.1f}M  /  Total: {n_total:.2f}B")

    opt = bnb.optim.AdamW8bit(
        trainable, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-8,
    )

    print(f"\n--- Running {args.steps} train steps (batch={args.batch_size}) ---")
    losses = []
    for step in range(args.steps):
        batch = [ds[(step * args.batch_size + i) % len(ds)] for i in range(args.batch_size)]
        out = model(batch)
        loss = out["action_loss"]
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.clip)
        opt.step()
        losses.append(loss.item())
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  step={step:>2}  loss={loss.item():>10.4f}  grad_norm={gn.item():>8.4f}  peak_mem={peak:.2f}GB")
        torch.cuda.reset_peak_memory_stats()

    print(f"\nloss trace: {[f'{l:.2f}' for l in losses]}")
    print("Full train step closed-loop OK.")


if __name__ == "__main__":
    main()
