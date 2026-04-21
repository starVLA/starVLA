#!/usr/bin/env python3
"""
Generate accelerate + DeepSpeed configs with the requested gradient accumulation steps.

starVLA's `trainer.gradient_accumulation_steps` is dead config — the actual value
is read from `starVLA/config/deepseeds/ds_config.yaml` at module-import time
because `train_starvla.py` constructs `Accelerator(deepspeed_plugin=...)` before
`cfg` is parsed (and DeepSpeed grad_accum can only come from its own JSON).

Instead of patching upstream, this helper writes a temp pair of config files with
the right value, and prints the path to the generated accelerate yaml so the
launcher can pass it via `--config_file`.

Usage:
  python _make_accelerate_config.py --grad-accum 4
  → writes /tmp/gemma4_ds_<ga>.yaml + /tmp/gemma4_accel_<ga>.yaml
  → prints /tmp/gemma4_accel_<ga>.yaml on stdout
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DS_TEMPLATE = REPO_ROOT / "starVLA" / "config" / "deepseeds" / "ds_config.yaml"
DEFAULT_ACCEL_TEMPLATE = REPO_ROOT / "starVLA" / "config" / "deepseeds" / "deepspeed_zero2.yaml"


def make_configs(
    grad_accum: int,
    num_processes: int,
    out_dir: Path | None = None,
    zero_stage: int = 2,
    cpu_offload: bool = False,
) -> Path:
    """
    Returns: Path to the generated accelerate yaml (suitable for `accelerate launch --config_file ...`).
    """
    out_dir = out_dir or Path(tempfile.gettempdir())
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ga{grad_accum}_z{zero_stage}{'_off' if cpu_offload else ''}"

    ds_path = out_dir / f"gemma4_ds_{tag}.yaml"
    accel_path = out_dir / f"gemma4_accel_{tag}.yaml"

    # --- DeepSpeed config (json with .yaml suffix is fine; HF loads either) ---
    ds_cfg = {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": grad_accum,
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": cpu_offload,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
    }
    if zero_stage == 3:
        ds_cfg["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
        if cpu_offload:
            ds_cfg["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
            ds_cfg["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    with open(ds_path, "w") as f:
        json.dump(ds_cfg, f, indent=2)

    # --- Accelerate yaml ---
    accel_yaml = (
        "compute_environment: LOCAL_MACHINE\n"
        "debug: false\n"
        "deepspeed_config:\n"
        f'  deepspeed_config_file: "{ds_path}"\n'
        "  deepspeed_multinode_launcher: standard\n"
        f"  zero3_init_flag: {'true' if zero_stage == 3 else 'false'}\n"
        "distributed_type: DEEPSPEED\n"
        "num_machines: 1\n"
        f"num_processes: {num_processes}\n"
    )
    with open(accel_path, "w") as f:
        f.write(accel_yaml)

    return accel_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--zero-stage", type=int, default=2, choices=[2, 3])
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    accel_path = make_configs(
        grad_accum=args.grad_accum,
        num_processes=args.num_processes,
        out_dir=out_dir,
        zero_stage=args.zero_stage,
        cpu_offload=args.cpu_offload,
    )
    # Print only the path to stdout so it can be captured into a shell variable.
    print(accel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
