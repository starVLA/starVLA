#!/usr/bin/env python3
"""
Generate accelerate + DeepSpeed configs with the requested gradient accumulation steps.

This mirrors the Gemma4 helper but writes MiniCPM-tagged temporary files.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def make_configs(
    grad_accum: int,
    num_processes: int,
    out_dir: Path | None = None,
    zero_stage: int = 2,
    cpu_offload: bool = False,
) -> Path:
    out_dir = out_dir or Path(tempfile.gettempdir())
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ga{grad_accum}_z{zero_stage}{'_off' if cpu_offload else ''}"

    ds_path = out_dir / f"minicpm_ds_{tag}.yaml"
    accel_path = out_dir / f"minicpm_accel_{tag}.yaml"

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
    print(accel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
