#!/usr/bin/env python3
"""
Submit LIBERO evaluation tasks to Labtasker.

Each task = one (ckpt × task_suite) pair.

Workflow:
  1. Edit the USER CONFIG section below.
  2. python submit.py
  3. Launch workers: CUDA_VISIBLE_DEVICES=<gpu> WORKER_CKPT=<ckpt> ... python run.py
"""

import itertools
import pathlib

try:
    import labtasker
except ImportError:
    print("[ERROR] Labtasker not installed. Install it with `pip install 'labtasker[plugins]'`")
    exit(1)

###############################################################################
# ============ USER CONFIG: modify this section ============
###############################################################################

# Checkpoints: auto-discover from a directory, or list explicitly
CKPT_DIR  = ""   # scan all .pt files in this directory
CKPT_LIST = [    # explicit list; overrides CKPT_DIR when non-empty
    "/path/to/starVLA/playground/Pretrained_models/StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt",
]

# Task suites to evaluate
TASK_SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    # "libero_90",
]

# Labtasker task settings
MAX_RETRIES = 3
PRIORITY    = 10  # 0-20

###############################################################################
# ============ END USER CONFIG ============
###############################################################################

def main() -> None:
    ckpts = CKPT_LIST if CKPT_LIST else sorted(
        str(p) for p in pathlib.Path(CKPT_DIR).glob("*.pt")
    )
    if not ckpts:
        raise ValueError("No checkpoints found. Set CKPT_LIST or CKPT_DIR.")

    # ── Submit tasks ──────────────────────────────────────────────────────────
    submitted = 0
    for ckpt, suite in itertools.product(ckpts, TASK_SUITES):
        stem = pathlib.Path(ckpt).stem
        result = labtasker.submit_task(
            task_name=f"{stem}_{suite}",
            args={"ckpt": ckpt, "task_suite": suite},
            metadata={"benchmark": "LIBERO", "ckpt_stem": stem, "task_suite": suite},
            max_retries=MAX_RETRIES,
            priority=PRIORITY,
        )
        print(f"  submitted  {stem} x {suite}  =>  {result.task_id}")
        submitted += 1
    print(f"\nDone. {submitted} tasks submitted.\n")



if __name__ == "__main__":
    main()
