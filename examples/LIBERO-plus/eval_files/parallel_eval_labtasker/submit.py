#!/usr/bin/env python3
"""Submit LIBERO-plus evaluation tasks to Labtasker.

Each task = one (ckpt × task_suite × [start_idx, end_idx)) slice.
Multiple workers can process different slices of the same suite in parallel.
"""

import itertools
import pathlib

try:
    import labtasker
except ImportError:
    print("[ERROR] Labtasker not installed. Install it with `pip install 'labtasker[plugins]'`")
    exit(1)

###############################################################################
# ============ USER CONFIG ============
###############################################################################

CKPT_DIR  = ""
CKPT_LIST = [
    "/path/to/starVLA/playground/Pretrained_models/StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt",
]

# Total number of tasks in each suite (determines the index range [0, size))
SUITE_SIZES = {
    "libero_goal":    2591,
    "libero_object":  2518,
    "libero_spatial": 2402,
    "libero_10": 2519,
}

# How many labtasker tasks to split each suite into.
# More slices = finer load balancing; workers consume from the queue regardless of count.
NUM_SLICES = {
    "libero_goal":    12,
    "libero_object":  12,
    "libero_spatial": 12,
    "libero_10": 24,
}

MAX_RETRIES = 3
PRIORITY    = 10

###############################################################################
# ============ END USER CONFIG ============
###############################################################################


def _slices(size: int, n: int):
    """Yield (start, end) pairs that evenly partition [0, size) into n chunks."""
    base, remainder = divmod(size, n)
    start = 0
    for i in range(n):
        end = start + base + (1 if i < remainder else 0)
        yield start, end
        start = end


def main() -> None:
    ckpts = CKPT_LIST if CKPT_LIST else sorted(
        str(p) for p in pathlib.Path(CKPT_DIR).glob("*.pt")
    )
    if not ckpts:
        raise ValueError("No checkpoints found. Set CKPT_LIST or CKPT_DIR.")

    submitted = 0
    for ckpt, suite in itertools.product(ckpts, SUITE_SIZES):
        stem = pathlib.Path(ckpt).stem
        size = SUITE_SIZES[suite]
        n = NUM_SLICES[suite]
        for start, end in _slices(size, n):
            result = labtasker.submit_task(
                task_name=f"{stem}_{suite}_{start}_{end}",
                args={"ckpt": ckpt, "task_suite": suite, "start_idx": start, "end_idx": end},
                metadata={"benchmark": "LIBERO-plus", "ckpt_stem": stem, "task_suite": suite},
                max_retries=MAX_RETRIES,
                priority=PRIORITY,
            )
            print(f"  submitted  {stem} x {suite} [{start}, {end})  =>  {result.task_id}")
            submitted += 1
    print(f"\nDone. {submitted} tasks submitted.\n")


if __name__ == "__main__":
    main()
