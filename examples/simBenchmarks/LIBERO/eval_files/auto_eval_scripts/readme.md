# Auto Eval Scripts for LIBERO

These scripts are used internally for batch evaluation across all LIBERO task suites.
The fully automated evaluation is tied to a specific compute platform and may require
adaptation to your own environment.
See `examples/simBenchmarks/LIBERO/README.md` for the underlying principles.

## Script Overview

| Script | Purpose |
|------|------|
| `auto_eval_libero.sh` | **Main entry point**. Define checkpoint dir/list, GPU list, and task suites; automatically assigns GPUs for parallel evaluation |
| `eval_libero_parall.sh` | Single evaluation script. Starts policy server → runs eval → shuts down server |
| `see_sr_auto.sh` | Scans evaluation logs and aggregates success rates |
| `rm_video.sh` | Cleans up evaluation recordings |

## Quick Start

### 1. Edit the config block at the top of `auto_eval_libero.sh`

```bash
# Which checkpoints to evaluate (auto-scans all .pt files under this dir)
CKPT_DIR="results/Checkpoints/0405_libero4in1_CosmoPredict2GR00T/checkpoints"

# Or specify a manual list (overrides CKPT_DIR when non-empty)
CKPT_LIST=(
    # "results/Checkpoints/.../steps_30000_pytorch_model.pt"
)

# Which task suites to evaluate
TASK_SUITES=(libero_10 libero_goal libero_object libero_spatial)

# Available GPUs (assigned round-robin)
GPU_LIST=(0 1 2 3 4 5 6 7)
```

### 2. Run

```bash
bash examples/simBenchmarks/LIBERO/eval_files/auto_eval_scripts/auto_eval_libero.sh
```

The script round-robins `ckpts × task_suites` jobs across GPUs; it waits for the
current batch to finish before launching the next one.

### 3. View Results

```bash
# Default: scans the 0405 experiment
bash examples/simBenchmarks/LIBERO/eval_files/auto_eval_scripts/see_sr_auto.sh

# Or specify a different experiment directory
bash examples/simBenchmarks/LIBERO/eval_files/auto_eval_scripts/see_sr_auto.sh results/Checkpoints/YOUR_EXP
```

## Scheduling Logic

- `eval_libero_parall.sh` accepts 4 arguments: `ckpt_path`, `task_suite_name`, `gpu_id`, `port`
- `auto_eval_libero.sh` assigns GPUs round-robin via `job_index % num_gpus`
- At most `num_gpus` jobs run concurrently; the script waits for one full round to complete before starting the next
- Each job uses an independent port `BASE_PORT + job_index` to avoid conflicts
