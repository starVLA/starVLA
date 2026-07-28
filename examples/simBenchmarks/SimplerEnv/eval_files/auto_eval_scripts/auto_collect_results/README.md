# `auto_collect_results/` — Automated Batch Evaluation & Result Collection

This directory provides a set of **batch scripts targeting training artifacts (`steps_*_pytorch_model.pt`)** that:

1. Submit SimplerEnv evaluation jobs in bulk on a SLURM cluster;
2. Scan logs, aggregate success rates, and produce CSV + line charts after completion;
3. Maintain the checkpoint directory (clean up `.pt` files).

File overview:

| File | Purpose | Input | Output |
| --- | --- | --- | --- |
| `schedule_widowx_eval.sh` | Batch-schedule **WidowX / BridgeData v2** (4 tasks) evaluation | `<ROOT_BASE>/<DIR_GLOB>/checkpoints/steps_*_pytorch_model.pt` | 4 `*.log.run1` files per ckpt (written by `star_bridge.sh`) |
| `schedule_google_eval.sh` | Given one ckpt, serially srun all **8 Google Robot** sub-evaluations | Single `MODEL_PATH` | `client_logs/steps_<step>/*.log` |
| `summarize_widowx_one.sh` | Parse WidowX logs for **one** experiment dir, call Python to plot | Experiment dir path | `success_summary/raw_success.txt`, `success_summary.csv`, `success_plot.png` |
| `summarize_widowx_all.sh` | Loop `summarize_widowx_one.sh` over all matching experiment dirs | Dir glob under `ROOT_BASE` | `success_summary/` per dir |
| `plot_widowx_results.py` | Actual plotting: aggregate success rate by step×task → CSV + PNG | `raw_success.txt` | `csv` + `png` |
| `rm_pt.sh` | Bulk-delete `.pt` files by glob (with dry-run toggle) | `ROOT_DIR` + `DIR_GLOB` + `FILE_GLOB` | Deleted files |

> **Naming note:** The `windox` typo in some legacy files refers to `widowx` (the WidowX arm from BridgeData v2). This has been corrected in the current naming.

---

## Typical Workflow

### A. Evaluate WidowX (Bridge)

```
[Training artifact: steps_*_pytorch_model.pt]
        │
        ▼
schedule_widowx_eval.sh           # Schedules srun; only runs ckpts with missing logs
        │
        ▼
star_bridge.sh   (each ckpt: 4 tasks × N seeds)
        │
        ▼
steps_<step>_pytorch_model_infer_<TASK>-v0.log.run1
        │
        ▼
summarize_widowx_all.sh           # Or use summarize_widowx_one.sh for a single dir
        │
        ▼
success_summary/{raw_success.txt, success_summary.csv, success_plot.png}
```

### B. Evaluate Google Robot

`schedule_google_eval.sh` is a **single-ckpt** script (edit `MODEL_DIR` / `step` at the top,
then run directly). It launches 8 parallel `srun` calls for `star_*.sh`
(drawer / move_near / pick_coke_can / put_in_drawer — both variant and visual_matching).

---

## Evaluating `0427_oxe_bridge_rt_1_QwenPI_v3`

`schedule_widowx_eval.sh` exposes the target experiment dir as a parameter whose
**default is already `0427_oxe_bridge_rt_1_QwenPI_v3`**, so simply run:

```bash
cd examples/simBenchmarks/SimplerEnv/eval_files/auto_eval_scripts/auto_collect_results

# Option 1: use the default DIR_GLOB
bash schedule_widowx_eval.sh

# Option 2: pass dir name / glob explicitly
bash schedule_widowx_eval.sh '0427_oxe_bridge_rt_1_QwenPI_v3'
bash schedule_widowx_eval.sh '0427_oxe_bridge*'

# Option 3: env-var form
DIR_GLOB='0427_oxe_bridge_rt_1_QwenPI_v3' \
SLURM_PARTITION=si SLURM_GRES=gpu:4 \
bash schedule_widowx_eval.sh
```

After the runs finish, aggregate results:

```bash
# Single experiment directory
bash summarize_widowx_one.sh \
  /mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints/0427_oxe_bridge_rt_1_QwenPI_v3

# Or edit DIR_GLOB at the top of summarize_widowx_all.sh and batch-run
DIR_GLOB='0427_oxe_bridge_rt_1_QwenPI_v3' bash summarize_widowx_all.sh
```

Results land in `<ckpt_dir>/success_summary/{raw_success.txt, success_summary.csv, success_plot.png}`.

---

## Configurable Environment Variables

`schedule_widowx_eval.sh`:

| Variable | Default | Description |
| --- | --- | --- |
| `ROOT_BASE` | `/mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints` | Root experiment directory |
| `DIR_GLOB` | `0427_oxe_bridge_rt_1_QwenPI_v3` | Glob for experiment subdirectories (also accepted as the first positional arg) |
| `SLURM_PARTITION` | `si` | srun partition |
| `SLURM_GRES` | `gpu:4` | srun resources |
| `SCRIPT_PATH` | `.../auto_eval_scripts/star_bridge.sh` | Bridge evaluation entry script |

`summarize_widowx_all.sh` / `summarize_widowx_one.sh`:

| Variable | Default | Description |
| --- | --- | --- |
| `ROOT_BASE` | `/mnt/.../results/Checkpoints` | Root experiment directory |
| `DIR_GLOB` | `0427_oxe_bridge_rt_1_QwenPI_v3` | Glob for experiments to scan |
| `RM_LOGS` | `false` | Whether to delete logs where no success was parsed |
