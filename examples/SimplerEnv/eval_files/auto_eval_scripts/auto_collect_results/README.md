# `auto_collect_results/` — 自动批量评测与结果收集

这一目录提供一组**面向训练产物（`steps_*_pytorch_model.pt`）的批处理脚本**，
负责：

1. 在 SLURM 集群上批量发起 SimplerEnv 评测任务；
2. 完成后扫描日志、聚合 success rate、产出 CSV + 折线图；
3. 维护 ckpt 目录（清理 `.pt`）。

文件作用一览：

| 文件 | 作用 | 输入 | 输出 |
| --- | --- | --- | --- |
| `schedule_widowx_eval.sh` | 批量调度 **WidowX / BridgeData v2**（4 任务）评测 | `<ROOT_BASE>/<DIR_GLOB>/checkpoints/steps_*_pytorch_model.pt` | 每个 ckpt 4 个 `*.log.run1`（由 `star_bridge.sh` 写入） |
| `schedule_google_eval.sh` | 给定一个 ckpt，串行 srun 起 **Google Robot** 全部 8 个子评测 | 单一 `MODEL_PATH` | `client_logs/steps_<step>/*.log` |
| `summarize_widowx_one.sh` | 解析**一个**实验目录下的 widowx 日志，调用 Python 出图 | 实验目录路径 | `success_summary/raw_success.txt` `success_summary.csv` `success_plot.png` |
| `summarize_widowx_all.sh` | 在所有匹配的实验目录上循环调用 `summarize_widowx_one.sh` | `ROOT_BASE` 下的目录 glob | 各目录的 `success_summary/` |
| `plot_widowx_results.py` | 实际绘图：按 step×task 聚合 success rate → CSV + PNG | `raw_success.txt` | `csv` + `png` |
| `rm_pt.sh` | 用 glob 批量清理 `.pt` 文件（含 dry-run 开关） | `ROOT_DIR` + `DIR_GLOB` + `FILE_GLOB` | 删除文件 |

> 命名说明：原文件中的 `windox` 是 `widowx`（BridgeData v2 的 WidowX 机械臂）的拼写错误，本次重命名一并修正。

---

## 典型工作流

### A. 评测 WidowX (Bridge)

```
[训练产出 steps_*_pytorch_model.pt]
        │
        ▼
schedule_widowx_eval.sh           # 调度 srun，缺日志的 ckpt 才会跑
        │
        ▼
star_bridge.sh   (每个 ckpt 4 个 task × N seed)
        │
        ▼
steps_<step>_pytorch_model_infer_<TASK>-v0.log.run1
        │
        ▼
summarize_widowx_all.sh           # 也可单目录用 summarize_widowx_one.sh
        │
        ▼
success_summary/{raw_success.txt, success_summary.csv, success_plot.png}
```

### B. 评测 Google Robot

`schedule_google_eval.sh` 是面向**单个 ckpt** 的脚本（修改顶部的
`MODEL_DIR` / `step` 后直接执行），会并行 srun 起 8 个 `star_*.sh`
（drawer / move_near / pick_coke_can / put_in_drawer 各 variant + visual_matching）。

---

## 怎么评测 `0427_oxe_bridge_rt_1_QwenPI_v3`

`schedule_widowx_eval.sh` 已经把目标实验目录抽成参数，**默认值就是
`0427_oxe_bridge_rt_1_QwenPI_v3`**，所以直接：

```bash
cd examples/SimplerEnv/eval_files/auto_eval_scripts/auto_collect_results

# 方式 1：用默认 DIR_GLOB
bash schedule_widowx_eval.sh

# 方式 2：显式传目录名 / glob
bash schedule_widowx_eval.sh '0427_oxe_bridge_rt_1_QwenPI_v3'
bash schedule_widowx_eval.sh '0427_oxe_bridge*'

# 方式 3：env var 形式
DIR_GLOB='0427_oxe_bridge_rt_1_QwenPI_v3' \
SLURM_PARTITION=si SLURM_GRES=gpu:4 \
bash schedule_widowx_eval.sh
```

跑完之后聚合结果：

```bash
# 单实验目录
bash summarize_widowx_one.sh \
  /mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints/0427_oxe_bridge_rt_1_QwenPI_v3

# 或者改 summarize_widowx_all.sh 顶部的 DIR_GLOB 后批量跑
DIR_GLOB='0427_oxe_bridge_rt_1_QwenPI_v3' bash summarize_widowx_all.sh
```

聚合结果会落在
`<ckpt_dir>/success_summary/{raw_success.txt, success_summary.csv, success_plot.png}`。

---

## 可调环境变量

`schedule_widowx_eval.sh`：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ROOT_BASE` | `/mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints` | 实验根目录 |
| `DIR_GLOB` | `0427_oxe_bridge_rt_1_QwenPI_v3` | 实验子目录的 glob（也可作为第一个位置参数） |
| `SLURM_PARTITION` | `si` | srun 的分区 |
| `SLURM_GRES` | `gpu:4` | srun 的资源 |
| `SCRIPT_PATH` | `.../auto_eval_scripts/star_bridge.sh` | bridge 评测入口 |

`summarize_widowx_all.sh` / `summarize_widowx_one.sh`：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ROOT_BASE` | `/mnt/.../results/Checkpoints` | 实验根目录 |
| `DIR_GLOB` | `0427_oxe_bridge_rt_1_QwenPI_v3` | 要扫描的实验 glob |
| `RM_LOGS` | `false` | 没解析到 success 的日志是否删除 |
