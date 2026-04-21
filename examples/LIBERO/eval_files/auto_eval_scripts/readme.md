
# Auto Eval Scripts for LIBERO

这里包含了内部使用的批量评测脚本，用于快速评测所有 LIBERO task suite。
全自动评测与使用的平台有绑定，可能需要根据自己的环境做适配。
原理参考 `examples/LIBERO/README.md`。

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `auto_eval_libero.sh` | **主入口**。定义 ckpt 目录/列表、GPU 列表、task suite，自动分配 GPU 并行评测 |
| `eval_libero_parall.sh` | 单次评测脚本。启动 policy server → 运行 eval → 关闭 server |
| `see_sr_auto.sh` | 扫描评测日志，汇总 success rate |
| `rm_video.sh` | 清理评测录像 |

## 快速使用

### 1. 修改 `auto_eval_libero.sh` 顶部的配置

```bash
# 评测哪些 checkpoint（自动扫描目录下所有 .pt）
CKPT_DIR="results/Checkpoints/0405_libero4in1_CosmoPredict2GR00T/checkpoints"

# 或手动指定列表（非空时覆盖 CKPT_DIR）
CKPT_LIST=(
    # "results/Checkpoints/.../steps_30000_pytorch_model.pt"
)

# 评测哪些 task suite
TASK_SUITES=(libero_10 libero_goal libero_object libero_spatial)

# 可用 GPU（round-robin 分配）
GPU_LIST=(0 1 2 3 4 5 6 7)
```

### 2. 运行

```bash
bash examples/LIBERO/eval_files/auto_eval_scripts/auto_eval_libero.sh
```

脚本会将 `ckpts × task_suites` 的任务轮流分配到 GPU 上，每填满一轮 GPU 后等待该批次完成再启动下一轮。

### 3. 查看结果

```bash
# 默认扫描 0405 实验
bash examples/LIBERO/eval_files/auto_eval_scripts/see_sr_auto.sh

# 或指定其他实验目录
bash examples/LIBERO/eval_files/auto_eval_scripts/see_sr_auto.sh results/Checkpoints/YOUR_EXP
```

## 调度逻辑

- `eval_libero_parall.sh` 接受 4 个参数：`ckpt_path`、`task_suite_name`、`gpu_id`、`port`
- `auto_eval_libero.sh` 按 `job_index % num_gpus` round-robin 分配 GPU
- 每轮最多同时跑 `num_gpus` 个任务，等一轮全部完成后再启动下一轮
- 每个任务使用独立端口 `BASE_PORT + job_index`，避免冲突

