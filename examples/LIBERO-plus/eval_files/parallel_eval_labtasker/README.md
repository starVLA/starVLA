# LIBERO-plus 并行评测：Labtasker 使用说明

本目录提供基于 [Labtasker](https://github.com/luocfprime/labtasker) 的 LIBERO-plus 并行评测脚本，支持多 GPU 弹性并行、失败自动重试、结果汇总。

---

## Labtasker 简介

[Labtasker](https://github.com/luocfprime/labtasker) 是一个轻量级任务队列系统，专为 ML 实验设计。核心思路是把 `for` 循环替换为任务队列：

- **submit**：把实验参数组合批量推入队列
- **worker**：从队列拉取任务并执行，多个 worker 并行互不干扰
- **retry**：任务失败自动重试，成功后将结果写入 summary

### 安装

```bash
pip install 'labtasker[plugins]'
```

### 启动 Labtasker Server

Labtasker 需要一个后端服务来管理任务队列。在远端机器上启动（建议放入 tmux）：

```bash
labtasker-server serve &
```

### 初始化队列

首次使用时，在项目根目录初始化配置并创建队列：

```bash
labtasker init          # 交互式填写 server 地址、队列名等，生成 .labtasker/config.toml
labtasker queue create-from-config
```

此后同一机器上的所有 submit / worker 命令自动读取 `.labtasker/config.toml`，无需重复配置。

---

## 架构说明

每个任务（checkpoint × task suite）完全自洽：worker 为该任务启动 policy server，eval 结束后关闭，再处理下一个任务。

```
Queue                                  Worker (GPU 0)
┌──────────────────────────────┐       ┌──────────────────────────────────────────┐
│  ckpt=A × libero_spatial     │──────►│  start server(ckpt=A)                   │
│  ckpt=A × libero_goal        │       │  run eval_libero.py(libero_spatial)      │
│  ckpt=B × libero_spatial     │       │  stop server                            │
│  ckpt=B × libero_goal        │       │  start server(ckpt=A)                   │
│  ...                         │       │  run eval_libero.py(libero_goal)         │
└──────────────────────────────┘       │  stop server                            │
                                       │  ...                                    │
                                       └──────────────────────────────────────────┘
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `submit.py` | 向任务队列提交 (checkpoint × task_suite) 评测任务 |
| `run.py` | Worker 脚本：循环拉取任务 → 启动 server → 运行评测 → 关闭 server → 上报结果 |
| `.env.example` | 环境变量模板，复制为 `.env` 后填入实际路径 |
| `collect_results.ipynb` | 结果汇总 notebook，含 per-suite 和 per-category（扰动类型）两个维度 |

任务粒度：**每个任务 = 一个 checkpoint × 一个 task suite**（默认 4 个 suite = 4 个任务/checkpoint）。

---

## 前提条件

LIBERO-plus 使用独立的 conda 环境（`libero_plus`），与运行 policy server 的 starVLA 环境分离。请确保 labtasker 在两个环境中都已安装：

```bash
# starVLA 环境（运行 run.py / policy server）
${STARVLA_PYTHON} -m pip install 'labtasker[plugins]'

# LIBERO-plus 环境（运行 eval_libero.py）
${LIBERO_PLUS_PYTHON} -m pip install labtasker
```

---

## 快速开始

### 第一步：配置环境变量

复制模板并填入实际路径：

```bash
cp .env.example .env
# 编辑 .env，填入 STARVLA_PYTHON、LIBERO_PLUS_PYTHON、LIBERO_PLUS_HOME
```

`.env` 内容说明：

```bash
# 必填
STARVLA_PYTHON=      # 运行 policy server 的 Python（starVLA conda env）
LIBERO_PLUS_PYTHON=  # 运行 eval_libero.py 的 Python（libero_plus conda env）
LIBERO_PLUS_HOME=    # LIBERO-plus repo 根目录（sylvestf/LIBERO-plus）

# 可选（括号内为默认值）
LIBERO_NUM_TRIALS=50   # 每个任务的 episode 数（50）
SERVER_TIMEOUT=300     # 等待 policy server 就绪的超时秒数（300）
```

### 第二步：提交任务

编辑 `submit.py` 顶部的 USER CONFIG，填写 `CKPT_LIST`（或 `CKPT_DIR`）和要评测的 `TASK_SUITES`：

```python
CKPT_LIST = [
    "/path/to/steps_30000_pytorch_model.pt",
]
TASK_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
```

提交：

```bash
${STARVLA_PYTHON} examples/LIBERO-plus/eval_files/parallel_eval_labtasker/submit.py
```

### 第三步：启动 Worker

每个 GPU 启动一个 worker，worker 自动从队列中拉取任务：

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 \
    ${STARVLA_PYTHON} examples/LIBERO-plus/eval_files/parallel_eval_labtasker/run.py \
    --env examples/LIBERO-plus/eval_files/parallel_eval_labtasker/.env &

# GPU 1
CUDA_VISIBLE_DEVICES=1 \
    ${STARVLA_PYTHON} examples/LIBERO-plus/eval_files/parallel_eval_labtasker/run.py \
    --env examples/LIBERO-plus/eval_files/parallel_eval_labtasker/.env &

wait
```

多个 worker 可以并行处理不同任务，labtasker 自动分配，不会重复。Worker 队列为空时自动退出。

---

## 查看进度

```bash
labtasker task count
labtasker task ls
labtasker task ls -s running
labtasker task ls -s failed  --no-pager
labtasker task ls -s success
```

## 汇总结果

打开 `collect_results.ipynb` 查看：
- 每个 (checkpoint × task_suite) 的成功率
- 按 checkpoint 的跨 suite 加权汇总
- 按扰动类别（Camera / Robot / Language / Light / Background / Noise / Layout）的成功率分析

## 失败重试

```bash
# 查看失败任务
labtasker task ls -s failed --no-pager

# 批量重置为 pending（重新排队）
labtasker task update -s failed -u "retries=0" --reset-pending --quiet
```

## 日志位置

```
<ckpt_dir>/
  logs/
    <task_suite>/
      <ckpt_stem>_server.log   # policy server 日志
      <ckpt_stem>.log          # eval_libero.py 完整输出
  videos/<task_suite>/<ckpt_stem>/
    rollout_*.mp4
```
