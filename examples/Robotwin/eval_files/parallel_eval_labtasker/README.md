# RoboTwin 并行评测：Labtasker 使用说明

本目录提供基于 [Labtasker](https://github.com/luocfprime/labtasker) 的 RoboTwin 并行评测脚本，支持多 GPU 弹性并行、失败自动重试、结果汇总。

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
# 指定数据目录（队列数据持久化在此）
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

每个任务（checkpoint × task × mode）完全自洽：worker 为该任务启动 policy server，eval 结束后关闭，再处理下一个任务。

```
Queue                                    Worker (GPU 0)
┌─────────────────────────────────┐      ┌──────────────────────────────────────────┐
│  ckpt=A × stack_blocks × clean  │─────►│  start server(ckpt=A)                   │
│  ckpt=A × stack_blocks × random │      │  run eval.sh(stack_blocks, demo_clean)  │
│  ckpt=A × place_shoe × clean    │      │  stop server                            │
│  ckpt=B × stack_blocks × clean  │      │  start server(ckpt=A)                   │
│  ...（50 tasks × 2 modes）       │      │  run eval.sh(stack_blocks, demo_random) │
└─────────────────────────────────┘      │  stop server                            │
                                         │  ...                                    │
                                         └──────────────────────────────────────────┘
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `submit.py` | 向任务队列提交 (ckpt × task_name × mode) 评测任务 |
| `run.py` | Worker 脚本：循环拉取任务 → 启动 server → 运行评测 → 关闭 server → 上报结果 |
| `.env.example` | 环境变量模板，复制为 `.env` 后填入实际路径 |
| `collect_results.ipynb` | 结果汇总 notebook |

任务粒度：**每个任务 = 一个 checkpoint × 一个 task × 一个 mode**（默认 50 tasks × 2 modes = 100 个任务）。

---

## 前提条件

请先按照 `examples/Robotwin/README.md` 完成依赖安装，并确保 labtasker 在两个环境中都已安装：

```bash
# starVLA 环境（运行 run.py / policy server）
${STARVLA_PYTHON} -m pip install 'labtasker[plugins]'

# RoboTwin 环境（运行 eval_policy.py）
${ROBOTWIN_PYTHON} -m pip install labtasker
```

---

## 快速开始

### 第一步：配置环境变量

复制模板并填入实际路径：

```bash
cp .env.example .env
# 编辑 .env，填入 STARVLA_PYTHON、ROBOTWIN_PYTHON、ROBOTWIN_PATH
```

`.env` 内容说明：

```bash
# 必填
STARVLA_PYTHON=   # 运行 policy server 的 Python（starVLA conda env）
ROBOTWIN_PYTHON=  # 运行 eval_policy.py 的 Python（robotwin conda env）
ROBOTWIN_PATH=    # RoboTwin repo 根目录

# 可选（括号内为默认值）
ROBOTWIN_SEED=0        # 评测随机种子（0）
SERVER_TIMEOUT=600     # 等待 policy server 就绪的超时秒数（600）
# ROBOTWIN_LOG_ROOT=   # 覆盖日志根目录（默认 <ckpt_dir>/robotwin_eval_logs/...）
```

### 第二步：提交任务

编辑 `submit.py` 顶部的 USER CONFIG，填写 `CKPT` 和 `POLICY_NAME`：

```python
CKPT = "/path/to/steps_30000_pytorch_model.pt"
POLICY_NAME = "my_run_v1"   # 用于日志目录命名
```

提交：

```bash
${STARVLA_PYTHON} examples/Robotwin/eval_files/parallel_eval_labtasker/submit.py
```

### 第三步：启动 Worker

每个 GPU 启动一个 worker，worker 自动从队列中拉取任务：

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 \
    ${STARVLA_PYTHON} examples/Robotwin/eval_files/parallel_eval_labtasker/run.py \
    --env examples/Robotwin/eval_files/parallel_eval_labtasker/.env &

# GPU 1
CUDA_VISIBLE_DEVICES=1 \
    ${STARVLA_PYTHON} examples/Robotwin/eval_files/parallel_eval_labtasker/run.py \
    --env examples/Robotwin/eval_files/parallel_eval_labtasker/.env &

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

打开 `collect_results.ipynb` 查看每个 (policy_name × task × mode) 的成功率，以及按 policy 和 mode 的平均汇总。

## 失败重试

```bash
# 查看失败任务
labtasker task ls -s failed --no-pager

# 批量重置为 pending（重新排队）
labtasker task update -s failed -u "retries=0" --reset-pending --quiet
```

## 日志位置

```
<ckpt_dir>/robotwin_eval_logs/<policy_name>_<ckpt_stem>/
  <ckpt_stem>_server.log          # policy server 日志
  <task_name>_<mode>_eval.log     # eval.sh 完整输出
```
