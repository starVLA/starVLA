# 🚀 VLA-Arena Training and Evaluation

This document describes how to train and evaluate StarVLA models on the [VLA-Arena](https://github.com/VLA-Arena/VLA-Arena) benchmark.

VLA-Arena covers 11 task suites, 3 difficulty levels, and 4 evaluation domains. The StarVLA integration follows the same WebSocket policy-server workflow used in other benchmarks.

---

## 📊 Benchmark Overview

| Domain | Suite | Tasks / Level |
| ------ | ----- | ------------- |
| **Safety** | safety_static_obstacles | 5 |
|  | safety_cautious_grasp | 5 |
|  | safety_hazard_avoidance | 5 |
|  | safety_state_preservation | 5 |
|  | safety_dynamic_obstacles | 5 |
| **Distractor** | distractor_static_distractors | 5 |
|  | distractor_dynamic_distractors | 5 |
| **Extrapolation** | extrapolation_preposition_combinations | 5 |
|  | extrapolation_task_workflows | 5 |
|  | extrapolation_unseen_objects | 5 |
| **Long Horizon** | long_horizon | 10 (L0) / 5 (L1, L2) |

Each suite has **3 difficulty levels** (L0 basic to L2 advanced).
Safety suites additionally report a **constraint cost** metric.

---

## 📦 0. Environment and Data Preparation

### 0.1 Install VLA-Arena

```bash
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena
pip install -e ".[base]"
```

Make sure the `vla_arena` package is importable in the evaluation environment.

### 0.2 Prepare Training Data (LeRobot Format)

The VLA-Arena L0 training data is available as three HuggingFace repos:

| Split | HuggingFace repo |
| ----- | ---------------- |
| Small | `VLA-Arena/VLA_Arena_L0_S_lerobot_openpi` |
| Medium | `VLA-Arena/VLA_Arena_L0_M_lerobot_openpi` |
| Large | `VLA-Arena/VLA_Arena_L0_L_lerobot_openpi` |

Run the provided preparation script to download all three splits and set up symlinks:

```bash
export DEST=/path/to/storage
bash examples/VLA-Arena/data_preparation.sh
```

This will:
1. Download the three repos under `$DEST/vla_arena/`
2. Create `playground/Datasets/VLA_ARENA_LEROBOT_DATA` → `$DEST/vla_arena/`
3. Copy `train_files/modality.json` into each dataset's `meta/` directory

Expected StarVLA keys (defined by `VLAArenaFrankaDataConfig` in `data_config.py`):

| Key | Description |
| --- | ----------- |
| `video.primary_image` | agent-view camera (mapped from dataset via modality.json) |
| `state.{x,y,z,roll,pitch,yaw,gripper}` | 7-dim EEF state |
| `action.{x,y,z,roll,pitch,yaw,gripper}` | 7-DoF delta EEF action |
| `annotation.human.action.task_description` | language instruction |

> **Note:** `train_files/modality.json` maps the raw dataset key
> `observation.images.agentview_rgb` to `video.primary_image`.
> If the actual primary camera key in the downloaded dataset differs,
> update `modality.json` accordingly before training.

---

## 🚀 1. Training

Before training, edit user configuration in `examples/VLA-Arena/train_files/run_vla_arena_train.sh`.

Start training from the StarVLA root:

```bash
bash examples/VLA-Arena/train_files/run_vla_arena_train.sh
```

Key config options in `examples/VLA-Arena/train_files/starvla_cotrain_vla_arena.yaml`:

| Parameter | Default | Notes |
| --------- | ------- | ----- |
| `framework.name` | `QwenGR00T` | Alternatives: `QwenOFT`, `QwenPI`, `QwenFast` |
| `framework.qwenvl.base_vlm` | `Qwen2.5-VL-3B` | Can be replaced by larger backbones |
| `datasets.vla_data.data_mix` | `vla_arena_all` | Supports all-suite or domain-specific mixes |
| `trainer.max_train_steps` | `80000` | Adjust by dataset size and compute budget |

Available `data_mix` values (defined in `starVLA/dataloader/gr00t_lerobot/mixtures.py`):

**L0 splits (downloaded via `data_preparation.sh`):**

- `vla_arena_L0_all` – all three L0 splits combined (recommended)
- `vla_arena_L0_S` – small split only
- `vla_arena_L0_M` – medium split only
- `vla_arena_L0_L` – large split only

---

## 🧪 2. Evaluation Workflow

Evaluation uses two terminals from the repository root:

- **starVLA environment**: policy server (PyTorch + VLM model)
- **VLA-Arena environment**: simulator and benchmark runner

### Step 1. Start Policy Server (starVLA environment)

```bash
bash examples/VLA-Arena/eval_files/run_policy_server.sh
```

Before running, set `your_ckpt`, `gpu_id`, and `port` in `run_policy_server.sh`.

### Step 2. Run Single-Suite Evaluation (VLA-Arena environment)

```bash
export PYTHONPATH=/path/to/VLA-Arena/vla_arena:$PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

bash examples/VLA-Arena/eval_files/eval_vla_arena.sh
```

Before running, set `task_suite_name`, `task_level`, and `your_ckpt` in `eval_vla_arena.sh`.

### Step 3. Run Parallel Evaluation for All 11 Suites

```bash
bash examples/VLA-Arena/eval_files/auto_eval_scripts/auto_eval_vla_arena.sh
```

This launches 11 background workers, each with its own GPU/port slot.

### Step 4. Check Aggregated Results

```bash
bash examples/VLA-Arena/eval_files/auto_eval_scripts/see_sr_auto.sh \
    results/Checkpoints/vla_arena_qwenoft_all
```

Evaluation results are also saved as JSON:

```text
results/vla_arena/starvla_vla_arena_<timestamp>.json
```

---

## 🔒 3. Safety Metric Notes

Safety suites report both success rate and constraint cost.

Use the following flag to enable safety-constraint filtering:

```bash
--args.apply-safety-constraint true
```

Default threshold is `10.0`.
For `safety_hazard_avoidance`, each step cost is additionally scaled by `0.05` before thresholding.

---

## 📌 4. Environment Split Summary

| Component | Environment |
| --------- | ----------- |
| Policy server | starVLA conda environment |
| Evaluation script | VLA-Arena conda environment |
| Communication | WebSocket (JSON + msgpack-numpy) |

This split is consistent with the other benchmark integrations in `examples/`.

---

## 📚 Citation

If you use VLA-Arena, please cite the original VLA-Arena paper (see the official repository for BibTeX).
