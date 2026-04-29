# Quick Start: Train & Evaluate Your First VLA with LIBERO

This guide walks you through the complete StarVLA workflow — from installation to training to evaluation — using the **LIBERO** benchmark as a concrete, end-to-end example. By the end you will have a trained VLA policy and know how to evaluate it in simulation.

> **Just want to evaluate a released checkpoint?** Jump to [Evaluate a Pretrained Checkpoint](#-evaluate-a-pretrained-checkpoint) — no training required.

---

## Table of Contents

- [0. Installation](#0-installation)
- [1. Verify Your Installation](#1-verify-your-installation)
- [2. Prepare Training Data](#2-prepare-training-data)
- [3. Prepare Pretrained Models](#3-prepare-pretrained-models)
- [4. Understanding the Training Config](#4-understanding-the-training-config)
- [5. Understanding the Training Script](#5-understanding-the-training-script)
- [6. Launch Training](#6-launch-training)
- [7. Evaluate a Pretrained Checkpoint](#7-evaluate-a-pretrained-checkpoint)
- [Next Steps](#next-steps)

---

## 0. Installation

```bash
git clone https://github.com/starVLA/starVLA
cd starVLA

conda create -n starVLA python=3.10 -y
conda activate starVLA

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```
<details>
<summary><b>⚠️ Common Issues</b></summary>

flash-attn can be tricky to install because it must match your system's CUDA toolkit (nvcc) and PyTorch versions. The `--no-build-isolation` flag resolves most issues, but on newer systems you may need to manually choose a compatible flash-attn version. Ensure your CUDA driver/toolkit and torch versions are aligned. Check your environment:

```bash
nvcc -V
pip list | grep -E 'torch|transformers|flash-attn'
```

If issues persist, pick a flash-attn release that matches your versions (CUDA and torch) or ask ChatGPT with its search function for help with the outputs above.

We have verified that `flash-attn==2.7.4.post1` works well with nvcc versions `12.0` and `12.4`.

</details>

---

## 1. Verify Your Installation

Run a quick smoke test to make sure the framework loads correctly:

```bash
python starVLA/model/framework/VLM4A/QwenGR00T.py
```

This requires [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) at `./playground/Pretrained_models/Qwen3-VL-4B-Instruct` (see [Step 3](#3-prepare-pretrained-models)). It should print the model architecture and run a forward pass on fake data without errors.

---

## 2. Prepare Training Data

StarVLA uses **LeRobot-format** datasets. We provide a one-command script to download all four LIBERO suites (Spatial, Object, Goal, Long-Horizon) plus the co-training VLM data:

```bash
# Set DEST to where you want to store the raw data (can be a shared disk)
export DEST=/path/to/your/data/directory
bash examples/LIBERO/data_preparation.sh
```

This script will:
1. Download 4 LIBERO subsets from HuggingFace (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`)
2. Download the VLM co-training data ([LLaVA-OneVision-COCO](https://huggingface.co/datasets/StarVLA/LLaVA-OneVision-COCO))
3. Create symlinks under `playground/Datasets/`
4. Copy `modality.json` into each dataset's `meta/` folder

<details>
<summary><b>Manual download (alternative)</b></summary>

```bash
# Download each dataset individually
huggingface-cli download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot
huggingface-cli download IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot  --repo-type dataset --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot
huggingface-cli download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot    --repo-type dataset --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot
huggingface-cli download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot      --repo-type dataset --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot

# Copy modality.json to each subset
for d in playground/Datasets/LEROBOT_LIBERO_DATA/*/; do
  cp examples/LIBERO/train_files/modality.json "$d/meta/"
done
```
</details>

After this step, your `playground/Datasets/` should look like:

```
playground/Datasets/
├── LEROBOT_LIBERO_DATA/
│   ├── libero_spatial_no_noops_1.0.0_lerobot/
│   │   ├── meta/
│   │   │   ├── modality.json        ← required
│   │   │   └── ...
│   │   └── ...
│   ├── libero_object_no_noops_1.0.0_lerobot/
│   ├── libero_goal_no_noops_1.0.0_lerobot/
│   └── libero_10_no_noops_1.0.0_lerobot/
└── LLaVA-OneVision-COCO/             ← VLM co-training data
```

### Verify Your Dataloader

To make sure the data can be loaded correctly:

```bash
python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

---

## 3. Prepare Pretrained Models

Download the base VLM to `playground/Pretrained_models/`:

```bash
# Qwen3-VL (recommended)
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct

# For FAST framework, use the action-extended version instead:
# huggingface-cli download StarVLA/Qwen3-VL-4B-Instruct-Action --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
```

See [Model Zoo](model_zoo.md) for all available base models and finetuned checkpoints.

---

## 4. Understanding the Training Config

The config YAML at [`examples/LIBERO/train_files/starvla_cotrain_libero.yaml`](../examples/LIBERO/train_files/starvla_cotrain_libero.yaml) defines the entire training setup. Here are the key sections:

### Framework

```yaml
framework:
  name: QwenGR00T                # which VLA architecture to use
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct
    attn_implementation: flash_attention_2
  action_model:
    action_dim: 7                # LIBERO: 7-DoF (xyz + rpy + gripper)
    state_dim: 7
    future_action_window_size: 7 # predict 7 future steps
    action_horizon: 8            # total action chunk size
```

StarVLA supports four framework variants — just change `framework.name`:

| Framework | Name | Description |
|-----------|------|-------------|
| StarVLA-OFT | `QwenOFT` | MLP action head, simple & fast |
| StarVLA-FAST | `QwenFAST` | Discrete action tokens, autoregressive |
| StarVLA-π | `QwenPI` | Flow-matching diffusion action head |
| StarVLA-GR00T | `QwenGR00T` | Dual-system: VLM (System 2) + Flow-matching (System 1) |

### Datasets

```yaml
datasets:
  vlm_data:                       # VLM co-training data (improves generalization)
    dataset_py: vlm_datasets
    per_device_batch_size: 4

  vla_data:                       # Robot action data
    dataset_py: lerobot_datasets
    data_root_dir: playground/Datasets/LEROBOT_LIBERO_DATA
    data_mix: libero_all          # all 4 suites; use "libero_goal" for single suite
    per_device_batch_size: 16
```

The `data_mix` field selects which datasets to combine. These mixtures are defined in [`examples/LIBERO/train_files/data_registry/data_config.py`](../examples/LIBERO/train_files/data_registry/data_config.py):

```python
DATASET_NAMED_MIXTURES = {
    "libero_all": [                              # all 4 suites
        ("libero_object_no_noops_1.0.0_lerobot",  1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot",    1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot",  1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot",      1.0, "libero_franka"),
    ],
    "libero_goal": [                             # single suite
        ("libero_goal_no_noops_1.0.0_lerobot",    1.0, "libero_franka"),
    ],
}
```


### Trainer

```yaml
trainer:
  max_train_steps: 100000
  save_interval: 10000            # save checkpoint every N steps
  eval_interval: 100             # log eval metrics every N steps
  learning_rate:
    base: 2.5e-05                # default LR
    qwen_vl_interface: 1.0e-05   # lower LR for VLM layers
    action_model: 1.0e-04        # higher LR for action head
  loss_scale:
    vla: 1.0                     # action loss weight
    vlm: 0.1                     # VLM co-training loss weight
```

---

## 5. Understanding the Training Script

The training script [`examples/LIBERO/train_files/run_libero_train.sh`](../examples/LIBERO/train_files/run_libero_train.sh) wraps around `accelerate launch`. Key variables to customize:

```bash
###########################################################################################
# === Modify these for your environment ===
Framework_name=QwenOFT              # QwenOFT | QwenFAST | QwenPI | QwenGR00T
freeze_module_list=''               # e.g. 'qwen_vl' to freeze VLM backbone
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all                 # or libero_goal for single suite
run_root_dir=./results/Checkpoints
run_id=my_first_libero_run          # unique experiment name
###########################################################################################
```

The script launches distributed training with DeepSpeed ZeRO-2:

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \                    # number of GPUs
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  ...
```

> **Note:** Command-line arguments override YAML config values. This lets you keep one base config and vary parameters per experiment.

---

## 6. Launch Training

```bash
# From the repository root
bash examples/LIBERO/train_files/run_libero_train.sh
```

### What to expect

- Checkpoints are saved to `results/Checkpoints/{run_id}/checkpoints/`
- Training logs go to W&B (set `WANDB_MODE=disabled` to skip)
- The script copies itself to the output directory for reproducibility
- With 8× A100/H800 GPUs, training on `libero_all` takes roughly 30K steps (~10 epochs)

### Adjusting GPU count

Edit `--num_processes` in the script and `num_processes` in `starVLA/config/deepseeds/deepspeed_zero2.yaml` to match your available GPUs.

---

## 7. Evaluate a Pretrained Checkpoint

Evaluation uses a **client-server architecture**: a policy server (in the `starVLA` env) serves model inference, while the simulation client (in a separate `LIBERO` env) sends observations and receives actions.

### Step 0: Set up the LIBERO environment

Create a separate conda environment for the LIBERO simulator:

```bash
conda create -n libero python=3.10 -y
conda activate libero
pip install mujoco==3.2.3

# Clone and install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..

# Install eval dependencies
pip install tyro matplotlib mediapy websockets msgpack numpy==1.24.4
```

Or use the provided script:

```bash
bash examples/LIBERO/eval_files/install_libero.sh
```

### Step 1: Download a checkpoint

Download a pretrained checkpoint from [🤗 StarVLA/bench-libero](https://huggingface.co/collections/StarVLA/bench-libero):

```bash
huggingface-cli download StarVLA/Qwen3-VL-OFT-LIBERO-4in1 \
  --local-dir playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1
```

### Step 2: Start the policy server (Terminal 1 — starVLA env)

Edit the checkpoint path in [`examples/LIBERO/eval_files/run_policy_server.sh`](../examples/LIBERO/eval_files/run_policy_server.sh):

```bash
CKPT=playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
```

Then launch:

```bash
conda activate starVLA
bash examples/LIBERO/eval_files/run_policy_server.sh
```

Wait until you see `server listening on 0.0.0.0:6694`.

### Step 3: Run evaluation (Terminal 2 — LIBERO env)

Edit the paths in [`examples/LIBERO/eval_files/eval_libero.sh`](../examples/LIBERO/eval_files/eval_libero.sh), then:

```bash
conda activate libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
bash examples/LIBERO/eval_files/eval_libero.sh
```

### What to expect

- Each task runs 50 episodes by default
- Videos are saved under `results/{task_suite}/{checkpoint_name}/`
- Success rates are printed at the end

> **Tip:** To evaluate your own trained checkpoint, just point `CKPT` to your checkpoint under `results/Checkpoints/{run_id}/checkpoints/steps_XXXXX_pytorch_model.pt`.

---

## Next Steps

- **Try a different framework:** Change `Framework_name` in the training script to `QwenGR00T`, `QwenPI`, or `QwenFAST`
- **Explore other benchmarks:** Check out [SimplerEnv](../examples/SimplerEnv/), [RoboTwin](../examples/Robotwin/), [Calvin](../examples/calvin/), or [Behavior-1K](../examples/Behavior/)
- **World-Model-for-Action:** Use video-generation models (Cosmos, Wan) as action backbones — see [WM4A](WM4A.md)
- **Co-train with VLM data:** Learn how multi-objective training works in [CoTrainVLM](../examples/CoTrainVLM/)
- **Deploy on real robots:** See the [Franka example](../examples/Franka/) for real-world deployment
- **RL post-training:** Check [StarVLA × RLinf](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/starvla.html)
