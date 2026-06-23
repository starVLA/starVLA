# RoboCasa Stage2 Restart Guide

This note describes how to restart the current RoboCasa Stage2 training run from the shipped code and Stage1 tokenizer assets.

## What Is Included

The required Stage1 tokenizer files are expected at:

```text
playground/Checkpoints/var_stage1_robocasa_gr1_e64_aeinit_productvq_g16_s1_2_4_8_16_batch256_rerun/
  epoch_027.ckpt
  config.yaml
  action_spec.json
  starvla_base_config.yaml
```

Only `epoch_027.ckpt` is required by the code path. The YAML and JSON files are included for provenance and config checks.

The Stage2 token cache is not included by default:

```text
playground/Checkpoints/var_stage1_robocasa_gr1_e64_aeinit_productvq_g16_s1_2_4_8_16_batch256_rerun/stage2_token_cache_epoch027.pt
```

Build it on the target server from the Stage1 checkpoint and RoboCasa training data.

## Required Local Assets

From the repository root, make sure these paths exist or override them on the command line:

```text
playground/Pretrained_models/Qwen3-VL-4B-Instruct-VARAction
playground/Datasets/RoboCasa-GR1/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot
playground/Checkpoints/var_stage1_robocasa_gr1_e64_aeinit_productvq_g16_s1_2_4_8_16_batch256_rerun/epoch_027.ckpt
```

The RoboCasa data should contain the `fourier_gr1_unified_local_1000` mix used by the Stage1 and Stage2 configs.

## Proprio State Conditioning

The current RoboCasa Stage2 path uses proprio/state input.

The Stage2 config enables state loading:

```yaml
datasets:
  vla_data:
    include_state: true
```

The model config enables a 58-dim proprio encoder:

```yaml
framework:
  proprio_state:
    enabled: true
    state_dim: 58
    add_context_token: true
    add_to_pooled: true
```

`QwenVARScaleParallel` reads `example["state"]`, encodes the last state vector, adds it to pooled context, and appends it as an extra context token.

## Build The Stage2 Token Cache

Run from the repository root:

```bash
bash examples/Robocasa_tabletop/stage2_files/build_productvq_g16_s124816_robocasa_epoch027_token_cache.sh
```

Useful overrides:

```bash
CACHE_DEVICE=cuda \
CACHE_BATCH_SIZE=256 \
CACHE_NUM_WORKERS=8 \
bash examples/Robocasa_tabletop/stage2_files/build_productvq_g16_s124816_robocasa_epoch027_token_cache.sh
```

For a quick smoke build:

```bash
CACHE_MAX_BATCHES=1 \
CACHE_NUM_WORKERS=0 \
bash examples/Robocasa_tabletop/stage2_files/build_productvq_g16_s124816_robocasa_epoch027_token_cache.sh
```

Expected full-cache output:

```text
playground/Checkpoints/var_stage1_robocasa_gr1_e64_aeinit_productvq_g16_s1_2_4_8_16_batch256_rerun/stage2_token_cache_epoch027.pt
```

## Start Stage2 Training

After the token cache exists, launch:

```bash
bash examples/Robocasa_tabletop/stage2_files/run_qwen_var_productvq_g16_s124816_robocasa_epoch027_100k.sh
```

The run script defaults to:

```text
CUDA_VISIBLE_DEVICES=2,3,4,5
NUM_PROCESSES=4
WANDB_MODE=disabled
MAIN_PROCESS_PORT=29553
```

Override as needed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
MAIN_PROCESS_PORT=29553 \
bash examples/Robocasa_tabletop/stage2_files/run_qwen_var_productvq_g16_s124816_robocasa_epoch027_100k.sh
```

If the base model or data live somewhere else, pass overrides directly to `train_starvla.py` instead of using the wrapper script:

```bash
accelerate launch \
  --num_processes 4 \
  --main_process_port 29553 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robocasa_tabletop/stage2_files/train_qwen_var_productvq_g16_s124816_robocasa_epoch027_100k.yaml \
  --framework.qwenvl.base_vlm /path/to/Qwen3-VL-4B-Instruct-VARAction \
  --datasets.vla_data.data_root_dir /path/to/LeRobot \
  --trainer.is_resume false \
  --run_id qwen_var_productvq_g16_s124816_robocasa_epoch027_100k_state
```

Use a new `run_id` when starting from scratch. Do not resume an older no-state RoboCasa Stage2 checkpoint into this config unless you intentionally handle partial checkpoint loading, because the current model has additional proprio-state parameters.

## Smoke Test

A minimal smoke run can verify the code path without committing to a full training job:

```bash
accelerate launch \
  --num_processes 1 \
  --main_process_port 29553 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robocasa_tabletop/stage2_files/train_qwen_var_productvq_g16_s124816_robocasa_epoch027_100k.yaml \
  --run_id smoke_robocasa_stage2_state \
  --trainer.is_resume false \
  --trainer.max_train_steps 1 \
  --datasets.vla_data.max_samples 2 \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.num_workers 0 \
  --trainer.freeze_modules qwen_vl_interface
```

The full unfrozen model is large; use multi-GPU for real training.
