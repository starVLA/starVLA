# Gemma 4 E2B Backbone for starVLA

Integrates [Google Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) (2.3B effective / 5.1B raw via PLE) as a VLM backbone for starVLA, alongside the existing Qwen-VL family.

## LIBERO 4-Suite Benchmark (50 trials/task, seed=7)

**Gemma4-E2B + PI head, 40K optimizer steps, effective BS=128 (8×H100)**

| Suite | Success Rate |
|---|---|
| LIBERO-Spatial | **98.4%** (492/500) |
| LIBERO-Object | **98.6%** (493/500) |
| LIBERO-Goal | **96.4%** (482/500) |
| LIBERO-10 | **90.6%** (453/500) |
| **Average** | **96.0%** (1920/2000) |

## Quick Start

### Requirements

- `transformers >= 5.5.0` (for `Gemma4ForConditionalGeneration`)
- `torch >= 2.1` with CUDA support
- Gemma 4 E2B weights: `google/gemma-4-E2B-it` from Hugging Face

### Smoke Test (single GPU)

```bash
conda activate <your_env>
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 python starVLA/model/modules/vlm/Gemma4.py --attn eager
CUDA_VISIBLE_DEVICES=0 python starVLA/model/framework/Gemma4PI.py --attn eager
```

### Training (multi-GPU with Slurm)

```bash
# Gemma4 + PI head, libero_all, 100K steps, effective BS=128
sbatch examples/Gemma4/submit_hpc3_libero.sh

# Switch to GR00T head
FRAMEWORK=Gemma4GR00T sbatch examples/Gemma4/submit_hpc3_libero.sh

# Single suite for quick ablation
DATA_MIX=libero_spatial MAX_STEPS=50000 sbatch examples/Gemma4/submit_hpc3_libero.sh
```

### Evaluation

```bash
export MUJOCO_GL=osmesa
CUDA_VISIBLE_DEVICES=0 python examples/Gemma4/eval_libero_local.py \
  --ckpt <checkpoint_path> \
  --task-suite libero_spatial \
  --num-trials 50
```

## Architecture

Only **3 new files + 1 dispatcher line** — no changes to existing starVLA code:

| File | Description |
|---|---|
| `starVLA/model/modules/vlm/Gemma4.py` | `_Gemma4_VL_Interface` — matches `_QWen3_VL_Interface` API |
| `starVLA/model/framework/VLM4A/Gemma4PI.py` | `Gemma4_PI(Qwen_PI)` thin subclass |
| `starVLA/model/framework/VLM4A/Gemma4GR00T.py` | `Gemma4_GR00T(Qwen_GR00T)` thin subclass |
| `starVLA/model/modules/vlm/__init__.py` | +4 lines in dispatcher |
