# Gemma 4 E2B Backbone for starVLA

Integrates [Google Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) (2.3B effective / 5.1B raw via PLE) as a VLM backbone for starVLA, alongside the existing Qwen-VL family.

## Key Differences from Qwen3-VL

| Parameter | Qwen3-VL-4B | Gemma 4 E2B | Notes |
|---|---|---|---|
| Effective params | ~4B | **2.3B** (PLE) | 42% fewer FLOPs |
| `hidden_size` | 2048 | **1536** | DiT `cross_attention_dim` must be set to 1536 |
| `num_kv_heads` | 2 | **1** | 4× smaller KV cache → faster inference |
| `num_hidden_layers` | 36 | 35 | Compatible with layerwise PI/GR00T heads |
| `sliding_window` | none | **512** | Image token budget needs management |
| `head_dim` | 128 | **256** | flash_attn ≤2.7 doesn't support this → use `sdpa` |

## LIBERO 4-Suite Benchmark (50 trials/task, seed=7)

**Gemma4-E2B + PI head, 40K optimizer steps, effective BS=128 (8×H100)**

| Suite | Success Rate |
|---|---|
| LIBERO-Spatial | **98.4%** (492/500) |
| LIBERO-Object | **98.6%** (493/500) |
| LIBERO-Goal | **96.4%** (482/500) |
| LIBERO-10 | 453/500 (90.6%) |
| **Average** | **96.0%** |

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

### Key Training Flags

```bash
ATTN_IMPL=sdpa              # Required: Gemma4 head_dim=256 > flash_attn limit
ENABLE_GRAD_CKPT=true       # Saves ~30-50% activation memory (recommended)
GRAD_ACCUM=8                # With BS=2×8GPUs×8acc = 128 effective
ZERO_STAGE=2                # DeepSpeed ZeRO stage
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
| `starVLA/model/framework/Gemma4PI.py` | `Gemma4_PI(Qwen_PI)` thin subclass |
| `starVLA/model/framework/Gemma4GR00T.py` | `Gemma4_GR00T(Qwen_GR00T)` thin subclass |
| `starVLA/model/modules/vlm/__init__.py` | +4 lines in dispatcher |

The Gemma4 interface handles:
- Per-Layer Embeddings (PLE) — each decoder layer receives a unique embedding injection
- Gradient checkpointing via `model.gradient_checkpointing_enable(use_reentrant=False)`
- Optional audio tower removal to save ~600MB (`drop_audio_tower: true`)
- `logits_to_keep=1` optimization — avoids materializing (B, L, 262144) logit tensor

## Known Limitations

- **`flash_attention_2` not supported**: Gemma 4's `head_dim=256` exceeds the flash_attn 2.x kernel limit (≤192 for most wheels). Use `attn_implementation=sdpa` instead.
- **Sliding window constraint**: Image tokens must fit within the 512-token window. For multi-view setups, control image budget via processor settings.
- **`transformers < 5.5` incompatible**: `Gemma4ForConditionalGeneration` is only available in transformers ≥ 5.5.0.
