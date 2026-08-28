#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

TOKEN_CACHE="playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g8/stage2_token_cache_full.pt"
WARMUP_CKPT="playground/Checkpoints/qwen_var_productvq_g8_stage2_warmup_2k_fullcache/checkpoints/steps_2000_pytorch_model.pt"
if [[ ! -f "${TOKEN_CACHE}" ]]; then
  echo "Missing ${TOKEN_CACHE}; build it first with examples/simBenchmarks/LIBERO/stage2_files/build_productvq_g8_full_token_cache.sh" >&2
  exit 1
fi
if [[ ! -f "${WARMUP_CKPT}" ]]; then
  echo "Missing ${WARMUP_CKPT}; finish Stage2 warmup first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/zhangfeihong/miniconda3/envs/starVLA/bin/accelerate}"

"${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_qwenpi_libero_stable.yaml \
  --main_process_port "${MAIN_PROCESS_PORT:-29525}" \
  --num_processes "${NUM_PROCESSES:-4}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_productvq_g8_overfit_16.yaml
