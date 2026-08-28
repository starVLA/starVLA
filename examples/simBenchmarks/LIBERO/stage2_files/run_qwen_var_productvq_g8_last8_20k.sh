#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

WARMUP_CKPT="playground/Checkpoints/qwen_var_productvq_g8_stage2_warmup_2k_fullcache/checkpoints/steps_2000_pytorch_model.pt"
if [[ ! -f "${WARMUP_CKPT}" ]]; then
  echo "Missing ${WARMUP_CKPT}; finish Stage2 warmup first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/zhangfeihong/miniconda3/envs/starVLA/bin/accelerate}"

"${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_qwenpi_libero_stable.yaml \
  --main_process_port "${MAIN_PROCESS_PORT:-29522}" \
  --num_processes "${NUM_PROCESSES:-4}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_productvq_g8_last8_20k.yaml
