#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

RUN_DIR="playground/Checkpoints/qwen_var_productvq_g8_stage2_last8_20k_fullcache"
if [[ ! -d "${RUN_DIR}/checkpoints" ]]; then
  echo "Missing ${RUN_DIR}/checkpoints; run last8 training first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/zhangfeihong/miniconda3/envs/starVLA/bin/accelerate}"

"${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_qwenpi_libero_stable.yaml \
  --main_process_port "${MAIN_PROCESS_PORT:-29523}" \
  --num_processes "${NUM_PROCESSES:-4}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_productvq_g8_last8_20k_resume.yaml
