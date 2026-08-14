#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangfeihong/starVLA

TOKEN_CACHE="playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s8_structemb/stage2_token_cache_full.pt"
if [[ ! -f "${TOKEN_CACHE}" ]]; then
  echo "Missing ${TOKEN_CACHE}; build it first with examples/simBenchmarks/LIBERO/stage2_files/build_productvq_g16_s8_structemb_full_token_cache.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7,8,9}"
export PYTHONPATH="${PYTHONPATH:-/home/zhangfeihong/starVLA}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false

RUN_DIR="playground/Checkpoints/qwen_var_productvq_g16_s8_structemb_time_parallel_100k_fullcache"
mkdir -p "${RUN_DIR}"

LOG_FILE="${RUN_DIR}/train.log"

/home/zhangfeihong/miniconda3/envs/starVLA/bin/accelerate launch \
  --num_processes "${NUM_PROCESSES:-4}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29533}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_productvq_g16_s8_structemb_time_parallel_100k.yaml \
  2>&1 | tee "${LOG_FILE}"
