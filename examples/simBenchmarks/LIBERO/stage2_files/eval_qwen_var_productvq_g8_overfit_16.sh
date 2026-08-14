#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

RUN_DIR="playground/Checkpoints/qwen_var_productvq_g8_stage2_overfit_16_fullcache"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/final_model/pytorch_model.pt}"
OUTPUT="${OUTPUT:-${RUN_DIR}/offline_eval_overfit16.json}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangfeihong/miniconda3/envs/starVLA/bin/python}"

"${PYTHON_BIN}" examples/simBenchmarks/LIBERO/eval_files/eval_var_stage2_offline.py \
  --config_yaml examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_productvq_g8_overfit_16.yaml \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  --token_cache playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g8/stage2_token_cache_full.pt \
  --device "${DEVICE:-cuda}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --num_workers 0 \
  --max_batches 0 \
  --num_debug_examples 8
