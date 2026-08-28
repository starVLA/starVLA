#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-/home/zhangfeihong/miniconda3/envs/starVLA/bin/python}"

"${PYTHON_BIN}" examples/simBenchmarks/LIBERO/eval_files/check_var_stage2_labels.py \
  --config_yaml examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_productvq_g8_overfit_16.yaml \
  --output playground/Checkpoints/var_stage2_productvq_g8_label_sanity_overfit16.json \
  --device "${DEVICE:-cpu}" \
  --num_samples "${NUM_SAMPLES:-16}" \
  --max_samples 16 \
  --validate_cache_online \
  --check_qwen_tokenizer
