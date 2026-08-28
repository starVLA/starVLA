#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-./playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
SAVE_DIR="${SAVE_DIR:-./playground/Pretrained_models/Qwen3-VL-4B-Instruct-VARAction}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-512}"
TOKENS_FILE="${TOKENS_FILE:-/tmp/var_action_tokens_${CODEBOOK_SIZE}.txt}"
DEVICE="${DEVICE:-cuda}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

python starVLA/model/modules/vlm/tools/add_qwen_special_tokens/write_var_action_tokens.py \
  --output "${TOKENS_FILE}" \
  --codebook_size "${CODEBOOK_SIZE}"

python starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py \
  --model-id "${MODEL_ID}" \
  --save-dir "${SAVE_DIR}" \
  --tokens-file "${TOKENS_FILE}" \
  --init-strategy avg \
  --as-special \
  --padding-side left \
  --device "${DEVICE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}"
