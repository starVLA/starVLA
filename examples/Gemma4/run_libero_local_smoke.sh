#!/bin/bash
# Local smoke test on LFT-A6000-1: forward + predict_action on a fake batch.
# Usage: bash examples/Gemma4/run_libero_local_smoke.sh

set -euo pipefail

# Force HF traffic out through the public route on this box.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# Activate the gemma-vla conda env (cloned from latent_wam2 with transformers>=5.5).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gemma_vla

cd "$(dirname "$0")/../.."   # cd into gemma-vla repo root
export PYTHONPATH="$PWD"

GPU_ID="${GPU_ID:-1}"        # GPU 0 is busy on LFT-A6000-1; default to GPU 1
ATTN="${ATTN:-eager}"        # flash_attention_2 may not yet have a Gemma4 kernel
MODEL_ID="${MODEL_ID:-google/gemma-4-E2B-it}"

LOG_DIR="results/smoke"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "[smoke] GPU=$GPU_ID  ATTN=$ATTN  MODEL=$MODEL_ID"

CUDA_VISIBLE_DEVICES="$GPU_ID" python starVLA/model/modules/vlm/Gemma4.py \
  --model_id "$MODEL_ID" \
  --attn "$ATTN" \
  2>&1 | tee "$LOG_DIR/${TS}_gemma4_vlm.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" python starVLA/model/framework/Gemma4PI.py \
  --model_id "$MODEL_ID" \
  --attn "$ATTN" \
  2>&1 | tee "$LOG_DIR/${TS}_gemma4pi.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" python starVLA/model/framework/Gemma4GR00T.py \
  --model_id "$MODEL_ID" \
  --attn "$ATTN" \
  2>&1 | tee "$LOG_DIR/${TS}_gemma4gr00t.log"

echo "[smoke] Done. Logs in $LOG_DIR/${TS}_*.log"
