#!/bin/bash
# Local smoke test: VLM forward + PI/GR00T fake-batch checks.
# Usage: bash examples/MiniCPM/run_libero_local_smoke.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"

GPU_ID="${GPU_ID:-0}"
ATTN="${ATTN:-sdpa}"
MODEL_ID="${MODEL_ID:-openbmb/MiniCPM-V-4.6}"

LOG_DIR="results/smoke"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "[smoke] GPU=$GPU_ID  ATTN=$ATTN  MODEL=$MODEL_ID"

CUDA_VISIBLE_DEVICES="$GPU_ID" python starVLA/model/modules/vlm/MiniCPM_V.py \
  --model_id "$MODEL_ID" \
  --attn "$ATTN" \
  2>&1 | tee "$LOG_DIR/${TS}_minicpm_vlm.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" python starVLA/model/framework/VLM4A/MiniCPMPI.py \
  --model_id "$MODEL_ID" \
  --attn "$ATTN" \
  2>&1 | tee "$LOG_DIR/${TS}_minicpmpi.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" python starVLA/model/framework/VLM4A/MiniCPMGR00T.py \
  --model_id "$MODEL_ID" \
  --attn "$ATTN" \
  2>&1 | tee "$LOG_DIR/${TS}_minicpmgr00t.log"

echo "[smoke] Done. Logs in $LOG_DIR/${TS}_*.log"
