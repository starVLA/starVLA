#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CKPT="${1:-${CKPT:-Qwen/Qwen3-VL-4B-Instruct}}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
PORT="${PORT:-6694}"
DTYPE="${DTYPE:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-8}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:--1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

infer_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    echo "${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -
  else
    echo "0"
  fi
}

launch_server() {
  local gpu_id="$1"
  local server_port="$2"
  local cmd=(
    "${STARVLA_PYTHON}" examples/VLN-CE/eval_files/qwenvl_vlm_server.py
    --ckpt_path "${CKPT}"
    --host "${SERVER_HOST}"
    --port "${server_port}"
    --dtype "${DTYPE}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
    --device_map "${DEVICE_MAP}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --num_video_frames "${NUM_VIDEO_FRAMES}"
    --idle_timeout "${IDLE_TIMEOUT}"
  )

  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    cmd+=(--trust_remote_code)
  fi

  echo "Starting QwenVL server: gpu=${gpu_id}, port=${server_port}, ckpt=${CKPT}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}"
}

if [[ "${GPU_IDS}" == "all" ]]; then
  GPU_IDS="$(infer_visible_gpus)"
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"

if [[ "${#GPU_LIST[@]}" -eq 1 ]]; then
  launch_server "${GPU_LIST[0]}" "${PORT}"
else
  PIDS=()
  cleanup() {
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
      kill "${PIDS[@]}" 2>/dev/null || true
    fi
  }
  trap cleanup INT TERM

  for idx in "${!GPU_LIST[@]}"; do
    gpu_id="${GPU_LIST[$idx]}"
    gpu_id="${gpu_id//[[:space:]]/}"
    server_port=$((PORT + idx))
    launch_server "${gpu_id}" "${server_port}" &
    PIDS+=("$!")
  done

  echo "Started ${#GPU_LIST[@]} QwenVL servers on ports ${PORT}-$((PORT + ${#GPU_LIST[@]} - 1)). Press Ctrl-C to stop."
  wait
fi
