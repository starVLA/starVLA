#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
star_vla_python="${star_vla_python:-python}"
port="${port:-10093}"
gpu_id="${gpu_id:-0}"
seed="${seed:-0}"
idle_timeout="${idle_timeout:--1}"
your_ckpt="${your_ckpt:-./results/Checkpoints/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt}"
base_vlm="${base_vlm:-Qwen/Qwen3-VL-4B-Instruct}"
USE_BF16="${USE_BF16:-1}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"

CMD=(
  "${star_vla_python}" deployment/model_server/server_policy.py
  --ckpt_path "${your_ckpt}"
  --port "${port}"
  --seed "${seed}"
  --idle_timeout "${idle_timeout}"
  --config_override "framework.qwenvl.base_vlm=${base_vlm}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" "${CMD[@]}" 2>&1 | tee "${log_file}"
