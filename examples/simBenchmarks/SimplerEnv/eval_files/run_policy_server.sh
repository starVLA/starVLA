#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
star_vla_python="${star_vla_python:-python}"
port="${port:-6678}"
gpu_id="${gpu_id:-0}"
your_ckpt="${your_ckpt:-./results/Checkpoints/0418_oxe_bridge_rt_1_QwenGR00T/checkpoints/steps_10000_pytorch_model.pt}"
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
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" "${CMD[@]}" 2>&1 | tee "${log_file}"
