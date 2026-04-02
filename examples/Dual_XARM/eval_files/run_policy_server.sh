#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

star_vla_python=${STAR_VLA_PYTHON:-/root/miniconda3/envs/starVLA/bin/python}
your_ckpt=${YOUR_CKPT:-./results/Checkpoints/replace_with_your_run/final_model/pytorch_model.pt}
port=${PORT:-5694}
gpu_id=${GPU_ID:-0}

ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"

CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"
