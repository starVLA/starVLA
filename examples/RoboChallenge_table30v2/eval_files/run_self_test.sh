#!/usr/bin/env bash
# Step-1 self-test: load checkpoint + dummy obs, no network.
set -euo pipefail
cd "$(dirname "$0")/../../.."

CKPT="${CKPT:-./playground/Checkpoints/robochallenge_table30v2_qwenoft_shred_paper_100step/checkpoints/steps_100_pytorch_model.pt}"
ROBOT_TAG="${ROBOT_TAG:-ur5}"
PROMPT="${PROMPT:-shred the paper}"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate starVLA_dev

export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.2/toolkit/12.2.2}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python examples/RoboChallenge_table30v2/eval_files/local_self_test.py \
    --checkpoint "${CKPT}" \
    --robot_tag "${ROBOT_TAG}" \
    --prompt "${PROMPT}" \
    "$@"
