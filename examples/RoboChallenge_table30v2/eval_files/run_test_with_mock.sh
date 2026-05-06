#!/usr/bin/env bash
# Step-2 self-test: drive upstream mock_robot_server.py with our policy.
# Requires:
#   * upstream repo cloned at $RC_REPO (default: ~/playground/Code/RoboChallengeInference)
#   * mock_robot_server.py already running on 127.0.0.1:9098
set -euo pipefail
cd "$(dirname "$0")/../../.."

CKPT="${CKPT:-./playground/Checkpoints/robochallenge_table30v2_qwenoft_shred_paper_100step/checkpoints/steps_100_pytorch_model.pt}"
ROBOT_TAG="${ROBOT_TAG:-ur5}"
PROMPT="${PROMPT:-shred the paper}"
RC_REPO="${RC_REPO:-$HOME/playground/Code/RoboChallengeInference}"
MAX_WAIT="${MAX_WAIT:-60}"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate starVLA_dev

export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.2/toolkit/12.2.2}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python examples/RoboChallenge_table30v2/eval_files/test_with_mock_server.py \
    --checkpoint "${CKPT}" \
    --robot_tag "${ROBOT_TAG}" \
    --prompt "${PROMPT}" \
    --rc_repo "${RC_REPO}" \
    --max_wait "${MAX_WAIT}" \
    "$@"
