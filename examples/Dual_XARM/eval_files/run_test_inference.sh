#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

star_vla_python=${STAR_VLA_PYTHON:-/root/miniconda3/envs/starVLA/bin/python}

${star_vla_python} examples/Dual_XARM/eval_files/test_policy_client.py \
  --host ${HOST:-127.0.0.1} \
  --port ${PORT:-5694} \
  --instruction "${INSTRUCTION:-Pick up the box and place it to the target area.}" \
  ${HEAD_IMAGE:+--head_image $HEAD_IMAGE} \
  ${LEFT_WRIST_IMAGE:+--left_wrist_image $LEFT_WRIST_IMAGE} \
  ${RIGHT_WRIST_IMAGE:+--right_wrist_image $RIGHT_WRIST_IMAGE}
