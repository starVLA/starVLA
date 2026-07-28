#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
METAWORLD_PYTHON="${METAWORLD_PYTHON:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10095}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
SEED="${SEED:-4042}"
LEVELS="${LEVELS:-easy,medium,hard,very_hard}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/metaworld_example/checkpoints/steps_50000_pytorch_model.pt}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"

FOLDER_NAME="$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"
MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
VIDEO_OUT_PATH="${MODEL_ROOT}/results/metaworld_mt50/${FOLDER_NAME}"

"${METAWORLD_PYTHON}" ./examples/simBenchmarks/MetaWorld/eval_files/eval_metaworld.py \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.episodes-per-task "${EPISODES_PER_TASK}" \
  --args.seed "${SEED}" \
  --args.levels "${LEVELS}" \
  --args.video-out-path "${VIDEO_OUT_PATH}"
