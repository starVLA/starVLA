#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18000}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_TASKS="${MAX_TASKS:-1}"
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:--1}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
RESULT_DIR="${RESULT_DIR:-outputs/openpi_libero_eval}"
RUN_ID="${RUN_ID:-pi05_${TASK_SUITE}_smoke}"
MODEL_SOURCE="${MODEL_SOURCE:-openpi}"
ASSETS_CHECKPOINT="${ASSETS_CHECKPOINT:-}"
DEBUG_DUMP_DIR="${DEBUG_DUMP_DIR:-}"
NOISE_SEED="${NOISE_SEED:-}"

mkdir -p "${RESULT_DIR}/${RUN_ID}"

cmd=(
  python examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py
  --model PI05
  --host "${HOST}"
  --port "${PORT}"
  --model-source "${MODEL_SOURCE}"
  --task-suite "${TASK_SUITE}"
  --num-trials "${NUM_TRIALS}"
  --max-tasks "${MAX_TASKS}"
  --task-start "${TASK_START}"
  --task-end "${TASK_END}"
  --replan-steps "${REPLAN_STEPS}"
  --result-json "${RESULT_DIR}/${RUN_ID}/result.json"
)

if [[ -n "${ASSETS_CHECKPOINT}" ]]; then
  cmd+=(--assets-checkpoint "${ASSETS_CHECKPOINT}")
fi

if [[ -n "${DEBUG_DUMP_DIR}" ]]; then
  cmd+=(--debug-dump-dir "${DEBUG_DUMP_DIR}")
fi

if [[ -n "${NOISE_SEED}" ]]; then
  cmd+=(--noise-seed "${NOISE_SEED}")
fi

"${cmd[@]}" "$@"
