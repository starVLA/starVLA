#!/usr/bin/env bash
set -euo pipefail

# Strict CALVIN ABC->D closed-loop evaluation entrypoint.
# Unlike the smoke script, this requires a real task_D_D/validation split.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

: "${CKPT:?Set CKPT to a WMH-trained checkpoint, e.g. .../checkpoints/steps_60000_pytorch_model.pt}"
: "${CALVIN_PYTHON:?Set CALVIN_PYTHON to the Python executable for the CALVIN env}"
: "${CALVIN_D_DATASET:?Set CALVIN_D_DATASET to the official CALVIN task_D_D dataset path}"
: "${CALVIN_CONFIG_PATH:?Set CALVIN_CONFIG_PATH to the CALVIN calvin_models/conf directory}"

case "${CKPT}" in
  *Qwen3-VL-OFT-LIBERO*|*LIBERO*|*Robotwin*|*robotwin*|*Robocasa*|*robocasa*|*Behavior*|*BEHAVIOR*|*SimplerEnv*|*qwenpi_calvin_task_D_D*)
    echo "Refusing action-trained upstream checkpoint: ${CKPT}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 3
fi

if [[ ! -f "${CALVIN_D_DATASET}/validation/.hydra/merged_config.yaml" ]]; then
  echo "Formal D eval requires ${CALVIN_D_DATASET}/validation/.hydra/merged_config.yaml" >&2
  exit 4
fi

if ! compgen -G "${CALVIN_D_DATASET}/validation/episode_*.npz" >/dev/null; then
  echo "Formal D eval requires validation episode_*.npz files under ${CALVIN_D_DATASET}/validation" >&2
  exit 5
fi

if [[ ! -f "${CALVIN_CONFIG_PATH}/annotations/new_playtable_validation.yaml" ]]; then
  echo "Missing CALVIN validation annotations under ${CALVIN_CONFIG_PATH}/annotations" >&2
  exit 6
fi

if [[ ! -f "${CALVIN_CONFIG_PATH}/callbacks/rollout/tasks/new_playtable_tasks.yaml" ]]; then
  echo "Missing CALVIN task oracle config under ${CALVIN_CONFIG_PATH}/callbacks/rollout/tasks" >&2
  exit 7
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5694}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
SEQUENCE_START="${SEQUENCE_START:-0}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-1}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-results/calvin_eval_d_formal}"
DEBUG="${DEBUG:-0}"
CALVIN_SEND_STATE="${CALVIN_SEND_STATE:-0}"
CALVIN_STATE_MODE="${CALVIN_STATE_MODE:-normal}"
CALVIN_STATE_SHUFFLE_BUFFER="${CALVIN_STATE_SHUFFLE_BUFFER:-32}"
CALVIN_ROOT="${CALVIN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/four/calvin}"
CALVIN_MODELS_PATH="${CALVIN_MODELS_PATH:-${CALVIN_ROOT}/calvin_models}"
CALVIN_ENV_PATH="${CALVIN_ENV_PATH:-${CALVIN_ROOT}/calvin_env}"
CALVIN_VENDOR_SITE="${CALVIN_VENDOR_SITE:-${STARVLA_ROOT}/examples/calvin_autoresearch/vendor/py38_site}"

mkdir -p "${EVAL_LOG_DIR}"
mkdir -p "${EVAL_LOG_DIR}/mplconfig"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${EVAL_LOG_DIR}/mplconfig}"
export CALVIN_USE_EGL="${CALVIN_USE_EGL:-0}"
export PYTHONPATH="${CALVIN_VENDOR_SITE}:${STARVLA_ROOT}:${CALVIN_MODELS_PATH}:${CALVIN_ENV_PATH}:${PYTHONPATH:-}"

args=(
  examples/calvin/eval_files/eval_calvin.py
  --args.pretrained-path "${CKPT}"
  --args.unnorm-key "${UNNORM_KEY}"
  --args.host "${HOST}"
  --args.port "${PORT}"
  --args.dataset_path "${CALVIN_D_DATASET}"
  --args.calvin_config_path "${CALVIN_CONFIG_PATH}"
  --args.eval_sequences_path examples/calvin/eval_files/eval_sequences.json
  --args.num_sequences "${NUM_SEQUENCES}"
  --args.sequence_start "${SEQUENCE_START}"
  --args.sequence_stride "${SEQUENCE_STRIDE}"
  --args.eval_log_dir "${EVAL_LOG_DIR}"
  --args.state_mode "${CALVIN_STATE_MODE}"
  --args.state_shuffle_buffer "${CALVIN_STATE_SHUFFLE_BUFFER}"
)

if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
  args+=(--args.debug)
fi

if [[ "${CALVIN_SEND_STATE}" == "1" || "${CALVIN_SEND_STATE}" == "true" || "${CALVIN_SEND_STATE}" == "True" ]]; then
  args+=(--args.send-state)
fi

exec "${CALVIN_PYTHON}" "${args[@]}"
