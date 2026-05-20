#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

: "${CKPT:?Set CKPT to the same newly trained WMH checkpoint served by run_policy_server.sh}"
: "${CALVIN_PYTHON:?Set CALVIN_PYTHON to the Python executable for your CALVIN env}"
: "${CALVIN_D_DATASET:?Set CALVIN_D_DATASET to the original CALVIN task_D_D dataset path}"
: "${CALVIN_CONFIG_PATH:?Set CALVIN_CONFIG_PATH to the CALVIN calvin_models/conf directory}"

case "${CKPT}" in
  *Qwen3-VL-OFT-LIBERO*|*LIBERO*|*Robotwin*|*robotwin*|*Robocasa*|*robocasa*|*Behavior*|*BEHAVIOR*|*SimplerEnv*|*qwenpi_calvin_task_D_D*)
    echo "Refusing action-trained upstream checkpoint: ${CKPT}" >&2
    exit 2
    ;;
esac

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5694}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1}"
SEQUENCE_START="${SEQUENCE_START:-0}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-1}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-results/calvin_eval_smoke}"
EVAL_DATASET_PATH="${CALVIN_D_DATASET}"
CALVIN_ROOT="${CALVIN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/four/calvin}"
CALVIN_MODELS_PATH="${CALVIN_MODELS_PATH:-${CALVIN_ROOT}/calvin_models}"
CALVIN_ENV_PATH="${CALVIN_ENV_PATH:-${CALVIN_ROOT}/calvin_env}"
CALVIN_VENDOR_SITE="${CALVIN_VENDOR_SITE:-${STARVLA_ROOT}/examples/calvin_autoresearch/vendor/py38_site}"

if [[ ! -f "${EVAL_DATASET_PATH}/validation/.hydra/merged_config.yaml" ]]; then
  if [[ -f "${CALVIN_D_DATASET}/training/.hydra/merged_config.yaml" ]]; then
    COMPAT_D_DATASET="${COMPAT_D_DATASET:-results/calvin_dataset_compat/task_D_D}"
    mkdir -p "${COMPAT_D_DATASET}"
    if [[ ! -e "${COMPAT_D_DATASET}/validation" && ! -L "${COMPAT_D_DATASET}/validation" ]]; then
      ln -s "${CALVIN_D_DATASET}/training" "${COMPAT_D_DATASET}/validation"
    fi
    EVAL_DATASET_PATH="${COMPAT_D_DATASET}"
  else
    echo "Missing CALVIN validation hydra config under ${CALVIN_D_DATASET}" >&2
    exit 4
  fi
fi

mkdir -p "${EVAL_LOG_DIR}/mplconfig"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${EVAL_LOG_DIR}/mplconfig}"
export CALVIN_USE_EGL="${CALVIN_USE_EGL:-0}"
export PYTHONPATH="${CALVIN_VENDOR_SITE}:${STARVLA_ROOT}:${CALVIN_MODELS_PATH}:${CALVIN_ENV_PATH}:${PYTHONPATH:-}"

exec "${CALVIN_PYTHON}" examples/calvin/eval_files/eval_calvin.py \
  --args.pretrained-path "${CKPT}" \
  --args.unnorm-key "${UNNORM_KEY}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.dataset_path "${EVAL_DATASET_PATH}" \
  --args.calvin_config_path "${CALVIN_CONFIG_PATH}" \
  --args.eval_sequences_path examples/calvin/eval_files/eval_sequences.json \
  --args.num_sequences "${NUM_SEQUENCES}" \
  --args.sequence_start "${SEQUENCE_START}" \
  --args.sequence_stride "${SEQUENCE_STRIDE}" \
  --args.eval_log_dir "${EVAL_LOG_DIR}"
