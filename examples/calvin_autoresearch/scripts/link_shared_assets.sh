#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220172}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing shared asset path: ${path}" >&2
    exit 2
  fi
}

safe_link() {
  local src="$1"
  local dst="$2"
  require_path "${src}"
  mkdir -p "$(dirname "${dst}")"

  if [[ -L "${dst}" ]]; then
    ln -sfn "${src}" "${dst}"
    return
  fi

  if [[ -e "${dst}" ]]; then
    local src_real dst_real
    src_real="$(readlink -f "${src}")"
    dst_real="$(readlink -f "${dst}")"
    if [[ "${src_real}" == "${dst_real}" ]]; then
      return
    fi
    echo "Refusing to overwrite non-symlink path: ${dst}" >&2
    exit 3
  fi

  ln -s "${src}" "${dst}"
}

MODEL_SHARED="${SHARED_ROOT}/shared/models/base/Qwen3-VL-4B-Instruct-Action"
ABC_SHARED="${SHARED_ROOT}/shared/datasets/calvin_lerobot/calvin_abc_train_v3.0"
D_SHARED="${SHARED_ROOT}/shared/datasets/calvin_original/task_D_D"

require_path "${MODEL_SHARED}"
require_path "${ABC_SHARED}"
require_path "${D_SHARED}"

safe_link "${MODEL_SHARED}" "${PROJECT_ROOT}/models/Qwen3-VL-4B-Instruct-Action"
safe_link "${ABC_SHARED}" "${PROJECT_ROOT}/data/calvin_lerobot/calvin_abc_train_v3.0"
safe_link "${D_SHARED}" "${PROJECT_ROOT}/data/calvin_original/task_D_D"

echo "Linked local project assets to ${SHARED_ROOT}"
