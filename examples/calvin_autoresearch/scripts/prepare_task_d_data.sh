#!/usr/bin/env bash
set -euo pipefail

# Download/extract the official CALVIN task_D_D dataset and repair the shared
# StarVLA/CALVIN links used by closed-loop D evaluation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh" >/dev/null 2>&1 || true
fi

SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
TASK_D_URL="${TASK_D_URL:-http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip}"
ZIP_DIR="${ZIP_DIR:-${SHARED_ROOT}/shared/downloads/calvin_original}"
ZIP_PATH="${ZIP_PATH:-${ZIP_DIR}/task_D_D.zip}"
EXTRACT_PARENT="${EXTRACT_PARENT:-${SHARED_ROOT}/shared/datasets/calvin_original/task_D_D_official_extract}"
TASK_D_DIR="${TASK_D_DIR:-${EXTRACT_PARENT}/task_D_D}"
SHARED_LINK="${SHARED_LINK:-${SHARED_ROOT}/shared/datasets/calvin_original/task_D_D}"
LOCAL_LINK="${LOCAL_LINK:-${PROJECT_ROOT}/data/calvin_original/task_D_D}"
PUBLIC_PARTIAL_ZIP="${PUBLIC_PARTIAL_ZIP:-/inspire/qb-ilm2/project/26summer-camp-10/public/four/calvin/dataset/task_D_D.zip}"
CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH:-/inspire/qb-ilm2/project/26summer-camp-10/public/four/calvin/calvin_models/conf}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"

have_zip_index() {
  local zip="$1"
  [[ -f "${zip}" ]] && unzip -l "${zip}" >/dev/null 2>&1
}

safe_link() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "${src}" ]]; then
    echo "Missing link source: ${src}" >&2
    exit 2
  fi

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
    echo "  existing resolves to: ${dst_real}" >&2
    echo "  wanted resolves to:   ${src_real}" >&2
    exit 3
  fi

  ln -s "${src}" "${dst}"
}

validate_task_d() {
  local data_dir="$1"
  local missing=0

  if [[ ! -f "${data_dir}/validation/.hydra/merged_config.yaml" ]]; then
    echo "Missing: ${data_dir}/validation/.hydra/merged_config.yaml" >&2
    missing=1
  fi

  if ! compgen -G "${data_dir}/validation/episode_*.npz" >/dev/null; then
    echo "Missing validation episodes under: ${data_dir}/validation" >&2
    missing=1
  fi

  if [[ ! -f "${CALVIN_CONFIG_PATH}/annotations/new_playtable_validation.yaml" ]]; then
    echo "Missing CALVIN annotation config: ${CALVIN_CONFIG_PATH}/annotations/new_playtable_validation.yaml" >&2
    missing=1
  fi

  if [[ ! -f "${CALVIN_CONFIG_PATH}/callbacks/rollout/tasks/new_playtable_tasks.yaml" ]]; then
    echo "Missing CALVIN task oracle config: ${CALVIN_CONFIG_PATH}/callbacks/rollout/tasks/new_playtable_tasks.yaml" >&2
    missing=1
  fi

  if [[ "${missing}" == "1" ]]; then
    return 1
  fi
}

if [[ ! -w "${SHARED_ROOT}" ]]; then
  echo "Cannot write shared workspace: ${SHARED_ROOT}" >&2
  echo "Run this from a node/session where /public/seven is writable, or override SHARED_ROOT/ZIP_DIR/EXTRACT_PARENT." >&2
  exit 1
fi

mkdir -p "${ZIP_DIR}" "${EXTRACT_PARENT}" "$(dirname "${SHARED_LINK}")" "$(dirname "${LOCAL_LINK}")"

if validate_task_d "${TASK_D_DIR}" >/dev/null 2>&1; then
  echo "[task_d] existing official task_D_D is valid: ${TASK_D_DIR}"
else
  if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
    if have_zip_index "${ZIP_PATH}"; then
      echo "[task_d] using existing complete zip: ${ZIP_PATH}"
    elif have_zip_index "${PUBLIC_PARTIAL_ZIP}"; then
      echo "[task_d] reusing existing complete zip: ${PUBLIC_PARTIAL_ZIP}"
      ln -sfn "${PUBLIC_PARTIAL_ZIP}" "${ZIP_PATH}"
    else
      echo "[task_d] downloading official CALVIN task_D_D -> ${ZIP_PATH}"
      echo "[task_d] expected compressed size: about 165 GiB"
      wget -c -O "${ZIP_PATH}" "${TASK_D_URL}"
    fi
  fi

  if [[ "${SKIP_EXTRACT}" != "1" ]]; then
    if ! have_zip_index "${ZIP_PATH}"; then
      echo "Zip is missing or incomplete: ${ZIP_PATH}" >&2
      exit 4
    fi

    echo "[task_d] extracting ${ZIP_PATH} -> ${EXTRACT_PARENT}"
    unzip -q -n "${ZIP_PATH}" -d "${EXTRACT_PARENT}"
  fi
fi

validate_task_d "${TASK_D_DIR}"

safe_link "${TASK_D_DIR}" "${SHARED_LINK}"
safe_link "${SHARED_LINK}" "${LOCAL_LINK}"

echo "[task_d] ready"
echo "  official data: ${TASK_D_DIR}"
echo "  shared link:   ${SHARED_LINK} -> $(readlink -f "${SHARED_LINK}")"
echo "  local link:    ${LOCAL_LINK} -> $(readlink -f "${LOCAL_LINK}")"
