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

HF_TMP_HOME="${HF_TMP_HOME:-/tmp/hf_starvla_assets}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3-VL-4B-Instruct-Action}"
PUBLIC_MODEL_DIR="${PUBLIC_MODEL_DIR:-/inspire/qb-ilm2/project/26summer-camp-10/public/Qwen3-VL-4B-Instruct-Action}"
LEROBOT_ROOT="${LEROBOT_ROOT:-playground/Datasets/calvin_lerobot}"
ABC_DIR="${ABC_DIR:-${LEROBOT_ROOT}/calvin_abc_train_v3.0}"
D_LEROBOT_DIR="${D_LEROBOT_DIR:-${LEROBOT_ROOT}/calvin_task_D_D_v3.0}"
CALVIN_ORIGINAL_ROOT="${CALVIN_ORIGINAL_ROOT:-playground/Datasets/calvin_original}"
SHARED_CALVIN_ROOT="${SHARED_CALVIN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d}"
DOWNLOAD_ORIGINAL_D="${DOWNLOAD_ORIGINAL_D:-0}"
DOWNLOAD_D_LEROBOT="${DOWNLOAD_D_LEROBOT:-0}"
DIRECT_HF="${DIRECT_HF:-1}"

run_hf() {
  if [[ "${DIRECT_HF}" == "1" ]]; then
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
      HF_HOME="${HF_TMP_HOME}" HF_HUB_CACHE="${HF_TMP_HOME}/hub" "$@"
  else
    HF_HOME="${HF_TMP_HOME}" HF_HUB_CACHE="${HF_TMP_HOME}/hub" "$@"
  fi
}

copy_modality() {
  local dataset_dir="$1"
  install -d "${dataset_dir}/meta"
  cp examples/calvin_autoresearch/train_files/modality_fywang_calvin_lerobot_v2.json \
    "${dataset_dir}/meta/modality.json"
}

install -d "$(dirname "${MODEL_DIR}")"
if [[ -d "${PUBLIC_MODEL_DIR}" && -f "${PUBLIC_MODEL_DIR}/config.json" ]]; then
  echo "[assets] linking public allowed base model -> ${MODEL_DIR}"
  if [[ -e "${MODEL_DIR}" || -L "${MODEL_DIR}" ]]; then
    echo "[assets] model path already exists: ${MODEL_DIR}"
  else
    ln -sfn "${PUBLIC_MODEL_DIR}" "${MODEL_DIR}"
  fi
elif [[ -e "${MODEL_DIR}" || -L "${MODEL_DIR}" ]]; then
  echo "[assets] model path already exists: ${MODEL_DIR}"
else
  echo "[assets] downloading allowed base model -> ${MODEL_DIR}"
  model_snapshot="$(run_hf hf download StarVLA/Qwen3-VL-4B-Instruct-Action \
    --repo-type model \
    --cache-dir "${HF_TMP_HOME}/hub" \
    --max-workers "${HF_MAX_WORKERS:-8}")"
  ln -sfn "${model_snapshot}" "${MODEL_DIR}"
fi

install -d "${LEROBOT_ROOT}"
if [[ -d "${SHARED_CALVIN_ROOT}/calvin_task_ABC_D" ]]; then
  echo "[assets] linking shared CALVIN ABC LeRobot data -> ${ABC_DIR}"
  if [[ -e "${ABC_DIR}" || -L "${ABC_DIR}" ]]; then
    echo "[assets] ABC path already exists: ${ABC_DIR}"
  else
    ln -sfn "${SHARED_CALVIN_ROOT}/calvin_task_ABC_D" "${ABC_DIR}"
  fi
else
  echo "[assets] downloading CALVIN ABC LeRobot data -> ${ABC_DIR}"
  run_hf hf download fywang/calvin-task-ABC-D-lerobot \
    --repo-type dataset \
    --local-dir "${ABC_DIR}" \
    --max-workers "${HF_MAX_WORKERS:-8}"
  copy_modality "${ABC_DIR}"
fi

if [[ "${DOWNLOAD_D_LEROBOT}" == "1" ]]; then
  echo "[assets] downloading CALVIN D-D LeRobot data -> ${D_LEROBOT_DIR}"
  run_hf hf download fywang/calvin-task-D-D-lerobot \
    --repo-type dataset \
    --local-dir "${D_LEROBOT_DIR}" \
    --max-workers "${HF_MAX_WORKERS:-8}"
  copy_modality "${D_LEROBOT_DIR}"
fi

install -d "${CALVIN_ORIGINAL_ROOT}"
if [[ -d "${SHARED_CALVIN_ROOT}/task_D_D" ]]; then
  echo "[assets] linking shared original CALVIN task_D_D -> ${CALVIN_ORIGINAL_ROOT}/task_D_D"
  if [[ -e "${CALVIN_ORIGINAL_ROOT}/task_D_D" || -L "${CALVIN_ORIGINAL_ROOT}/task_D_D" ]]; then
    echo "[assets] original D path already exists: ${CALVIN_ORIGINAL_ROOT}/task_D_D"
  else
    ln -sfn "${SHARED_CALVIN_ROOT}/task_D_D" "${CALVIN_ORIGINAL_ROOT}/task_D_D"
  fi
elif [[ "${DOWNLOAD_ORIGINAL_D}" == "1" ]]; then
  zip_path="${CALVIN_ORIGINAL_ROOT}/task_D_D.zip"
  echo "[assets] downloading official CALVIN task_D_D zip -> ${zip_path}"
  wget -c -O "${zip_path}" http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
  echo "[assets] extracting official CALVIN task_D_D -> ${CALVIN_ORIGINAL_ROOT}"
  unzip -q -n "${zip_path}" -d "${CALVIN_ORIGINAL_ROOT}"
fi

echo "[assets] done"
