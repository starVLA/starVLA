#!/usr/bin/env bash
set -euo pipefail

# Sequential n300 CALVIN ABC->D evaluation for WMH candidate checkpoints.
# Keep long checkpoint/report paths here so users can launch this through ./wmh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

MEMBER="${MEMBER:-WMH}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
RUN_DIR="${RUN_DIR:-${SHARED_ROOT}/members/${MEMBER}/runs}"
LOG_DIR="${LOG_DIR:-${SHARED_ROOT}/members/${MEMBER}/logs}"
REPORT_DIR="${REPORT_DIR:-${SHARED_ROOT}/members/${MEMBER}/reports}"

TOTAL_SEQUENCES="${TOTAL_SEQUENCES:-300}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
BASE_PORT="${BASE_PORT:-7400}"
RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT:-14400}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
CALVIN_SEND_STATE="${CALVIN_SEND_STATE:-1}"
DEBUG="${DEBUG:-0}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
CANDIDATES="${CANDIDATES:-base8k aug_hardv2 mirror_hardv2 lora2000}"

REPORT_ROOT="${REPORT_ROOT:-${REPORT_DIR}/eval_compare_d_n${TOTAL_SEQUENCES}_${TS}}"
LOG_ROOT="${LOG_ROOT:-${LOG_DIR}/eval_compare_d_n${TOTAL_SEQUENCES}_${TS}}"
mkdir -p "${REPORT_ROOT}" "${LOG_ROOT}"

latest_ckpt() {
  local run_pattern="$1"
  find "${RUN_DIR}" -maxdepth 1 -type d -name "${run_pattern}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2- \
    | while read -r run; do
        [[ -n "${run}" ]] || exit 0
        find "${run}/checkpoints" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' -printf '%T@ %p\n' 2>/dev/null \
          | sort -nr \
          | head -1 \
          | cut -d' ' -f2-
      done
}

candidate_ckpt() {
  local name="$1"
  case "${name}" in
    base8k)
      printf '%s\n' "${RUN_DIR}/abc_state8_connector_8h200_bs96_8k_0519_083200/checkpoints/steps_8000_pytorch_model.pt"
      ;;
    aug_hardv2)
      latest_ckpt 'abc_aug_hardv2_*'
      ;;
    mirror_hardv2)
      latest_ckpt 'abc_mirror_hardv2_*'
      ;;
    lora2000)
      latest_ckpt 'abc_lora_explore_ft2000_*'
      ;;
    moe15000)
      printf '%s\n' "${RUN_DIR}/abc_augmented_moe_adaptive_WMH_bs64_s15k_0519_121249/checkpoints/steps_15000_pytorch_model.pt"
      ;;
    *)
      if [[ -f "${name}" ]]; then
        printf '%s\n' "${name}"
      else
        echo "Unknown candidate '${name}'. Use one of: base8k aug_hardv2 mirror_hardv2 lora2000 moe15000, or pass a checkpoint path." >&2
        return 2
      fi
      ;;
  esac
}

echo "[eval-candidates] total_sequences=${TOTAL_SEQUENCES}"
echo "[eval-candidates] candidates=${CANDIDATES}"
echo "[eval-candidates] gpu_ids=${GPU_IDS}"
echo "[eval-candidates] workers_per_gpu=${WORKERS_PER_GPU}"
echo "[eval-candidates] report_root=${REPORT_ROOT}"
echo "[eval-candidates] log_root=${LOG_ROOT}"

for candidate in ${CANDIDATES}; do
  ckpt="$(candidate_ckpt "${candidate}")"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "[eval-candidates] missing checkpoint for ${candidate}: ${ckpt}" >&2
    exit 3
  fi

  eval_dir="${REPORT_ROOT}/${candidate}"
  log_file="${LOG_ROOT}/${candidate}.log"
  mkdir -p "${eval_dir}"

  echo "[eval-candidates] start ${candidate}"
  echo "[eval-candidates] ckpt=${ckpt}" | tee "${eval_dir}/checkpoint.txt"
  {
    echo "[eval-candidates] candidate=${candidate}"
    echo "[eval-candidates] ckpt=${ckpt}"
    echo "[eval-candidates] eval_dir=${eval_dir}"
    TOTAL_SEQUENCES="${TOTAL_SEQUENCES}" \
    GPU_IDS="${GPU_IDS}" \
    WORKERS_PER_GPU="${WORKERS_PER_GPU}" \
    BASE_PORT="${BASE_PORT}" \
    RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT}" \
    SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT}" \
    CALVIN_SEND_STATE="${CALVIN_SEND_STATE}" \
    DEBUG="${DEBUG}" \
    DRY_RUN="${DRY_RUN}" \
    CKPT="${ckpt}" \
    EVAL_LOG_DIR="${eval_dir}" \
      bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh
  } > "${log_file}" 2>&1

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[eval-candidates] dry-run done ${candidate}: ${eval_dir}"
    continue
  fi

  if [[ ! -f "${eval_dir}/metrics.json" ]]; then
    echo "[eval-candidates] ${candidate} finished without metrics.json; see ${log_file}" >&2
    exit 4
  fi

  echo "[eval-candidates] done ${candidate}: ${eval_dir}/metrics.json"
  PYTHONDONTWRITEBYTECODE=1 python examples/calvin_autoresearch/scripts/summarize_eval_metrics.py "${eval_dir}/metrics.json" \
    > "${eval_dir}/summary.txt" 2>&1 || true
  cat "${eval_dir}/summary.txt" || true
done

echo "[eval-candidates] all done: ${REPORT_ROOT}"
