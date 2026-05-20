#!/usr/bin/env bash
set -euo pipefail

# Evaluate WMH's best public checkpoints from public/seven.
# Intended for teammates: set MEMBER to your own initials so logs/reports land
# under public/seven/starvla_calvin/members/${MEMBER}/.

PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
RUNTIME_ENV="${RUNTIME_ENV:-${PUBLIC_ROOT}/shared/runtime/starvla_env.sh}"
CODE_ROOT="${CODE_ROOT:-${PUBLIC_ROOT}/members/WMH/code/starVLA_ft_aug}"

MEMBER="${MEMBER:?Set MEMBER to your initials, e.g. MEMBER=GTY}"
TOTAL_SEQUENCES="${TOTAL_SEQUENCES:-300}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
BASE_PORT="${BASE_PORT:-7400}"
CANDIDATES="${CANDIDATES:-aug_hardv2 mirror_hardv2 lora2000 base8k}"
CALVIN_SEND_STATE="${CALVIN_SEND_STATE:-1}"
CALVIN_STATE_MODE="${CALVIN_STATE_MODE:-normal}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT:-14400}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date +%m%d_%H%M%S)}"

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

if [[ ! -f "${RUNTIME_ENV}" ]]; then
  echo "Missing runtime env: ${RUNTIME_ENV}" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${RUNTIME_ENV}"

export PATH="${STARVLA_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
export STARVLA_ROOT="${CODE_ROOT}"
export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"

if [[ ! -d "${STARVLA_ROOT}" ]]; then
  echo "Missing public WMH code root: ${STARVLA_ROOT}" >&2
  exit 3
fi

cd "${STARVLA_ROOT}"

RUN_ROOT="${PUBLIC_ROOT}/members/WMH/runs"
OUT_ROOT="${OUT_ROOT:-${PUBLIC_ROOT}/members/${MEMBER}/reports/eval_wmh_best_d_n${TOTAL_SEQUENCES}_${TS}}"
LOG_ROOT="${LOG_ROOT:-${PUBLIC_ROOT}/members/${MEMBER}/logs/eval_wmh_best_d_n${TOTAL_SEQUENCES}_${TS}}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

candidate_ckpt() {
  local name="$1"
  case "${name}" in
    aug_hardv2)
      printf '%s\n' "${RUN_ROOT}/abc_aug_hardv2_8000_0519_171848/checkpoints/steps_8000_pytorch_model.pt"
      ;;
    mirror_hardv2)
      printf '%s\n' "${RUN_ROOT}/abc_mirror_hardv2_8000_0519_171848/checkpoints/steps_8000_pytorch_model.pt"
      ;;
    lora2000)
      printf '%s\n' "${RUN_ROOT}/abc_lora_explore_ft2000_0519_210816/checkpoints/steps_2000_pytorch_model.pt"
      ;;
    base8k)
      printf '%s\n' "${RUN_ROOT}/abc_state8_connector_8h200_bs96_8k_0519_083200/checkpoints/steps_8000_pytorch_model.pt"
      ;;
    *)
      if [[ -f "${name}" ]]; then
        printf '%s\n' "${name}"
      else
        echo "Unknown candidate '${name}'" >&2
        return 2
      fi
      ;;
  esac
}

echo "[public-eval] member=${MEMBER}"
echo "[public-eval] candidates=${CANDIDATES}"
echo "[public-eval] total_sequences=${TOTAL_SEQUENCES}"
echo "[public-eval] gpu_ids=${GPU_IDS}"
echo "[public-eval] workers_per_gpu=${WORKERS_PER_GPU}"
echo "[public-eval] out_root=${OUT_ROOT}"
echo "[public-eval] log_root=${LOG_ROOT}"

offset=0
for candidate in ${CANDIDATES}; do
  ckpt="$(candidate_ckpt "${candidate}")"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "[public-eval] missing checkpoint for ${candidate}: ${ckpt}" >&2
    exit 4
  fi

  eval_dir="${OUT_ROOT}/${candidate}"
  log_file="${LOG_ROOT}/${candidate}.log"
  mkdir -p "${eval_dir}"
  printf '%s\n' "${ckpt}" > "${eval_dir}/checkpoint.txt"

  candidate_port=$((BASE_PORT + offset))
  offset=$((offset + 100))

  echo "[public-eval] start ${candidate}"
  echo "[public-eval] ckpt=${ckpt}"
  {
    TOTAL_SEQUENCES="${TOTAL_SEQUENCES}" \
    GPU_IDS="${GPU_IDS}" \
    WORKERS_PER_GPU="${WORKERS_PER_GPU}" \
    BASE_PORT="${candidate_port}" \
    SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT}" \
    RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT}" \
    CALVIN_SEND_STATE="${CALVIN_SEND_STATE}" \
    CALVIN_STATE_MODE="${CALVIN_STATE_MODE}" \
    DRY_RUN="${DRY_RUN}" \
    CKPT="${ckpt}" \
    EVAL_LOG_DIR="${eval_dir}" \
      bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh
  } > "${log_file}" 2>&1

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[public-eval] dry-run done ${candidate}: ${eval_dir}"
    continue
  fi

  if [[ -f "${eval_dir}/metrics.json" ]]; then
    "${STARVLA_PYTHON}" examples/calvin_autoresearch/scripts/summarize_eval_metrics.py "${eval_dir}/metrics.json" \
      > "${eval_dir}/summary.txt" 2>&1 || true
    cat "${eval_dir}/summary.txt" || true
  else
    echo "[public-eval] ${candidate} finished without metrics.json; inspect ${log_file}" >&2
    exit 5
  fi
done

echo "[public-eval] all done: ${OUT_ROOT}"
