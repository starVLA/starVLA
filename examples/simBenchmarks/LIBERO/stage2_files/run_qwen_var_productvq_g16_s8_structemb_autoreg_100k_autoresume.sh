#!/usr/bin/env bash
set -uo pipefail

cd /home/zhangfeihong/starVLA

RUN_DIR="playground/Checkpoints/qwen_var_productvq_g16_s8_structemb_autoreg_100k_fullcache"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
SUPERVISOR_LOG="${RUN_DIR}/autoresume_supervisor.log"
SLEEP_SECONDS="${AUTO_RESUME_SLEEP_SECONDS:-60}"
MAX_RESTARTS="${AUTO_RESUME_MAX_RESTARTS:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29545}"
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "${RUN_DIR}"

latest_checkpoint() {
  find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' -printf '%f\n' 2>/dev/null \
    | sed -E 's/^steps_([0-9]+)_pytorch_model\.pt$/\1 &/' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  latest="$(latest_checkpoint)"
  {
    echo "[$(date '+%F %T')] autoreg autoresume attempt ${attempt}"
    echo "[$(date '+%F %T')] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"
    if [[ -n "${latest}" ]]; then
      echo "[$(date '+%F %T')] latest checkpoint: ${CHECKPOINT_DIR}/${latest}"
    else
      echo "[$(date '+%F %T')] no checkpoint found; training code will start from scratch"
    fi
  } | tee -a "${SUPERVISOR_LOG}"

  examples/simBenchmarks/LIBERO/stage2_files/run_qwen_var_productvq_g16_s8_structemb_autoreg_100k.sh
  exit_code=$?

  if [[ "${exit_code}" -eq 0 ]]; then
    echo "[$(date '+%F %T')] training exited cleanly; supervisor stopping" | tee -a "${SUPERVISOR_LOG}"
    exit 0
  fi

  echo "[$(date '+%F %T')] training failed with exit code ${exit_code}; will resume after ${SLEEP_SECONDS}s" | tee -a "${SUPERVISOR_LOG}"

  if [[ "${MAX_RESTARTS}" -gt 0 && "${attempt}" -ge "${MAX_RESTARTS}" ]]; then
    echo "[$(date '+%F %T')] reached AUTO_RESUME_MAX_RESTARTS=${MAX_RESTARTS}; supervisor stopping" | tee -a "${SUPERVISOR_LOG}"
    exit "${exit_code}"
  fi

  sleep "${SLEEP_SECONDS}"
done
