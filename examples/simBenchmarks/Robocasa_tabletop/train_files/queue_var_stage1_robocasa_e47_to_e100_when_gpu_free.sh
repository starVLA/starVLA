#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../.."

OUTPUT_DIR="/root/feihong/starVLA/Checkpoints/var_stage1_robocasa_gr1_abs_productvq_g16_s124816_e256_closebalanced_resume_e47_to_e100"
RUN_SCRIPT="examples/simBenchmarks/Robocasa_tabletop/train_files/run_var_stage1_robocasa_gr1_abs_productvq_g16_s1_2_4_8_16_e256_closebalanced_resume_e47_to_e100.sh"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "${OUTPUT_DIR}"
exec 8>"${OUTPUT_DIR}/queue.lock"
if ! flock -n 8; then
  echo "A GPU-wait queue already owns ${OUTPUT_DIR}/queue.lock." >&2
  exit 1
fi

exec > >(tee -a "${OUTPUT_DIR}/queue.log") 2>&1
echo "[$(date --iso-8601=seconds)] Queued guarded RoboCasa Stage-1 resume; waiting for exclusive GPU access."

poll_count=0
while true; do
  if [[ -f "${OUTPUT_DIR}/STOPPED_EARLY.json" ]]; then
    echo "Run was already stopped by its MAE guardrail; refusing to relaunch."
    exit 2
  fi

  busy_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sed '/^$/d' || true)"
  if [[ -z "${busy_pids}" ]]; then
    echo "[$(date --iso-8601=seconds)] GPU is free; starting guarded RoboCasa Stage-1 resume."
    exec bash "${RUN_SCRIPT}"
  fi

  if (( poll_count % 20 == 0 )); then
    echo "[$(date --iso-8601=seconds)] GPU still busy with PID(s): ${busy_pids}; waiting."
  fi
  poll_count=$((poll_count + 1))
  sleep "${POLL_SECONDS}"
done
