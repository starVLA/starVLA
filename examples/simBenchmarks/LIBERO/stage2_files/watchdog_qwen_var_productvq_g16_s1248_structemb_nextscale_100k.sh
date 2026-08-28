#!/usr/bin/env bash
set -u

cd /home/zhangfeihong/starVLA

RUN_DIR="playground/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_100k_fullcache"
LOG_FILE="${RUN_DIR}/watchdog.log"
mkdir -p "${RUN_DIR}"

while true; do
  echo "[$(date --iso-8601=seconds)] starting nextscale stage2 training" >> "${LOG_FILE}"

  CUDA_VISIBLE_DEVICES=2,3,4,5 \
  NUM_PROCESSES=4 \
  MAIN_PROCESS_PORT=29624 \
  WANDB_MODE=online \
  PYTHONPATH=/home/zhangfeihong/starVLA \
  TOKENIZERS_PARALLELISM=false \
  examples/simBenchmarks/LIBERO/stage2_files/run_qwen_var_productvq_g16_s1248_structemb_nextscale_100k.sh

  code=$?
  echo "[$(date --iso-8601=seconds)] training exited with code ${code}" >> "${LOG_FILE}"

  if [ "${code}" -eq 0 ]; then
    echo "[$(date --iso-8601=seconds)] training finished successfully; watchdog stops" >> "${LOG_FILE}"
    break
  fi

  echo "[$(date --iso-8601=seconds)] restarting after 60 seconds" >> "${LOG_FILE}"
  sleep 60
done
