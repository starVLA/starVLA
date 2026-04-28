#!/usr/bin/env bash
# RoboCasa365 — download all `target/human` LeRobot bundles (50 tasks: 18
# atomic + 32 composite). Run from the repo root inside the `robocasa365`
# conda env.
#
# Output goes to ${DATASET_BASE_PATH} configured in
#   playground/Code/robocasa365/robocasa/macros_private.py
# i.e. ./playground/Datasets/robocasa365/v1.0/target/{atomic,composite}/...
#
# The upstream script asks for `y/N` confirmation before each task; we pipe
# `yes` so it works under nohup.
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "robocasa365" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate robocasa365
fi

mkdir -p tmp/logs
LOG=${LOG:-tmp/logs/download_robocasa365_target_human.log}

echo "[robocasa365] downloading target/human LeRobot bundles -> $LOG"
yes | python -m robocasa.scripts.download_datasets \
  --split target \
  --source human \
  2>&1 | tee "${LOG}"

echo "[robocasa365] done. listing downloaded tasks:"
ls playground/Datasets/robocasa365/v1.0/target/atomic    2>/dev/null || true
ls playground/Datasets/robocasa365/v1.0/target/composite 2>/dev/null || true
