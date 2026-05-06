#!/usr/bin/env bash
# RoboChallenge Table30v2 — download all 30 tasks in the background, then
# convert each task to LeRobot v2.1 (the format consumed by starVLA's
# gr00t_lerobot loader). After each task is converted, the raw extracted
# directory is removed to free disk; the converted lerobot tree symlinks
# back to .mp4s, so we KEEP the raw videos until conversion succeeds.
#
# Usage:
#   nohup bash examples/RoboChallenge_table30v2/train_files/download_table30v2.sh \
#     > tmp/logs/download_table30v2.log 2>&1 &
#
# Re-run any time — already-downloaded / already-converted tasks are skipped.
set -uo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "starVLA" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate starVLA
fi

RAW_ROOT=${RAW_ROOT:-/project/vonneumann1/jye624/Datasets/RoboChallenge_table30v2/raw}
LEROBOT_ROOT=${LEROBOT_ROOT:-/project/vonneumann1/jye624/Datasets/RoboChallenge_table30v2/lerobot}
mkdir -p "${RAW_ROOT}" "${LEROBOT_ROOT}" tmp/logs

# Step 1: pull the HF parts and untar to ${RAW_ROOT}/<task>/. Removes
# ``.tar`` + ``.tar.part-*`` shards as soon as a task is fully extracted.
python examples/RoboChallenge_table30v2/train_files/download_table30v2.py \
  --raw-root "${RAW_ROOT}"

# Step 2: convert every extracted task that does NOT already have a lerobot dir.
# (Conversion is idempotent with --overwrite; skip if already converted.)
echo "[convert] scanning ${RAW_ROOT} for tasks to convert ..."
for task_dir in "${RAW_ROOT}"/*/; do
  task=$(basename "${task_dir}")
  # Skip stray dirs (e.g. *_extracted from manual untars during bring-up).
  if [[ ! -f "${task_dir}/meta/task_info.json" ]]; then
    continue
  fi
  if [[ -f "${LEROBOT_ROOT}/${task}/meta/info.json" ]]; then
    echo "[convert][skip] ${task} (already converted)"
    continue
  fi
  echo "[convert] ${task}"
  python examples/RoboChallenge_table30v2/train_files/convert_robochallenge_to_lerobot.py \
      --raw-root "${RAW_ROOT}" \
      --task "${task}" \
      --out-root "${LEROBOT_ROOT}" \
    || { echo "[convert][FAIL] ${task}"; continue; }
done

echo "[done] download + convert pipeline finished."
echo "Tasks converted:"
ls "${LEROBOT_ROOT}"
