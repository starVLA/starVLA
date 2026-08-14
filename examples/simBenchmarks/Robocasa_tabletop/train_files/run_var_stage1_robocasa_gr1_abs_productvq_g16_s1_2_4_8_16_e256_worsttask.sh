#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

if [[ "${ALLOW_SHARED_GPU:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  BUSY_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | sed '/^$/d' || true)"
  if [[ -n "${BUSY_PIDS}" ]]; then
    echo "CUDA device is busy with existing compute process(es): ${BUSY_PIDS}. Set ALLOW_SHARED_GPU=1 to run anyway." >&2
    exit 1
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
CONFIG_YAML="${CONFIG_YAML:-examples/simBenchmarks/Robocasa_tabletop/train_files/train_var_stage1_robocasa_gr1_abs_productvq_g16_s1_2_4_8_16_e256_worsttask.yaml}"
INIT_CKPT="/root/feihong/starVLA/Checkpoints/var_stage1_robocasa_gr1_abs_pure_ae_e256_worsttask/best_worst_task_mae.ckpt"
if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "Missing abs e256 worst-task pure AE init checkpoint: ${INIT_CKPT}" >&2
  exit 1
fi
OUTPUT_DIR="$(${PYTHON_BIN} - <<PY2
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CONFIG_YAML}')
print(cfg.experiment.output_dir)
PY2
)"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" starVLA/training/train_var_stage1.py   --config_yaml "${CONFIG_YAML}"   2>&1 | tee -a "${OUTPUT_DIR}/train.log"
