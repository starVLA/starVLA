#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

INIT_CKPT="playground/Checkpoints/var_stage1_metaworld_mt50_pure_ae_e32/best_recon.ckpt"
if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "Missing ${INIT_CKPT}; run examples/simBenchmarks/MetaWorld/train_files/run_var_stage1_metaworld_mt50_pure_ae_e32.sh first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" starVLA/training/train_var_stage1.py \
  --config_yaml examples/simBenchmarks/MetaWorld/train_files/train_var_stage1_metaworld_mt50_e32_aeinit_productvq_g16_s1_2_4_8.yaml
