#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangfeihong/starVLA

export CUDA_VISIBLE_DEVICES=2
export NO_ALBUMENTATIONS_UPDATE=1
export NUMBA_CACHE_DIR=/tmp/numba_cache
export MPLCONFIGDIR=/tmp/mplconfig

CONFIG_YAML="${CONFIG_YAML:-examples/simBenchmarks/LIBERO/train_files/train_var_stage1_pi05_libero_posw1024.yaml}"
PYTHON="${PYTHON:-/home/zhangfeihong/miniconda3/envs/starVLA/bin/python}"

"${PYTHON}" starVLA/training/train_var_stage1.py \
  --config_yaml "${CONFIG_YAML}"
