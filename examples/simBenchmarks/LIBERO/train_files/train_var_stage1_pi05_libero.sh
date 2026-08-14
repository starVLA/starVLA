#!/usr/bin/env bash
set -euo pipefail

CONFIG_YAML="${CONFIG_YAML:-examples/simBenchmarks/LIBERO/train_files/train_var_stage1_pi05_libero.yaml}"

python starVLA/training/train_var_stage1.py \
  --config_yaml "${CONFIG_YAML}"

