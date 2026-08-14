#!/usr/bin/env bash
set -euo pipefail

CONFIG_YAML="${CONFIG_YAML:-examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_libero.yaml}"

python starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}"
