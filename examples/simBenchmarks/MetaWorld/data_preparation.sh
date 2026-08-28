#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export DEST=/path/to/datasets && bash examples/simBenchmarks/MetaWorld/data_preparation.sh
# or
#   bash examples/simBenchmarks/MetaWorld/data_preparation.sh /path/to/datasets
#
# After this script:
#   playground/Datasets/metaworld_starvla -> $DEST/metaworld_starvla
#   playground/Datasets/metaworld_starvla/metaworld_mt50_lerobot/meta/modality.json
#   playground/Datasets/metaworld_starvla/metaworld_mt50_lerobot/meta/stats_gr00t.json

DEST="${DEST:-${1:-}}"
if [[ -z "${DEST}" ]]; then
  echo "ERROR: DEST is not set."
  echo "  export DEST=/path/to/datasets && bash examples/simBenchmarks/MetaWorld/data_preparation.sh"
  echo "  or: bash examples/simBenchmarks/MetaWorld/data_preparation.sh /path/to/datasets"
  exit 1
fi

STARVLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${DEST}/metaworld_starvla"
DATASET_DIR="${DATA_ROOT}/metaworld_mt50_lerobot"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${DATA_ROOT}"

"${PYTHON_BIN}" -m pip install -U "huggingface-hub==0.35.3"
hf download HuggingFaceVLA/metaworld_mt50 \
  --repo-type dataset \
  --local-dir "${DATASET_DIR}"

mkdir -p "${STARVLA_DIR}/playground/Datasets"
ln -sfn "${DATA_ROOT}" "${STARVLA_DIR}/playground/Datasets/metaworld_starvla"

"${PYTHON_BIN}" "${STARVLA_DIR}/examples/simBenchmarks/MetaWorld/train_files/prepare_metaworld_metadata.py" \
  "${DATASET_DIR}"

echo ""
echo "Done. Dataset layout:"
echo "  playground/Datasets/metaworld_starvla -> ${DATA_ROOT}"
echo ""
echo "Available data_mix value for training:"
echo "  metaworld_mt50"
