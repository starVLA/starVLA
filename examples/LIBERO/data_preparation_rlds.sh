#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export DEST=/path/to/dir && bash examples/LIBERO/data_preparation_rlds.sh
# or
#   bash examples/LIBERO/data_preparation_rlds.sh /path/to/dir

DEST="${DEST:-${1:-}}"
if [[ -z "${DEST}" ]]; then
  echo "ERROR: DEST is not set."
  echo "  export DEST=/path/to/dir && bash examples/LIBERO/data_preparation_rlds.sh"
  echo "  or: bash examples/LIBERO/data_preparation_rlds.sh /path/to/dir"
  exit 1
fi

CUR="$(pwd)"
TARGET="${DEST}/modified_libero_rlds"

if [[ ! -d "${TARGET}" ]]; then
  git lfs install
  git clone https://huggingface.co/datasets/openvla/modified_libero_rlds "${TARGET}"
else
  echo "Found existing dataset at ${TARGET}, skip clone."
fi

mkdir -p "${CUR}/playground/Datasets"
ln -sfn "${TARGET}" "${CUR}/playground/Datasets/MODIFIED_LIBERO_RLDS"

echo "Done. RLDS dataset linked at: ${CUR}/playground/Datasets/MODIFIED_LIBERO_RLDS"
