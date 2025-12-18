#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export DEST=/path/to/dir && bash examples/LIBERO/data_preparation.sh
# or
#   bash examples/LIBERO/data_preparation.sh /path/to/dir

DEST="${DEST:-${1:-}}"
if [[ -z "${DEST}" ]]; then
  echo "ERROR: DEST is not set."
  echo "  export DEST=/path/to/dir && bash examples/LIBERO/data_preparation.sh"
  echo "  or: bash examples/LIBERO/data_preparation.sh /path/to/dir"
  exit 1
fi

CUR="$(pwd)"
mkdir -p "$DEST"

python -m pip install -U "huggingface-hub==0.35.3"

for repo in \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
do
  hf download "$repo" --repo-type dataset --local-dir "$DEST/libero/${repo##*/}"
done

hf download "StarVLA/LLaVA-OneVision-COCO" --repo-type dataset --local-dir "$DEST/LLaVA-OneVision-COCO"
unzip -- "$DEST/LLaVA-OneVision-COCO/sharegpt4v_coco.zip" -d "$DEST/LLaVA-OneVision-COCO/"

mkdir -p "$CUR/playground/Datasets"
ln -s "$DEST/libero" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA"
ln -s "$DEST/LLaVA-OneVision-COCO" "$CUR/playground/Datasets/LLaVA-OneVision-COCO"

## move modality
cp "$CUR/examples/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/meta"
