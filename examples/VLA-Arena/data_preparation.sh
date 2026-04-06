#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export DEST=/path/to/dir && bash examples/VLA-Arena/data_preparation.sh
# or
#   bash examples/VLA-Arena/data_preparation.sh /path/to/dir
#
# Downloads the three VLA-Arena L0 datasets (Small / Medium / Large) from
# HuggingFace in LeRobot (openpi) format and wires them up for StarVLA training.
#
# HuggingFace repos:
#   VLA-Arena/VLA_Arena_L0_S_lerobot_openpi
#   VLA-Arena/VLA_Arena_L0_M_lerobot_openpi
#   VLA-Arena/VLA_Arena_L0_L_lerobot_openpi
#
# After this script:
#   playground/Datasets/VLA_ARENA_LEROBOT_DATA/ -> $DEST/vla_arena/
#
# NOTE: The modality.json maps dataset keys to StarVLA keys.
#   If the primary camera key in your dataset differs from
#   "observation.images.agentview_rgb", update train_files/modality.json
#   (video.primary_image.original_key) before training.

DEST="${DEST:-${1:-}}"
if [[ -z "${DEST}" ]]; then
  echo "ERROR: DEST is not set."
  echo "  export DEST=/path/to/dir && bash examples/VLA-Arena/data_preparation.sh"
  echo "  or: bash examples/VLA-Arena/data_preparation.sh /path/to/dir"
  exit 1
fi

CUR="$(pwd)"
mkdir -p "$DEST/vla_arena"

python -m pip install -U "huggingface-hub==0.35.3"

for repo in \
  VLA-Arena/VLA_Arena_L0_L_lerobot_openpi \
  # VLA-Arena/VLA_Arena_L0_M_lerobot_openpi \
  # VLA-Arena/VLA_Arena_L0_S_lerobot_openpi
do
  hf download "$repo" --repo-type dataset --local-dir "$DEST/vla_arena/${repo##*/}"
done

mkdir -p "$CUR/playground/Datasets"
ln -sfn "$DEST/vla_arena"       "$CUR/playground/Datasets/VLA_ARENA_LEROBOT_DATA"

## copy modality.json into each dataset's meta/ directory
for dataset in \
  VLA_Arena_L0_L_lerobot_openpi \
  # VLA_Arena_L0_M_lerobot_openpi \
  # VLA_Arena_L0_S_lerobot_openpi
do
  cp "$CUR/examples/VLA-Arena/train_files/modality.json" \
     "$DEST/vla_arena/${dataset}/meta/modality.json"
done

echo ""
echo "Done. Dataset layout:"
echo "  playground/Datasets/VLA_ARENA_LEROBOT_DATA -> $DEST/vla_arena"
echo ""
echo "Available data_mix values for training:"
echo "  vla_arena_L0_L   - large split"
