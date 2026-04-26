#!/bin/bash
set -euo pipefail

ROOT_DIR="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/"
DIR_GLOB="1013*"          # to match the directory prefix or wildcard: like 0822* or 0822_best*
FILE_GLOB="*.pt"          # to match the file wildcard: like '*.pt' or 'steps_*_pytorch_model.pt' or '*pytorch_model*.pt'

# dry-run: 1 = only list, 0 = really delete
DRY_RUN=0

echo "ROOT_DIR = $ROOT_DIR"
echo "DIR_GLOB = $DIR_GLOB"
echo "FILE_GLOB = $FILE_GLOB"
echo "DRY_RUN  = $DRY_RUN"
echo

for dir in "$ROOT_DIR"/$DIR_GLOB; do
  [ -d "$dir" ] || continue
  echo "Processing directory: $dir"
  # find all matching files, safely handle file names with spaces
  while IFS= read -r -d '' f; do
    if [ "$DRY_RUN" -eq 1 ]; then
      printf "WILL DELETE: %s\n" "$f"
    else
      printf "DELETING: %s\n" "$f"
      rm -v -- "$f"
    fi
  done < <(find "$dir" -type f -name "$FILE_GLOB" -print0)
done

echo "Done."