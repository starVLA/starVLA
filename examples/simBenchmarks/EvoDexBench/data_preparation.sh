#!/usr/bin/env bash
# Download the Evo-DexBench LeRobot v3 data and arrange it for the 15-task recipes:
#   ${DATA_ROOT}/single/m01 ... m15 and ${DATA_ROOT}/dual/b01 ... b15, each with meta/modality.json.
# Run from the StarVLA repository root. Set TASKS (e.g. "m01 m02") to prepare a subset.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-playground/Datasets/EvoDexBench}"
RAW="${DATA_ROOT}/raw"
TASKS="${TASKS:-}"

patterns=("lerobot/*")
if [[ -n "${TASKS}" ]]; then
  patterns=()
  for task in ${TASKS}; do patterns+=("lerobot/*/${task}/*"); done
fi
# snapshot_download keeps every pattern on all huggingface_hub versions (CLI --include parsing differs between them).
python - "${RAW}" "${patterns[@]}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download("MINT-SJTU/Evo-DexBench-dataset", repo_type="dataset", local_dir=sys.argv[1], allow_patterns=sys.argv[2:])
PY

mkdir -p "${DATA_ROOT}/single" "${DATA_ROOT}/dual"
for task_dir in "${RAW}"/lerobot/*/*/; do
  task="$(basename "${task_dir}")"
  case "${task}" in
    m[0-9][0-9]) embodiment=single ;;
    b[0-9][0-9]) embodiment=dual ;;
    *) continue ;;
  esac
  out="${DATA_ROOT}/${embodiment}/${task}"
  rm -rf "${out}"
  mkdir -p "${out}"
  cp -r "${task_dir}meta" "${out}/meta"
  cp -al "${task_dir}videos" "${out}/videos"
  # The sidecar maps the released 111D/222D state to the 31D/62D policy state and names the three cameras.
  cp "${HERE}/train_files/modality_${embodiment}.json" "${out}/meta/modality.json"
  # Copy the data without the depth columns: the recipes do not use depth, and the Array2D depth features
  # trip a pandas extension-dtype bug in datasets 3.x when trajectories are loaded. Other columns are unchanged.
  python - "${task_dir}" "${out}" <<'PY'
import json, os, sys
from pathlib import Path
import pyarrow.parquet as pq

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for f in sorted(src.glob("data/*/*.parquet")):
    table = pq.read_table(f)
    depth = [c for c in table.column_names if c.startswith("observation.depth.")]
    table = table.select([c for c in table.column_names if c not in depth])
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        hf = json.loads(metadata[b"huggingface"])
        for c in depth:
            hf.get("info", {}).get("features", {}).pop(c, None)
        metadata[b"huggingface"] = json.dumps(hf).encode()
    target = dst / f.relative_to(src)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(metadata), target)
info_path = dst / "meta/info.json"
info = json.loads(info_path.read_text())
info["features"] = {k: v for k, v in info["features"].items() if not k.startswith("observation.depth.")}
info_path.write_text(json.dumps(info, indent=4))
PY
done
echo "Prepared: $(ls "${DATA_ROOT}/single" | tr '\n' ' ')$(ls "${DATA_ROOT}/dual" | tr '\n' ' ')"
