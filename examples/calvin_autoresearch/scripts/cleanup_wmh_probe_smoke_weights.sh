#!/usr/bin/env bash
set -euo pipefail

# Remove only large model weight files from WMH probe/smoke runs while keeping
# small metadata files such as config.yaml, config.full.yaml, summary.jsonl,
# dataset_statistics.json, and logs for provenance.
#
# Usage:
#   bash examples/calvin_autoresearch/scripts/cleanup_wmh_probe_smoke_weights.sh --dry-run
#   bash examples/calvin_autoresearch/scripts/cleanup_wmh_probe_smoke_weights.sh --yes

MODE="${1:---dry-run}"
case "${MODE}" in
  --dry-run) DO_DELETE=0 ;;
  --yes) DO_DELETE=1 ;;
  *)
    echo "Usage: $0 [--dry-run|--yes]" >&2
    exit 2
    ;;
esac

SHARED_ROOT="${SEVEN_STARVLA_CALVIN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
RUN_ROOT="${SHARED_ROOT}/members/WMH/runs"

RUNS=(
  "abc_probe_bs16_w8_h200_0518_161001"
  "abc_state8_smoke20_0519_075852"
  "abc_state8_connector_smoke20_0519_081222"
  "abc_state8_connector_probe200_8h200_0519_081756"
  "abc_state8_connector_probe200_8h200_bs64_0519_082244"
  "abc_state8_connector_balanced_aug_probe200_4gpu_0519_090319"
  "abc_state8_connector_balanced_probe200_4gpu_0519_091520"
  "abc_state8_connector_balanced_lang_taskaug_probe200_0519_104717"
  "abc_state8_connector_balanced_lang_taskaug_lrmirror_probe200_0519_122858"
)

echo "[cleanup] mode=${MODE}"
echo "[cleanup] run_root=${RUN_ROOT}"
echo "[cleanup] target: only *.pt/*.bin/*.safetensors model weight files"

total_bytes=0
found_files=0

for run_id in "${RUNS[@]}"; do
  run_dir="${RUN_ROOT}/${run_id}"
  if [[ ! -d "${run_dir}" ]]; then
    echo "[cleanup] skip missing: ${run_dir}"
    continue
  fi

  echo
  echo "[cleanup] scan: ${run_id}"
  mapfile -d '' files < <(
    find "${run_dir}" -type f \
      \( -name '*pytorch_model*.pt' -o -name '*.safetensors' -o -name '*.bin' \) \
      -print0
  )

  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "[cleanup] no weight files found"
    continue
  fi

  for f in "${files[@]}"; do
    size=$(stat -c '%s' "${f}")
    total_bytes=$((total_bytes + size))
    found_files=$((found_files + 1))
    printf '[cleanup] %10s  %s\n' "$(numfmt --to=iec --suffix=B "${size}")" "${f}"
    if [[ "${DO_DELETE}" == "1" ]]; then
      rm -f -- "${f}"
    fi
  done
done

echo
printf '[cleanup] matched files: %d\n' "${found_files}"
printf '[cleanup] matched size:  %s\n' "$(numfmt --to=iec --suffix=B "${total_bytes}")"

if [[ "${DO_DELETE}" == "1" ]]; then
  echo "[cleanup] deletion complete"
else
  echo "[cleanup] dry-run only; rerun with --yes to delete"
fi

echo
df -h /inspire/qb-ilm2/project/26summer-camp-10/public 2>/dev/null || true
