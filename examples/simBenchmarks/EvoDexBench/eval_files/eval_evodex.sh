#!/usr/bin/env bash
# Run the standard 100-episode evaluation on the 15 tasks of one embodiment against a running server.
# Run in the Evo-DexBench simulator environment with the benchmark assets activated.
#   CASE=qwenoft-single|...  CKPT_PATH=<path the server reports>  HOST=127.0.0.1  PORT=10101  OUT=results/<case>
# Re-running skips tasks with a complete log in OUT and moves incomplete logs aside.
set -euo pipefail

CASE="${CASE:-qwenoft-single}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10101}"
OUT="${OUT:-results/${CASE}}"
model="${CASE%-*}"
embodiment="${CASE#*-}"
prefix=$([[ "${embodiment}" == single ]] && echo m || echo b)
mkdir -p "${OUT}"

for index in $(seq -w 1 15); do
  task="${prefix}${index}"
  kwargs="{\"host\":\"${HOST}\",\"port\":${PORT},\"embodiment\":\"${embodiment}\""
  [[ -n "${CKPT_PATH:-}" ]] && kwargs+=",\"expected_checkpoint_path\":\"${CKPT_PATH}\""
  kwargs+="}"
  log="${OUT}/${task}.jsonl"
  # A finished log ends with the summary event; dexbench refuses to write to an existing log.
  if [[ -f "${log}" ]] && grep -q '"event": "summary"' "${log}"; then
    echo "skip ${task}: ${log} is complete"
    continue
  fi
  [[ -f "${log}" ]] && mv "${log}" "${log}.incomplete.$(date +%s)"
  dexbench evaluate --policy "starvla-${model}" --task-id "${task}_v1" --obs-mode rgb \
    --policy-kwargs "${kwargs}" --log "${log}"
done
