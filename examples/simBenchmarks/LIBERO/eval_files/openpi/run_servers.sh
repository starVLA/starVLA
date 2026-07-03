#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL="${MODEL:-PI0}"
GPU_LIST="${GPU_LIST:-0 1}"
PORT_BASE="${PORT_BASE:-18000}"
RESULT_DIR="${RESULT_DIR:-outputs/openpi_libero_eval}"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}/servers}"
MODEL_SOURCE="${MODEL_SOURCE:-openpi}"
CHECKPOINT="${CHECKPOINT:-}"
PRECISION="${PRECISION:-float32}"
DEVICE="${DEVICE:-cuda}"
SCRIPT="examples/simBenchmarks/LIBERO/eval_files/openpi/server_starvla_openpi.py"

mkdir -p "${LOG_DIR}"
read -r -a GPUS <<< "${GPU_LIST}"
pids=()
server_args=(
  --model "${MODEL}"
  --model-source "${MODEL_SOURCE}"
  --precision "${PRECISION}"
  --device "${DEVICE}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  server_args+=(--checkpoint "${CHECKPOINT}")
fi

cleanup() {
  status=$?
  if [[ "${status}" != 0 ]]; then
    echo "stopping servers ..."
    for pid in "${pids[@]:-}"; do
      kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for i in "${!GPUS[@]}"; do
  port=$((PORT_BASE + i))
  echo "server ${i}: GPU=${GPUS[$i]} port=${port}"
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python "${SCRIPT}" \
    --port "${port}" \
    "${server_args[@]}" \
    "$@" \
    > >(sed -u "s/^/[server ${i}:${port}] /" | tee "${LOG_DIR}/${MODEL}_${MODEL_SOURCE}_${port}.log") 2>&1 &
  pids+=("$!")
done

wait
