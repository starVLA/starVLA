#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <ckpt_path> [port]" >&2
  exit 1
fi

CKPT_PATH="$1"
PORT="${2:-5694}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT_PATH" \
  --port "$PORT" \
  --use_bf16

