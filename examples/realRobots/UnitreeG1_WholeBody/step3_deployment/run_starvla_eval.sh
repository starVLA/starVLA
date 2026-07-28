#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
DEPLOY_DIR="$ROOT_DIR/examples/realRobots/UnitreeG1_WholeBody/step3_deployment"

export PYTHONPATH="$ROOT_DIR:$DEPLOY_DIR:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <ckpt_path> [server_port]" >&2
  exit 1
fi

CKPT_PATH="$1"
SERVER_PORT="${2:-5694}"

python "$DEPLOY_DIR/eval_files/local_self_test.py" \
  --ckpt-path "$CKPT_PATH" \
  --server-host 127.0.0.1 \
  --server-port "$SERVER_PORT"

