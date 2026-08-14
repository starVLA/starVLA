#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_VENV="${UMI_TOOLS_VENV:-${UMI_DATA_ROOT:-$SCRIPT_DIR}/.umi-tools-venv}"

if command -v hf >/dev/null 2>&1 || python3 -c 'import huggingface_hub, requests' >/dev/null 2>&1; then
  exec python3 "$SCRIPT_DIR/umi_pipeline.py" all "$@"
fi

if [[ ! -x "$TOOLS_VENV/bin/python" ]]; then
  echo "Bootstrapping UMI downloader environment at $TOOLS_VENV" >&2
  python3 -m venv "$TOOLS_VENV"
fi
if ! "$TOOLS_VENV/bin/python" -c 'import huggingface_hub, requests' >/dev/null 2>&1; then
  "$TOOLS_VENV/bin/python" -m pip install --disable-pip-version-check \
    -r "$SCRIPT_DIR/requirements-download.txt"
fi
exec "$TOOLS_VENV/bin/python" "$SCRIPT_DIR/umi_pipeline.py" all "$@"
