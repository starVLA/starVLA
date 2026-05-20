#!/usr/bin/env bash
set -euo pipefail

# Policy server launcher for WMH QwenGR00T_MoE_LoRA checkpoints.
# It registers GTY MoE modules plus the local MoE+LoRA framework before
# delegating to StarVLA's deployment server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"
GTY_ROOT="${GTY_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/GTY}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

: "${CKPT:?Set CKPT to a QwenGR00T_MoE_LoRA checkpoint}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 3
fi

GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-5694}"
PYTHON_BIN="${STARVLA_PYTHON:-python}"

export PYTHONPATH="${STARVLA_ROOT}/examples/calvin_autoresearch/train_files:${GTY_ROOT}/train_files:${STARVLA_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export GTY_ROOT

exec "${PYTHON_BIN}" - \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --use_bf16 \
  "$@" <<'PY'
import logging
import os
import sys

local_train_files = os.path.join(
    os.getcwd(), "examples", "calvin_autoresearch", "train_files"
)
if local_train_files not in sys.path:
    sys.path.insert(0, local_train_files)

gty_root = os.environ.get("GTY_ROOT")
gty_train_files = os.path.join(gty_root, "train_files")
if gty_train_files not in sys.path:
    sys.path.insert(0, gty_train_files)

# Trigger framework registry side effects.
from moe.qwen_gr00t_moe import QwenGR00T_MoE  # noqa: F401
from moe_lora.qwen_gr00t_moe_lora import QwenGR00T_MoE_LoRA  # noqa: F401

# Merge GTY/WMH CALVIN ABC registry entries so PolicyNormProcessor can resolve
# new_embodiment/calvin_abc_augmented from the checkpoint config.
try:
    from data_registry import data_config
    from starVLA.dataloader.gr00t_lerobot import registry as starvla_registry

    for name in (
        "ROBOT_TYPE_CONFIG_MAP",
        "ROBOT_TYPE_TO_EMBODIMENT_TAG",
        "DATASET_NAMED_MIXTURES",
    ):
        value = getattr(data_config, name, None)
        target = getattr(starvla_registry, name, None)
        if isinstance(value, dict) and isinstance(target, dict):
            target.update(value)
    logging.info("Registered CALVIN augmented data registry for MoE+LoRA eval")
except Exception:
    logging.exception("Failed to register CALVIN augmented data registry")
    raise

from deployment.model_server.server_policy import build_argparser
from deployment.model_server.server_policy import main as server_main

logging.basicConfig(level=logging.INFO, force=True)
parser = build_argparser()
args = parser.parse_args()
server_main(args)
PY
