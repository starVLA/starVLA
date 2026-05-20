#!/usr/bin/env bash
set -euo pipefail

# Policy server launcher for GTY MoE / MoE-Adaptive checkpoints.
# It registers both framework classes before delegating to StarVLA's
# deployment server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# GTY MoE/Adaptive checkpoints were trained with the shared runtime copy.
# Keep that as the default for checkpoint compatibility; callers can override
# STARVLA_ROOT if they intentionally want to test another code checkout.
STARVLA_ROOT="${STARVLA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/runtime/code/starVLA}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"
GTY_ROOT="${GTY_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/GTY}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

: "${CKPT:?Set CKPT to a QwenGR00T_MoE or QwenGR00T_MoE_Adaptive checkpoint}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 3
fi

GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-5694}"
PYTHON_BIN="${STARVLA_PYTHON:-python}"

export PYTHONPATH="${GTY_ROOT}/train_files:${STARVLA_ROOT}:${PYTHONPATH:-}"
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

gty_root = os.environ.get("GTY_ROOT")
train_files = os.path.join(gty_root, "train_files")
if train_files not in sys.path:
    sys.path.insert(0, train_files)

# Trigger framework registry side effects.
from moe.qwen_gr00t_moe import QwenGR00T_MoE  # noqa: F401
from moe.qwen_gr00t_moe_adaptive import QwenGR00T_MoE_Adaptive  # noqa: F401

# GTY MoE/Adaptive checkpoints were trained with member-local data registry
# entries such as calvin_abc_augmented.  The deployment server resolves
# robot_type through the global StarVLA registry, so merge those entries before
# PolicyNormProcessor reads the checkpoint config.
try:
    from data_registry import data_config as gty_data_config
    from starVLA.dataloader.gr00t_lerobot import registry as starvla_registry

    if hasattr(gty_data_config, "ROBOT_TYPE_CONFIG_MAP"):
        starvla_registry.ROBOT_TYPE_CONFIG_MAP.update(
            gty_data_config.ROBOT_TYPE_CONFIG_MAP
        )
    if hasattr(gty_data_config, "ROBOT_TYPE_TO_EMBODIMENT_TAG"):
        starvla_registry.ROBOT_TYPE_TO_EMBODIMENT_TAG.update(
            gty_data_config.ROBOT_TYPE_TO_EMBODIMENT_TAG
        )
    if hasattr(gty_data_config, "DATASET_NAMED_MIXTURES"):
        starvla_registry.DATASET_NAMED_MIXTURES.update(
            gty_data_config.DATASET_NAMED_MIXTURES
        )
    logging.info(
        "Registered GTY data registry entries: mixtures=%s robot_types=%s",
        sorted(getattr(gty_data_config, "DATASET_NAMED_MIXTURES", {}).keys()),
        sorted(getattr(gty_data_config, "ROBOT_TYPE_CONFIG_MAP", {}).keys()),
    )
except Exception:
    logging.exception("Failed to register GTY data registry entries")
    raise

from deployment.model_server.server_policy import build_argparser
from deployment.model_server.server_policy import main as server_main

logging.basicConfig(level=logging.INFO, force=True)
parser = build_argparser()
args = parser.parse_args()
server_main(args)
PY
