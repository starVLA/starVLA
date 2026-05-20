#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh" >/dev/null 2>&1 || true
fi

cd "${STARVLA_ROOT}"

PYTHON_BIN="${STARVLA_PYTHON:-python}"
BASE_VLM="${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
DATA_ROOT="${DATA_ROOT:-playground/Datasets/calvin_lerobot}"
TRAIN_DATASET="${TRAIN_DATASET:-calvin_abc_train_v3.0}"
D_EVAL_DATASET="${D_EVAL_DATASET:-calvin_task_D_D_v3.0}"
CALVIN_ORIGINAL_D="${CALVIN_ORIGINAL_D:-playground/Datasets/calvin_original/task_D_D}"
CONFIG_YAML="${CONFIG_YAML:-examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc.yaml}"
STRICT_ASSETS="${STRICT_ASSETS:-0}"
CHECK_ORIGINAL_D="${CHECK_ORIGINAL_D:-0}"

bad_ckpt_re='(Qwen3-VL-OFT-LIBERO|LIBERO|Robotwin|robotwin|Robocasa|robocasa|Behavior|BEHAVIOR|SimplerEnv|qwenpi_calvin_task_D_D|qwenoft)'
bad_assignment_re="(CKPT|your_ckpt|MODEL_PATH|PRETRAINED_CHECKPOINT)[^#]*${bad_ckpt_re}"

echo "[verify] repo: ${STARVLA_ROOT}"
echo "[verify] config: ${CONFIG_YAML}"

if rg -n "${bad_assignment_re}" examples/calvin_autoresearch >/tmp/starvla_calvin_bad_refs.$$ 2>/dev/null; then
  echo "[verify] forbidden action-trained checkpoint default under examples/calvin_autoresearch:" >&2
  cat /tmp/starvla_calvin_bad_refs.$$ >&2
  rm -f /tmp/starvla_calvin_bad_refs.$$
  exit 2
fi
rm -f /tmp/starvla_calvin_bad_refs.$$

if rg -n "${bad_ckpt_re}" examples/calvin/eval_files >/tmp/starvla_calvin_upstream_refs.$$ 2>/dev/null; then
  echo "[verify] note: upstream examples/calvin eval/server scripts still contain trained-checkpoint defaults."
  echo "[verify]       Do not use them for this baseline; use examples/calvin_autoresearch/scripts instead."
  rm -f /tmp/starvla_calvin_upstream_refs.$$
fi

"${PYTHON_BIN}" - "${CONFIG_YAML}" "${TRAIN_DATASET}" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

cfg_path = Path(sys.argv[1])
train_dataset = sys.argv[2]
cfg = OmegaConf.load(cfg_path)
assert cfg.framework.name == "QwenGR00T", cfg.framework.name
assert "Qwen3-VL-4B-Instruct-Action" in cfg.framework.qwenvl.base_vlm, cfg.framework.qwenvl.base_vlm
assert cfg.datasets.vla_data.data_mix in {"calvin_abc_train_v3.0", "calvin_abc_train_state_v3.0"}, cfg.datasets.vla_data.data_mix
assert train_dataset == "calvin_abc_train_v3.0", train_dataset
assert cfg.datasets.vla_data.lerobot_version == "v2.0", cfg.datasets.vla_data.lerobot_version
assert "pretrained_checkpoint" not in cfg.trainer, "trainer.pretrained_checkpoint must not be set"

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

for key in ("calvin_abc_train_v3.0", "calvin_abc_train_state_v3.0", "calvin_task_D_D_v3.0"):
    assert key in DATASET_NAMED_MIXTURES, f"missing mixture: {key}"

print("[verify] config and registry checks passed")
PY

missing=0
for path in "${BASE_VLM}" "${DATA_ROOT}/${TRAIN_DATASET}"; do
  if [[ -e "${path}" ]]; then
    echo "[verify] found: ${path}"
  else
    echo "[verify] missing: ${path}"
    missing=1
  fi
done

if [[ -e "${DATA_ROOT}/${D_EVAL_DATASET}" ]]; then
  echo "[verify] found optional D LeRobot: ${DATA_ROOT}/${D_EVAL_DATASET}"
else
  echo "[verify] missing optional D LeRobot: ${DATA_ROOT}/${D_EVAL_DATASET}"
fi

dataset_dirs=("${DATA_ROOT}/${TRAIN_DATASET}")
if [[ -e "${DATA_ROOT}/${D_EVAL_DATASET}" ]]; then
  dataset_dirs+=("${DATA_ROOT}/${D_EVAL_DATASET}")
fi

for dataset_dir in "${dataset_dirs[@]}"; do
  if [[ -e "${dataset_dir}" && ! -f "${dataset_dir}/meta/modality.json" ]]; then
    echo "[verify] missing modality metadata: ${dataset_dir}/meta/modality.json" >&2
    missing=1
  fi
done

if [[ "${CHECK_ORIGINAL_D}" == "1" ]]; then
  if [[ -f "${CALVIN_ORIGINAL_D}/validation/.hydra/merged_config.yaml" ]] && compgen -G "${CALVIN_ORIGINAL_D}/validation/episode_*.npz" >/dev/null; then
    echo "[verify] found original CALVIN D: ${CALVIN_ORIGINAL_D}"
  else
    echo "[verify] missing formal original CALVIN D validation split: ${CALVIN_ORIGINAL_D}"
    missing=1
  fi
fi

if [[ "${STRICT_ASSETS}" == "1" && "${missing}" == "1" ]]; then
  echo "[verify] STRICT_ASSETS=1 and at least one required asset is missing." >&2
  exit 3
fi

echo "[verify] baseline preflight completed"
