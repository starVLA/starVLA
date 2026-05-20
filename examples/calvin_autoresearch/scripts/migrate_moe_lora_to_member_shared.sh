#!/usr/bin/env bash
set -euo pipefail

# Migrate the current WMH StarVLA MoE95k+LoRA training code to a shared member
# directory so GPU machines can launch it without relying on the private
# /26220172/WMH/starVLA checkout.
#
# Output:
#   ${SHARED_ROOT}/members/${MEMBER}/code/starVLA_moe_lora/
#   ${SHARED_ROOT}/members/${MEMBER}/run_moe_lora_aug8.sh
#   ${SHARED_ROOT}/members/${MEMBER}/run_moe_lora_mirror8.sh
#
# This copies code only. Checkpoints, datasets, and base models stay in the
# existing shared locations and are referenced by absolute paths.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="${SOURCE_REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
MEMBER="${MEMBER:-WMH}"
GTY_MEMBER="${GTY_MEMBER:-GTY}"

MEMBER_ROOT="${SHARED_ROOT}/members/${MEMBER}"
TARGET_CODE="${TARGET_CODE:-${MEMBER_ROOT}/code/starVLA_moe_lora}"
BACKUP_ROOT="${MEMBER_ROOT}/backups/moe_lora_migration"
TS="${TS:-$(date +%m%d_%H%M%S)}"

required_files=(
  "${SOURCE_REPO}/wmh"
  "${SOURCE_REPO}/examples/calvin_autoresearch/scripts/run_train_moe95k_lora_h200.sh"
  "${SOURCE_REPO}/examples/calvin_autoresearch/train_files/run_train_moe_lora_entry.py"
  "${SOURCE_REPO}/examples/calvin_autoresearch/train_files/moe_lora/qwen_gr00t_moe_lora.py"
  "${SOURCE_REPO}/examples/calvin_autoresearch/train_files/starvla_qwen3vl_calvin_abc_augmented_moe_lora.yaml"
  "${SOURCE_REPO}/examples/calvin_autoresearch/train_files/starvla_qwen3vl_calvin_abc_augmented_moe_lora_lrmirror.yaml"
  "${SOURCE_REPO}/starVLA/model/framework/VLM4A/QwenGR00T.py"
  "${SOURCE_REPO}/starVLA/dataloader/gr00t_lerobot/datasets.py"
  "${SOURCE_REPO}/starVLA/training/trainer_utils/trainer_tools.py"
)

for path in "${required_files[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[migrate-moe-lora] missing required source file: ${path}" >&2
    exit 3
  fi
done

for path in \
  "${SHARED_ROOT}/shared/runtime/starvla_env.sh" \
  "${SHARED_ROOT}/shared/env/tmux.shared.sh" \
  "${SHARED_ROOT}/shared/models/base/Qwen3-VL-4B-Instruct-Action/config.json" \
  "${SHARED_ROOT}/shared/datasets/calvin_lerobot/calvin_abc_train_v3.0/meta/modality.json" \
  "${SHARED_ROOT}/members/${GTY_MEMBER}/runs/gty_moe_posttrain_8h_GTY_0519_182014/checkpoints/steps_95000_pytorch_model.pt" \
  "${SHARED_ROOT}/members/${GTY_MEMBER}/train_files/data_registry/data_config.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "[migrate-moe-lora] missing required shared asset: ${path}" >&2
    exit 3
  fi
done

mkdir -p "${MEMBER_ROOT}/code" "${BACKUP_ROOT}"

if [[ -e "${TARGET_CODE}" ]]; then
  backup="${BACKUP_ROOT}/starVLA_moe_lora_${TS}"
  echo "[migrate-moe-lora] existing target found, moving to backup:"
  echo "  ${backup}"
  mv "${TARGET_CODE}" "${backup}"
fi

echo "[migrate-moe-lora] copying repo:"
echo "  from ${SOURCE_REPO}"
echo "  to   ${TARGET_CODE}"

mkdir -p "${TARGET_CODE}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='wandb/' \
    --exclude='results/' \
    --exclude='wmh_links/' \
    --exclude='playground/Datasets' \
    --exclude='playground/Pretrained_models' \
    --exclude='playground/Checkpoints' \
    "${SOURCE_REPO}/" "${TARGET_CODE}/"
else
  tar -C "${SOURCE_REPO}" \
    --exclude='./.git' \
    --exclude='./__pycache__' \
    --exclude='*.pyc' \
    --exclude='./wandb' \
    --exclude='./results' \
    --exclude='./wmh_links' \
    --exclude='./playground/Datasets' \
    --exclude='./playground/Pretrained_models' \
    --exclude='./playground/Checkpoints' \
    -cf - . | tar -C "${TARGET_CODE}" -xf -
fi

mkdir -p "${TARGET_CODE}/playground"
ln -sfn "${SHARED_ROOT}/shared/datasets" "${TARGET_CODE}/playground/Datasets"
ln -sfn "${SHARED_ROOT}/shared/models/base" "${TARGET_CODE}/playground/Pretrained_models"
ln -sfn "${MEMBER_ROOT}/runs" "${TARGET_CODE}/playground/Checkpoints"
ln -sfn "${SHARED_ROOT}/members/${GTY_MEMBER}" "${TARGET_CODE}/examples/GTY_calvin"
ln -sfn "${TARGET_CODE}" "${MEMBER_ROOT}/code/latest_starVLA_moe_lora"

chmod +x \
  "${TARGET_CODE}/wmh" \
  "${TARGET_CODE}/examples/calvin_autoresearch/scripts/run_train_moe95k_lora_h200.sh"

cat > "${MEMBER_ROOT}/run_moe_lora_aug8.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SHARED_ROOT="${SHARED_ROOT}"
CODE_ROOT="${TARGET_CODE}"
source "\${SHARED_ROOT}/shared/runtime/starvla_env.sh"
if [[ -f "\${SHARED_ROOT}/shared/env/tmux.shared.sh" ]]; then
  source "\${SHARED_ROOT}/shared/env/tmux.shared.sh"
fi
export STARVLA_ROOT="\${CODE_ROOT}"
export PYTHONPATH="\${CODE_ROOT}:\${CODE_ROOT}/examples/calvin_autoresearch/train_files:\${SHARED_ROOT}/members/${GTY_MEMBER}/train_files:\${PYTHONPATH:-}"
cd "\${CODE_ROOT}"
exec ./wmh moe-lora-aug8
EOF

cat > "${MEMBER_ROOT}/run_moe_lora_mirror8.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SHARED_ROOT="${SHARED_ROOT}"
CODE_ROOT="${TARGET_CODE}"
source "\${SHARED_ROOT}/shared/runtime/starvla_env.sh"
if [[ -f "\${SHARED_ROOT}/shared/env/tmux.shared.sh" ]]; then
  source "\${SHARED_ROOT}/shared/env/tmux.shared.sh"
fi
export STARVLA_ROOT="\${CODE_ROOT}"
export PYTHONPATH="\${CODE_ROOT}:\${CODE_ROOT}/examples/calvin_autoresearch/train_files:\${SHARED_ROOT}/members/${GTY_MEMBER}/train_files:\${PYTHONPATH:-}"
cd "\${CODE_ROOT}"
exec ./wmh moe-lora-mirror8
EOF

cat > "${MEMBER_ROOT}/tail_moe_lora_aug.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${TARGET_CODE}"
exec ./wmh tail-moe-lora-aug
EOF

cat > "${MEMBER_ROOT}/tail_moe_lora_mirror.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${TARGET_CODE}"
exec ./wmh tail-moe-lora-mirror
EOF

chmod +x \
  "${MEMBER_ROOT}/run_moe_lora_aug8.sh" \
  "${MEMBER_ROOT}/run_moe_lora_mirror8.sh" \
  "${MEMBER_ROOT}/tail_moe_lora_aug.sh" \
  "${MEMBER_ROOT}/tail_moe_lora_mirror.sh"

cat > "${MEMBER_ROOT}/MOE_LORA_README.txt" <<EOF
MoE95k + fresh LoRA training entrypoints
=======================================

Code copy:
  ${TARGET_CODE}

Run on 8-card machine A, no mirror:
  bash ${MEMBER_ROOT}/run_moe_lora_aug8.sh

Run on 8-card machine B, with left/right mirror:
  bash ${MEMBER_ROOT}/run_moe_lora_mirror8.sh

Tail logs:
  bash ${MEMBER_ROOT}/tail_moe_lora_aug.sh
  bash ${MEMBER_ROOT}/tail_moe_lora_mirror.sh

Default training:
  MAX_TRAIN_STEPS=2500
  SAVE_INTERVAL=250
  BATCH_SIZE=16
  source checkpoint: GTY MoE steps_95000
  data: calvin_abc_augmented only

Override examples:
  MAX_TRAIN_STEPS=1000 SAVE_INTERVAL=200 bash ${MEMBER_ROOT}/run_moe_lora_aug8.sh
  MAX_TRAIN_STEPS=1000 SAVE_INTERVAL=200 bash ${MEMBER_ROOT}/run_moe_lora_mirror8.sh
EOF

echo "[migrate-moe-lora] done"
echo "Code: ${TARGET_CODE}"
echo "No mirror: bash ${MEMBER_ROOT}/run_moe_lora_aug8.sh"
echo "Mirror:    bash ${MEMBER_ROOT}/run_moe_lora_mirror8.sh"
