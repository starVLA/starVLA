#!/usr/bin/env bash
set -euo pipefail

# Export the current WMH StarVLA runtime into public/seven so team members can
# train from the shared path without accessing the private project directory.

SRC_PROJECT_ROOT="${SRC_PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220172}"
SRC_STARVLA_ROOT="${SRC_STARVLA_ROOT:-${SRC_PROJECT_ROOT}/WMH/starVLA}"
SRC_ENV="${SRC_ENV:-${SRC_PROJECT_ROOT}/envs/starvla-next}"

SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${SHARED_ROOT}/shared/runtime}"

CODE_DST="${RUNTIME_ROOT}/code/starVLA"
ENV_DST="${RUNTIME_ROOT}/envs/starvla-next"
OLD_ENV_PREFIX="${SRC_ENV}"

MODEL_DIR="${SHARED_ROOT}/shared/models/base/Qwen3-VL-4B-Instruct-Action"
DATA_ROOT="${SHARED_ROOT}/shared/datasets/calvin_lerobot"
WMH_CKPT="${SHARED_ROOT}/shared/checkpoints/wmh_trained/best_abc_to_d_steps_60000_pytorch_model.pt"

log() {
  printf '[export-seven-runtime] %s\n' "$*"
}

require_path() {
  local path="$1"
  local desc="$2"
  if [[ ! -e "${path}" ]]; then
    printf 'Missing %s: %s\n' "${desc}" "${path}" >&2
    exit 2
  fi
}

replace_text_prefix() {
  local root="$1"
  local old="$2"
  local new="$3"

  if [[ ! -d "${root}" ]]; then
    return 0
  fi

  log "rewriting text prefix under ${root}"
  export OLD_PREFIX="${old}"
  export NEW_PREFIX="${new}"

  grep -IlR -- "${OLD_PREFIX}" "${root}" 2>/dev/null | while IFS= read -r file; do
    perl -0pi -e 's/\Q$ENV{OLD_PREFIX}\E/$ENV{NEW_PREFIX}/g' "${file}"
  done
}

copy_code() {
  local tmp="${RUNTIME_ROOT}/code/starVLA.tmp.$$"

  log "copying StarVLA code to ${CODE_DST}"
  rm -rf "${tmp}"
  mkdir -p "${tmp}"

  tar \
    --exclude='.git' \
    --exclude='results' \
    --exclude='wandb' \
    --exclude='playground/Datasets' \
    --exclude='playground/Pretrained_models' \
    --exclude='playground/Checkpoints' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -C "${SRC_STARVLA_ROOT}" \
    -cpf - . | tar -C "${tmp}" -xpf -

  rm -rf "${CODE_DST}"
  mkdir -p "$(dirname "${CODE_DST}")"
  mv "${tmp}" "${CODE_DST}"

  mkdir -p "${CODE_DST}/playground/Pretrained_models" \
           "${CODE_DST}/playground/Datasets" \
           "${RUNTIME_ROOT}/results" \
           "${RUNTIME_ROOT}/results/Checkpoints" \
           "${RUNTIME_ROOT}/cache/huggingface" \
           "${RUNTIME_ROOT}/cache/torch" \
           "${RUNTIME_ROOT}/cache/pip"

  rm -rf "${CODE_DST}/playground/Checkpoints" "${CODE_DST}/results"
  ln -sfn "${MODEL_DIR}" "${CODE_DST}/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action"
  ln -sfn "${DATA_ROOT}" "${CODE_DST}/playground/Datasets/calvin_lerobot"
  ln -sfn "${RUNTIME_ROOT}/results/Checkpoints" "${CODE_DST}/playground/Checkpoints"
  ln -sfn "${RUNTIME_ROOT}/results" "${CODE_DST}/results"
}

copy_env() {
  local tmp="${RUNTIME_ROOT}/envs/starvla-next.tmp.$$"

  if [[ "${SKIP_ENV_COPY:-0}" == "1" && -x "${ENV_DST}/bin/python" ]]; then
    log "SKIP_ENV_COPY=1 and existing env found: ${ENV_DST}"
    return 0
  fi

  log "copying Python env to ${ENV_DST}; this is about 8G and can take several minutes"
  rm -rf "${tmp}"
  mkdir -p "${tmp}"
  tar -C "${SRC_ENV}" -cpf - . | tar -C "${tmp}" -xpf -

  rm -rf "${ENV_DST}"
  mkdir -p "$(dirname "${ENV_DST}")"
  mv "${tmp}" "${ENV_DST}"

  replace_text_prefix "${ENV_DST}" "${OLD_ENV_PREFIX}" "${ENV_DST}"

  "${ENV_DST}/bin/python" - <<'PY'
import sys
print("[export-seven-runtime] relocated python:", sys.executable)
print("[export-seven-runtime] sys.prefix:", sys.prefix)
PY
}

write_runtime_env() {
  log "writing shared runtime env scripts"
  mkdir -p "${RUNTIME_ROOT}/env" "${RUNTIME_ROOT}/scripts"

  cat > "${RUNTIME_ROOT}/starvla_env.sh" <<EOF
#!/usr/bin/env bash

export SEVEN_STARVLA_CALVIN_ROOT="${SHARED_ROOT}"
export STARVLA_RUNTIME_ROOT="${RUNTIME_ROOT}"
export STARVLA_ROOT="\${STARVLA_RUNTIME_ROOT}/code/starVLA"
export STARVLA_ENV="\${STARVLA_RUNTIME_ROOT}/envs/starvla-next"
export STARVLA_PYTHON="\${STARVLA_ENV}/bin/python"

export PATH="\${STARVLA_ENV}/bin:\${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="\${STARVLA_ROOT}:\${PYTHONPATH:-}"

export HF_HOME="\${STARVLA_RUNTIME_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="\${HF_HOME}/hub"
export TORCH_HOME="\${STARVLA_RUNTIME_ROOT}/cache/torch"
export PIP_CACHE_DIR="\${STARVLA_RUNTIME_ROOT}/cache/pip"
export WANDB_DIR="\${STARVLA_RUNTIME_ROOT}/wandb"

export CUDA_HOME="\${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="\${STARVLA_ENV}/lib:\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1

accelerate() {
  "\${STARVLA_PYTHON}" -m accelerate.commands.accelerate_cli "\$@"
}
export -f accelerate

cd "\${STARVLA_ROOT}"
EOF

  cp "${RUNTIME_ROOT}/starvla_env.sh" "${RUNTIME_ROOT}/env/starvla_runtime_env.sh"
  chmod a+rx "${RUNTIME_ROOT}/starvla_env.sh" "${RUNTIME_ROOT}/env/starvla_runtime_env.sh"

  if [[ -d "${SHARED_ROOT}/shared/env" ]]; then
    cat > "${SHARED_ROOT}/shared/env/starvla_env.shared.sh" <<EOF
#!/usr/bin/env bash
source "${RUNTIME_ROOT}/starvla_env.sh"

echo "Seven StarVLA CALVIN runtime:"
echo "  repo: \${STARVLA_ROOT}"
echo "  env:  \${STARVLA_ENV}"
echo "  model: ${MODEL_DIR}"
echo "  calvin abc: ${DATA_ROOT}/calvin_abc_train_v3.0"
echo "  WMH checkpoint: ${WMH_CKPT}"
EOF
    chmod a+rx "${SHARED_ROOT}/shared/env/starvla_env.shared.sh"
  fi
}

write_launchers() {
  log "writing shared training launchers"

  cat > "${RUNTIME_ROOT}/scripts/train_abc_headonly_h200.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/runtime}"
source "${RUNTIME_ROOT}/starvla_env.sh"

MEMBER_CANDIDATE="${MEMBER-}"
if [[ -z "${MEMBER_CANDIDATE}" ]]; then
  MEMBER_CANDIDATE="${SUDO_USER-}"
fi
if [[ -z "${MEMBER_CANDIDATE}" ]]; then
  MEMBER_CANDIDATE="${USER-}"
fi
MEMBER="${MEMBER_CANDIDATE:-shared}"
SHARED_ROOT="${SEVEN_STARVLA_CALVIN_ROOT}"

export BASE_VLM="${BASE_VLM:-${SHARED_ROOT}/shared/models/base/Qwen3-VL-4B-Instruct-Action}"
export DATA_ROOT="${DATA_ROOT:-${SHARED_ROOT}/shared/datasets/calvin_lerobot}"
export DATA_MIX="${DATA_MIX:-calvin_abc_train_v3.0}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SHARED_ROOT}/members/${MEMBER}/runs}"
export RUN_ID="${RUN_ID:-abc_headonly_${MEMBER}_$(date +%m%d_%H%M%S)}"

export NUM_PROCESSES="${NUM_PROCESSES:-3}"
export GPU_IDS="${GPU_IDS:-0,1,2}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-60000}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "${RUN_ROOT_DIR}"

exec bash "${STARVLA_ROOT}/examples/calvin_autoresearch/scripts/run_train_abc_pretrain_h200.sh"
EOF

  cat > "${RUNTIME_ROOT}/scripts/train_abc_headonly_from_wmh_60k_h200.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/runtime}"
source "${RUNTIME_ROOT}/starvla_env.sh"

MEMBER_CANDIDATE="${MEMBER-}"
if [[ -z "${MEMBER_CANDIDATE}" ]]; then
  MEMBER_CANDIDATE="${SUDO_USER-}"
fi
if [[ -z "${MEMBER_CANDIDATE}" ]]; then
  MEMBER_CANDIDATE="${USER-}"
fi
MEMBER="${MEMBER_CANDIDATE:-shared}"
SHARED_ROOT="${SEVEN_STARVLA_CALVIN_ROOT}"

CONFIG_YAML="${CONFIG_YAML:-examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc.yaml}"
BASE_VLM="${BASE_VLM:-${SHARED_ROOT}/shared/models/base/Qwen3-VL-4B-Instruct-Action}"
DATA_ROOT="${DATA_ROOT:-${SHARED_ROOT}/shared/datasets/calvin_lerobot}"
DATA_MIX="${DATA_MIX:-calvin_abc_train_v3.0}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${SHARED_ROOT}/shared/checkpoints/wmh_trained/best_abc_to_d_steps_60000_pytorch_model.pt}"
RELOAD_MODULES="${RELOAD_MODULES:-action_model}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SHARED_ROOT}/members/${MEMBER}/runs}"
RUN_ID="${RUN_ID:-abc_headonly_from_wmh60k_${MEMBER}_$(date +%m%d_%H%M%S)}"

NUM_PROCESSES="${NUM_PROCESSES:-3}"
GPU_IDS="${GPU_IDS:-0,1,2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-1}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-1}"

if [[ "${DATA_MIX}" != "calvin_abc_train_v3.0" ]]; then
  echo "This shared launcher is ABC-only. Refusing DATA_MIX=${DATA_MIX}" >&2
  exit 2
fi

case "${DATA_ROOT}/${DATA_MIX}" in
  *task_D_D*|*ABCD-D*|*abcd-d*|*calvin-task-D-D*|*calvin-task-ABCD-D*)
    echo "This shared launcher must not train on CALVIN D or ABCD-D data." >&2
    echo "Refusing dataset path: ${DATA_ROOT}/${DATA_MIX}" >&2
    exit 2
    ;;
esac

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${STARVLA_PYTHON}" - "${NUM_PROCESSES}" <<'PY'
import sys
import torch

required = int(sys.argv[1])
available = torch.cuda.device_count() if torch.cuda.is_available() else 0
if available < required:
    raise SystemExit(
        f"This H200 launcher requires at least {required} visible CUDA devices, "
        f"but PyTorch sees {available}. Run inside a GPU allocation and check nvidia-smi."
    )
print(f"[train-from-wmh60k] visible CUDA devices: {available}")
PY
fi

STRICT_ASSETS="${STRICT_ASSETS:-1}" \
BASE_VLM="${BASE_VLM}" \
DATA_ROOT="${DATA_ROOT}" \
TRAIN_DATASET="${DATA_MIX}" \
"${STARVLA_ROOT}/examples/calvin_autoresearch/scripts/verify_assets.sh"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"

mkdir -p "${RUN_ROOT_DIR}"

cmd=(
  "${STARVLA_PYTHON}" -m accelerate.commands.accelerate_cli launch
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
  --num_processes "${NUM_PROCESSES}"
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}"
  --run_id "${RUN_ID}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --framework.qwenvl.base_vlm "${BASE_VLM}"
  --datasets.vla_data.data_root_dir "${DATA_ROOT}"
  --datasets.vla_data.data_mix "${DATA_MIX}"
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.logging_frequency "${LOGGING_FREQUENCY}"
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --datasets.vla_data.num_workers "${DATALOADER_NUM_WORKERS}"
  --datasets.vla_data.prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
  --datasets.vla_data.pin_memory "${DATALOADER_PIN_MEMORY}"
  --datasets.vla_data.persistent_workers "${DATALOADER_PERSISTENT_WORKERS}"
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
  --trainer.reload_modules "${RELOAD_MODULES}"
)

printf '[train-from-wmh60k] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${cmd[@]}"
EOF

  cat > "${RUNTIME_ROOT}/scripts/smoke_test_runtime.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/runtime}"
source "${RUNTIME_ROOT}/starvla_env.sh"

"${STARVLA_PYTHON}" - <<'PY'
import os
import torch
import transformers
import starVLA

print("python:", os.sys.executable)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("starVLA:", starVLA.__file__)
PY

DRY_RUN=1 \
MAX_TRAIN_STEPS=1 \
RUN_ROOT_DIR="${RUNTIME_ROOT}/smoke_runs" \
bash "${RUNTIME_ROOT}/scripts/train_abc_headonly_h200.sh"
EOF

  chmod a+rx "${RUNTIME_ROOT}/scripts/"*.sh
}

write_readme() {
  log "writing README"
  cat > "${RUNTIME_ROOT}/README.md" <<EOF
# Shared StarVLA CALVIN Runtime

This runtime was exported from:

\`\`\`bash
${SRC_PROJECT_ROOT}
\`\`\`

Use it from any GPU allocation that can access \`public/seven\`.

## Environment Smoke Test

\`\`\`bash
bash ${RUNTIME_ROOT}/scripts/smoke_test_runtime.sh
\`\`\`

## Train ABC Head-Only From Base Qwen

\`\`\`bash
MEMBER=GTY \\
bash ${RUNTIME_ROOT}/scripts/train_abc_headonly_h200.sh
\`\`\`

## Continue Head-Only Training From WMH 60k Action Head

\`\`\`bash
MEMBER=GTY MAX_TRAIN_STEPS=10000 \\
bash ${RUNTIME_ROOT}/scripts/train_abc_headonly_from_wmh_60k_h200.sh
\`\`\`

Default assets:

- Base VLM: \`${MODEL_DIR}\`
- ABC LeRobot root: \`${DATA_ROOT}\`
- WMH action-head checkpoint: \`${WMH_CKPT}\`

The shared launchers are ABC-only and refuse D / ABCD-D training paths.
EOF
  chmod a+r "${RUNTIME_ROOT}/README.md"
}

main() {
  require_path "${SRC_STARVLA_ROOT}/starVLA/training/train_starvla.py" "StarVLA source repo"
  require_path "${SRC_ENV}/bin/python" "source Python env"
  require_path "${MODEL_DIR}" "shared base VLM"
  require_path "${DATA_ROOT}/calvin_abc_train_v3.0" "shared ABC dataset"
  require_path "${WMH_CKPT}" "WMH 60k checkpoint"

  mkdir -p "${RUNTIME_ROOT}"
  copy_code
  copy_env
  write_runtime_env
  write_launchers
  write_readme

  if [[ -f "${CODE_DST}/examples/calvin_autoresearch/scripts/patch_seven_runtime_nccl_barrier.sh" ]]; then
    RUNTIME_ROOT="${RUNTIME_ROOT}" bash "${CODE_DST}/examples/calvin_autoresearch/scripts/patch_seven_runtime_nccl_barrier.sh"
  fi

  chmod -R a+rX "${RUNTIME_ROOT}"

  log "done"
  log "runtime: ${RUNTIME_ROOT}"
  log "smoke test: bash ${RUNTIME_ROOT}/scripts/smoke_test_runtime.sh"
}

main "$@"
