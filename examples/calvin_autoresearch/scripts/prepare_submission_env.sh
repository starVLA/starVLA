#!/usr/bin/env bash
set -euo pipefail

# Prepare stable links/manifests expected by the final image/test harness.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
PUBLIC_D="${PUBLIC_D:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d}"
RUN_ID="${RUN_ID:-abc_pretrain_qwen3vl_gr00t_headonly_h200_60k_0518_163437}"
RUN_DIR="${RUN_DIR:-${SHARED_ROOT}/members/WMH/runs/${RUN_ID}}"
BEST_CKPT="${BEST_CKPT:-${RUN_DIR}/checkpoints/steps_60000_pytorch_model.pt}"

cd "${STARVLA_ROOT}"

if [[ ! -d "${PUBLIC_D}" || ! -f "${PUBLIC_D}/validation/.hydra/merged_config.yaml" ]]; then
  echo "Missing formal CALVIN D dataset: ${PUBLIC_D}" >&2
  exit 3
fi

if [[ ! -f "${BEST_CKPT}" ]]; then
  echo "Missing best checkpoint: ${BEST_CKPT}" >&2
  exit 4
fi

ensure_symlink() {
  local target="$1"
  local link_path="$2"
  local parent
  parent="$(dirname "${link_path}")"

  if [[ -e "${link_path}" || -L "${link_path}" ]]; then
    if [[ "$(readlink -f "${link_path}" 2>/dev/null)" == "$(readlink -f "${target}" 2>/dev/null)" ]]; then
      echo "[submission-env] link already valid: ${link_path}"
      return 0
    fi
  fi

  mkdir -p "${parent}"
  if [[ ! -w "${parent}" ]]; then
    echo "Cannot update ${link_path}; parent directory is not writable: ${parent}" >&2
    echo "Run this in a shell with write access:" >&2
    printf '  mkdir -p %q && ln -sfnT %q %q\n' "${parent}" "${target}" "${link_path}" >&2
    exit 5
  fi
  ln -sfnT "${target}" "${link_path}"
}

mkdir -p "${PROJECT_ROOT}/data/calvin_original" "${SHARED_ROOT}/shared/checkpoints/wmh_trained" "${SHARED_ROOT}/shared/manifests"

ensure_symlink "${PUBLIC_D}" "${PROJECT_ROOT}/data/calvin_original/task_D_D"
ensure_symlink "${PUBLIC_D}" "${SHARED_ROOT}/shared/datasets/calvin_original/task_D_D"
ensure_symlink "${BEST_CKPT}" "${SHARED_ROOT}/shared/checkpoints/wmh_trained/best_abc_to_d_steps_60000_pytorch_model.pt"

cat > "${SHARED_ROOT}/shared/manifests/wmh_submission.yaml" <<EOF
project_root: ${PROJECT_ROOT}
starvla_root: ${STARVLA_ROOT}
best_checkpoint: ${SHARED_ROOT}/shared/checkpoints/wmh_trained/best_abc_to_d_steps_60000_pytorch_model.pt
training_run_dir: ${RUN_DIR}
base_model: ${PROJECT_ROOT}/models/Qwen3-VL-4B-Instruct-Action
calvin_abc_lerobot: ${PROJECT_ROOT}/data/calvin_lerobot/calvin_abc_train_v3.0
calvin_d_eval: ${PUBLIC_D}
oneclick_train: examples/calvin_autoresearch/scripts/run_train_abc_h200_oneclick.sh
oneclick_eval: examples/calvin_autoresearch/scripts/run_eval_abc_to_d_oneclick.sh
guardrail: no upstream action-trained policy checkpoints are used
EOF

CHECK_ORIGINAL_D=1 STRICT_ASSETS=1 bash examples/calvin_autoresearch/scripts/verify_assets.sh
echo "[submission-env] manifest: ${SHARED_ROOT}/shared/manifests/wmh_submission.yaml"
