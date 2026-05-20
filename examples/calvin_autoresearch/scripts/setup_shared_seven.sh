#!/usr/bin/env bash
set -euo pipefail

# Initialize the group-shared StarVLA/CALVIN workspace under /public/seven.
#
# This script is intentionally conservative:
# - it stores large common assets as symlinks, not copies;
# - it refuses to overwrite real files/directories;
# - it separates shared assets from per-member work areas;
# - it does not preload upstream action-trained policy checkpoints.

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220172}"
STARVLA_ROOT="${STARVLA_ROOT:-${PROJECT_ROOT}/WMH/starVLA}"
SEVEN_ROOT="${SEVEN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven}"
WORKSPACE_NAME="${WORKSPACE_NAME:-starvla_calvin}"
SHARED_ROOT="${SHARED_ROOT:-${SEVEN_ROOT}/${WORKSPACE_NAME}}"

# Members can be provided as either space- or comma-separated text:
#   MEMBERS="WMH zhangsan lisi" bash ...
#   MEMBERS="WMH,zhangsan,lisi" bash ...
MEMBERS="${MEMBERS:-WMH}"
CURRENT_MEMBER="${CURRENT_MEMBER:-WMH}"

MODEL_SOURCE="${MODEL_SOURCE:-/inspire/qb-ilm2/project/26summer-camp-10/public/Qwen3-VL-4B-Instruct-Action}"
ABC_SOURCE="${ABC_SOURCE:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D}"
PREPARED_D_SOURCE="${SHARED_ROOT}/shared/datasets/calvin_original/task_D_D_official_extract/task_D_D"
PUBLIC_D_SOURCE="/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d"
LEGACY_D_SOURCE="/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/task_D_D"
if [[ -e "${PREPARED_D_SOURCE}" ]]; then
  DEFAULT_D_SOURCE="${PREPARED_D_SOURCE}"
elif [[ -e "${PUBLIC_D_SOURCE}" ]]; then
  DEFAULT_D_SOURCE="${PUBLIC_D_SOURCE}"
else
  DEFAULT_D_SOURCE="${LEGACY_D_SOURCE}"
fi
D_SOURCE="${D_SOURCE:-${DEFAULT_D_SOURCE}}"
REQUIRE_D_SOURCE="${REQUIRE_D_SOURCE:-0}"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required source path: ${path}" >&2
    exit 2
  fi
}

safe_link() {
  local src="$1"
  local dst="$2"
  require_path "${src}"
  mkdir -p "$(dirname "${dst}")"

  if [[ -L "${dst}" ]]; then
    ln -sfn "${src}" "${dst}"
    return
  fi

  if [[ -e "${dst}" ]]; then
    local src_real dst_real
    src_real="$(readlink -f "${src}")"
    dst_real="$(readlink -f "${dst}")"
    if [[ "${src_real}" == "${dst_real}" ]]; then
      return
    fi
    echo "Refusing to overwrite non-symlink path: ${dst}" >&2
    echo "  existing resolves to: ${dst_real}" >&2
    echo "  wanted resolves to:   ${src_real}" >&2
    exit 3
  fi

  ln -s "${src}" "${dst}"
}

write_file_if_changed() {
  local path="$1"
  local tmp
  tmp="$(mktemp)"
  cat > "${tmp}"
  mkdir -p "$(dirname "${path}")"
  if [[ -f "${path}" ]] && cmp -s "${tmp}" "${path}"; then
    rm -f "${tmp}"
  else
    mv "${tmp}" "${path}"
  fi
}

require_path "${MODEL_SOURCE}"
require_path "${ABC_SOURCE}"

has_d_source=0
if [[ -e "${D_SOURCE}" ]]; then
  has_d_source=1
elif [[ "${REQUIRE_D_SOURCE}" == "1" ]]; then
  require_path "${D_SOURCE}"
else
  echo "Warning: CALVIN task_D_D source is not ready yet: ${D_SOURCE}" >&2
  echo "         Run examples/calvin_autoresearch/scripts/prepare_task_d_data.sh before formal D eval." >&2
fi

if ! mkdir -p "${SHARED_ROOT}" 2>/dev/null; then
  echo "Cannot write shared workspace: ${SHARED_ROOT}" >&2
  echo "Run this script in a session where /public/seven is writable." >&2
  exit 1
fi

mkdir -p \
  "${SHARED_ROOT}/shared/models/base" \
  "${SHARED_ROOT}/shared/datasets/calvin_lerobot" \
  "${SHARED_ROOT}/shared/datasets/calvin_original" \
  "${SHARED_ROOT}/shared/env" \
  "${SHARED_ROOT}/shared/manifests" \
  "${SHARED_ROOT}/shared/docs" \
  "${SHARED_ROOT}/shared/checkpoints/wmh_trained" \
  "${SHARED_ROOT}/shared/reports" \
  "${SHARED_ROOT}/templates" \
  "${SHARED_ROOT}/members"

safe_link "${MODEL_SOURCE}" "${SHARED_ROOT}/shared/models/base/Qwen3-VL-4B-Instruct-Action"
safe_link "${ABC_SOURCE}" "${SHARED_ROOT}/shared/datasets/calvin_lerobot/calvin_abc_train_v3.0"
if [[ "${has_d_source}" == "1" ]]; then
  safe_link "${D_SOURCE}" "${SHARED_ROOT}/shared/datasets/calvin_original/task_D_D"
fi

normalized_members="${MEMBERS//,/ }"
for member in ${normalized_members}; do
  [[ -z "${member}" ]] && continue
  mkdir -p \
    "${SHARED_ROOT}/members/${member}/runs" \
    "${SHARED_ROOT}/members/${member}/checkpoints" \
    "${SHARED_ROOT}/members/${member}/logs" \
    "${SHARED_ROOT}/members/${member}/notes" \
    "${SHARED_ROOT}/members/${member}/reports" \
    "${SHARED_ROOT}/members/${member}/scratch"

  write_file_if_changed "${SHARED_ROOT}/members/${member}/README.md" <<EOF
# ${member} Workspace

Use this directory for member-owned StarVLA/CALVIN outputs.

Recommended layout:

- runs/: training and evaluation run folders
- checkpoints/: checkpoints trained by this member
- logs/: server, training, and eval logs
- reports/: experiment notes and result summaries
- notes/: free-form notes
- scratch/: temporary files safe to delete

Do not put upstream action-trained policy checkpoints here.
Use shared base models and datasets from ../../shared/.
EOF
done

write_file_if_changed "${SHARED_ROOT}/README.md" <<EOF
# StarVLA CALVIN Shared Workspace

Root:

\`\`\`text
${SHARED_ROOT}
\`\`\`

This workspace separates shared assets from member-owned outputs.

## Shared Assets

\`\`\`text
shared/models/base/Qwen3-VL-4B-Instruct-Action
shared/datasets/calvin_lerobot/calvin_abc_train_v3.0
shared/datasets/calvin_original/task_D_D
shared/checkpoints/wmh_trained
shared/env
shared/manifests
\`\`\`

Allowed shared assets:

- base VLMs such as Qwen/Cosmos
- CALVIN datasets
- WMH-trained checkpoints produced by this project

Forbidden preloaded assets:

- upstream LIBERO action-trained policy checkpoints
- upstream Robotwin/Robocasa/Behavior/SimplerEnv action-trained policy checkpoints
- upstream CALVIN action-trained policy checkpoints such as qwenpi_calvin_task_D_D

## Member Workspaces

Each member should use:

\`\`\`text
members/<member>/
\`\`\`

Do not use another member's directory for scratch or run outputs.
EOF

write_file_if_changed "${SHARED_ROOT}/shared/manifests/assets.yaml" <<EOF
models:
  qwen3vl_action_base:
    policy: allowed_base_model
    source: ${MODEL_SOURCE}
    shared_path: ${SHARED_ROOT}/shared/models/base/Qwen3-VL-4B-Instruct-Action
    expected_files:
      - config.json
      - model.safetensors.index.json
      - tokenizer.json

datasets:
  calvin_abc_lerobot:
    source: ${ABC_SOURCE}
    shared_path: ${SHARED_ROOT}/shared/datasets/calvin_lerobot/calvin_abc_train_v3.0
    use: train_smoke_and_abc_imitation
    reader: lerobot_v2

  calvin_original_task_d_d:
    source: ${D_SOURCE}
    shared_path: ${SHARED_ROOT}/shared/datasets/calvin_original/task_D_D
    use: closed_loop_d_evaluation

guardrails:
  forbidden_preloaded_policy_weights:
    - LIBERO
    - Robotwin
    - Robocasa
    - Behavior
    - SimplerEnv
    - qwenpi_calvin_task_D_D
EOF

write_file_if_changed "${SHARED_ROOT}/shared/env/starvla_env.shared.sh" <<EOF
#!/usr/bin/env bash

export SEVEN_STARVLA_CALVIN_ROOT="${SHARED_ROOT}"
export SEVEN_STARVLA_MODEL_DIR="\${SEVEN_STARVLA_CALVIN_ROOT}/shared/models/base/Qwen3-VL-4B-Instruct-Action"
export SEVEN_CALVIN_ABC_DIR="\${SEVEN_STARVLA_CALVIN_ROOT}/shared/datasets/calvin_lerobot/calvin_abc_train_v3.0"
export SEVEN_CALVIN_D_DIR="\${SEVEN_STARVLA_CALVIN_ROOT}/shared/datasets/calvin_original/task_D_D"
export SEVEN_WMH_CHECKPOINT_DIR="\${SEVEN_STARVLA_CALVIN_ROOT}/shared/checkpoints/wmh_trained"

echo "Seven StarVLA CALVIN assets:"
echo "  model: \${SEVEN_STARVLA_MODEL_DIR}"
echo "  calvin abc: \${SEVEN_CALVIN_ABC_DIR}"
echo "  calvin d: \${SEVEN_CALVIN_D_DIR}"
echo "  WMH checkpoints: \${SEVEN_WMH_CHECKPOINT_DIR}"
EOF
chmod +x "${SHARED_ROOT}/shared/env/starvla_env.shared.sh"

write_file_if_changed "${SHARED_ROOT}/templates/link_member_project.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to your project root}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"

mkdir -p "${PROJECT_ROOT}/models" "${PROJECT_ROOT}/data/calvin_lerobot" "${PROJECT_ROOT}/data/calvin_original"
ln -sfn "${SHARED_ROOT}/shared/models/base/Qwen3-VL-4B-Instruct-Action" "${PROJECT_ROOT}/models/Qwen3-VL-4B-Instruct-Action"
ln -sfn "${SHARED_ROOT}/shared/datasets/calvin_lerobot/calvin_abc_train_v3.0" "${PROJECT_ROOT}/data/calvin_lerobot/calvin_abc_train_v3.0"
ln -sfn "${SHARED_ROOT}/shared/datasets/calvin_original/task_D_D" "${PROJECT_ROOT}/data/calvin_original/task_D_D"
EOF
chmod +x "${SHARED_ROOT}/templates/link_member_project.sh"

cp "${PROJECT_ROOT}/STARVLA_SETUP.md" "${SHARED_ROOT}/shared/docs/STARVLA_SETUP.md" 2>/dev/null || true
cp "${STARVLA_ROOT}/STARVLA_env_check.txt" "${SHARED_ROOT}/shared/env/STARVLA_env_check.txt" 2>/dev/null || true
cp "${STARVLA_ROOT}/STARVLA_pip_freeze.txt" "${SHARED_ROOT}/shared/env/STARVLA_pip_freeze.txt" 2>/dev/null || true
cp "${STARVLA_ROOT}/STARVLA_pip_list.txt" "${SHARED_ROOT}/shared/env/STARVLA_pip_list.txt" 2>/dev/null || true

write_file_if_changed "${SHARED_ROOT}/shared/checkpoints/wmh_trained/README.md" <<'EOF'
# WMH-Trained Checkpoints Only

Only put checkpoints trained by this team/project here.

For each run, keep:

- config.yaml
- config.full.yaml
- dataset_statistics.json
- summary.jsonl if available
- checkpoints/steps_*_pytorch_model.pt or model.safetensors

Do not place upstream action-trained checkpoints here.
EOF

if [[ "${LINK_CURRENT_PROJECT:-0}" == "1" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT}" SHARED_ROOT="${SHARED_ROOT}" \
    "${SHARED_ROOT}/templates/link_member_project.sh"
fi

echo "Initialized ${SHARED_ROOT}"
echo "Members: ${normalized_members}"
echo "To link a member project:"
echo "  PROJECT_ROOT=/path/to/member/project bash ${SHARED_ROOT}/templates/link_member_project.sh"
