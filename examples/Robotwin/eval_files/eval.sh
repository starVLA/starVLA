#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: bash examples/Robotwin/eval_files/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <policy_ckpt_path> [policy_port] [policy_host]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ROBOTWIN_PATH="${ROBOTWIN_PATH:-/mnt/data/gaoning/code_repos/RoboTwin}"
if [[ ! -d "${ROBOTWIN_PATH}" ]]; then
    echo "ROBOTWIN_PATH does not exist: ${ROBOTWIN_PATH}" >&2
    exit 1
fi

robotwin_eval_script="${ROBOTWIN_PATH}/script/eval_policy.py"
if [[ ! -f "${robotwin_eval_script}" ]]; then
    echo "RoboTwin eval entry does not exist: ${robotwin_eval_script}" >&2
    exit 1
fi

patch_check_tool=""
if command -v rg >/dev/null 2>&1; then
    patch_check_tool="rg"
    patch_check_cmd=(rg -q "policy_ckpt_path" "${robotwin_eval_script}")
elif command -v grep >/dev/null 2>&1; then
    patch_check_tool="grep"
    patch_check_cmd=(grep -q "policy_ckpt_path" "${robotwin_eval_script}")
else
    echo "Neither rg nor grep is available, so the RoboTwin patch check cannot run." >&2
    exit 1
fi

if ! "${patch_check_cmd[@]}"; then
    echo "Your third-party RoboTwin checkout is missing the required policy_ckpt_path patch: ${robotwin_eval_script}" >&2
    echo "Patch check used: ${patch_check_tool}" >&2
    echo "Apply the documented patch in your own RoboTwin repo; see examples/Robotwin/README.md." >&2
    exit 1
fi

policy_name="${ROBOTWIN_POLICY_NAME:-model2robotwin_interface}"
task_name="$1"
task_config="$2"
ckpt_setting="${3:-starvla_demo}"
seed="${4:-0}"
gpu_id="${5:-0}"
policy_ckpt_path="$6"
policy_port="${7:-${ROBOTWIN_POLICY_PORT:-5694}}"
policy_host="${8:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
robotwin_python="${ROBOTWIN_PYTHON:-python}"
deploy_policy_template="${DEPLOY_POLICY_TEMPLATE_PATH:-${SCRIPT_DIR}/deploy_policy.yml}"

if [[ ! -f "${deploy_policy_template}" ]]; then
    echo "Deploy policy template does not exist: ${deploy_policy_template}" >&2
    exit 1
fi

runtime_deploy_policy="$(mktemp "${TMPDIR:-/tmp}/robotwin_deploy_policy.XXXXXX.yml")"
cleanup() {
    rm -f "${runtime_deploy_policy}"
}
trap cleanup EXIT

sed \
    -e "s/^host:.*/host: \"${policy_host}\"/" \
    -e "s/^port:.*/port: ${policy_port}/" \
    "${deploy_policy_template}" > "${runtime_deploy_policy}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

EVAL_FILES_PATH="${SCRIPT_DIR}"
STARVLA_PATH="${REPO_ROOT}"

export PYTHONPATH="${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export PYTHONPATH="${STARVLA_PATH}:${PYTHONPATH}"
export PYTHONPATH="${EVAL_FILES_PATH}:${PYTHONPATH}"

cd "${ROBOTWIN_PATH}"

echo "PYTHONPATH: ${PYTHONPATH}"
echo "task_name: ${task_name}"
echo "task_config: ${task_config}"
echo "ckpt_setting: ${ckpt_setting}"
echo "policy_port: ${policy_port}"

PYTHONWARNINGS=ignore::UserWarning \
"${robotwin_python}" script/eval_policy.py --config "${runtime_deploy_policy}" \
    --policy_ckpt_path "${policy_ckpt_path}" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}"
