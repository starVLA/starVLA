#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: bash examples/DOMINO/eval_files/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <policy_ckpt_path> [policy_port] [policy_host]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DOMINO_PATH="${DOMINO_PATH:-/path/to/DOMINO}"
if [[ ! -d "${DOMINO_PATH}" ]]; then
    echo "DOMINO_PATH does not exist: ${DOMINO_PATH}" >&2
    echo "Please clone https://github.com/h-embodvis/DOMINO and set DOMINO_PATH." >&2
    exit 1
fi

domino_eval_script="${DOMINO_PATH}/script/eval_policy.py"
if [[ ! -f "${domino_eval_script}" ]]; then
    echo "DOMINO eval entry does not exist: ${domino_eval_script}" >&2
    exit 1
fi

policy_name="${DOMINO_POLICY_NAME:-model2robotwin_interface}"
task_name="$1"
task_config="$2"
ckpt_setting="${3:-starvla_demo}"
seed="${4:-0}"
gpu_id="${5:-0}"
policy_ckpt_path="$6"
policy_port="${7:-${DOMINO_POLICY_PORT:-5694}}"
policy_host="${8:-${DOMINO_POLICY_HOST:-127.0.0.1}}"
domino_python="${DOMINO_PYTHON:-python}"
deploy_policy_template="${DEPLOY_POLICY_TEMPLATE_PATH:-${SCRIPT_DIR}/deploy_policy.yml}"

if [[ ! -f "${deploy_policy_template}" ]]; then
    echo "Deploy policy template does not exist: ${deploy_policy_template}" >&2
    exit 1
fi

runtime_deploy_policy="$(mktemp "${TMPDIR:-/tmp}/domino_deploy_policy.XXXXXX.yml")"
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

export PYTHONPATH="${DOMINO_PATH}:${PYTHONPATH:-}"
export PYTHONPATH="${STARVLA_PATH}:${PYTHONPATH}"
export PYTHONPATH="${EVAL_FILES_PATH}:${PYTHONPATH}"

cd "${DOMINO_PATH}"

echo "PYTHONPATH: ${PYTHONPATH}"
echo "task_name: ${task_name}"
echo "task_config: ${task_config}"
echo "ckpt_setting: ${ckpt_setting}"
echo "policy_port: ${policy_port}"

# DOMINO's script/eval_policy.py accepts extra key/value pairs via --overrides
# and merges them into the config dict. We forward policy_ckpt_path this way,
# so no patch to the upstream DOMINO repo is required.
PYTHONWARNINGS=ignore::UserWarning \
"${domino_python}" script/eval_policy.py --config "${runtime_deploy_policy}" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --policy_ckpt_path "${policy_ckpt_path}"
