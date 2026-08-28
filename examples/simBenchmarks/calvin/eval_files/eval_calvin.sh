#!/usr/bin/env bash
set -euo pipefail

###########################################################################################
# === Please modify the following paths according to your environment ===
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
CALVIN_ROOT="${CALVIN_ROOT:-${STARVLA_DIR}/third_party/calvin}"
export PYTHONPATH="${STARVLA_DIR}:${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${PYTHONPATH:-}"
calvin_python="${calvin_python:-${STARVLA_DIR}/.venv-calvin-eval/bin/python}"

host="127.0.0.1"
base_port=5694
unnorm_key="franka"
your_ckpt=results/Checkpoints/0123_starvla_qwen3_calvin_task_D_D/checkpoints/steps_30000_pytorch_model.pt
dataset_path="${dataset_path:-/path/to/calvin/task_D_D/}"
calvin_config_path="${calvin_config_path:-${CALVIN_ROOT}/calvin_models/conf}"
eval_sequences_path="${eval_sequences_path:-${STARVLA_DIR}/examples/simBenchmarks/calvin/eval_files/eval_sequences.json}"

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

cd "${STARVLA_DIR}"
${calvin_python} ./examples/simBenchmarks/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path ${your_ckpt} \
    --args.unnorm-key ${unnorm_key} \
    --args.host "$host" \
    --args.port $base_port \
    --args.dataset_path "${dataset_path}" \
    --args.calvin_config_path "${calvin_config_path}" \
    --args.eval_sequences_path "${eval_sequences_path}" \
    --args.num_sequences 1000
