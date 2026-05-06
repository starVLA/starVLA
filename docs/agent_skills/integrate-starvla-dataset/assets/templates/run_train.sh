#!/usr/bin/env bash
# <<TODO_BENCH>> — launch script (smoke-test profile by default).
# Copy to: examples/<<TODO_BENCH>>/train_files/run_<<TODO_BENCH>>_train.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."   # always run from repo root

###########################################################################################
# === Edit these for your environment ===
Framework_name=<<TODO_FRAMEWORK>>          # QwenOFT | QwenFAST | QwenPI | QwenPI_v3 | QwenGR00T
freeze_module_list=''                      # e.g. 'qwen_vl' to freeze the VLM backbone
config_yaml=./examples/<<TODO_BENCH>>/train_files/starvla_<<TODO_FRAMEWORK>>_<<TODO_BENCH>>.yaml
data_root_dir=playground/Datasets/<<TODO_DATASET_PARENT_DIR>>
data_mix=<<TODO_MIXTURE_NAME>>
run_root_dir=./results/Checkpoints
run_id=starvla_<<TODO_FRAMEWORK>>_<<TODO_BENCH>>_smoke
num_processes=2                            # number of GPUs
###########################################################################################

mkdir -p "${run_root_dir}/${run_id}"
cp "$0" "${run_root_dir}/${run_id}/$(basename "$0")"   # snapshot the launcher
cp "${config_yaml}" "${run_root_dir}/${run_id}/$(basename "${config_yaml}")"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --framework.name "${Framework_name}" \
    --datasets.vla_data.data_root_dir "${data_root_dir}" \
    --datasets.vla_data.data_mix "${data_mix}" \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    ${freeze_module_list:+--trainer.freeze_modules "${freeze_module_list}"}
