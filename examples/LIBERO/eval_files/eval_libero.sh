#!/bin/bash
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/home/jye624/Projcets/starVLA

cd ${STARVLA_DIR}
# === Checkpoint ===
CKPT=${STARVLA_DIR}/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt

cd /root/starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/root/starVLA/LIBERO-master
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/root/miniconda3/envs/libero/bin/python
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find websocket tools from repo root

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl


host="127.0.0.1"
base_port=6694
unnorm_key="franka"
your_ckpt=/root/model/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
unset DEBUG

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}



task_suite_name=libero_spatial
num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"


${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
