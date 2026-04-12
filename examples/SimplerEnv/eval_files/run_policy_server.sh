

# Environment setup
cd /home/jye624/Projcets/starVLA
export star_vla_python=/home/jye624/.conda/envs/starVLA/bin/python
export sim_python=/home/jye624/.conda/envs/simpler_env/bin/python
export SimplerEnv_PATH=/project/vonneumann1/jye624/Projcets/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
export LD_LIBRARY_PATH=/home/jye624/.conda/envs/simpler_env/lib:${LD_LIBRARY_PATH}
port=6678 
gpu_id=2

your_ckpt=./playground/Pretrained_models/StarVLA/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt

#### build output directory #####
ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"


#### run server #####
CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"