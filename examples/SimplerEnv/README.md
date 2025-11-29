This document provides instructions for reproducing our **experimental results** with SimplerEnv.  


The evaluation process consists of two main parts:  

1. Setting up the `simpler_env` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `simpler_env` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.  

## 📦 1. Environment Setup

To set up the environment, please first follow the official [SimplerEnv repository](https://github.com/simpler-env/SimplerEnv) to install the base `simpler_env` environment. 

Afterwards, inside the `simpler_env` environment, install the following dependencies:  

```bash
conda activate simpler_env
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

⚠️ **Common Issues**
When testing SimplerEnv on NVIDIA A100, you may encounter the following error:
`libvulkan.so.1: cannot open shared object file: No such file or directory`
You can refer to this link to fix: [Installation Guide – Vulkan Section](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)


## 🔧 Verification Your SimplerEnv
We provide a minimal environment verification script:

```bash
python examples/SimplerEnv/test_your_simplerEnv.py
```

If you see the "✅ Env built successfully" message, it means SimplerEnv is installed correctly and ready to use.


---


## 🚀 2. Eval SimplerEnv

The evaluation should be run **from the repository root** using **two separate terminals/machines**, one for each environment:  

- **starVLA environment**: runs the policy inference server.  
- **simpler_env environment**: runs the simulation eval code.  



### Step 0. Download offical checkpoint

1) Download the checkpoint:[Qwen3VL-GR00T-Bridge-RT-1](https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1)



### Step 1. Start the server (starVLA environment)

In the first terminal, activate the `starVLA` conda environment and run:  

```bash
bash examples/SimplerEnv/eval_files/run_policy_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in  
`run_policy_server.sh`  

---


### Step 2. Start the simulation (simpler_env environment)

In the second terminal, activate the `simpler_env` conda environment and run:  

```bash
export MODEL_PATH=.../checkpoints/steps_50000_pytorch_model.pt # for read normonization json and get name to save video under ckpt dir
bash examples/SimplerEnv/start_simpler_env.sh ${MODEL_PATH} 
```
This script will automatically launch the WidowX Robot evaluation tasks, reproducing the benchmark results reported above.

⚠️ **Note:** Please ensure that you specify the correct `SimplerEnv_PATH`in 
`start_simpler_env.sh`  



We also provide a parallel evaluation script:

```bash
check_pt=StarVLA/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt
bash examples/SimplerEnv/star_bridge_parall_eval.sh ${check_pt}
```

Before running star_bridge.sh, set the following three paths:
- star_vla_python: Python interpreter for the StarVLA environment.
- sim_python: Python interpreter for the SimplerEnv environment.
- SimplerEnv_PATH: Local path to the SimplerEnv project.
Alternatively, edit these variables directly at the top of `star_bridge.sh`.

# 🚀 Training on OXE

## Data Preparation

1. Prepare the OXE data following the GR00T / Open-X Embodiment procedure (download + convert). Keep only the subsets you need (e.g. bridge / rt1, etc.).
2. Move `./train_files/modality.json` to your `${lerobot_data}/meta/modality.json`
3. YAML parameter snippet: (`examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml`)

```
datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: playground/Datasets/OXE_LEROBOT_DATASET
    data_mix: bridge_rt_1   # change or extend if you add more mixture
```


### Check Your Dataoader
We provide a simple way to check your dataloader. Make sure you can load batched data:

```bash
python starVLA/dataloader/lerobot_datasets.py --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
```

## Framework Preparation

Before running, you need to ensure that your framework can `forward` and `predict_action` using a fake data example.

Try the following command:

```bash
python starVLA/model/framework/QwenGR00T.py --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
```

Note: You can modify the following code snippet to align with your dataset:

```python
    # Generate a fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),  # action_chunk, action_dim
        "image": [image, image],  # two views
        "lang": "This is a fake for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }
```

## Training

Once everything is ready, use our provided script to start training:

```bash
bash ./examples/SimplerEnv/train_files/run_oxe_train.sh
```

⚠️ **Note:** Ensure that the script explicitly uses the validated config path in `run_lerobot_datasets.sh`. If not already passed, add the `--config_yaml` argument.


