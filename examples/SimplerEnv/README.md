This document provides instructions for reproducing our **experimental results** with SimplerEnv.  


## 📦 1. Environment Setup

To set up the environment, please first follow the official [SimplerEnv repository](https://github.com/simpler-env/SimplerEnv) to install the base `simpler_env` environment. 

Afterwards, inside the `simpler_env` environment, install the following dependencies:  

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

⚠️ **Common Issues**
When testing SimplerEnv on NVIDIA A100, you may encounter the following error:
`libvulkan.so.1: cannot open shared object file: No such file or directory`
You can refer to this link to fix: [Installation Guide – Vulkan Section](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)


## 🔧 Verification Method
We provide a minimal environment verification script:

```bash
python examples/SimplerEnv/test_your_simplerEnv.py

```

If you see the "✅ Env built successfully" message, it means SimplerEnv is installed correctly and ready to use.


---


## 🚀 2. Eval SimplerEnv

Steps:
1) Download the checkpoint:[Qwen3VL-GR00T-Bridge-RT-1](https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1)

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




