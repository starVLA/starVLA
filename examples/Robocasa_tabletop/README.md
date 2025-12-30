# 🚀 Robocasa-GR1-Tabletop-Tasks Evaluation

This document provides instructions for reproducing our **experimental results** with [robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks).  
The evaluation process consists of two main parts:  

1. Setting up the `robocasa` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `robocasa` environments.  

We have verified that this workflow runs successfully on **NVIDIA A100** GPUs.  


# Evaluation

![Eval Videos](https://github.com/user-attachments/assets/a5ff9bdd-b47d-4eb0-95ac-c09556fb4b48)


## ⬇️ 0. Download Checkpoints
First, download the checkpoints from 
- [Qwen3VL-GR00T](https://huggingface.co/StarVLA/Qwen3-VL-GR00T-Robocasa-gr1)
- [Qwen3VL-OFT](https://huggingface.co/StarVLA/Qwen3-VL-OFT-Robocasa)

## 📦 1. Environment Setup

To set up the environment, please first follow the [official RoboCasa installation guide](https://github.com/robocasa/robocasa-gr1-tabletop-tasks?tab=readme-ov-file#getting-started) to install the base `robocasa-gr1-tabletop-tasks` environment.  

than pip soceket support

'''bash
pip install tyro
'''

---

## 🚀 2. Evaluation Workflow

### Step 1. Start the server (starVLA environment)

In the first terminal, activate the `starVLA` conda environment and run:  

```bash
python deployment/model_server/server_policy.py \
        --ckpt_path ${your_ckpt} \
        --port 5678 \
        --use_bf16
```

---

### Step 2. Start the simulation (robocasa environment)

In the second terminal, activate the `robocasa` conda environment and run:  

```bash
export PYTHONPATH=$(pwd):${PYTHONPATH}
your_ckpt=StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt

python examples/Robocasa_tabletop/eval_files/simulation_env.py\
   --args.env_name ${env_name} \
   --args.port 5678 \
   --args.n_episodes 50 \
   --args.n_envs 1 \
   --args.max_episode_steps 720 \
   --args.n_action_steps 12 \
   --args.video_out_path ${video_out_path} \
   --args.pretrained_path ${your_ckpt}
```


### Optional: Batch Evaluation

If you have more GPU, you can use the batch evaluation script:
```bash
bash examples/Robocasa_tabletop/batch_eval_args.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `batch_eval_args.sh`  

---
## 📊 Experimental Results

# RoboCasa GR1 Tabletop Tasks Evaluation Results

| Task | GR00T-N1.6 | Qwen3GR00T | Qwen3PI | Qwen3OFT | Qwen3FAST |
|------|------------|------------|---------|----------|-----------|
| **PnP Bottle To Cabinet Close** | 51.5 | 46.0 | 26.0 | **30.0** | 38.0 |
| **PnP Can To Drawer Close** | 13.0 | 80.0 | 62.0 | **76.0** | 44.0 |
| **PnP Cup To Drawer Close** | 8.5 | 54.0 | 42.0 | **44.0** | 56.0 |
| **PnP Milk To Microwave Close** | 14.0 | 48.0 | 50.0 | **44.0** | 44.0 |
| **PnP Potato To Microwave Close** | 41.5 | 28.0 | 42.0 | **32.0** | 14.0 |
| **PnP Wine To Cabinet Close** | 16.5 | 46.0 | 32.0 | **36.0** | 14.0 |
| | | | | | |
| **PnP Novel From Cuttingboard To Basket** | 58.0 | 48.0 | 40.0 | **50.0** | 54.0 |
| **PnP Novel From Cuttingboard To Cardboardbox** | 46.5 | 40.0 | 46.0 | **40.0** | 42.0 |
| **PnP Novel From Cuttingboard To Pan** | 68.5 | 68.0 | 60.0 | **70.0** | 58.0 |
| **PnP Novel From Cuttingboard To Pot** | 65.0 | 52.0 | 40.0 | **54.0** | 58.0 |
| **PnP Novel From Cuttingboard To Tieredbasket** | 46.5 | 56.0 | 44.0 | **38.0** | 40.0 |
| | | | | | |
| **PnP Novel From Placemat To Basket** | 58.5 | 42.0 | 44.0 | **32.0** | 36.0 |
| **PnP Novel From Placemat To Bowl** | 57.5 | 44.0 | 52.0 | **58.0** | 38.0 |
| **PnP Novel From Placemat To Plate** | 63.0 | 48.0 | 50.0 | **52.0** | 42.0 |
| **PnP Novel From Placemat To Tieredshelf** | 28.5 | 18.0 | 28.0 | **24.0** | 18.0 |
| | | | | | |
| **PnP Novel From Plate To Bowl** | 57.0 | 60.0 | 52.0 | **60.0** | 52.0 |
| **PnP Novel From Plate To Cardboardbox** | 43.5 | 50.0 | 40.0 | **50.0** | 30.0 |
| **PnP Novel From Plate To Pan** | 51.0 | 54.0 | 36.0 | **66.0** | 48.0 |
| **PnP Novel From Plate To Plate** | 78.7 | 70.0 | 48.0 | **68.0** | 50.0 |
| | | | | | |
| **PnP Novel From Tray To Cardboardbox** | 51.5 | 38.0 | 34.0 | **44.0** | 28.0 |
| **PnP Novel From Tray To Plate** | 71.0 | 56.0 | 64.0 | **56.0** | 34.0 |
| **PnP Novel From Tray To Pot** | 64.5 | 50.0 | 44.0 | **62.0** | 46.0 |
| **PnP Novel From Tray To Tieredbasket** | 57.0 | 36.0 | 50.0 | **54.0** | 36.0 |
| **PnP Novel From Tray To Tieredshelf** | 31.5 | 16.0 | 28.0 | **30.0** | 16.0 |
| | | | | | |
| **Average** | **47.6** | **47.8** | **43.9** | **48.8** | **39.0** |

*Note: All values are success rates in percentage (%). A single model was trained for all 24 tasks. Results are reported over 50 rollouts per task.*

---


# 🚀 Reproduce Training Results
## 📦 Step0: Download the training dataset
Download the PhysicalAI-Robotics-GR00T-X-Embodiment-Sim directory datasets from [HuggingFace](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) to the playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim directory

To download only the relevant finetuning folders, you can refer [GR00T-N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/examples/RoboCasa#-1-dataset-preparation) repo's instruction. 
Or using the script download the *_1000 folders.

```bash
python examples/Robocasa_tabletop/download_gr00t_ft_data.py
```

## 🚀 Step1: Start Training
Different datasets can be selected by modifying the parameter `data_mix`, and the following script can be used to fine-tune the `*_1000` datasets:
```bash
bash examples/Robocasa_tabletop/train_files/run_robocasa.sh
```

