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


| Task | GR00T-N1.6 | Qwen3GR00T | Qwen3PI | Qwen3OFT | Qwen3FAST |
|------|------------|------------|---------|----------|-----------|
| **PnPBottleToCabinetClose** | 51.5 | | | **30.0** | |
| **PnPCanToDrawerClose** | 13.0 | | | **76.0** | |
| **PnPCupToDrawerClose** | 8.5 | | | **44.0** | |
| **PnPMilkToMicrowaveClose** | 14.0 | | | **44.0** | |
| **PnPPotatoToMicrowaveClose** | 41.5 | | | **32.0** | |
| **PnPWineToCabinetClose** | 16.5 | | | **36.0** | |
| **PnPNovelFromCuttingboardToBasket** | 58.0 | | | **50.0** | |
| **PnPNovelFromCuttingboardToCardboardbox** | 46.5 | | | **40.0** | |
| **PnPNovelFromCuttingboardToPan** | 68.5 | | | **70.0** | |
| **PnPNovelFromCuttingboardToPot** | 65.0 | | | **54.0** | |
| **PnPNovelFromCuttingboardToTieredbasket** | 46.5 | | | **38.0** | |
| **PnPNovelFromPlacematToBasket** | 58.5 | | | **32.0** | |
| **PnPNovelFromPlacematToBowl** | 57.5 | | | **58.0** | |
| **PnPNovelFromPlacematToPlate** | 63.0 | | | **52.0** | |
| **PnPNovelFromPlacematToTieredshelf** | 28.5 | | | **24.0** | |
| **PnPNovelFromPlateToBowl** | 57.0 | | | **60.0** | |
| **PnPNovelFromPlateToCardboardbox** | 43.5 | | | **50.0** | |
| **PnPNovelFromPlateToPan** | 51.0 | | | **66.0** | |
| **PnPNovelFromPlateToPlate** | 78.7 | | | **68.0** | |
| **PnPNovelFromTrayToCardboardbox** | 51.5 | | | **44.0** | |
| **PnPNovelFromTrayToPlate** | 71.0 | | | **56.0** | |
| **PnPNovelFromTrayToPot** | 64.5 | | | **62.0** | |
| **PnPNovelFromTrayToTieredbasket** | 57.0 | | | **54.0** | |
| **PnPNovelFromTrayToTieredshelf** | 31.5 | | | **30.0** | |
| **Average** | **47.6** | | | **48.8** | |

**Note:** A single model was trained for all 24 tasks. Results are reported over 50 rollouts per task (average success rate with 250 rollouts: 48.97%).

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

