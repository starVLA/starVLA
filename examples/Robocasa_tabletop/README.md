# 🚀 Robocasa-GR1-Tabletop-Tasks Evaluation

This document provides instructions for reproducing our **experimental results** with [robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks).  
The evaluation process consists of two main parts:  

1. Setting up the `robocasa` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `robocasa` environments.  

We have verified that this workflow runs successfully on **NVIDIA A100** GPUs.  

---
## 📊 Experimental Results
| Environment                                                                 | GR00T-N1.5 | Qwen3VL-GR00T | Qwen3VL-Pi | Qwen3VL-oft | Qwen3VL-FAST |
|-----------------------------------------------------------------------------|--------------|-------------|-----------|------------|-------------|
| gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env             | 0.38         | 0.18        |           |            |             |
| gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env       | 0.32         | 0.24        |           |            |             |
| gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env         | 0.60         | 0.52        |           |            |             |
| gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env         | 0.54         | 0.46        |           |            |             |
| gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env           | 0.38         | 0.22        |           |            |             |
| gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env             | 0.50         | 0.16        |           |            |             |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env | 0.38 | 0.06 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 0.46 | 0.24 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env | 0.58 | 0.42 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env | 0.62 | 0.22 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env | 0.28 | 0.38 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env | 0.30 | 0.18 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env | 0.60 | 0.34 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 0.56 | 0.26 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env | 0.36 | 0.12 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env | 0.58 | 0.44 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 0.44 | 0.10 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env | 0.60 | 0.34 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 0.64 | 0.22 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 0.52 | 0.26 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 0.48 | 0.12 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env | 0.60 | 0.16 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env | 0.52 | 0.30 |           |            |             |
| gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env | 0.32 | 0.06 |           |            |             |
| **Average** | **0.48** | **0.25** |           |            |             |

All the above tasks are evaluated at 50 rollouts each.

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
bash scripts/run_scripts/Robocasa/run_lerobot_datasets_qwenGR00T.sh
```

# Evaluation
## ⬇️ 0. Download Checkpoints
First, download the checkpoints from 
- [Qwen3VL-GR00T](https://huggingface.co/StarVLA/Qwen3-VL-GR00T-Robocasa-gr1)


## 📦 1. Environment Setup

To set up the environment, please first follow the [official RoboCasa installation guide](https://github.com/robocasa/robocasa-gr1-tabletop-tasks?tab=readme-ov-file#getting-started) to install the base `robocasa-gr1-tabletop-tasks` environment.  

---

## 🚀 2. Evaluation Workflow

### Step 1. Start the server (starVLA environment)

In the first terminal, activate the `starVLA` conda environment and run:  

```bash
python deployment/model_server/server_policy.py \
        --ckpt_path ${your_ckpt} \
        --port ${port} \
        --use_bf16
```

---

### Step 2. Start the simulation (robocasa environment)

In the second terminal, activate the `robocasa` conda environment and run:  

```bash
python examples/Robocasa_tabletop/simulation_env.py \
        --args.env_name ${env_name} \
        --args.port ${port} \
        --args.n_episodes 50 \
        --args.n_envs 1 \
        --args.max_episode_steps 720 \
        --args.n_action_steps 12 \
        --args.video_out_path ${video_out_path} \
        --args.pretrained_path ${your_ckpt}
```

If you have more GPU, you can use the batch evaluation script:
```bash
bash examples/Robocasa_tabletop/batch_eval_args.sh
```


