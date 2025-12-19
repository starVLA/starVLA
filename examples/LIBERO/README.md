# 🚀 LIBERO Evaluation

This document provides instructions for reproducing our **experimental results** with LIBERO.  
The evaluation process consists of two main parts:  

1. Setting up the `LIBERO` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `LIBERO` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.  

---


## ⬇️ 0. Download Checkpoints


We provide a collection of pretrained checkpoints on Hugging Face to make community evaluation easier: [🤗 StarVLA/bench-libero](https://huggingface.co/collections/StarVLA/bench-libero). Their corresponding results on LIBERO are summarized in the table below.

### 📊 Experimental Results

| Model               | Steps | Epochs | Spatial | Object | Goal | Long  | Avg   |
|---------------------|-------|--------|---------|--------|------|-------|-------|
| $\pi_0$+FAST | -     | -      | 96.4    | 96.8   | 88.6 | 60.2  | 85.5  |
| OpenVLA-OFT | 175K  | 223    | 97.6    | 98.4   | 97.9 | 94.5  | 97.1  |
| $\pi_0$             | -     | -      | 96.8    | 98.8   | 95.8 | 85.2  | 94.1  |
| GR00T-N1.5 | 20K   | 203    | 92.0    | 92.0   | 86.0 | 76.0  | 86.5  |
| **Qwen2.5-VL-FAST** | 30K   | 9.54   | 97.3    | 97.2   | 96.1 | 90.2  | 95.2  |
| **Qwen2.5-VL-OFT**  | 30K   | 9.54   | 97.4    | 98.0   | 96.8 | 92.0  | 96.1  |
| **Qwen2.5-VL-GR00T**| 30K   | 9.54   | 97.8    | 98.2   | 94.6 | 90.8  | 95.4  |
| **Qwen3-VL-FAST**   | 30K   | 9.54   | 97.3    | 97.4   | 96.3 | 90.6  | 95.4  |
| **Qwen3-VL-OFT**    | 30K   | 9.54   | 97.8    | 98.6   | 96.2 | 93.8  | 96.6  |
| **Qwen3-VL-GR00T**  | 30K   | 9.54   | 97.8    | 98.8   | 97.4 | 92.0  | 96.5  |

We train one policy for all 4 suites. All
scores are averaged over 500 trials for each task suite (10 tasks × 50 episodes).

---


## 📦 1. Environment Setup

To set up the environment, please first follow the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) to install the base `LIBERO` environment.  

⚠️ **Common issue:** LIBERO defaults to Python 3.8, but the syntax updates between 3.8 and 3.10 are substantial. **We verified that using Python 3.10 avoids many issues**. 


Afterwards, inside the `LIBERO` environment, install the following dependencies:  

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

---

## 🚀 2. Evaluation Workflow

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **starVLA environment**: runs the inference server.  
- **LIBERO environment**: runs the simulation.  

### Step 1. Start the server (starVLA environment)

In the first terminal, activate the `starVLA` conda environment and run:  

```bash
bash examples/LIBERO/eval_files/run_policy_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `examples/LIBERO/eval_files/run_policy_server.sh`  


---

### Step 2. Start the simulation (LIBERO environment)

In the second terminal, activate the `LIBERO` conda environment and run:  

```bash
bash examples/LIBERO/eval_files/eval_libero.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `eval_libero.sh` to load action unnormalization stats. 

Also ensure the environment variables at the top of `eval_libero.sh` are correctly set.

Finally, each result will also save a video for visualization, as shown below:

![Example](example.gif)

---


# 🚀 LIBERO Training

## 📦 Step 0: Download the training dataset
Download the datasets to the playground/Datasets/LEROBOT_LIBERO_DATA directory:
- [LIBERO-spatial](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot)
- [LIBERO-object](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot)
- [LIBERO-goal](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot)
- [LIBERO-10](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot)

And move `modality.json` to each `$LEROBOT_LIBERO_DATA/subset/meta/modality.json`.

You could quickly prepare these by running:
```bash
# Set DEST to the directory where you want to store the data
export DEST=/path/to/your/data/directory
bash examples/LIBERO/data_preparation.sh
```


## 🚀 Step1: Start Training

Most of the required training files have been organized in [train_files](train_files).  

Please run the following command to start training:

```bash
bash examples/LIBERO/train_files/run_libero_train.sh
```
⚠️ **Note:** Please ensure that you specify the correct path in `examples/LIBERO/train_files/run_libero_train.sh`

