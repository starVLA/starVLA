# 🚀 LIBERO-plus zero shot Evaluation

This document provides instructions for reproducing our **zero shot experimental results** with LIBERO-plus.  
The evaluation process consists of two main parts:  

1. Setting up the `LIBERO-plus` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `LIBERO-plus` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.  

---


## ⬇️ 0. Download Checkpoints

We use models trained exclusively on LIBERO to perform zero-shot evaluation on LIBERO-plus.: [🤗 StarVLA/bench-libero](https://huggingface.co/collections/StarVLA/bench-libero). Their corresponding results on LIBERO-plus are summarized in the table below.

### 📊 Experimental Results

| Model               | Camera | Robot | Language|Light|Background|Noise|Layout|Total|
|---------------------|-------|--------|---------|--------|------|-------|-------|-------|
| OpenVLA  | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT  | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| ABot-M0  | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | 80.5 |
| **Qwen2.5-VL-FAST**   | 19.6 | 27.6 | 74.5 | 75.2 | 71.0 | 27.4 | 62.7 | 48.9 |
| **Qwen2.5-VL-GR00T**   | 32.9 | 50.8 | 86.3 | 96.2 | 85.7 | 62.0 | 73.6 | 66.4 |
| **Qwen2.5-VL-OFT**   | 34.4 | 63.7 | 82.1 | 86.9 | 88.8 | 53.3 | 74.6 | 67.2 |
| **Qwen3-VL-OFT**   | 47.0 | 60.1 | 87.0 | 96.3 | 95.3 | 73.1 | 79.2 | 75.0 |
| **Qwen3-VL-PI**   | 64.3 | 57.2 | 82.8 | 94.2 | 94.0 | 79.6 | 78.2 | 77.0 |



---


## 📦 1. Environment Setup

To set up the environment, please first follow the official [LIBERO-plus repository](https://github.com/sylvestf/LIBERO-plus) to install the base `LIBERO-plus` environment.  



Afterwards, inside the `LIBERO-plus` environment, install the following dependencies:  

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

---

## 🚀 2. Evaluation Workflow

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **starVLA environment**: runs the inference server.  
- **LIBERO-plus environment**: runs the simulation.  

### Step 1. Start the server (starVLA environment)

In the first terminal, activate the `starVLA` conda environment and run:  

```bash
bash examples/LIBERO-plus/eval_files/run_policy_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `examples/LIBERO-plus/eval_files/run_policy_server.sh`  


---

### Step 2. Start the simulation (LIBERO-plus environment)

In the second terminal, activate the `LIBERO-plus` conda environment and run:  

```bash
bash examples/LIBERO-plus/eval_files/eval_libero.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `eval_libero.sh` to load action unnormalization stats. 

Also ensure the environment variables at the top of `eval_libero.sh` are correctly set.


---

⚠️ **Note:** Since LIBERO-plus has 10,030 tasks, completing all the evaluations will take an extremely long time. It is recommended to run multiple model instances in parallel for the evaluations. We provide code and scripts for parallel testing on cluster `./parallel_eval/run_nebula_libero_plus`. Please modify them to fit your own cluster.


