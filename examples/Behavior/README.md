<!-- # Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Zixuan Wang / HKUST] in [2025]. -->
# 🚧Under construction
This document provides instructions to run our framework with the [BEHAVIOR-1K Benchmark](https://github.com/StanfordVL/BEHAVIOR-1K). We follow the structure of [2025 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/index.html) so that you can train and evaluate on the 50 full-length household tasks. 


The evaluation process consists of two main parts:  

1. Setting up the `behavior` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `behavior` environments.  

Note that to run evaluation on Behavior benchmark, you should **not** use GPUs without RT Cores (A100, H100). Otherwise you may encounter problems of Segmentation fault or low resolution. See [this](https://github.com/StanfordVL/BEHAVIOR-1K/issues/1872#issuecomment-3455002820) and [this](https://github.com/StanfordVL/BEHAVIOR-1K/issues/1875#issuecomment-3444246495) for discussion.


## 📦 1. Environment Setup
To set up the conda enviroment for `behavior` environment, use the following command:

```
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
conda create -n behavior python=3.10 -y
conda activate behavior
cd BEHAVIOR-1K
pip install "setuptools<=79"
./setup.sh --omnigibson --bddl --joylo --dataset
conda install -c conda-forge libglu
pip install rich omegaconf hydra-core msgpack websockets av pandas google-auth
```

Also in starVLA env:
```
pip install websockets
```

## 🚀 2. Eval in Behavior

Steps:
1) Download the checkpoint:``
2) Choose the script below according to your need

(A) parallel evaluation script. 

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/Behavior/start_parallel_eval.sh
```

Before running start_parallel_eval.sh, set the following three paths:
- `star_vla_python`: Python interpreter for the StarVLA environment.
- `sim_python`: Python interpreter for the Behavior environment.
- `TASKS_JSONL_PATH`: This file contains the task description downloaded from the [training dataset](https://huggingface.co/datasets/behavior-1k/2025-challenge-demos), we have included it with path examples/Behavior/tasks.jsonl
- `BEHAVIOR_ASSET_PATH`: Local path to the behavior asset path. The default path is in BEHAVIOR-1K/datasets after installing behavior with `./setup.sh --omnigibson --bddl --joylo --dataset`

(B) For the ease of debugging, you may also start the client (evaluation environment) and server (policy) in two separate terminal:

```bash
bash examples/Behavior/start_server.sh
bash examples/Behavior/start_client.sh
```
The above debugging files will conduct evaluation on train set

(C) To prevent memory overflow, we implemented another file `start_parallel_eval_per_task.sh`:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/Behavior/start_parallel_eval.sh
```
- The script will run evaluation for each task in `INSTANCE_NAMES` iteratively
- For each tasks, allocate all instances from `TEST_EVAL_INSTANCE_IDS` across GPUs 
- Wait for the previous task to finish, then procede to the next task

## 🔧 3. Notes

### Wrapper
1. RGBLowResWrapper: only use rgb as visual observation and camera resolutions of 224 * 224. Only using low-res RGB can help speed up the simulator and thus reduce evaluation time compared to the two other example wrappers. This wrapper is ok to use in standard track.
2. DefaultWrapper: wrapper with the default observation config used during data collection (rgb + depth + segmentation, 720p for head camera and 480p for wrist camera). This wrapper is ok to use in standard track, but evaluation will be considerably slower compared to RGBLowResWrapper.
3. RichObservationWrapper: this will load additional observation modalities, such as normal and flow, as well as privileged task information. This wrapper can only be used in privileged information track.


### Action Dim
BEHAVIOR has action dim = 23

```
"R1Pro": {
    "base": np.s_[0:3],        # Indices 0-2
    "torso": np.s_[3:7],       # Indices 3-6  
    "left_arm": np.s_[7:14],   # Indices 7-13
    "left_gripper": np.s_[14:15], # Index 14
    "right_arm": np.s_[15:22], # Indices 15-21
    "right_gripper": np.s_[22:23], # Index 22
}
```

### Video Saving:
The video will be saved in the format of {task_name}\_{idx}\_{epi}.mp4, where idx is the instance number, epi is the episode number

### Common Bugs
Segmentation fault (core dumped): a likely reason is vulkan is not successfully installed. Check this [link](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)

ImportError: libGL.so.1: cannot open shared object file: No such file or directory:  
`apt-get install ffmpeg libsm6 libxext6  -y`
