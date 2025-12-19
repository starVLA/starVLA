# Copyright
<!-- # Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Zixuan Wang / HKUST University] in [2025].

# Draft Version so far

# Enviroment Setup -->

# Enviroment Setup
```
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
conda create -n behavior python=3.10 -y
cd $PATH_TO_BEHAVIOR_1K
pip install "setuptools<=79"
pip install transformers==4.57.0
pip install qwen-vl-utils
pip install timm
pip install accelerate==1.5.2
./setup.sh --omnigibson --bddl --joylo --dataset
conda install -c conda-forge libglu
pip install websockets
<!-- pip install decord
pip install numpydantic
pip install albumentations
pip install pytorch3d -->
```

# Wrapper
1. RGBLowResWrapper: only use rgb as visual observation and camera resolutions of 224 * 224. Only using low-res RGB can help speed up the simulator and thus reduce evaluation time compared to the two other example wrappers. This wrapper is ok to use in standard track.
2. DefaultWrapper: wrapper with the default observation config used during data collection (rgb + depth + segmentation, 720p for head camera and 480p for wrist camera). This wrapper is ok to use in standard track, but evaluation will be considerably slower compared to RGBLowResWrapper.
3. RichObservationWrapper: this will load additional observation modalities, such as normal and flow, as well as privileged task information. This wrapper can only be used in privileged information track.

# Configure Robot Action Space
Currently not supported. 

# Action Dim
BEHAVIOR is having action dim = 23

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

# Video Saving:
The video will be saved in the format of {task_name}_{idx}_{epi}.mp4, where idx is the instance number, epi is the episode number

# TODO:
1. Clear all TODOs in the code
