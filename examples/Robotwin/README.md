# 🚀 RoboTwin 2.0 Evaluation

This document provides instructions for reproducing our **experimental results** with [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin).  
The evaluation process consists of two main parts:  

1. Setting up the `robotwin` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `robotwin` environments.  

We have verified that this workflow runs successfully on **NVIDIA 4090** GPUs.  

# Evaluation

## 📦 1. Environment Setup

To set up the environment, please first follow the [official RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) to install the base `robotwin` environment.  

than pip install additional requirements

```bash
pip install -r examples/Robotwin/eval_files/requirements.txt
```

and edit `ROBOTWIN_PATH` in `examples/Robotwin/eval_files/eval.sh`.

## 🚀 2. Evaluation Workflow

### Step 1. Start the server (starVLA environment)

In the first terminal, activate the `starVLA` conda environment and run:  

```bash
python examples/Robotwin/eval_files/run_policy_server.sh
```

Edit your checkpoint path in `examples/Robotwin/eval_files/deploy_policy.yml` and `examples/Robotwin/eval_files/run_policy_server.sh`.

---

### Step 2. Start the simulation (robotwin environment)

In the second terminal, activate the `robotwin` conda environment and run:  

```bash
conda activate robotwin
cd examples/Robotwin/eval_files
bash eval.sh task_name demo_clean my_test_v1 0 0
```

all tasks in RoboTwin 2.0 include:

```txt
adjust_bottle
beat_block_hammer
blocks_ranking_rgb
blocks_ranking_size
click_alarmclock
click_bell
dump_bin_bigbin
grab_roller
handover_block
handover_mic
hanging_mug
lift_pot
move_can_pot
move_pillbottle_pad
move_playingcard_away
move_stapler_pad
open_laptop
open_microwave
pick_diverse_bottles
pick_dual_bottles
place_a2b_left
place_a2b_right
place_bread_basket
place_bread_skillet
place_burger_fries
place_can_basket
place_cans_plasticbox
place_container_plate
place_dual_shoes
place_empty_cup
place_fan
place_mouse_pad
place_object_basket
place_object_scale
place_object_stand
place_phone_stand
place_shoe
press_stapler
put_bottles_dustbin
put_object_cabinet
rotate_qrcode
scan_object
shake_bottle_horizontally
shake_bottle
stack_blocks_three
stack_blocks_two
stack_bowls_three
stack_bowls_two
stamp_seal
turn_switch
```

and all modes include `demo_clean` and `demo_randomized`.