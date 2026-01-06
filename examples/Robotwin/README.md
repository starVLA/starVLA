# 🚀 RoboTwin 2.0 Evaluation

This document provides instructions for reproducing our **experimental results** with [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin).  
The evaluation process consists of two main parts:  

1. Setting up the `robotwin` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `robotwin` environments.  

We have verified that this workflow runs successfully on **NVIDIA 4090** GPUs.  

# Results


<details open>
<summary><b>RoboTwin 2.0 Benchmark Results over 48 Tasks </b></summary>


| Task Name | RDT Easy | RDT Hard | Pi0 Easy | Pi0 Hard | ACT Easy | ACT Hard | DP Easy | DP Hard | DP3 Easy | DP3 Hard | Qwen3OFT Easy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Adjust Bottle | 81 | 75 | 90 | 56 | 97 | 23 | 97 | 0 | 99 | 3 | 96 |
| Beat Block Hammer | 77 | 37 | 43 | 21 | 56 | 3 | 42 | 0 | 72 | 8 | 58 |
| Blocks Ranking RGB | 3 | 0 | 19 | 5 | 1 | 0 | 0 | 0 | 3 | 0 | 45 |
| Blocks Ranking Size | 0 | 0 | 7 | 1 | 0 | 0 | 1 | 0 | 2 | 0 | 27 |
| Click Alarmclock | 61 | 12 | 63 | 11 | 32 | 4 | 61 | 5 | 77 | 14 | 91 |
| Click Bell | 80 | 9 | 44 | 3 | 58 | 3 | 54 | 0 | 90 | 0 | 94 |
| Dump Bin Bigbin | 64 | 32 | 83 | 24 | 68 | 1 | 49 | 0 | 85 | 53 | 68 |
| Grab Roller | 74 | 43 | 96 | 80 | 94 | 25 | 98 | 0 | 98 | 2 | 93 |
| Handover Block | 45 | 14 | 45 | 8 | 42 | 0 | 10 | 0 | 70 | 0 | 0 |
| Handover Mic | 90 | 31 | 98 | 13 | 85 | 0 | 53 | 0 | 100 | 3 | 39 |
| Hanging Mug | 23 | 16 | 11 | 3 | 7 | 0 | 8 | 0 | 17 | 1 | 15 |
| Lift Pot | 72 | 9 | 84 | 36 | 88 | 0 | 39 | 0 | 97 | 0 | 0 |
| Move Can Pot | 25 | 12 | 58 | 21 | 22 | 4 | 39 | 0 | 70 | 6 | 50 |
| Move Pillbottle Pad | 8 | 0 | 21 | 1 | 0 | 0 | 1 | 0 | 41 | 0 | 54 |
| Move Playingcard Away | 43 | 11 | 53 | 22 | 36 | 0 | 47 | 0 | 68 | 3 | 69 |
| Move Stapler Pad | 2 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 12 | 0 | 12 |
| Open Laptop | 59 | 32 | 85 | 46 | 56 | 0 | 49 | 0 | 82 | 7 | 31 |
| Open Microwave | 37 | 20 | 80 | 50 | 86 | 0 | 5 | 0 | 61 | 22 | -- |
| Pick Diverse Bottles | 2 | 0 | 27 | 6 | 7 | 0 | 6 | 0 | 52 | 1 | 30 |
| Pick Dual Bottles | 42 | 13 | 57 | 12 | 31 | 0 | 24 | 0 | 60 | 1 | 43 |
| Place A2B Left | 3 | 1 | 31 | 1 | 1 | 0 | 2 | 0 | 46 | 2 | 20 |
| Place A2B Right | 1 | 1 | 27 | 6 | 0 | 0 | 13 | 0 | 49 | 0 | 22 |
| Place Bread Basket | 10 | 2 | 17 | 4 | 6 | 0 | 14 | 0 | 26 | 1 | 52 |
| Place Bread Skillet | 5 | 1 | 23 | 1 | 7 | 0 | 11 | 0 | 19 | 0 | 56 |
| Place Burger Fries | 50 | 27 | 80 | 4 | 49 | 0 | 72 | 0 | 72 | 18 | 96 |
| Place Can Basket | 19 | 6 | 41 | 5 | 1 | 0 | 18 | 0 | 67 | 2 | 63 |
| Place Cans Plasticbox | 6 | 5 | 34 | 2 | 16 | 0 | 40 | 0 | 48 | 3 | 81 |
| Place Container Plate | 78 | 17 | 88 | 45 | 72 | 1 | 41 | 0 | 86 | 1 | 99 |
| Place Dual Shoes | 4 | 4 | 15 | 0 | 9 | 0 | 8 | 0 | 13 | 0 | 28 |
| Place Empty Cup | 56 | 7 | 37 | 11 | 61 | 0 | 37 | 0 | 65 | 1 | 72 |
| Place Fan | 12 | 2 | 20 | 10 | 1 | 0 | 3 | 0 | 36 | 1 | 28 |
| Place Mouse Pad | 1 | 0 | 7 | 1 | 0 | 0 | 0 | 0 | 4 | 1 | 9 |
| Place Object Basket | 33 | 17 | 16 | 2 | 15 | 0 | 15 | 0 | 65 | 0 | 40 |
| Place Object Scale | 1 | 0 | 10 | 0 | 0 | 0 | 1 | 0 | 15 | 0 | 19 |
| Place Object Stand | 15 | 5 | 36 | 11 | 1 | 0 | 22 | 0 | 60 | 0 | 48 |
| Place Phone Stand | 15 | 6 | 35 | 7 | 2 | 0 | 13 | 0 | 44 | 2 | 24 |
| Place Shoe | 35 | 7 | 28 | 6 | 5 | 0 | 23 | 0 | 58 | 2 | 63 |
| Press Stapler | 41 | 24 | 62 | 29 | 31 | 6 | 6 | 0 | 69 | 3 | 60 |
| Put Bottles Dustbin | 21 | 4 | 54 | 13 | 27 | 1 | 22 | 0 | 60 | 21 | -- |
| Put Object Cabinet | 33 | 18 | 68 | 18 | 15 | 0 | 42 | 0 | 72 | 1 | 35 |
| Rotate QRcode | 50 | 5 | 68 | 15 | 1 | 0 | 13 | 0 | 74 | 1 | 50 |
| Scan Object | 4 | 1 | 18 | 1 | 2 | 0 | 9 | 0 | 31 | 1 | 13 |
| Shake Bottle Horizontally | 84 | 51 | 99 | 51 | 63 | 4 | 59 | 18 | 100 | 25 | 98 |
| Shake Bottle | 74 | 45 | 97 | 60 | 74 | 10 | 65 | 8 | 98 | 19 | 98 |
| Stack Blocks Three | 2 | 0 | 17 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 41 |
| Stack Blocks Two | 21 | 2 | 42 | 1 | 25 | 0 | 7 | 0 | 24 | 0 | 83 |
| Stack Bowls Three | 51 | 17 | 66 | 24 | 48 | 0 | 63 | 0 | 57 | 5 | 62 |
| Stack Bowls Two | 76 | 30 | 91 | 41 | 82 | 0 | 61 | 0 | 83 | 6 | 90 |
| Stamp Seal | 1 | 0 | 3 | 4 | 2 | 0 | 2 | 0 | 18 | 0 | 27 |
| Turn Switch | 35 | 15 | 27 | 23 | 5 | 2 | 36 | 1 | 46 | 8 | 26 |
| **Average** | **34.50** | **13.72** | **46.42** | **16.34** | **29.74** | **1.74** | **28.04** | **0.64** | **55.24** | **4.96** | **50.38** |

*Note: All 50 tasks are trained within a single model, using 50 demonstrations per task (50×50 total demonstrations). Checkpoints can be downloaded at [Qwen3-VL-OFT-Robotwin2](https://huggingface.co/StarVLA/Qwen3-VL-OFT-Robotwin2)*.

</details>


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
