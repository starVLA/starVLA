# 🚀 RoboTwin 2.0 Evaluation

This document provides instructions for reproducing our **experimental results** with [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin).  
The evaluation process consists of two main parts:  

1. Setting up the `robotwin` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `robotwin` environments.  

We have verified that this workflow runs successfully on **NVIDIA 4090** GPUs.  

# Results


<details open>
<summary><b>RoboTwin 2.0 Benchmark Results over 50 Tasks (data-scaling settings) </b></summary>

### Training Dataset

The model is trained using the official **RoboTwin 2.0 dataset**.

* Clean Demonstrations: 50 tasks × 50 trajectories per task
* Randomized Demonstrations: 50 tasks × 500 trajectories per task

| Task Name | StarVLA-OFT Easy | StarVLA-OFT Hard | π0 Easy | π0 Hard | π0.5 Easy | π0.5 Hard | X-VLA Easy | X-VLA Hard | Motus Easy | Motus Hard | lingbot-vla w/o depth Easy | lingbot-vla w/o depth Hard | lingbot-vla w/ depth Easy | lingbot-vla w/ depth Hard |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Adjust Bottle | 100 | 99 | 99 | 95 | 100 | 99 | 100 | 99 | 89 | 93 | 100 | 100 | 100 | 100 |
| Beat Block Hammer | 93 | 92 | 79 | 84 | 96 | 93 | 92 | 88 | 95 | 88 | 87 | 91 | 92 | 89 |
| Blocks Ranking RGB | 99 | 98 | 80 | 63 | 92 | 85 | 83 | 83 | 99 | 97 | 92 | 91 | 92 | 91 |
| Blocks Ranking Size | 79 | 80 | 14 | 5 | 49 | 26 | 67 | 74 | 75 | 63 | 66 | 73 | 76 | 70 |
| Click Alarmclock | 58 | 51 | 77 | 68 | 98 | 89 | 99 | 99 | 100 | 100 | 93 | 26 | 97 | 43 |
| Click Bell | 23 | 27 | 71 | 48 | 99 | 66 | 100 | 100 | 100 | 100 | 32 | 19 | 43 | 36 |
| Dump Bin Bigbin | 91 | 94 | 88 | 83 | 92 | 97 | 79 | 77 | 95 | 91 | 97 | 92 | 97 | 97 |
| Grab Roller | 100 | 100 | 98 | 94 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 100 | 100 |
| Handover Block | 97 | 93 | 47 | 31 | 66 | 57 | 73 | 37 | 86 | 73 | 80 | 83 | 83 | 95 |
| Handover Mic | 98 | 96 | 97 | 97 | 98 | 97 | 0 | 0 | 78 | 63 | 94 | 98 | 94 | 99 |
| Hanging Mug | 34 | 29 | 14 | 11 | 18 | 17 | 23 | 27 | 38 | 38 | 32 | 27 | 34 | 53 |
| Lift Pot | 100 | 100 | 80 | 72 | 96 | 85 | 99 | 100 | 96 | 99 | 100 | 99 | 100 | 100 |
| Move Can Pot | 91 | 90 | 68 | 48 | 51 | 55 | 89 | 86 | 34 | 74 | 79 | 84 | 89 | 87 |
| Move Pillbottle Pad | 98 | 100 | 67 | 46 | 84 | 61 | 73 | 71 | 93 | 96 | 93 | 94 | 92 | 90 |
| Move Playingcard Away | 100 | 98 | 74 | 65 | 96 | 84 | 93 | 98 | 100 | 96 | 96 | 99 | 98 | 100 |
| Move Stapler Pad | 74 | 90 | 41 | 24 | 56 | 42 | 78 | 73 | 83 | 85 | 74 | 49 | 74 | 48 |
| Open Laptop | 98 | 100 | 71 | 81 | 90 | 96 | 93 | 100 | 95 | 91 | 96 | 96 | 98 | 96 |
| Open Microwave | 28 | 39 | 4 | 32 | 34 | 77 | 79 | 71 | 95 | 91 | 91 | 75 | 91 | 92 |
| Pick Diverse Bottles | 87 | 86 | 69 | 31 | 81 | 71 | 58 | 36 | 90 | 91 | 79 | 86 | 88 | 85 |
| Pick Dual Bottles | 91 | 93 | 59 | 37 | 93 | 63 | 47 | 36 | 96 | 90 | 82 | 95 | 99 | 90 |
| Place A2B Left | 90 | 95 | 43 | 47 | 87 | 82 | 48 | 49 | 82 | 79 | 86 | 83 | 89 | 85 |
| Place A2B Right | 88 | 95 | 39 | 34 | 87 | 84 | 36 | 36 | 90 | 87 | 74 | 77 | 80 | 80 |
| Place Bread Basket | 91 | 78 | 62 | 46 | 77 | 64 | 81 | 71 | 91 | 94 | 92 | 93 | 95 | 93 |
| Place Bread Skillet | 89 | 80 | 66 | 49 | 85 | 66 | 77 | 67 | 86 | 83 | 90 | 89 | 90 | 92 |
| Place Burger Fries | 100 | 100 | 81 | 76 | 94 | 87 | 94 | 94 | 98 | 98 | 95 | 96 | 98 | 94 |
| Place Can Basket | 75 | 75 | 55 | 46 | 62 | 62 | 49 | 52 | 81 | 76 | 68 | 78 | 75 | 72 |
| Place Cans Plasticbox | 100 | 99 | 63 | 45 | 94 | 84 | 97 | 98 | 98 | 94 | 97 | 100 | 100 | 98 |
| Place Container Plate | 99 | 99 | 97 | 92 | 99 | 95 | 97 | 95 | 98 | 99 | 99 | 99 | 99 | 100 |
| Place Dual Shoes | 91 | 89 | 59 | 51 | 75 | 75 | 79 | 88 | 93 | 87 | 80 | 83 | 87 | 86 |
| Place Empty Cup | 100 | 100 | 91 | 85 | 100 | 99 | 100 | 98 | 99 | 98 | 100 | 100 | 100 | 100 |
| Place Fan | 94 | 95 | 66 | 71 | 87 | 85 | 80 | 75 | 91 | 87 | 91 | 79 | 92 | 87 |
| Place Mouse Pad | 87 | 94 | 20 | 20 | 60 | 39 | 70 | 70 | 66 | 68 | 82 | 78 | 86 | 79 |
| Place Object Basket | 93 | 94 | 67 | 70 | 80 | 76 | 44 | 39 | 81 | 87 | 90 | 91 | 90 | 88 |
| Place Object Scale | 93 | 93 | 57 | 52 | 86 | 80 | 52 | 74 | 88 | 85 | 84 | 90 | 90 | 88 |
| Place Object Stand | 99 | 98 | 82 | 68 | 91 | 85 | 86 | 88 | 98 | 97 | 97 | 93 | 93 | 88 |
| Place Phone Stand | 86 | 95 | 49 | 53 | 81 | 81 | 88 | 87 | 87 | 86 | 92 | 93 | 90 | 87 |
| Place Shoe | 96 | 100 | 76 | 76 | 92 | 93 | 96 | 95 | 99 | 97 | 99 | 94 | 99 | 99 |
| Press Stapler | 99 | 96 | 44 | 37 | 87 | 83 | 92 | 98 | 93 | 98 | 90 | 88 | 86 | 93 |
| Put Bottles Dustbin | 90 | 85 | 65 | 56 | 84 | 79 | 74 | 77 | 81 | 79 | 88 | 92 | 92 | 93 |
| Put Object Cabinet | 89 | 91 | 73 | 60 | 80 | 79 | 46 | 48 | 88 | 71 | 92 | 86 | 85 | 88 |
| Rotate QRcode | 88 | 90 | 74 | 70 | 89 | 87 | 34 | 33 | 89 | 73 | 93 | 84 | 86 | 82 |
| Scan Object | 94 | 91 | 55 | 42 | 72 | 65 | 14 | 36 | 67 | 66 | 91 | 97 | 92 | 96 |
| Shake Bottle Horizontally | 100 | 100 | 98 | 92 | 99 | 99 | 100 | 100 | 100 | 98 | 100 | 100 | 99 | 98 |
| Shake Bottle | 100 | 100 | 94 | 91 | 99 | 97 | 99 | 100 | 100 | 97 | 99 | 100 | 100 | 99 |
| Stack Blocks Three | 94 | 86 | 72 | 52 | 91 | 76 | 6 | 10 | 91 | 95 | 92 | 99 | 96 | 95 |
| Stack Blocks Two | 100 | 100 | 93 | 79 | 97 | 100 | 92 | 87 | 100 | 98 | 100 | 100 | 100 | 99 |
| Stack Bowls Three | 95 | 91 | 77 | 75 | 77 | 71 | 76 | 86 | 79 | 87 | 72 | 83 | 71 | 77 |
| Stack Bowls Two | 99 | 100 | 94 | 95 | 95 | 96 | 96 | 93 | 98 | 98 | 92 | 95 | 90 | 97 |
| Stamp Seal | 86 | 90 | 46 | 33 | 79 | 55 | 76 | 82 | 93 | 92 | 76 | 86 | 74 | 77 |
| Turn Switch | 65 | 62 | 41 | 42 | 62 | 54 | 40 | 61 | 84 | 78 | 61 | 65 | 67 | 63 |
| **Average** | **88.18** | **88.32** | **65.92** | **58.40** | **82.74** | **76.76** | **72.80** | **72.84** | **88.66** | **87.02** | **86.50** | **85.34** | **88.56** | **86.68** |

*Note: All 50 tasks are trained within a single model, using 50 clean and 500 randomized demonstrations per task for co-training. Checkpoints can be downloaded at [Qwen3-VL-OFT-Robotwin2-All](https://huggingface.co/StarVLA/Qwen3-VL-OFT-RoboTwin2-All)*.

</details>


<details close>
<summary><b>RoboTwin 2.0 Benchmark Results over 48 Tasks (50×50 setting) </b></summary>


| Task Name | RDT Easy | RDT Hard | Pi0 Easy | Pi0 Hard | ACT Easy | ACT Hard | DP Easy | DP Hard | DP3 Easy | DP3 Hard | StarVLA-OFT Easy |
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


<details open>
<summary><b>RoboTwin 2.0 data-scaling settings </b></summary>

### Training Dataset

The model is trained using the official **RobotWin 2.0 dataset**.

* Clean Demonstrations: 50 tasks × 50 trajectories per task
* Randomized Demonstrations: 50 tasks × 500 trajectories per task

### StarVLA-OFT

| Task                      | Easy       | Hard       |
| ------------------------- | ---------- | ---------- |
| stack_blocks_two          | 1.0000     | 1.0000     |
| place_cans_plasticbox     | 1.0000     | 0.9900     |
| grab_roller               | 1.0000     | 1.0000     |
| place_empty_cup           | 1.0000     | 1.0000     |
| shake_bottle_horizontally | 1.0000     | 1.0000     |
| lift_pot                  | 1.0000     | 1.0000     |
| place_burger_fries        | 1.0000     | 1.0000     |
| move_playingcard_away     | 1.0000     | 0.9800     |
| adjust_bottle             | 1.0000     | 0.9900     |
| shake_bottle              | 1.0000     | 1.0000     |
| blocks_ranking_rgb        | 0.9900     | 0.9800     |
| stack_bowls_two           | 0.9900     | 1.0000     |
| place_container_plate     | 0.9900     | 0.9900     |
| press_stapler             | 0.9900     | 0.9600     |
| place_object_stand        | 0.9900     | 0.9800     |
| open_laptop               | 0.9800     | 1.0000     |
| handover_mic              | 0.9800     | 0.9600     |
| move_pillbottle_pad       | 0.9800     | 1.0000     |
| handover_block            | 0.9700     | 0.9300     |
| place_shoe                | 0.9600     | 1.0000     |
| stack_bowls_three         | 0.9500     | 0.9100     |
| place_fan                 | 0.9400     | 0.9500     |
| scan_object               | 0.9400     | 0.9100     |
| stack_blocks_three        | 0.9400     | 0.8600     |
| place_object_basket       | 0.9300     | 0.9400     |
| beat_block_hammer         | 0.9300     | 0.9200     |
| place_object_scale        | 0.9300     | 0.9300     |
| place_dual_shoes          | 0.9100     | 0.8900     |
| pick_dual_bottles         | 0.9100     | 0.9300     |
| place_bread_basket        | 0.9100     | 0.7800     |
| dump_bin_bigbin           | 0.9100     | 0.9400     |
| move_can_pot              | 0.9100     | 0.9000     |
| put_bottles_dustbin       | 0.9000     | 0.8500     |
| place_a2b_left            | 0.9000     | 0.9500     |
| place_bread_skillet       | 0.8900     | 0.8000     |
| put_object_cabinet        | 0.8900     | 0.9100     |
| place_a2b_right           | 0.8800     | 0.9500     |
| rotate_qrcode             | 0.8800     | 0.9000     |
| pick_diverse_bottles      | 0.8700     | 0.8600     |
| place_mouse_pad           | 0.8700     | 0.9400     |
| stamp_seal                | 0.8600     | 0.9000     |
| place_phone_stand         | 0.8600     | 0.9500     |
| blocks_ranking_size       | 0.7900     | 0.8000     |
| place_can_basket          | 0.7500     | 0.7500     |
| move_stapler_pad          | 0.7400     | 0.9000     |
| turn_switch               | 0.6500     | 0.6200     |
| click_alarmclock          | 0.5800     | 0.5100     |
| hanging_mug               | 0.3400     | 0.2900     |
| open_microwave            | 0.2800     | 0.3900     |
| click_bell                | 0.2300     | 0.2700     |
| **Average**               | **0.8818** | **0.8832** |

*Note: All 50 tasks are trained within a single model, using 50 + 500 demonstrations per task (50×550 total demonstrations). Checkpoints can be downloaded at [Qwen3-VL-OFT-Robotwin2-All](https://huggingface.co/StarVLA/Qwen3-VL-OFT-RoboTwin2-All)*.


</details>

---



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
bash examples/Robotwin/eval_files/run_policy_server.sh
```

Edit your checkpoint path in `examples/Robotwin/eval_files/deploy_policy.yml` and `examples/Robotwin/eval_files/run_policy_server.sh`.
If your checkpoint was trained with percentile normalization, set `normalization_mode: "q99"` in `examples/Robotwin/eval_files/deploy_policy.yml`.
Available options are `min_max` and `q99`. The default is `min_max` to preserve the previous RoboTwin deployment behavior.

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
