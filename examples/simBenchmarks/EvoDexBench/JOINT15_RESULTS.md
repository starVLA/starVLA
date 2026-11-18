# Evo-DexBench 15-task joint training

This page reports one checkpoint per embodiment, trained jointly on the 15
single-arm or 15 dual-arm tasks and evaluated on all of them with the Evo-DexBench
standard protocol. Every task, including Gen tasks, is seen during training, so
these results are not comparable with Core-only or Gen-only training.

## Models

| | QwenOFT | Wan-OFT | Cosmos-GR00T |
| --- | --- | --- | --- |
| Framework | `QwenOFT` | `EvoWanOFT` | `EvoCosmosGR00T` |
| Backbone | Qwen3-VL-4B-Instruct, frozen | Wan2.2-TI2V-5B, frozen | Cosmos-Predict2-2B, frozen |
| Action head | MLP | MLP | DiT-B flow matching |
| Action horizon / executed steps | 50 / 10 | 8 / 8 | 8 / 8 |
| Learning rate, base / action head | 1e-5 / 1e-4 | 1e-4 / 1e-4 | 1e-4 / 1e-4 |
| GPUs x per-device batch | 2 x 2 | 1 x 4 | 1 x 4 |
| Single-arm recipe | `single15_qwenoft_h50_q99.yaml` | `single15_wan_h8_mean_q99.yaml` | `single15_cosmos_h8_mean_q99.yaml` |
| Dual-arm recipe | `dual15_qwenoft_h50_q99.yaml` | `dual15_wan_h8_mean_q99.yaml` | `dual15_cosmos_h8_mean_q99.yaml` |

## Training

Each checkpoint uses 624 demonstrations: 50 from each of 12 Core tasks and 8 from
each of 3 Gen tasks. Tasks are sampled with equal weight, and q99 normalization
statistics are merged across the 15 tasks. All runs use global batch 4, seed 42,
bf16, 5,000 warmup steps, cosine decay to 5e-7 and gradient clipping of 1.0.
Training ran on NVIDIA A100 80 GB GPUs.

| Checkpoint | Steps | Final model SHA-256 | Statistics SHA-256 |
| --- | ---: | --- | --- |
| QwenOFT single | 300k | `3506fa1e98873d8185b81532f67a615eca8b409e7317c8aa138cdbcf42a31f9a` | `082d6e13c063e757…` |
| QwenOFT dual | 300k | `7999faef4f62e9f3f9615d93d04aed6f52b6cf8cd2f1b0a5747d0742174a5f6c` | `70eecef9651dde31…` |
| QwenOFT single | 600k | `d314ed09ec577363f4d97324d2fde59959032022985abdc8f3d396e76b07f72a` | `082d6e13c063e757…` |
| QwenOFT dual | 600k | `442afb5b3081d9a912c22757c1014fe95ba63bfa244659678e3721fc79230a8c` | `70eecef9651dde31…` |
| Wan-OFT single | 300k | `e898d0d3cd19bebb25e45aeb172bbf7dd82265fd680cb84cdfe73db142edbdc3` | `082d6e13c063e757…` |
| Wan-OFT dual | 300k | `08ce15c174da8503f4598d3e7f5b3ed7a2f67547dc78b04490f146887a235171` | `70eecef9651dde31…` |
| Cosmos-GR00T single | 300k | `c7c10bcebdd83d0d1998c7bbf733e87083eeed0111917befbd5441137971eb18` | `082d6e13c063e757…` |
| Cosmos-GR00T dual | 300k | `0141131fb61cc87914e9067e29122e23ea399a6d87d6e2fd0d0f81bdeb03af79` | `70eecef9651dde31…` |

The 600k runs start from the same initialization as the 300k runs. They are not
continuations, because the cosine schedule depends on the total step count.

## Evaluation protocol

Each task runs 100 episodes: five conditions of 20 (Nominal, Camera, Lighting,
Physical, Spatial), layouts 34/33/33, seeds 1001-1005, four parallel environments,
30 Hz control, 120 Hz simulation and the step limit in the task YAML. SR is the
success rate and Stage is the mean normalized stage score.

| Tasks | Evaluator | Task-specific variant |
| --- | --- | --- |
| M01-M15, B11-B15 | pre-release revision `b25905b` | M12: instance-binding repair r1 |
| B01-B10 | pre-release revision `229faec` | B03: dynamic-object override repair r1; B04, B05: hand-joint-limits-r1 |
| M01-M04 re-evaluation | public [Evo-DexBench](https://github.com/MINT-SJTU/Evo-DexBench) | M01, M03: updated physics; independent stage scoring |

Simulator environment: Python 3.10.21, Torch 2.5.1+cu124, SAPIEN 3.0.3, ManiSkill
3.0.1. All checkpoints were evaluated on NVIDIA A100 80 GB GPUs.

## Results

All values are between 0 and 1.

### Single arm

| Task | Role | QwenOFT 300k SR | Stage | QwenOFT 600k SR | Stage | Wan-OFT 300k SR | Stage | Cosmos-GR00T 300k SR | Stage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M01 | Core | 0.22 | 0.006 | 0.07 | 0.006 | 0.22 | 0.038 | 0.20 | 0.091 |
| M02 | Core | 0.16 | 0.412 | 0.82 | 0.880 | 0.57 | 0.711 | 0.31 | 0.529 |
| M03 | Core | 0.06 | 0.084 | 0.03 | 0.039 | 0.02 | 0.032 | 0.01 | 0.019 |
| M04 | Core | 0.19 | 0.658 | 0.23 | 0.674 | 0.33 | 0.726 | 0.47 | 0.778 |
| M05 | Gen | 0.00 | 0.081 | 0.00 | 0.018 | 0.00 | 0.069 | 0.09 | 0.243 |
| M06 | Core | 0.02 | 0.012 | 0.00 | 0.057 | 0.16 | 0.155 | 0.02 | 0.038 |
| M07 | Core | 0.00 | 0.022 | 0.02 | 0.285 | 0.03 | 0.077 | 0.14 | 0.198 |
| M08 | Core | 0.17 | 0.246 | 0.06 | 0.398 | 0.01 | 0.342 | 0.13 | 0.506 |
| M09 | Core | 0.86 | 0.860 | 0.85 | 0.856 | 0.41 | 0.522 | 0.53 | 0.747 |
| M10 | Gen | 0.03 | 0.240 | 0.27 | 0.508 | 0.06 | 0.135 | 0.04 | 0.256 |
| M11 | Core | 0.00 | 0.020 | 0.00 | 0.000 | 0.00 | 0.005 | 0.00 | 0.065 |
| M12 | Core | 0.00 | 0.000 | 0.00 | 0.003 | 0.00 | 0.000 | 0.00 | 0.007 |
| M13 | Core | 0.00 | 0.040 | 0.00 | 0.117 | 0.00 | 0.159 | 0.00 | 0.140 |
| M14 | Core | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.005 |
| M15 | Gen | 0.00 | 0.000 | 0.00 | 0.013 | 0.00 | 0.000 | 0.00 | 0.000 |

### Dual arm

| Task | Role | QwenOFT 300k SR | Stage | QwenOFT 600k SR | Stage | Wan-OFT 300k SR | Stage | Cosmos-GR00T 300k SR | Stage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B01 | Core | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.024 | 0.00 | 0.009 |
| B02 | Core | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.000 |
| B03 | Core | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.032 |
| B04 | Core | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.000 |
| B05 | Gen | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.000 |
| B06 | Core | 0.00 | 0.008 | 0.01 | 0.038 | 0.00 | 0.020 | 0.00 | 0.122 |
| B07 | Core | 0.00 | 0.022 | 0.00 | 0.030 | 0.00 | 0.098 | 0.00 | 0.158 |
| B08 | Core | 0.00 | 0.263 | 0.00 | 0.196 | 0.00 | 0.230 | 0.00 | 0.290 |
| B09 | Core | 0.02 | 0.055 | 0.12 | 0.180 | 0.01 | 0.098 | 0.08 | 0.152 |
| B10 | Gen | 0.00 | 0.024 | 0.00 | 0.118 | 0.00 | 0.000 | 0.01 | 0.052 |
| B11 | Core | 0.23 | 0.350 | 0.02 | 0.059 | 0.01 | 0.013 | 0.19 | 0.358 |
| B12 | Core | 0.00 | 0.247 | 0.00 | 0.196 | 0.00 | 0.178 | 0.15 | 0.455 |
| B13 | Core | 0.00 | 0.400 | 0.00 | 0.400 | 0.00 | 0.406 | 0.07 | 0.448 |
| B14 | Core | 0.08 | 0.362 | 0.12 | 0.399 | 0.05 | 0.353 | 0.04 | 0.361 |
| B15 | Gen | 0.00 | 0.016 | 0.00 | 0.012 | 0.01 | 0.058 | 0.02 | 0.076 |

### Macro averages

| Checkpoint | SR, 15 tasks | Stage, 15 tasks | SR, 12 Core tasks | SR, 3 Gen tasks |
| --- | ---: | ---: | ---: | ---: |
| QwenOFT single 300k | 0.1140 | 0.1788 | 0.1400 | 0.0100 |
| QwenOFT single 600k | 0.1567 | 0.2568 | 0.1733 | 0.0900 |
| Wan-OFT single 300k | 0.1207 | 0.1980 | 0.1458 | 0.0200 |
| Cosmos-GR00T single 300k | 0.1293 | 0.2414 | 0.1508 | 0.0433 |
| QwenOFT dual 300k | 0.0220 | 0.1165 | 0.0275 | 0.0000 |
| QwenOFT dual 600k | 0.0180 | 0.1085 | 0.0225 | 0.0000 |
| Wan-OFT dual 300k | 0.0053 | 0.0985 | 0.0058 | 0.0033 |
| Cosmos-GR00T dual 300k | 0.0373 | 0.1676 | 0.0442 | 0.0100 |

### M01-M04 with the updated evaluator

Stage values here use independent stage scoring and should not be compared with
the tables above.

| Task | QwenOFT 300k SR | Stage | QwenOFT 600k SR | Stage | Wan-OFT 300k SR | Stage | Cosmos-GR00T 300k SR | Stage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M01 | 0.24 | 0.261 | 0.06 | 0.072 | 0.17 | 0.203 | 0.12 | 0.156 |
| M02 | 0.16 | 0.412 | 0.82 | 0.880 | 0.57 | 0.711 | 0.31 | 0.529 |
| M03 | 0.07 | 0.088 | 0.00 | 0.003 | 0.05 | 0.065 | 0.02 | 0.045 |
| M04 | 0.19 | 0.676 | 0.23 | 0.692 | 0.33 | 0.732 | 0.47 | 0.788 |
| Mean | 0.1650 | 0.3593 | 0.2775 | 0.4118 | 0.2800 | 0.4278 | 0.2300 | 0.3795 |

### Results by condition

Condition columns are success rates over 20 episodes.

<details>
<summary>QwenOFT single 300k</summary>

| Task | SR | Stage | Nominal | Camera | Lighting | Physical | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M01 | 0.22 | 0.006 | 0.30 | 0.25 | 0.20 | 0.15 | 0.20 |
| M02 | 0.16 | 0.412 | 0.15 | 0.25 | 0.25 | 0.00 | 0.15 |
| M03 | 0.06 | 0.084 | 0.15 | 0.00 | 0.10 | 0.00 | 0.05 |
| M04 | 0.19 | 0.658 | 0.20 | 0.35 | 0.20 | 0.00 | 0.20 |
| M05 | 0.00 | 0.081 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M06 | 0.02 | 0.012 | 0.00 | 0.05 | 0.00 | 0.00 | 0.05 |
| M07 | 0.00 | 0.022 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M08 | 0.17 | 0.246 | 0.10 | 0.20 | 0.20 | 0.25 | 0.10 |
| M09 | 0.86 | 0.860 | 1.00 | 1.00 | 1.00 | 1.00 | 0.30 |
| M10 | 0.03 | 0.240 | 0.00 | 0.00 | 0.05 | 0.00 | 0.10 |
| M11 | 0.00 | 0.020 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M12 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M13 | 0.00 | 0.040 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M14 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M15 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

</details>

<details>
<summary>QwenOFT single 600k</summary>

| Task | SR | Stage | Nominal | Camera | Lighting | Physical | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M01 | 0.07 | 0.006 | 0.00 | 0.15 | 0.15 | 0.00 | 0.05 |
| M02 | 0.82 | 0.880 | 0.85 | 0.70 | 0.85 | 1.00 | 0.70 |
| M03 | 0.03 | 0.039 | 0.05 | 0.00 | 0.10 | 0.00 | 0.00 |
| M04 | 0.23 | 0.674 | 0.30 | 0.30 | 0.35 | 0.00 | 0.20 |
| M05 | 0.00 | 0.018 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M06 | 0.00 | 0.057 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M07 | 0.02 | 0.285 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 |
| M08 | 0.06 | 0.398 | 0.00 | 0.05 | 0.00 | 0.15 | 0.10 |
| M09 | 0.85 | 0.856 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 |
| M10 | 0.27 | 0.508 | 0.20 | 0.20 | 0.35 | 0.40 | 0.20 |
| M11 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M12 | 0.00 | 0.003 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M13 | 0.00 | 0.117 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M14 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M15 | 0.00 | 0.013 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

</details>

<details>
<summary>Wan-OFT single 300k</summary>

| Task | SR | Stage | Nominal | Camera | Lighting | Physical | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M01 | 0.22 | 0.038 | 0.25 | 0.10 | 0.30 | 0.25 | 0.20 |
| M02 | 0.57 | 0.711 | 0.90 | 0.20 | 0.70 | 0.70 | 0.35 |
| M03 | 0.02 | 0.032 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 |
| M04 | 0.33 | 0.726 | 0.45 | 0.15 | 0.60 | 0.35 | 0.10 |
| M05 | 0.00 | 0.069 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M06 | 0.16 | 0.155 | 0.35 | 0.00 | 0.05 | 0.20 | 0.20 |
| M07 | 0.03 | 0.077 | 0.00 | 0.05 | 0.00 | 0.00 | 0.10 |
| M08 | 0.01 | 0.342 | 0.00 | 0.05 | 0.00 | 0.00 | 0.00 |
| M09 | 0.41 | 0.522 | 0.70 | 0.25 | 0.35 | 0.45 | 0.30 |
| M10 | 0.06 | 0.135 | 0.15 | 0.00 | 0.10 | 0.05 | 0.00 |
| M11 | 0.00 | 0.005 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M12 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M13 | 0.00 | 0.159 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M14 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M15 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

</details>

<details>
<summary>Cosmos-GR00T single 300k</summary>

| Task | SR | Stage | Nominal | Camera | Lighting | Physical | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M01 | 0.20 | 0.091 | 0.35 | 0.10 | 0.20 | 0.25 | 0.10 |
| M02 | 0.31 | 0.529 | 0.25 | 0.00 | 0.45 | 0.45 | 0.40 |
| M03 | 0.01 | 0.019 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 |
| M04 | 0.47 | 0.778 | 0.70 | 0.10 | 0.65 | 0.20 | 0.70 |
| M05 | 0.09 | 0.243 | 0.20 | 0.00 | 0.15 | 0.05 | 0.05 |
| M06 | 0.02 | 0.038 | 0.05 | 0.00 | 0.05 | 0.00 | 0.00 |
| M07 | 0.14 | 0.198 | 0.15 | 0.15 | 0.15 | 0.10 | 0.15 |
| M08 | 0.13 | 0.506 | 0.10 | 0.05 | 0.05 | 0.20 | 0.25 |
| M09 | 0.53 | 0.747 | 0.45 | 0.30 | 0.75 | 0.85 | 0.30 |
| M10 | 0.04 | 0.256 | 0.05 | 0.05 | 0.00 | 0.00 | 0.10 |
| M11 | 0.00 | 0.065 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M12 | 0.00 | 0.007 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M13 | 0.00 | 0.140 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M14 | 0.00 | 0.005 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M15 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

</details>

<details>
<summary>QwenOFT dual 300k</summary>

| Task | SR | Stage | Nominal | Camera | Lighting | Physical | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B01 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B02 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B03 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B04 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B05 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B06 | 0.00 | 0.008 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B07 | 0.00 | 0.022 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B08 | 0.00 | 0.263 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B09 | 0.02 | 0.055 | 0.00 | 0.05 | 0.05 | 0.00 | 0.00 |
| B10 | 0.00 | 0.024 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B11 | 0.23 | 0.350 | 0.30 | 0.10 | 0.35 | 0.30 | 0.10 |
| B12 | 0.00 | 0.247 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B13 | 0.00 | 0.400 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B14 | 0.08 | 0.362 | 0.05 | 0.00 | 0.20 | 0.15 | 0.00 |
| B15 | 0.00 | 0.016 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

</details>

<details>
<summary>QwenOFT dual 600k</summary>

| Task | SR | Stage | Nominal | Camera | Lighting | Physical | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B01 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B02 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B03 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B04 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B05 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B06 | 0.01 | 0.038 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 |
| B07 | 0.00 | 0.030 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B08 | 0.00 | 0.196 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B09 | 0.12 | 0.180 | 0.10 | 0.05 | 0.15 | 0.15 | 0.15 |
| B10 | 0.00 | 0.118 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B11 | 0.02 | 0.059 | 0.00 | 0.05 | 0.00 | 0.00 | 0.05 |
| B12 | 0.00 | 0.196 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B13 | 0.00 | 0.400 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B14 | 0.12 | 0.399 | 0.25 | 0.05 | 0.05 | 0.15 | 0.10 |
| B15 | 0.00 | 0.012 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

</details>

<details>
<summary>Wan-OFT dual 300k</summary>

| Task | SR | Stage | Nominal | Camera | Lighting | Physical | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B01 | 0.00 | 0.024 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B02 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B03 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B04 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B05 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B06 | 0.00 | 0.020 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B07 | 0.00 | 0.098 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B08 | 0.00 | 0.230 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B09 | 0.01 | 0.098 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 |
| B10 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B11 | 0.01 | 0.013 | 0.00 | 0.05 | 0.00 | 0.00 | 0.00 |
| B12 | 0.00 | 0.178 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B13 | 0.00 | 0.406 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B14 | 0.05 | 0.353 | 0.00 | 0.00 | 0.10 | 0.10 | 0.05 |
| B15 | 0.01 | 0.058 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 |

</details>

<details>
<summary>Cosmos-GR00T dual 300k</summary>

| Task | SR | Stage | Nominal | Camera | Lighting | Physical | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B01 | 0.00 | 0.009 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B02 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B03 | 0.00 | 0.032 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B04 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B05 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B06 | 0.00 | 0.122 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B07 | 0.00 | 0.158 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B08 | 0.00 | 0.290 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B09 | 0.08 | 0.152 | 0.10 | 0.05 | 0.00 | 0.10 | 0.15 |
| B10 | 0.01 | 0.052 | 0.00 | 0.00 | 0.05 | 0.00 | 0.00 |
| B11 | 0.19 | 0.358 | 0.30 | 0.00 | 0.20 | 0.25 | 0.20 |
| B12 | 0.15 | 0.455 | 0.25 | 0.00 | 0.15 | 0.20 | 0.15 |
| B13 | 0.07 | 0.448 | 0.15 | 0.05 | 0.10 | 0.05 | 0.00 |
| B14 | 0.04 | 0.361 | 0.10 | 0.00 | 0.05 | 0.05 | 0.00 |
| B15 | 0.02 | 0.076 | 0.00 | 0.05 | 0.00 | 0.05 | 0.00 |

</details>

<details>
<summary>M01-M04 with the updated evaluator, QwenOFT 300k</summary>

| Task | SR | Stage | Nominal | Camera | Lighting | Physical | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M01 | 0.24 | 0.261 | 0.30 | 0.25 | 0.25 | 0.30 | 0.10 |
| M02 | 0.16 | 0.412 | 0.15 | 0.25 | 0.25 | 0.00 | 0.15 |
| M03 | 0.07 | 0.088 | 0.15 | 0.00 | 0.15 | 0.00 | 0.05 |
| M04 | 0.19 | 0.676 | 0.20 | 0.35 | 0.20 | 0.00 | 0.20 |

</details>

<details>
<summary>M01-M04 with the updated evaluator, QwenOFT 600k</summary>

| Task | SR | Stage | Nominal | Camera | Lighting | Physical | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M01 | 0.06 | 0.072 | 0.00 | 0.25 | 0.05 | 0.00 | 0.00 |
| M02 | 0.82 | 0.880 | 0.85 | 0.70 | 0.85 | 1.00 | 0.70 |
| M03 | 0.00 | 0.003 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M04 | 0.23 | 0.692 | 0.30 | 0.30 | 0.35 | 0.00 | 0.20 |

</details>

<details>
<summary>M01-M04 with the updated evaluator, Wan-OFT 300k</summary>

| Task | SR | Stage | Nominal | Camera | Lighting | Physical | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M01 | 0.17 | 0.203 | 0.15 | 0.15 | 0.25 | 0.15 | 0.15 |
| M02 | 0.57 | 0.711 | 0.90 | 0.20 | 0.70 | 0.70 | 0.35 |
| M03 | 0.05 | 0.065 | 0.05 | 0.05 | 0.15 | 0.00 | 0.00 |
| M04 | 0.33 | 0.732 | 0.45 | 0.15 | 0.60 | 0.35 | 0.10 |

</details>

<details>
<summary>M01-M04 with the updated evaluator, Cosmos-GR00T 300k</summary>

| Task | SR | Stage | Nominal | Camera | Lighting | Physical | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M01 | 0.12 | 0.156 | 0.05 | 0.05 | 0.15 | 0.25 | 0.10 |
| M02 | 0.31 | 0.529 | 0.25 | 0.00 | 0.45 | 0.45 | 0.40 |
| M03 | 0.02 | 0.045 | 0.05 | 0.05 | 0.00 | 0.00 | 0.00 |
| M04 | 0.47 | 0.788 | 0.70 | 0.10 | 0.65 | 0.20 | 0.70 |

</details>

QwenOFT, Wan-OFT and Cosmos-GR00T differ in backbone, action head, horizon and
learning rate, and each configuration has one training seed, so a gap between them
cannot be attributed to a single factor.

## Reproduce

Prepare the data with [`data_preparation.sh`](README.md#2-data), which places one
LeRobot v3 dataset per task (`m01` to `m15`, or `b01` to `b15`) with its
`meta/modality.json` under `playground/Datasets/EvoDexBench/{single,dual}`.

Train QwenOFT (600k by default; add `--trainer.max_train_steps 300000` and a new
`run_id` for the 300k run):

```bash
FRAMEWORK=QwenOFT EMBODIMENT=single \
  bash examples/simBenchmarks/EvoDexBench/train_files/run_evodex_train.sh \
  --framework.qwenvl.base_vlm /path/to/Qwen3-VL-4B-Instruct
```

Train Wan-OFT or Cosmos-GR00T after installing the
[model extension](README.md#wan-oft-and-cosmos-gr00t) and setting the official
checkpoint and base configuration paths in the recipe:

```bash
FRAMEWORK=EvoCosmosGR00T EMBODIMENT=single \
  bash examples/simBenchmarks/EvoDexBench/train_files/run_evodex_train.sh
```

Use `FRAMEWORK=EvoWanOFT` for Wan-OFT and `EMBODIMENT=dual` for the dual-arm
recipes. To evaluate, start the matching server and run the 15 tasks as
described in [Evaluation](README.md#4-evaluation).

Checkpoints: the QwenOFT 600k, Wan-OFT 300k and Cosmos-GR00T 300k single-arm and
dual-arm checkpoints are in
[MINT-SJTU/Evo-DexBench_StarVLA_checkpoints](https://huggingface.co/MINT-SJTU/Evo-DexBench_StarVLA_checkpoints),
with their configurations, statistics, model extension and per-task results.
