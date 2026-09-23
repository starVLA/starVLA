# Model Zoo

We release a series of modified models and finetuning checkpoints to facilitate reproduction and downstream use.

## Available Modified Models

| Model | Description | Link |
| --- | --- | --- |
| **Qwen2.5-VL-3B-Action** | Extend Qwen2.5-VL's vocabulary with Fast Tokens | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action) |
| **Qwen3-VL-4B-Action** | Extend Qwen3-VL's vocabulary with Fast Tokens | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action) |

## VLAct Backbone and Downstream Checkpoints

[VLAct](https://starvla.github.io/VLAct/) is a reusable Qwen3-VL-4B backbone continued-pretrained on
heterogeneous robot data. It can initialize a fresh StarVLA OFT, PI, or GR00T action head by setting
`framework.qwenvl.base_vlm: StarVLA/VLAct_Qwen3_Pretrain`.

| Artifact | Description | Link |
| --- | --- | --- |
| **VLAct Qwen3-VL-4B** | Reusable continued-pretrained backbone; try it as a direct replacement for `Qwen/Qwen3-VL-4B-Instruct` | [🤗 Hugging Face](https://huggingface.co/StarVLA/VLAct_Qwen3_Pretrain) |
| **VLAct LIBERO-Plus** | PI downstream checkpoint; the paper's 82.6% baseline uses OFT | [🤗 Hugging Face](https://huggingface.co/StarVLA/VLAct_Qwen3PI_Libero_Plus_Finetune) |
| **VLAct RoboTwin 2.0** | OFT, PI, and GR00T checkpoints for base and scaling settings | [🤗 Collection](https://huggingface.co/collections/StarVLA/vlact-6a903c2e0c176179da425c96) |
| **VLAct VLA-Arena** | PI downstream checkpoint; 54.8% | [🤗 Hugging Face](https://huggingface.co/StarVLA/VLAct_Qwen3PI_VLA_Arena_Finetune) |
| **VLAct DOMINO** | OFT downstream checkpoint; 18.50 SR / 34.20 MS | [🤗 Hugging Face](https://huggingface.co/StarVLA/VLAct_Qwen3OFT_Domino_Finetune) |
| **VLAct RoboDojo** | OFT downstream checkpoint; 10.66 score / 7.60% success | [🤗 Hugging Face](https://huggingface.co/StarVLA/VLAct-Qwen3VL4B-OFT-RoboDojo) |

For methods and exact evaluation protocols, see the [VLAct paper](https://arxiv.org/abs/2608.27550)
and [repository](https://github.com/starVLA/VLAct). If you use these artifacts or results, please use
the [VLAct citation](../README.md#citation).

## Available Finetuning Checkpoints

| Model | Description | WidowX | Link |
| --- | --- | --- | --- |
| **QWen2.5-FAST-Bridge-RT-1** | Training on [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) and [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) | 58.6 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-FAST-Bridge-RT-1) |
| **QWen2.5-OFT-Bridge-RT-1** | Training on [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) and [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) | 41.8 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-OFT-Bridge-RT-1) |
| **QWen2.5-PI-Bridge-RT-1** | Training on [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) and [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) | 62.5 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-FM-Bridge-RT-1) |
| **QWen2.5-GR00T-Bridge-RT-1** | Training on [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) and [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) | 63.6 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-PI-Bridge-RT-1) |
| **QWen-GR00T-Bridge** | Training only on [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) | 71.4 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-GR00T-Bridge) |
| **QWen3VL-OFT-Bridge-RT-1** | Training on [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) and [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) | 42.7 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen3VL-OFT-Bridge-RT-1) |
| **QWen3VL-GR00T-Bridge-RT-1** | Training on [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) and [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) | 65.3 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1) |
| **QWen3VL-PI_v3-Bridge-RT-1** | Training on [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) and [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) | 69.8 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen3VL-PI_v3-Bridge-RT_1) |

| Model | Description | Avg. Length | Link |
| --- | --- | --- | --- |
| **QWen2.5VL-GR00T-Calvin_D_D** | Training on [Calvin_D_D](https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani/tree/lerobot/examples/simBenchmarks/calvin) | 3.786 | [🤗 Hugging Face](https://huggingface.co/Simplicissimus-S/StarVLA-QwenGR00T_Qwen2.5-VL-3B-Instruct-Action_calvin_D_D) |

## VM4A

VM4A provides from-scratch visuomotor baselines that use StarVLA's shared data, normalization, training, and deployment interfaces. See [VM4A: VisuoMotor for Action](VM4A.md) for configuration and implementation details.

| Model | Description | Checkpoint |
| --- | --- | --- |
| **ACT** | LeRobot ACTPolicy wrapper for direct image-and-state action chunking | real-robot validated; public checkpoint TBD per maintainer discussion |
| **Diffusion Policy** | Vendored non-hybrid image policy with a ResNet-18 encoder and DDPM action generation | real-robot validated; public checkpoint TBD per maintainer discussion |
