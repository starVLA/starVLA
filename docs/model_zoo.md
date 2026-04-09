# Model Zoo

We release a series of modified models and finetuning checkpoints to facilitate reproduction and downstream use.

## Available Modified Models

| Model | Description | Link |
| --- | --- | --- |
| **Qwen2.5-VL-3B-Action** | Extend Qwen2.5-VL's vocabulary with Fast Tokens | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action) |
| **Qwen3-VL-4B-Action** | Extend Qwen3-VL's vocabulary with Fast Tokens | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action) |

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

| Model | Description | Avg. Length | Link |
| --- | --- | --- | --- |
| **QWen2.5VL-GR00T-Calvin_D_D** | Training on [Calvin_D_D](https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani/tree/lerobot/examples/calvin) | 3.786 | [🤗 Hugging Face](https://huggingface.co/Simplicissimus-S/StarVLA-QwenGR00T_Qwen2.5-VL-3B-Instruct-Action_calvin_D_D) |
