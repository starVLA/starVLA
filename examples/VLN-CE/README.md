
This guide outlines the process for VLN-CE tasks using StarVLA (Vision-Language-Action) framework.

---



## Evaluation

Example VLN-CE model trained based on Qwen3VL-4B: [Ricky06662/StarVLA-VLNCE-Qwen3VL-4B](https://huggingface.co/Ricky06662/StarVLA-VLNCE-Qwen3VL-4B).

### 1.Start QwenVL VLM Server

VLN-CE evaluation can run QwenVL as a standalone websocket server, then call it from the evaluator instead of loading the model locally.

Start the server with a checkpoint path:

```bash
bash examples/VLN-CE/eval_files/run_qwenvl_vlm_server.sh /path/to/qwenvl/checkpoint
```

You can also specify the checkpoint and runtime options with environment variables:

```bash
CKPT=/path/to/qwenvl/checkpoint \
GPU_ID=0 \
SERVER_HOST=0.0.0.0 \
PORT=6694 \
bash examples/VLN-CE/eval_files/run_qwenvl_vlm_server.sh
```

To start one independent server on every available GPU, use `GPU_IDS=all`. The script assigns ports sequentially from `PORT`: GPU 0 uses `PORT`, GPU 1 uses `PORT + 1`, and so on.

```bash
GPU_IDS=all \
PORT=6694 \
bash examples/VLN-CE/eval_files/run_qwenvl_vlm_server.sh /path/to/qwenvl/checkpoint
```

You can also select specific GPUs:

```bash
GPU_IDS=0,1,2,3 \
PORT=6694 \
bash examples/VLN-CE/eval_files/run_qwenvl_vlm_server.sh /path/to/qwenvl/checkpoint
```

For multi-process evaluation, route each eval process to a different port, for example `6694`, `6695`, `6696`, `6697`.

Common options:

- `CKPT`: local checkpoint path or HuggingFace model id. The first script argument has higher priority than this variable.
- `GPU_ID`: single visible GPU id, default `0`. Kept for backward compatibility.
- `GPU_IDS`: comma-separated GPU ids, or `all` to use all visible GPUs. When multiple GPUs are used, the script starts one server per GPU.
- `SERVER_HOST`: server bind host, default `0.0.0.0`. Do not use `HOST`, because some shells/conda environments predefine it as a platform string.
- `PORT`: base websocket port, default `6694`.
- `DTYPE`: model dtype, default `bf16`.
- `MAX_NEW_TOKENS`: generation length, default `128`.

The server script is `examples/VLN-CE/eval_files/run_qwenvl_vlm_server.sh`, and the Python server implementation is `examples/VLN-CE/eval_files/qwenvl_vlm_server.py`.

---


### 2. Configure the VLN-CE Simulation Environment

Before running evaluation, set up the simulator and task environment required by VLN-CE. This includes preparing the VLN-CE evaluation codebase, Habitat/Habitat-Sim dependencies, scene datasets, and task configuration files.

> [!TIP]
> We separate the StarVLA environment from the simulator environment. The simulator should be installed in a completely new environment and should not be coupled with the StarVLA environment. The two environments communicate through websocket during evaluation.

For detailed setup instructions, refer to [StarVLA-VLN-CE-Evaluation](https://github.com/LiuRicky/StarVLA-VLN-CE-Evaluation).

----

## Training

### 📦 1. Multi-Modal Data Preparation

The VLM data must adhere to the [QwenVL Conversations JSON Data Structure](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md).


#### Required Format:
* Each data instance is a JSON object.
* It links an **image file path** to a list of **human-GPT conversational turns**.

```json
{
    "image": ["path/to/images/001.jpg", ..., "path/to/images/008.jpg"],
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nWhat's the main object in this picture?"
        },
        {
            "from": "gpt",
            "value": "A red apple on a wooden table"
        }
    ]
}
````

#### Quick Start

You can download R2R and RxR from [NaVILA-Dataset](https://huggingface.co/datasets/a8cheng/NaVILA-Dataset/tree/main).  
Unzip R2R and RxR files and place them in `playground/Datasets/VLN-CE`.

The resulting file structure will look like this:

``` bash
.../VLN-CE
├── R2R
  ├── train
  └── annotations.json
├── RxR
  ├── train
  └── annotations.json
```

Reformat the annotation files using [annotation_processing.py](examples/VLN-CE/train_files/annotation_processing.py):

```bash
python examples/VLN-CE/train_files/annotation_processing.py --data_path playground/Datasets/VLN-CE/R2R/annotations.json --dataset R2R
python examples/VLN-CE/train_files/annotation_processing.py --data_path playground/Datasets/VLN-CE/RxR/annotations.json --dataset RxR
```

The data format follows the [QwenVL Conversations JSON Data Structure](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune). Each data instance is a JSON object linking an **image file path** to a list of **human-GPT conversational turns**.

-----

### ⚙️ 2. Dataset Configuration

R2R and RxR are pre-registered in [qwen_data_config.py](../../starVLA/dataloader/qwenvl_llavajson/qwen_data_config.py):

```python
vlnce_root = "./playground/Datasets/VLN-CE"

R2R = {
    "annotation_path": f"{vlnce_root}/R2R/annotations.json",
    "data_path": f"{vlnce_root}/R2R/train/",
}

RXR = {
    "annotation_path": f"{vlnce_root}/RxR/annotations.json",
    "data_path": f"{vlnce_root}/RxR/train/",
}

data_dict = {
    "r2r": R2R,
    "rxr": RXR,
}
```

-----

### 🚀 3. Training Execution

Use this for VLM-specific pre-training or fine-tuning.

  * **Script:** `starVLA/training/train_starvln.py`

```bash
bash examples/VLN-CE/train_files/run_vlnce_train.sh
```

You can change the `batch_size=8` and `grad_accum_steps=2` parameters to adjust the batch size and gradient accumulation steps, to fit the memory of your GPU.
