
This guide outlines the process for VLN-CE tasks using StarVLA (Vision-Language-Action) framework.

[TODO] The evaluation is remind todo

---

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
