
## StarVLA Co-Training Guide: VLA with VLM Data 🚀

This guide outlines the process for integrating VLM (Vision-Language Model) data to co-train the StarVLA (Vision-Language-Action) framework, enhancing its general visual and language understanding.

---

### 📦 1. Multi-Modal Data Preparation

The VLM data must adhere to the [QwenVL Conversations JSON Data Structure](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune).


#### Required Format:
* Each data instance is a JSON object.
* It links an **image file path** to a list of **human-GPT conversational turns**.

```json
{
    "image": "path/to/images/001.jpg",
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

You can download our case dataset [LLaVA-OneVision-COCO](https://huggingface.co/datasets/StarVLA/LLaVA-OneVision-COCO).  
Unzip `sharegpt4v_coco.zip` and place it in `playground/Datasets/LLaVA-OneVision-COCO`.

The resulting file structure will look like this:

``` bash
.../LLaVA-OneVision-COCO
├── images
│   └── sharegpt4v_coco
└── llava_jsons
    └── sharegpt4v_coco.json
```

-----

### ⚙️ 2. VLM Dataset Configuration

To add a custom VLM dataset, follow these steps:

#### 2.1 Register Dataset (Python)

Register your dataset by adding it to the `data_dict` in [qwen_data_config.py](./starVLA/dataloader/qwenvl_llavajson/qwen_data_config.py#L17).


```python
# Example Registration

SHAREGPT4V_COCO = {
    "annotation_path": f"{json_root}/sharegpt4v_coco.json",
    "data_path": f"{image_root}/",
}

data_dict = {
    "sharegpt4v_coco": SHAREGPT4V_COCO, # Use this name in the YAML config
}
```

#### 2.2 Update Training YAML

Include the VLM dataset configuration in your training YAML file (`your_train_config.yaml`).

```yaml
datasets:
  vlm_data:
    dataset_py: vlm_datasets
    dataformat: llava_json
    dataset_use: sharegpt4v_coco # Must match the name registered in 2.1
```

**Tip:** You can verify the VLM dataloader by running:

```bash
python starVLA/dataloader/vlm_datasets.py --config_yaml your_train_config.yaml
```


-----

### 🚀 3. Training Execution

Choose the appropriate script based on whether you want to train *only* on VLM data or *co-train* with VLA data.

#### Option A: Train with VLM Data Only

Use this for VLM-specific pre-training or fine-tuning.

  * **Script:** `starVLA/training/train_starvla_vlm.py`

<!-- end list -->

```bash
bash examples/CoTrainVLM/train_files/run_train_starvlm.sh
```

#### Option B: Co-Train VLA with VLM Data

This simultaneously trains the model on both robotics (VLA) and multi-modal (VLM) data.

  * **Script:** `starVLA/training/train_starvla_cotrain.py`

<!-- end list -->

```bash
bash examples/CoTrainVLM/train_files/run_libero_cotrain.sh
```

