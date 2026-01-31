# 🚀 Calvin Training and Evaluation

This document describes how to **train and evaluate StarVLA models on the Calvin benchmark**, including dataset preparation, training configuration, and evaluation procedures.

> **Note:** Calvin benchmark experiments were conducted by the UNT team. For inquiries, please contact Zhijie Song (1600013008@pku.edu.cn) or Feng Yan (bphengyan@163.com).


---

## 📊 Benchmark Results (Calvin)

| Model                                     | Avg. Length | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
| ----------------------------------------- | ----------- | ------ | ------ | ------ | ------ | ------ |
| **PI0***                                  | 2.954       | 84.8%  | 70.4%  | 55.9%  | 46.6%  | 37.7%  |
| **PI0.5***                                | 3.885       | 92.5%  | 84.0%  | 76.6%  | 71.0%  | 64.4%  |
| qwenpi (qwen2.5-vl-3B-instruct-action)    | 3.576       | 90.9%  | 79.5%  | 69.6%  | 62.2%  | 55.4%  |
| qwenpi (qwen3-vl-4B-instruct)             | 3.472       | 87.7%  | 75.2%  | 67.4%  | 61.8%  | 55.1%  |
| qwengr00t (qwen2.5-vl-3B-instruct)        | 3.697       | 91.7%  | 81.9%  | 72.7%  | 65.3%  | 58.1%  |
| qwengr00t (qwen2.5-vl-3B-instruct-action) | 3.786       | 92.5%  | 83.9%  | 74.4%  | 67.9%  | 59.9%  |
| qwengr00t (qwen3-vl-4B-instruct)          | 3.645       | 89.8%  | 79.9%  | 71.8%  | 64.9%  | 58.1%  |
| qwengr00t (qwen3-vl-4B-instruct-action)   | 3.757       | 91.1%  | 81.8%  | 74.1%  | 67.6%  | 61.1%  |


* **Models marked with `*` are trained by us.**
Other experimental results will be released soon.

---

## 📦 0. Dataset Preparation

### Training Dataset (LeRobot Format)

For training, the Calvin dataset must be converted into **LeRobot format**.

1. Convert the original Calvin dataset to LeRobot format.
   Please refer to **RoboTron-Mani (ICCV 2025)** for detailed instructions:
   👉 [https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani/tree/lerobot/examples/calvin](https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani/tree/lerobot/examples/calvin)

2. Copy the modality definition file:

   ```bash
   cp examples/calvin/train_files/modality.json <lerobot_dataset_path>/meta/modality.json
   ```

---

### Evaluation Dataset (Original Calvin Format)

For evaluation, use the **original Calvin dataset format**.

* The dataset path should directly contain the `validation/` directory.

⚠️ **Important Note**

Training and evaluation use **different dataset formats**:

* **Training**: LeRobot format (converted from original Calvin)
* **Evaluation**: Original Calvin format (with `validation/` folder)

---

## ⚙️ 1. Configure Data Mix

Configure the Calvin data mix in:

```
starVLA/dataloader/gr00t_lerobot/mixtures.py
```

Example configuration:

```python
"calvin_task_D_D": [
    ("task_D_D", 1.0, "libero_franka"),
],
```

Make sure the key name (e.g. `calvin_task_D_D`) matches the one used during training.

---

## 🚀 2. Training

⚠️ **Before training**, please double-check the following paths in
`examples/calvin/train_files/run_calvin_train.sh`:

* `calvin_data_root`: Path to the **LeRobot-format Calvin dataset**
* `data_mix`: Must match the key defined in `mixtures.py`

Start training with:

```bash
bash examples/calvin/train_files/run_calvin_train.sh
```

---

## 🧪 3. Evaluation

Evaluation requires **two terminals**, both launched from the **repository root**, but using **different conda environments**.

---

### Step 1. Launch Policy Server (StarVLA Environment)

In the first terminal, activate the `starVLA` conda environment and start the inference server:

```bash
bash examples/calvin/eval_files/run_policy_server.sh
```

⚠️ **Note:**
Ensure the checkpoint path specified in `run_policy_server.sh` is correct.

---

### Step 2. Run Calvin Evaluation (Calvin Environment)

In the second terminal, activate the `Calvin` conda environment and run:

```bash
bash examples/calvin/eval_files/eval_calvin.sh
```

⚠️ **Note:**
Please verify the following paths in `eval_calvin.sh`:

* `dataset_path`: Path to the **original Calvin dataset** (containing `validation/`)
* `your_ckpt`: Checkpoint path (must match the one used by the policy server)

⚠️ **Note:**
Additionally, you need to modify the following paths in `eval_calvin.py`:

* `dataset_path`: Path to Calvin dataset (default: `"/path/to/calvin/task_D_D"`)
* `calvin_config_path`: Path to Calvin models configuration directory (default: `"/path/to/calvin/calvin_models/conf"`)
* `eval_sequences_path`: Path to evaluation sequences JSON file (default: `"/path/to/calvin/eval_sequences.json"`)



