# MiniCPM-RobotManip × LIBERO Fine-tuning

Fine-tune the released **[MiniCPM-RobotManip](https://github.com/OpenBMB/MiniCPM-Robot)**
generalist VLA on LIBERO inside starVLA.

This integration is contributed by members of the MiniCPM-RobotManip team to
provide a compact, upstream-friendly training example for the released model.

MiniCPM-RobotManip is a 1.5B generalist manipulation policy: a MiniCPM-V-4.6
backbone + a flow-matching GR00T action head, trained on a unified **80-D**
action space. This example loads the released checkpoint
[`openbmb/MiniCPM-RobotManip`](https://huggingface.co/openbmb/MiniCPM-RobotManip)
**as-is** (via its shipped `trust_remote_code` model class) and fine-tunes it on
LIBERO's four task suites.

> **How this differs from `examples/modelExtensions/MiniCPM`.**
> That example plugs the plain `openbmb/MiniCPM-V-4.6` VLM into starVLA and trains
> a *fresh* 7-D action head from scratch. **This** example loads the *released
> MiniCPM-RobotManip* weights (backbone **and** a pretrained 80-D action head) and
> continues training them — i.e. it showcases the open-sourced model itself.

## Action space

State and action share a unified 80-D layout; LIBERO is single-arm, so the 10-D
absolute EE6D target (`observation.xvla_abs_ee6d` = xyz(3) + rot6d(6) +
gripper(1)) occupies the left-arm end-effector slot `[7:17]`. All other channels
are masked out of the loss. `embodiment_id = 0`, `action_horizon = 30`.

## Requirements

- `transformers` new enough to load `openbmb/MiniCPM-V-4.6` (`AutoModelForImageTextToText`)
- `trust_remote_code=True` is used to load the released `MiniCPMV_VLA` class
- LIBERO LeRobot-v3 data with the `observation.xvla_abs_ee6d` column

## Data preparation

This recipe requires the filtered, 20 Hz LeRobot-v3 conversion with absolute
EE6D targets. The converted dataset is available at
[`openbmb/MiniCPM-RobotManip-LIBERO`](https://huggingface.co/datasets/openbmb/MiniCPM-RobotManip-LIBERO).
Download it with:

```bash
huggingface-cli download openbmb/MiniCPM-RobotManip-LIBERO \
  --repo-type dataset \
  --local-dir playground/Datasets/LIBERO_EE6D
```

Do not substitute `lerobot/libero_*_image`: those public datasets contain 7-D
actions at 10 Hz and do not provide `observation.xvla_abs_ee6d`.

The expected layout is:

```
playground/Datasets/LIBERO_EE6D/
  libero_10_agentview_rot180_wrist_raw_lerobot_v30_xvla_abs6d/
  libero_goal_agentview_rot180_wrist_raw_lerobot_v30_xvla_abs6d/
  libero_object_agentview_rot180_wrist_raw_lerobot_v30_xvla_abs6d/
  libero_spatial_agentview_rot180_wrist_raw_lerobot_v30_xvla_abs6d/
```

The launcher validates all four suites and installs `meta/modality.json`, which
maps `observation.xvla_abs_ee6d` into state and action modalities.

## Training (8-GPU node)

```bash
VLM_PATH=openbmb/MiniCPM-RobotManip \
LIBERO_EE6D_ROOT=playground/Datasets/LIBERO_EE6D \
bash examples/modelExtensions/MiniCPM-RobotManip/train_files/run_libero_train.sh
```

Key settings (`train_files/minicpm_robotmanip_libero.yaml`):

| Parameter | Value |
|---|---|
| action_dim / state_dim | 80 |
| action_horizon | 30 |
| image resolution | 448 × 448 |
| base LR / action-head LR | 1e-7 / 3e-6 |
| total steps / warmup | 1 500 / 100 |
| batch size per GPU | 12 |
| repeated diffusion steps | 8 |

The training path follows the LIBERO branch of the released model's mixed
post-training setup: the same robot/action/FPS prompt, action offsets 1–30,
unified 80-D mask, repeated diffusion steps 8, length-balanced sampling, and
removal of incomplete action chunks. Both camera views are resized to the same
448×448 resolution used by the released model and evaluation interface. The
clean-action loss is xyz masked-MSE ×500 + rotation6D masked-MSE ×10 + gripper
masked-L1. The example is intended to provide a correct, runnable
full-parameter training path; downstream performance depends on the dataset and
training setup and is not guaranteed by this recipe.

## Exporting a checkpoint

Training checkpoints use starVLA's framework-prefixed state dict. To create a
local Hugging Face model directory for downstream use, strip that prefix and
copy the released model's code and tokenizer files:

```bash
python examples/modelExtensions/MiniCPM-RobotManip/train_files/export_checkpoint.py \
  --ckpt playground/Checkpoints/minicpm_robotmanip_libero_full_finetune/checkpoints/steps_1500_pytorch_model.pt \
  --base /path/to/MiniCPM-RobotManip \
  --out playground/Exported/minicpm_robotmanip_libero_step1500
```

The exported directory is local output; this recipe does not require publishing
a separate fine-tuned checkpoint.

## LIBERO results

| Recipe | LIBERO-10 | Goal | Object | Spatial | Overall |
|---|---:|---:|---:|---:|---:|
| Simple LIBERO full-parameter fine-tuning, step 1 500 | 93.4 | 98.8 | 99.6 | 94.6 | 96.60 |

This is the result produced by the compact, single-dataset LIBERO recipe in
this PR. The released generalist model uses a broader multi-embodiment
post-training setup, which is outside the scope of this example.

## Files

| File | Description |
|---|---|
| `starVLA/model/framework/VLM4A/MiniCPMRobotManip.py` | framework: loads released model + adds training loss |
| `train_files/data_registry/data_config.py` | 80-D EE6D LIBERO data config (auto-discovered) |
| `train_files/modality.json` | maps `observation.xvla_abs_ee6d` → state/action |
| `train_files/install_modality.py` | validates each suite and installs the mapping |
| `train_files/minicpm_robotmanip_libero.yaml` | training config |
| `train_files/run_libero_train.sh` | 8-GPU launch script |
| `train_files/export_checkpoint.py` | export a trainer checkpoint to HF format |
