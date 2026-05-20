# CALVIN Improvement Plan: State, Connector, Hard-Task Sampling, Gripper

Last updated: 2026-05-19
Owner path: `examples/calvin_autoresearch`
Primary goal: improve the compliant CALVIN ABC -> D baseline without using upstream action-trained checkpoints.

## Non-Negotiable Constraints

- Training data for this line must stay CALVIN ABC-only unless a later decision explicitly opens a separate experiment.
- Do not load upstream action-trained StarVLA/GR00T/OFT checkpoints such as LIBERO, RoboTwin, RoboCasa, Behavior, or CALVIN-D action checkpoints.
- Allowed initialization is base VLM only, currently `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action`.
- Final closed-loop evaluation target is CALVIN D using the official raw task_D_D environment/data.
- Each major change must have its own config/run id and must be evaluated against the current action-only baseline.

## Current Baseline Snapshot

Baseline checkpoint:

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_pretrain_qwen3vl_gr00t_headonly_h200_60k_0518_163437/checkpoints/steps_60000_pytorch_model.pt
```

Baseline config facts:

- Framework: `QwenGR00T`
- VLM backbone: `Qwen3-VL-4B-Instruct-Action`
- Action head: GR00T flow-matching DiT head
- Training data: `calvin_abc_train_v3.0`
- `include_state: false`
- `freeze_modules: qwen_vl_interface`
- Loss: action-only VLA loss

Baseline eval/report locations:

```text
formal n1000:
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/reports/eval_abc_to_d_parallel_n1000_0519_053605

debug GIF n128:
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/reports/eval_abc_to_d_debug_gif_n128_0519_071808

snapshot doc:
examples/calvin_autoresearch/docs/baseline_snapshot_2026-05-19.md
```

## Stage Overview

| Stage | Change | Status | Main Risk | Required Validation |
| --- | --- | --- | --- | --- |
| 0 | Freeze baseline evidence | Completed | Losing comparability | Saved metrics and config |
| 1 | 5.1 `include_state=true`, 8-D proprio | Smoke passed, pending real train | train/eval state normalization mismatch | dataset sample + train smoke + eval smoke |
| 2 | 5.2 lightweight VLM-to-action connector, no LoRA | Implemented, pending GPU smoke | accidentally unfreezing Qwen 4B | trainable param audit |
| 3 | 6 hard-task balanced sampler + language paraphrase + task-aware light image aug | Implemented, CPU validated, pending GPU probe | wrong natural-language to canonical-task mapping or too-strong image aug | sampled task distribution report + language/image previews + 200-step probe |
| 4A | 7.1 eval-time gripper hysteresis | Pending | delayed or over-sticky gripper | same checkpoint A/B eval |
| 4B | 7.1 separate gripper head | Pending | larger architecture/load compatibility change | forward/load/eval compatibility tests |
| 5 | Combined training and formal ABC->D eval | Pending | attribution becomes unclear | ablation matrix |

## Stage 0: Baseline Evidence

Purpose: keep one fixed comparison point before touching model/data behavior.

Tasks:

- Save current training config and checkpoint path in this document.
- Save latest D eval directory and summarized metrics.
- Save hard-task failure table.
- Save action statistics:
  - mean absolute action
  - max absolute action
  - saturation rate
  - jitter
  - gripper switch rate

Validation command:

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh

python examples/calvin_autoresearch/scripts/summarize_eval_metrics.py \
  /path/to/baseline_eval_dir
```

Completion criteria:

- `baseline_snapshot_2026-05-19.md` contains the checkpoint, config facts, D eval directory, hard-task failure table, and action statistics.
- This document has the baseline eval directory filled in.

## Stage 1: 5.1 Enable Proprio/State

Purpose: give the action head the robot state it already supports architecturally.

Repo facts:

- CALVIN LeRobot ABC state is 8-D:
  `state.x, state.y, state.z, state.roll, state.pitch, state.yaw, state.pad, state.gripper`.
- QwenGR00T already reads optional `example["state"]`.
- GR00T action head already has `state_encoder`, so this is mainly data/config/eval plumbing.
- Current eval client sends image + language only.

Required code/config changes:

- Add or finalize `calvin_franka` data registry with explicit 8-D state keys.
- Add state-aware data mix:
  - logical mix: `calvin_abc_train_state_v3.0`
  - physical dataset path remains `calvin_abc_train_v3.0`
- Add state config:
  - `include_state: true`
  - `framework.action_model.state_dim: 8`
  - `data_mix: calvin_abc_train_state_v3.0`
- Add eval option:
  - `CALVIN_SEND_STATE=1`
  - map CALVIN `robot_obs` to `robot_obs[:8]`, shape `(1, 8)`.
  - note: the LeRobot CALVIN state names are `x, y, z, roll, pitch, yaw, pad, gripper`; the last two entries align with CALVIN's two gripper-state values in `robot_obs[6:8]`.
- Decide and enforce one state scaling path:
  - Preferred: training normalizes state and policy server applies the same state transform at eval.
  - Temporary acceptable path: raw state in both training/eval, explicitly documented.

Do not use:

- `scene_obs` as final model input.
- Any D demonstrations in training.

Implemented files:

- `examples/calvin_autoresearch/train_files/data_registry/data_config.py`
- `examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8.yaml`
- `examples/calvin_autoresearch/scripts/run_train_abc_state_h200.sh`
- `examples/calvin_autoresearch/scripts/run_train_abc_pretrain_h200.sh`
- `examples/calvin_autoresearch/scripts/verify_assets.sh`
- `examples/calvin/eval_files/eval_calvin.py`
- `deployment/model_server/policy_norm_processor.py`
- `deployment/model_server/policy_wrapper.py`
- `examples/calvin_autoresearch/scripts/run_eval_d_formal.sh`
- `examples/calvin_autoresearch/scripts/run_eval_abc_to_d_oneclick.sh`
- `examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh`

CPU-side validation completed on 2026-05-19:

- `py_compile` passed for modified Python files.
- `bash -n` passed for modified shell scripts.
- `DRY_RUN=1` passed for the original action-only config and the new state8 config.
- Dataset sample check returned `state_shape (1, 8)` and `action_shape (8, 7)`.

GPU smoke completed on 2026-05-19:

- Run id: `abc_state8_smoke20_0519_075852`
- Checkpoint: `members/WMH/runs/abc_state8_smoke20_0519_075852/checkpoints/steps_20_pytorch_model.pt`
- Eval command used `CALVIN_SEND_STATE=1`, `NUM_SEQUENCES=10`.
- Server metadata confirmed `model_state_dim: 8`, `model_action_dim: 7`, and CALVIN state keys.
- Eval completed and wrote metrics to:
  `/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/reports/eval_abc_to_d_formal_n10_0519_080016/metrics.json`
- Result was 0/10 first-task success, which is expected for a 20-step smoke checkpoint and should be treated only as pipeline validation.

Validation:

```bash
# 1. Static/syntax checks
python -m py_compile examples/calvin/eval_files/eval_calvin.py
bash -n examples/calvin_autoresearch/scripts/run_train_abc_pretrain_h200.sh
bash -n examples/calvin_autoresearch/scripts/run_eval_d_formal.sh

# 2. Dataset sample check
python - <<'PY'
from omegaconf import OmegaConf
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
cfg = OmegaConf.load("examples/calvin_autoresearch/train_files/<state_config>.yaml")
ds = get_vla_dataset(cfg.datasets.vla_data)
sample = ds[0]
print(sample.keys())
print(sample["state"].shape)
print(sample["action"].shape)
PY

# 3. Train smoke on GPU
MAX_TRAIN_STEPS=20 DATA_MIX=calvin_abc_train_state_v3.0 CONFIG_YAML=examples/calvin_autoresearch/train_files/<state_config>.yaml \
  bash examples/calvin_autoresearch/scripts/run_train_abc_pretrain_h200.sh

# 4. Eval smoke
CALVIN_SEND_STATE=1 NUM_SEQUENCES=10 GPU_ID=0 PORT=5694 \
  bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_oneclick.sh
```

Completion criteria:

- Train smoke saves a valid checkpoint.
- Eval smoke finishes and reports metrics.
- Policy server metadata or logs make state usage auditable.

## Stage 2: 5.2 Lightweight Connector, No LoRA

Purpose: allow the Qwen hidden states to adapt to CALVIN action prediction while keeping Qwen 4B frozen.

Important repo mismatch:

- The suggested `qwen_vl_interface.adapter` and `qwen_vl_interface.layernorm` modules do not exist in the current `QwenGR00T`.
- The trainer does not support `train_modules`; it supports `freeze_modules` plus per-module learning rates.

Correct local design:

- Add `vl_connector` inside `Qwen_GR00T`.
- Apply it after Qwen `last_hidden` and before `action_model`.
- Suggested connector:

```text
LayerNorm(hidden_dim)
Linear(hidden_dim, hidden_dim)
GELU
Linear(hidden_dim, hidden_dim)
```

Config pattern:

```yaml
trainer:
  freeze_modules: qwen_vl_interface
  learning_rate:
    base: 2.5e-05
    vl_connector: 5.0e-05
    action_model: 1.0e-04
```

Required code/config changes:

- Add connector config under `framework`, e.g.:

```yaml
framework:
  vl_connector:
    enabled: true
    type: mlp
    hidden_dim: 2560
    dropout: 0.0
```

- Add connector module to `Qwen_GR00T.__init__`.
- Use connector in both `forward()` and `predict_action()`.
- Ensure old checkpoints without connector remain loadable, or keep connector experiments in new checkpoints only.

Implemented design:

- Added `VLMTokenConnector` in `starVLA/model/framework/VLM4A/QwenGR00T.py`.
- Default `framework.vl_connector.enabled=false`, so old configs/checkpoints keep the old no-connector path.
- Enabled connector uses residual MLP:
  `LayerNorm -> Linear(H, 512) -> GELU -> Linear(512, H)`, with zero-initialized final linear.
- Because it is residual and zero-initialized, the connector starts as exact identity and then learns an adapter delta.
- Connector is applied after Qwen hidden states and before the GR00T action head in both training and inference.
- `trainer.freeze_modules: qwen_vl_interface` still freezes Qwen; `vl_connector` is a separate top-level module and remains trainable.
- `TrainerUtils.print_trainable_parameters()` now prints top-level trainable parameter counts for audit.

Implemented files:

- `starVLA/model/framework/VLM4A/QwenGR00T.py`
- `starVLA/training/trainer_utils/trainer_tools.py`
- `examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector.yaml`
- `examples/calvin_autoresearch/scripts/run_train_abc_state_connector_h200.sh`

CPU-side validation completed on 2026-05-19:

- `py_compile` passed for modified Python files.
- `bash -n` passed for modified shell scripts.
- Connector unit test passed:
  - enabled: `True`
  - output shape: `(2, 3, 8)`
  - trainable params in toy 8-D adapter: `92`
  - initial residual identity max abs diff: `0.0`
- `DRY_RUN=1 STRICT_ASSETS=1 bash examples/calvin_autoresearch/scripts/run_train_abc_state_connector_h200.sh` passed.
- `git diff --check` passed.

Validation:

```bash
# Print trainable params and confirm Qwen is frozen.
MAX_TRAIN_STEPS=1 CONFIG_YAML=examples/calvin_autoresearch/train_files/<state_connector_config>.yaml \
  bash examples/calvin_autoresearch/scripts/run_train_abc_pretrain_h200.sh
```

Required audit from logs:

- `qwen_vl_interface` has zero trainable params.
- `vl_connector` is trainable.
- `action_model` is trainable.

Completion criteria:

- 20-step smoke succeeds.
- 200-step probe succeeds.
- D eval smoke succeeds.

## Stage 3: 6 Hard-Task Data Strategy

Purpose: increase exposure to tasks that dominate failure while staying ABC-only.

High-risk follow-up planning is tracked separately in:

```text
examples/calvin_autoresearch/docs/high_risk_augmentation_plan.md
```

Repo facts:

- LeRobot ABC has natural-language task strings in `meta/tasks.jsonl` and per-episode task info in `meta/episodes.jsonl`.
- CALVIN eval uses canonical task ids, e.g. `turn_off_lightbulb`, `move_slider_left`.
- Therefore we need a mapping layer from natural language variants to canonical ids.
- This repo's `LeRobotMixtureDataset` already samples data inside `sample_step()`:
  - sample dataset by `dataset_sampling_weights`
  - sample trajectory by `trajectory_sampling_weights`
  - sample a random `base_index` inside that trajectory
- Therefore task balancing should first modify trajectory weights, not wrap the dataloader with a PyTorch sampler.
- Visual transforms should be applied in the dataset path after raw frame read and before PIL packing/model preprocessing.

### Stage 3A: Hard-Task Balanced Sampler

First implementation: sampler only, no content augmentation.

Disturbance implemented:

- Sampling distribution perturbation only.
- It changes how often an ABC episode/trajectory is sampled.
- It does not change image, wrist image, state, action, or language values.
- It never samples from D.

Target task families:

- `turn_off_lightbulb`
- `turn_off_led`
- `close_drawer`
- `open_drawer`
- `move_slider_left`
- `push_red_block_right`
- `push_blue_block_right`
- `push_pink_block_right`

Canonical mapping plan:

- Add a CALVIN task text mapper under `examples/calvin_autoresearch/`.
- Input: one natural-language task string from `meta/episodes.jsonl`.
- Output: canonical eval-like id or `other`.
- Initial mapping should be deterministic regex/string rules, not an LLM.
- Example rules:
  - contains `drawer` and `close`/`push`/`handle` with closing phrasing -> `close_drawer`
  - contains `drawer` and `open`/`pull` with opening phrasing -> `open_drawer`
  - contains `turn off`/`toggle`/`switch` and `yellow`/`lamp`/`light bulb`/`light` -> `turn_off_lightbulb`
  - contains `turn off`/`button` and `led`/`green light` -> `turn_off_led`
  - contains `slide`/`move` and `door`/`slider` and `left` -> `move_slider_left`
  - contains `push`/`slide`/`sweep`, block color, and `right` -> `push_{color}_block_right`

Repo implementation plan:

- In `LeRobotSingleDataset`, build an optional `trajectory_task_labels` array from `meta/episodes.jsonl`.
- In `LeRobotMixtureDataset.__init__`, after the normal length-based trajectory weights are computed, optionally multiply per-trajectory weights by the configured oversampling factor.
- Normalize weights after multiplication.
- Add a small report method or script that prints:
  - raw episode counts by canonical id
  - raw weighted sampling probability by canonical id
  - balanced weighted sampling probability by canonical id
  - unmatched `other` examples

Config shape:

```yaml
datasets:
  vla_data:
    sampler:
      type: task_balanced
      report_top_k: 30
      oversample_tasks:
        turn_off_lightbulb: 5.0
        close_drawer: 4.0
        move_slider_left: 3.0
        turn_off_led: 3.0
        push_red_block_right: 3.0
        push_blue_block_right: 2.0
        push_pink_block_right: 2.0
        open_drawer: 2.0
```

Concrete perturbation formula:

```text
base_weight_i = trajectory_length_i if balance_trajectory_weights else 1
canonical_i = map_language_to_canonical(episode_i.tasks[0])
factor_i = oversample_tasks.get(canonical_i, 1.0)
effective_weight_i = base_weight_i * factor_i
trajectory_sampling_weights = normalize(effective_weight)
```

Expected effect:

- More batches contain hard tasks without duplicating files or changing labels.
- Because the data itself is untouched, the risk of action-label corruption is very low.

Validation:

```bash
# Dry-run 10k samples and print canonical task distribution.
python examples/calvin_autoresearch/scripts/check_calvin_task_sampling.py \
  --dataset playground/Datasets/calvin_lerobot/calvin_abc_train_v3.0 \
  --config examples/calvin_autoresearch/train_files/<balanced_config>.yaml \
  --num-samples 10000
```

Required output:

- Before/after canonical task distribution.
- Confirm sampled paths are still under ABC dataset only.
- Confirm `other` is not accidentally dominant due to bad mapping.

Completion criteria:

- Distribution matches intended oversampling within tolerance.
- 200-step train probe does not show dataloader slowdown or repeated failure.
- D eval `NUM_SEQUENCES=100` reports per-task changes.

Implemented files:

- `starVLA/dataloader/gr00t_lerobot/datasets.py`
- `examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced.yaml`
- `examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_h200.sh`
- `examples/calvin_autoresearch/scripts/check_calvin_task_sampling.py`

CPU-side validation completed on 2026-05-19:

- `py_compile` passed for modified dataloader and sampling script.
- `bash -n` passed for the balanced training wrapper.
- `DRY_RUN=1 STRICT_ASSETS=1 bash examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_h200.sh` passed.
- Dataset sample check returned `state_shape (1, 8)` and `action_shape (8, 7)`.
- Sampling check on 10k samples showed the intended hard-task lift:
  - `turn_off_lightbulb`: base `0.0311` -> balanced `0.1049`
  - `close_drawer`: base `0.0311` -> balanced `0.0859`
  - `move_slider_left`: base `0.0314` -> balanced `0.0650`
  - `turn_off_led`: base `0.0280` -> balanced `0.0581`

After the 2026-05-19 switch up/down mapping refinement, the latest diagnostic showed:

- `push/move/slide the switch down` maps to `turn_off_lightbulb`.
- `push/move/slide the switch up` maps to `turn_on_lightbulb`.
- lightbulb ABC task counts are balanced: `on=525`, `off=525`.
- LED ABC task counts are balanced: `on=525`, `off=526`.

Recommended GPU probe:

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh

TS=$(date +%m%d_%H%M%S)
RUN_ID=abc_state8_connector_balanced_probe200_${TS}
LOG_DIR=/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/logs
mkdir -p "${LOG_DIR}"

nohup bash -c "
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh

STRICT_ASSETS=1 \
GPU_IDS=0,1,2,3,4,5,6,7 \
NUM_PROCESSES=8 \
BATCH_SIZE=96 \
DATALOADER_NUM_WORKERS=8 \
DATALOADER_PREFETCH_FACTOR=2 \
MAX_TRAIN_STEPS=200 \
SAVE_INTERVAL=100 \
RUN_ID=${RUN_ID} \
bash examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_h200.sh
" > "${LOG_DIR}/${RUN_ID}.log" 2>&1 &

echo "RUN_ID=${RUN_ID}"
echo "LOG=${LOG_DIR}/${RUN_ID}.log"
```

### Stage 3B: Light Visual Augmentation

Second implementation: apply low-risk image-only perturbations to hard tasks.

Disturbance implemented:

- Only RGB images are perturbed:
  - static camera image
  - wrist camera image
- State, action, gripper, and language remain unchanged.
- No left/right flip in this stage.
- No scene-state oracle input.
- No D data.

Why image-only and light:

- CALVIN D failures involve small affordances: drawer handle, sliding door handle, switch, light/LED buttons.
- Heavy random crop can erase these cues at 224x224.
- Geometry augmentation without action/language remapping can poison supervision.
- Therefore Stage 3B starts with photometric and very small translation/crop perturbations only.

Target tasks:

- Apply augmentation only to hard-task canonical ids from Stage 3A by default.
- Optionally apply a weaker global augmentation to all tasks later if D eval improves.

Config shape:

```yaml
datasets:
  vla_data:
    image_augmentation:
      enabled: true
      apply_to: hard_tasks
      probability: 0.5
      brightness: 0.08
      contrast: 0.08
      saturation: 0.06
      hue: 0.015
      max_translate_ratio: 0.04
      scale_range: [0.96, 1.00]
      protect_small_affordances: true
```

Concrete perturbations:

- Brightness:
  - multiply image intensity by a factor sampled from `[0.92, 1.08]`.
- Contrast:
  - move pixels away/toward per-image mean with factor `[0.92, 1.08]`.
- Saturation:
  - use PIL/torchvision color jitter with factor `[0.94, 1.06]`.
- Hue:
  - tiny hue shift `[-0.015, 0.015]`.
- Small crop/translation:
  - sample crop scale from `[0.96, 1.00]`.
  - sample horizontal/vertical offset within `4%` of image size.
  - resize back to model input size.
  - keep this disabled for wrist camera if it causes gripper/handle disappearance in GIF inspection.

Repo implementation plan:

- Add an optional augmentation module under `examples/calvin_autoresearch/train_files/data_registry/` or `starVLA/dataloader/gr00t_lerobot/transform/`.
- Prefer implementing it in the dataset path before `_pack_sample()` converts images to PIL at fixed `224x224`.
- The transform should receive the canonical task id from the sampled trajectory.
- If `apply_to: hard_tasks`, skip augmentation when canonical id is not in the configured hard-task set.
- Use deterministic seeding from `(epoch, index, trajectory_id, base_index)` so distributed workers produce reproducible augmentations for a given sample.
- Keep all augmentation parameters in YAML and default `enabled: false`.

Concrete code path options:

- Minimal invasive option:
  - In `LeRobotMixtureDataset.__getitem__`, after `raw_data = dataset.get_step_data(...)`, infer canonical task from the selected trajectory.
  - Before `data = dataset.transforms(raw_data)`, apply image augmentation to keys in `dataset.modality_keys["video"]`.
  - This works because raw video arrays are still numpy arrays at that point.
- Cleaner option:
  - Add a `CalvinTaskAwareImageAugment` transform and include it in `CalvinFrankaDataConfig.transform()`.
  - This requires passing task id into `data`, so it is a slightly larger change.

Recommended first implementation:

- Use the minimal invasive option in `LeRobotMixtureDataset.__getitem__`.
- Keep the code guarded by `datasets.vla_data.image_augmentation.enabled`.
- Add distribution/preview script to save a few before/after images for each hard task.

Validation:

```bash
python examples/calvin_autoresearch/scripts/preview_calvin_image_aug.py \
  --config examples/calvin_autoresearch/train_files/<balanced_aug_config>.yaml \
  --output /tmp/calvin_aug_preview \
  --tasks turn_off_lightbulb close_drawer move_slider_left turn_off_led \
  --probability 1.0
```

Completion criteria:

- Preview images keep drawer handle, slider door, switches, LED/light visible.
- 200-step probe runs without dataloader slowdown.
- Compare `balanced-only` vs `balanced+visual-aug` on `NUM_SEQUENCES=100` before long training.

Implemented files:

- `starVLA/dataloader/gr00t_lerobot/datasets.py`
- `examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_aug.yaml`
- `examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_aug_h200.sh`
- `examples/calvin_autoresearch/scripts/preview_calvin_image_aug.py`

CPU-side validation completed on 2026-05-19:

- `py_compile` passed for the preview script.
- `bash -n` passed for the balanced+aug training wrapper.
- `DRY_RUN=1 STRICT_ASSETS=1 bash examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_aug_h200.sh` passed.
- Dataset sample check returned `state_shape (1, 8)` and `action_shape (8, 7)`.
- Preview images were saved to `/tmp/calvin_aug_preview_force`.
- Preview records showed non-zero mean pixel changes for `close_drawer`, `move_slider_left`, `turn_off_led`, and `turn_off_lightbulb`.

Recommended GPU probe:

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh

TS=$(date +%m%d_%H%M%S)
RUN_ID=abc_state8_connector_balanced_aug_probe200_${TS}
LOG_DIR=/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/logs
mkdir -p "${LOG_DIR}"

nohup bash -c "
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh

STRICT_ASSETS=1 \
GPU_IDS=0,1,2,3,4,5,6,7 \
NUM_PROCESSES=8 \
BATCH_SIZE=96 \
DATALOADER_NUM_WORKERS=8 \
DATALOADER_PREFETCH_FACTOR=2 \
MAX_TRAIN_STEPS=200 \
SAVE_INTERVAL=100 \
RUN_ID=${RUN_ID} \
bash examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_aug_h200.sh
" > "${LOG_DIR}/${RUN_ID}.log" 2>&1 &

echo "RUN_ID=${RUN_ID}"
echo "LOG=${LOG_DIR}/${RUN_ID}.log"
```

### Stage 3C: Language Paraphrase

Third implementation: replace hard-task language variants with canonical-safe paraphrases.

Disturbance implemented:

- Only task text is perturbed.
- Images, wrist images, state, action, and gripper labels remain unchanged.
- Canonical task id remains unchanged.
- No left/right swapping and no action sign change.

Config:

```text
examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang.yaml
```

Wrapper:

```text
examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_lang_h200.sh
```

Implemented files:

- `starVLA/dataloader/gr00t_lerobot/datasets.py`
- `examples/calvin_autoresearch/scripts/preview_calvin_language_aug.py`
- `examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang.yaml`
- `examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_lang_h200.sh`

CPU-side validation completed on 2026-05-19:

- `py_compile` passed.
- `bash -n` passed.
- `DRY_RUN=1 STRICT_ASSETS=1` passed.
- Dataset sample check returned `state (1, 8)` and `action (8, 7)`.
- Language preview saved 28 records across 14 hard-task ids to `/tmp/calvin_language_aug_preview.json`.
- Preview confirmed direction/object preserving examples, e.g.:
  - `push the switch downwards` -> `turn the light off`
  - `move down the switch` -> `deactivate the yellow lamp`
  - `slide the door to the left` -> `move the sliding door left`

Recommended GPU probe:

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh

TS=$(date +%m%d_%H%M%S)
RUN_ID=abc_state8_connector_balanced_lang_probe200_${TS}
LOG_DIR=/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/logs
mkdir -p "${LOG_DIR}"

nohup bash -c "
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh

STRICT_ASSETS=1 \
GPU_IDS=0,1,2,3 \
NUM_PROCESSES=4 \
BATCH_SIZE=96 \
DATALOADER_NUM_WORKERS=8 \
DATALOADER_PREFETCH_FACTOR=2 \
MAX_TRAIN_STEPS=200 \
SAVE_INTERVAL=100 \
RUN_ID=${RUN_ID} \
bash examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_lang_h200.sh
" > "${LOG_DIR}/${RUN_ID}.log" 2>&1 &

echo "RUN_ID=${RUN_ID}"
echo "LOG=${LOG_DIR}/${RUN_ID}.log"
```

### Stage 3D: Task-Aware Light Augmentation

Fourth implementation: combine hard-task balancing, language paraphrase, and conservative task/camera-aware image augmentation.

Disturbance implemented:

- Language paraphrase as in Stage 3C.
- Hard-task static camera image augmentation:
  - photometric jitter.
  - crop scale `[0.98, 1.00]` and max translation `0.02` for drawer/slider/light/LED tasks.
- Wrist camera image augmentation:
  - photometric jitter only.
  - crop/translation disabled.
- State, action, and gripper labels remain unchanged.

Config:

```text
examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang_taskaug.yaml
```

Wrapper:

```text
examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_lang_taskaug_h200.sh
```

Implemented files:

- `starVLA/dataloader/gr00t_lerobot/datasets.py`
- `examples/calvin_autoresearch/scripts/preview_calvin_image_aug.py`
- `examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang_taskaug.yaml`
- `examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_lang_taskaug_h200.sh`

CPU-side validation completed on 2026-05-19:

- `py_compile` passed.
- `bash -n` passed.
- `DRY_RUN=1 STRICT_ASSETS=1` passed.
- Dataset sample check returned `state (1, 8)` and `action (8, 7)`.
- Image preview saved records to `/tmp/calvin_taskaware_aug_preview/records.json`.
- Preview records confirmed:
  - `video.primary_image`: `photometric=true`, `crop_translate=true`, `max_translate_ratio=0.02`, `scale_range=[0.98, 1.0]`.
  - `video.wrist_image`: `photometric=true`, `crop_translate=false`.

Recommended GPU probe:

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh

TS=$(date +%m%d_%H%M%S)
RUN_ID=abc_state8_connector_balanced_lang_taskaug_probe200_${TS}
LOG_DIR=/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/logs
mkdir -p "${LOG_DIR}"

nohup bash -c "
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh

STRICT_ASSETS=1 \
GPU_IDS=0,1,2,3 \
NUM_PROCESSES=4 \
BATCH_SIZE=96 \
DATALOADER_NUM_WORKERS=8 \
DATALOADER_PREFETCH_FACTOR=2 \
MAX_TRAIN_STEPS=200 \
SAVE_INTERVAL=100 \
RUN_ID=${RUN_ID} \
bash examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_lang_taskaug_h200.sh
" > "${LOG_DIR}/${RUN_ID}.log" 2>&1 &

echo "RUN_ID=${RUN_ID}"
echo "LOG=${LOG_DIR}/${RUN_ID}.log"
```

### Deferred: Geometry Augmentation

Do not implement in Stage 3A/3B/3C/3D:

- Horizontal flip.
- Left/right language swap.
- Action sign swap.

Reason: left/right augmentation must coordinate static image, wrist image, language, canonical task, and action dimensions. A sign mistake poisons the dataset.

## Stage 4A: 7.1 Eval-Time Gripper Hysteresis

Purpose: cheap diagnostic before changing architecture.

Current behavior:

- Model outputs 7D action.
- Eval turns gripper into binary with `action[-1] > 0`.

Required code/config changes:

- Add optional env flags:

```bash
CALVIN_GRIPPER_HYSTERESIS=1
CALVIN_GRIPPER_OPEN_TH=0.25
CALVIN_GRIPPER_CLOSE_TH=-0.25
```

- Keep previous gripper state when raw value is in the deadband.

Validation:

- Same checkpoint, same eval sequence count, hysteresis off vs on.
- Compare:
  - success rate
  - gripper switch rate
  - close/open related tasks
  - failure step

Completion criteria:

- Hysteresis is retained only if it improves or does not hurt D eval.

## Stage 4B: 7.1 Separate Gripper Head

Purpose: train continuous motion and binary gripper with appropriate losses.

Required architecture changes:

- Flow/Diffusion branch predicts first 6 action dims.
- New gripper head predicts horizon-wise binary logits.
- Training target maps gripper action:
  - `-1` -> class 0
  - `+1` -> class 1
- Total loss:

```text
loss = flow_6d_loss + lambda_gripper * gripper_bce_or_focal_loss
```

Recommended additions:

- Transition-frame weighting for gripper changes.
- Metrics:
  - gripper BCE
  - gripper accuracy
  - transition precision/recall

Required compatibility work:

- `predict_action()` still returns normalized or env-compatible 7D action.
- Policy server remains compatible with existing eval client.
- Checkpoint load must be explicit: gripper-head configs should not silently load old 7D-head checkpoints as if equivalent.

Validation:

```bash
# Forward smoke
MAX_TRAIN_STEPS=20 CONFIG_YAML=examples/calvin_autoresearch/train_files/<gripper_head_config>.yaml \
  bash examples/calvin_autoresearch/scripts/run_train_abc_pretrain_h200.sh

# Eval smoke
NUM_SEQUENCES=10 GPU_ID=0 PORT=5694 \
  bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_oneclick.sh
```

Completion criteria:

- 20-step and 200-step training runs are stable.
- Eval server returns 7D actions.
- D eval does not regress on place/lift tasks while improving gripper-sensitive tasks.

## Stage 5: Ablation Matrix

Run experiments in this order:

| ID | State | Connector | Balanced Sampler | Hysteresis | Gripper Head |
| --- | --- | --- | --- | --- | --- |
| A0 | no | no | no | no | no |
| A1 | yes | no | no | no | no |
| A2 | yes | yes | no | no | no |
| A3 | yes | yes | yes | no | no |
| A4 | yes | yes | yes | yes | no |
| A5 | yes | yes | yes | optional | yes |

Training schedule per experiment:

```text
20-step smoke -> 200-step probe -> 5k/10k short run -> 60k formal run
```

Eval schedule per experiment:

```text
NUM_SEQUENCES=10    flow smoke
NUM_SEQUENCES=50    quick direction check
NUM_SEQUENCES=100   hard-task comparison
NUM_SEQUENCES=1000  formal result
```

Minimum report fields:

- checkpoint path
- config path
- train command
- eval command
- eval dir
- avg sequence length
- chain success rates 1/5 through 5/5
- conditional success by task position
- hard-task per-atomic success
- failure step stats
- near-miss rates
- raw action stats
- env action stats
- gripper switch rate

## Run Log

Append new runs here.

### Run Template

```text
Date:
Stage:
Run ID:
Config:
Checkpoint:
Training data:
Command:
Eval dir:
Summary:
Decision:
```

## Open Decisions

- State scaling path:
  - Option A: raw state in train/eval.
  - Option B: normalized state in train and server-side normalized eval.
  - Preferred: Option B, but requires `PolicyNormProcessor.apply_state()`.
- Connector hidden size:
  - Start with hidden dim equal to Qwen hidden size.
  - Reduce only if memory or speed becomes a problem.
- Hard-task task mapping:
  - Need explicit canonical mapping report before sampler is trusted.
- Gripper head loss:
  - Start with BCE.
  - Use focal loss only if transition or class metrics justify it.

## Maintenance Rules

- Update this document before starting a new stage.
- After each run, add a Run Log entry.
- If a validation step fails, record the failure and do not mark the stage complete.
- Keep every experiment command reproducible from this document or a linked script.
- Do not overwrite old eval reports; create new timestamped directories.
