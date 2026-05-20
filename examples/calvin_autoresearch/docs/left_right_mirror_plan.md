# CALVIN Left/Right Mirror Plan

Last updated: 2026-05-19

## Goal

Evaluate and, only if validated, add horizontal left/right mirror augmentation for CALVIN ABC training without using any upstream action-trained checkpoint or D demonstrations.

This plan is deliberately diagnostic-first. The current best training line is state-conditioned (`include_state=true`), so an image-only mirror is not safe. A valid mirror sample must keep image, language, canonical task id, action chunk, and robot proprio/state mutually consistent.

## Safety Snapshot

Current progress was saved before this plan at:

```text
safety_snapshots/left_right_mirror_plan_20260519_111307
```

Contents:

- `base_commit.txt`
- `status_short.txt`
- `tracked_changes.stat.txt`
- `tracked_changes.patch`
- `untracked_files.txt`
- `untracked_files.tar.gz`
- `manifest.txt`

Rollback principle:

- Do not use destructive git rollback commands casually.
- If a later mirror implementation is abandoned, save a fresh snapshot first.
- Restore tracked edits from `tracked_changes.patch` and untracked files from `untracked_files.tar.gz` deliberately.

## Current Baseline For This Branch

Current non-mirror training line:

```text
state8 + vl_connector + hard-task balanced sampler + language paraphrase + task-aware light image augmentation
```

Probe run already completed:

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_state8_connector_balanced_lang_taskaug_probe200_0519_104717
```

Important config facts:

- Training data: `calvin_abc_train_state_v3.0`
- Physical dataset path: `playground/Datasets/calvin_lerobot/calvin_abc_train_v3.0`
- VLM base: `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action`
- `include_state: true`
- `framework.action_model.state_dim: 8`
- `trainer.freeze_modules: qwen_vl_interface`
- Trainable modules: `vl_connector` and `action_model`
- Action horizon: `8`

State layout:

```text
state:  x, y, z, roll, pitch, yaw, pad, gripper
```

Action layout:

```text
action: x, y, z, roll, pitch, yaw, gripper
```

Current dataset statistics from the probe run:

```text
state.x  min=-0.4322 max=0.4215 mean=0.0399 q01=-0.3220 q99=0.2930
action.x min=-1.0000 max=1.0000 mean=0.0009 q01=-0.7076 q99=0.6852
action.yaw min=-1.0000 max=1.0000 mean=-0.0051 q01=-1.0000 q99=1.0000
```

Earlier left/right action diagnostic:

```text
move_slider_left   action.x sum mean ~= -5.87
move_slider_right  action.x sum mean ~= +4.52
push_*_left        action.x sum mean ~= -5.73 to -6.38
push_*_right       action.x sum mean ~= +5.84 to +6.29
```

Interpretation:

- Left/right direction is primarily represented by `action.x`.
- `action.yaw` also changes with direction but may not be a simple sign flip.
- `state.x` is absolute robot pose, not an action delta; it cannot be naively multiplied by `-1`.
- State orientation (`roll`, `pitch`, `yaw`) under mirror is physically nontrivial and should not be changed without explicit evidence.

## Scope

Initial mirror candidates cover only tasks with simple left/right symmetry:

```text
move_slider_left  <-> move_slider_right
push_red_block_left   <-> push_red_block_right
push_blue_block_left  <-> push_blue_block_right
push_pink_block_left  <-> push_pink_block_right
```

Explicitly out of scope for the first implementation:

- `rotate_*_left/right`
- drawer open/close
- light/LED on/off
- lift/place/stack tasks
- any task where left/right is not the main spatial instruction

Reason: rotation tasks need orientation/action-yaw treatment, and non-directional tasks can be poisoned by arbitrary reflection.

## Required Mirror Components

For an eligible sample, a valid mirror transform must update:

- `video.primary_image`: horizontal flip.
- `video.wrist_image`: configurable; default should be disabled until preview/eval says otherwise.
- canonical task id: swap left/right pair.
- language: swap left/right words or replace with canonical mirrored templates.
- action chunk: transform all 8 horizon steps.
- state: transform current 8-D proprio only if the state transform candidate is validated.

Unchanged fields:

- gripper action.
- gripper state.
- z translation.
- non-eligible tasks.
- D data remains unused.

## Ordering In The Current Dataloader

Current order in `LeRobotMixtureDataset.__getitem__`:

```text
sample step
raw_data = dataset.get_step_data(...)
canonical_task = dataset.get_trajectory_canonical_task(...)
language_augmentation(raw_data, canonical_task)
image_augmentation(raw_data, canonical_task)
dataset.transforms(raw_data)
pack_sample(...)
```

Mirror should be inserted before language paraphrase and before image augmentation:

```text
sample step
raw_data = dataset.get_step_data(...)
canonical_task = dataset.get_trajectory_canonical_task(...)
raw_data, canonical_task = left_right_mirror(raw_data, canonical_task)
language_augmentation(raw_data, canonical_task)
image_augmentation(raw_data, canonical_task)
dataset.transforms(raw_data)
pack_sample(...)
```

Reason:

- If `move_slider_left` is mirrored into `move_slider_right`, later language paraphrase must sample from the right-task templates.
- Image crop/photo profiles should also use the mirrored canonical task.
- Action/state transforms must happen before `StateActionTransform` normalization.

## Stage 0: Snapshot

Status: completed.

Validation:

```bash
ls -lh safety_snapshots/left_right_mirror_plan_20260519_111307
cat safety_snapshots/left_right_mirror_plan_20260519_111307/manifest.txt
```

Completion criteria:

- Snapshot exists.
- Tracked diff patch exists.
- Untracked archive exists.

## Stage 1: Diagnostics Only

Status: completed for the first diagnostic pass on 2026-05-19.

No training behavior changes in this stage.

Saved reports:

```text
examples/calvin_autoresearch/reports/lr_mirror_diagnostics_20260519/candidate_summary.md
examples/calvin_autoresearch/reports/lr_mirror_diagnostics_20260519/candidate_summary.json
examples/calvin_autoresearch/reports/lr_mirror_diagnostics_20260519/preview_records.json
```

Temporary preview images were generated under:

```text
/tmp/lr_mirror_preview_all
```

Diagnostic result summary:

- `action.x` negation is strongly supported across all four left/right task pairs.
- `action.roll` negation improves the roll distribution in all four pairs.
- `action.yaw` negation improves the yaw distribution in all four pairs, but the improvement is weaker for block-push tasks than for slider.
- `state.x` mirroring is supported by distribution diagnostics, but center choice matters:
  - global mean center works across all pairs.
  - task-pair-specific centers are best for some pairs but add complexity and risk.
  - global midrange is not robust.
- Primary-camera mirror preview succeeded.
- Two visual preview variants were generated:
  - primary-only flip.
  - primary+wrist flip.
- Primary-only flip creates a real multi-view consistency risk: the static view is mirrored but the wrist view still reports the original local left/right arrangement.
- Primary+wrist flip is more consistent between views in image space, but it may create a wrist-camera geometry that the policy never sees at evaluation time.

Current diagnostic recommendation:

```text
If mirror proceeds to implementation, the image branch must be treated as an ablation:
  candidate V0: flip primary image only
  candidate V1: flip primary image and wrist image

The numeric transform candidate should be:
  swap left/right canonical task and language
  action.x *= -1
  action.roll *= -1
  action.yaw *= -1
  state.x = 2 * global_state_x_mean - state.x

Neither V0 nor V1 is approved for long training until a config-gated implementation
passes CPU sample checks, 200-step probe, and D n100/n200 eval. If only one can be
tried first, V1 is now the more geometrically consistent candidate, while V0 remains
useful as a controlled ablation.
```

### 1.1 Candidate Transform Report

Implemented script:

```text
examples/calvin_autoresearch/scripts/check_lr_mirror_candidates.py
```

Inputs:

```bash
--dataset playground/Datasets/calvin_lerobot/calvin_abc_train_v3.0
--config examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang_taskaug.yaml
--max-episodes-per-task 200
--output /tmp/lr_mirror_candidates
```

The script should:

- Read `meta/episodes.jsonl`.
- Map task text to canonical ids with `canonicalize_calvin_task()`.
- Load underlying LeRobot parquet for selected episodes.
- Collect raw unnormalized state/action arrays.
- Compare real left-task distributions against real right-task distributions.
- Compare mirrored-left candidate distributions against real right distributions.
- Compare mirrored-right candidate distributions against real left distributions.

Task pairs:

```text
move_slider_left/right
push_red_block_left/right
push_blue_block_left/right
push_pink_block_left/right
```

Candidate action transforms:

```text
A0: action.x *= -1
A1: action.x *= -1, action.yaw *= -1
A2: action.x *= -1, action.roll *= -1
A3: action.x *= -1, action.yaw *= -1, action.roll *= -1
```

Candidate state transforms:

```text
S0: no state transform
S1: state.x = 2 * 0.0 - state.x
S2: state.x = 2 * global_state_x_mean - state.x
S3: state.x = 2 * global_state_x_midrange - state.x
S4: state.x = 2 * pair_specific_center - state.x
```

Candidate centers:

```text
global_state_x_mean     ~= 0.0399
global_state_x_midrange ~= (-0.4322 + 0.4215) / 2 ~= -0.0053
pair_specific_center    = (mean_state_x_left + mean_state_x_right) / 2
```

Metrics:

- per-dimension mean/std/q01/q50/q99
- absolute mean gap before vs after mirror
- 1D Wasserstein distance for `action.x`, `action.yaw`, `state.x`
- bounds violation rate against original dataset min/max
- action direction score:
  - left tasks should have negative summed `action.x`
  - right tasks should have positive summed `action.x`
- roundtrip error:
  - applying the same transform twice should recover the original numeric values within tolerance.

Hard pass gates:

- `action.x` mirror must reduce left/right distribution distance for every task pair.
- Mirrored values must stay within original action/state ranges except for a tiny tail.
- `action.yaw` transform is enabled only if it improves distribution distance for most task pairs and does not worsen others strongly.
- `state.x` transform is enabled only if it improves the state distribution and has low bounds violation.
- If no state candidate passes, mirror must not be used in the current state-conditioned long training line.

### 1.2 Visual Preview Report

Implemented script:

```text
examples/calvin_autoresearch/scripts/preview_lr_mirror_aug.py
```

Outputs:

```text
/tmp/lr_mirror_preview/
  records.json
  <task>_<episode>_primary_before.png
  <task>_<episode>_primary_after.png
  <task>_<episode>_wrist_before.png
  <task>_<episode>_wrist_after.png
```

Preview variants:

```text
V0: flip primary only
V1: flip primary + wrist
```

Saved preview artifacts from the first pass:

```text
examples/calvin_autoresearch/reports/lr_mirror_diagnostics_20260519/lr_mirror_preview_contact_sheet.jpg
examples/calvin_autoresearch/reports/lr_mirror_diagnostics_20260519/lr_mirror_preview_contact_sheet_flip_wrist.jpg
examples/calvin_autoresearch/reports/lr_mirror_diagnostics_20260519/preview_records_with_permanent_paths.json
examples/calvin_autoresearch/reports/lr_mirror_diagnostics_20260519/preview_records_flip_wrist_with_permanent_paths.json
```

Manual review checklist:

- primary image is mirrored correctly.
- wrist flip does not create visually implausible gripper/camera geometry.
- slider/block target direction now visually matches mirrored language.
- no handle, block, slider, or gripper is cropped away by existing image aug after mirror.

Recommendation:

- Do not treat `flip_wrist=false` as the default safe choice.
- Compare `flip_wrist=false` and `flip_wrist=true` as two explicit ablations.
- If training budget permits only one first probe, prefer `flip_wrist=true` because it avoids an obvious cross-view left/right contradiction.

### 1.3 Language Swap Report

Implemented in `preview_lr_mirror_aug.py`; it prints and saves:

```text
original canonical -> mirrored canonical
original language  -> mirrored language
```

Rules:

- Swap only word-boundary `left` and `right`.
- Use temporary placeholders to avoid double replacement.
- Do not alter unrelated words.
- For safety, replacement is allowed only when canonical id is in the eligible task-pair map.

Examples:

```text
move_slider_left  + "slide the door to the left"  -> move_slider_right + "slide the door to the right"
push_red_block_right + "move the red block right" -> push_red_block_left + "move the red block left"
```

Hard pass gates:

- 100 sampled mirror language records have no wrong direction.
- Every mirrored canonical id belongs to the paired task.

## Stage 2: Config-Gated Implementation

Status: completed on 2026-05-19.

Stage 1 diagnostics passed and the primary+wrist visual candidate was manually accepted for the first probe.

The dataloader default remains disabled when this block is absent or `enabled: false`.
The dedicated lrmirror probe config enables this block explicitly:

```yaml
datasets:
  vla_data:
    spatial_augmentation:
      left_right_mirror:
        enabled: false
        probability: 0.25
        apply_to: lr_tasks
        flip_primary_image: true
        flip_wrist_image: true
        action_transform:
          x: negate
          roll: negate
          yaw: negate
        state_transform:
          x: mirror_center
          x_center: 0.03991219401359558
        tasks:
          move_slider_left: move_slider_right
          move_slider_right: move_slider_left
          push_red_block_left: push_red_block_right
          push_red_block_right: push_red_block_left
          push_blue_block_left: push_blue_block_right
          push_blue_block_right: push_blue_block_left
          push_pink_block_left: push_pink_block_right
          push_pink_block_right: push_pink_block_left
```

Implementation functions in `starVLA/dataloader/gr00t_lerobot/datasets.py`:

```text
_parse_lr_mirror_cfg(data_cfg)
swap_left_right_text(text)
swap_left_right_task(canonical_task, cfg)
apply_calvin_lr_mirror(raw_data, video_keys, language_keys, canonical_task, rng, cfg)
```

Added training config and wrapper:

```text
examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang_taskaug_lrmirror.yaml
examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_lang_taskaug_lrmirror_h200.sh
```

Required behavior:

- deterministic per-sample RNG, same seed pattern as existing language/image aug.
- return `(raw_data, canonical_task)`.
- copy `raw_data` before edits.
- transform all action horizon rows, not only the first action.
- transform state only if config says so.
- preserve dtype and shape.
- no-op for non-eligible tasks.

Code insertion point:

```text
after dataset.get_step_data(...)
before apply_calvin_language_augmentation(...)
```

Roundtrip tests:

- text swap twice returns original.
- canonical task swap twice returns original.
- image flip twice returns original.
- action transform twice returns original.
- state.x centered transform twice returns original.

## Stage 3: CPU Validation

Status: completed on 2026-05-19.

Commands:

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh

python -m py_compile \
  starVLA/dataloader/gr00t_lerobot/datasets.py \
  examples/calvin_autoresearch/scripts/check_lr_mirror_candidates.py \
  examples/calvin_autoresearch/scripts/preview_lr_mirror_aug.py

python examples/calvin_autoresearch/scripts/check_lr_mirror_candidates.py \
  --dataset playground/Datasets/calvin_lerobot/calvin_abc_train_v3.0 \
  --config examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang_taskaug.yaml \
  --max-episodes-per-task 200 \
  --output /tmp/lr_mirror_candidates

python examples/calvin_autoresearch/scripts/preview_lr_mirror_aug.py \
  --config examples/calvin_autoresearch/train_files/<mirror_config>.yaml \
  --output /tmp/lr_mirror_preview \
  --max-per-task 3 \
  --flip-wrist
```

Dataset sample check:

```bash
python - <<'PY'
from omegaconf import OmegaConf
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
cfg = OmegaConf.load("examples/calvin_autoresearch/train_files/<mirror_config>.yaml")
ds = get_vla_dataset(cfg.datasets.vla_data)
for idx in range(16):
    sample = ds[idx]
    print(idx, sample["state"].shape, sample["action"].shape, sample["lang"])
PY
```

Required outputs:

- no exceptions.
- no NaN/inf in sampled state/action.
- `state` remains `(1, 8)`.
- `action` remains `(8, 7)`.
- mirrored language direction matches mirrored canonical task.

Observed validation results:

- `python -m py_compile` passed for dataloader and diagnostic scripts.
- `bash -n` passed for the new H200 wrapper and base H200 launcher.
- YAML load confirmed `enabled=true`, `probability=0.25`, primary+wrist flip, `action.x/roll/yaw=negate`, and `state.x` center mirror.
- `DRY_RUN=1` launcher preflight passed and printed the expected ABC-only training command.
- Roundtrip checks passed for text swap, canonical task swap, image flip, action negation, and centered `state.x` mirror.
- Raw assertion on a real `move_slider_left` trajectory passed:
  - mirrored canonical task became `move_slider_right`;
  - language changed from `slide the door to the left` to `slide the door to the right`;
  - primary and wrist images matched horizontal flips;
  - `action.x`, `action.roll`, and `action.yaw` matched sign negation;
  - `state.x` matched `2 * 0.03991219401359558 - state.x`;
  - non-mirrored fields stayed unchanged.
- Packed sample check passed for four samples:
  - `action` shape `(8, 7)`;
  - `state` shape `(1, 8)`;
  - all sampled action/state values finite.

## Stage 4: GPU Probe Matrix

Do not start with long training.

Probe A: no-state sanity line, mirror action/image/language only.

- Purpose: isolate whether visual/language/action mirror works without proprio complications.
- Uses `include_state=false` config only for diagnostic training, not final comparison.

Probe B: state8 with validated `action.x/roll/yaw` mirror and validated `state.x` candidate.

- Purpose: actual candidate for the current line.
- Only run if Stage 1 validates `state.x`.

Probe C: state8 with primary-only flip.

- Purpose: isolate whether leaving wrist unflipped helps or hurts.
- This is now considered a risky ablation because it creates cross-view inconsistency.

Recommended first GPU command shape:

```bash
STRICT_ASSETS=1 \
GPU_IDS=0,1,2,3 \
NUM_PROCESSES=4 \
BATCH_SIZE=96 \
MAX_TRAIN_STEPS=200 \
SAVE_INTERVAL=100 \
RUN_ID=abc_state8_connector_balanced_lang_taskaug_lrmirror_probe200_${TS} \
bash examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_lang_taskaug_lrmirror_h200.sh
```

Training pass gates:

- 200 steps complete.
- no OOM.
- no NaN loss.
- loss curve is not clearly worse than non-mirror probe.
- trainable parameter audit remains:
  - Qwen frozen.
  - connector trainable.
  - action head trainable.

## Stage 5: Eval Matrix

Every mirror checkpoint must be compared to the non-mirror branch.

Smoke:

```bash
CALVIN_SEND_STATE=1 NUM_SEQUENCES=10 GPU_ID=0 PORT=5695 CKPT=<mirror_ckpt> \
bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_oneclick.sh
```

Quick comparison:

```bash
TOTAL_SEQUENCES=100 GPU_IDS=0,1,2,3 WORKERS_PER_GPU=1 BASE_PORT=6500 \
bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh
```

Required metrics:

- average successful sequence length.
- 1/5 to 5/5 chain success.
- per-atomic task success.
- first-task failure distribution.
- left/right task success:
  - `move_slider_left/right`
  - `push_red/blue/pink_block_left/right`
- non-left/right hard tasks:
  - drawer
  - light/LED
- action stats:
  - mean abs action
  - max abs action
  - saturation rate
  - jitter
  - gripper switch rate

Pass gates for longer training:

- left/right task success improves or shows a clear positive trend.
- no major regression on non-left/right hard tasks.
- no increase in action saturation or jitter that suggests label corruption.
- D n100 result is at least competitive with non-mirror branch.

## Stage 6: Long Training Gate

Mirror can enter long training only if all of these are true:

- Stage 1 transform diagnostics pass.
- Stage 2 implementation is config-gated and disabled by default.
- Stage 3 CPU validation passes.
- Stage 4 200-step probe passes.
- Stage 5 D n100/n200 eval does not regress materially.
- Manual image/language preview review is complete.

Recommended first long-train setting if accepted:

```yaml
left_right_mirror:
  enabled: true
  probability: 0.15
  flip_primary_image: true
  flip_wrist_image: true
```

Do not exceed `probability: 0.25` until eval shows benefit.

## Known Risks

- Wrong `state.x` center silently corrupts proprio input.
- Incorrect yaw/roll transform can corrupt contact orientation.
- Flipping wrist image may produce a camera geometry the policy never sees at eval.
- Left/right text swap can be wrong if applied outside eligible canonical ids.
- Mirroring only image/language/action while leaving state unchanged may teach contradictory supervision.
- Gains may be task-specific and hurt non-left/right tasks if overused.

## Decision Summary

## Incorporating `analysis(2).txt`

The mirror branch should absorb the recommendation as a conservative Stage-A
data-augmentation experiment, not as a reason to unfreeze Qwen.

Applied policy for subsequent mirror runs:

- Keep `qwen_vl_interface` frozen. Qwen visual encoder, LLM backbone, and main
  weights are not updated.
- Train only the bridge/action side available in this repo:
  `vl_connector` plus `action_model`. In this QwenGR00T implementation,
  `vl_connector` is the local equivalent of the suggested
  Qwen-to-action adapter.
- Lower connector LR from `5e-5` to `3e-5`; keep action/state side at `1e-4`.
  This follows the recommendation that adapter/connector LR should be lower
  than action-head LR.
- Do not enable Qwen LoRA or top-layer unfreeze for mirror until the full
  1000-sequence eval and state/connector ablations show that the frozen-Qwen
  line is genuinely saturated.
- Keep mirror probability at or below `0.25`; lower to `0.15` if D n100 shows
  regressions on non-left/right tasks.
- Continue logging gradient norms so we can verify the connector and action
  head are receiving gradients.

Not applied yet:

- Global `weight_decay=0.01`. The current optimizer path does not separate
  LayerNorm/bias no-decay parameters, so switching the whole model to `0.01`
  is deferred until param-group weight decay is implemented.
- Gripper BCE / auxiliary heads. These are separate action-head changes and
  should not be mixed into the first mirror-only conclusion.

Recommended next action:

1. Run/retain the completed 200-step lrmirror GPU probe as a training-sanity
   check.
2. Run D n10 smoke with `CALVIN_SEND_STATE=1` for the probe checkpoint if not
   already done.
3. For overnight exploration, train from the fixed 8k state8+connector source
   checkpoint with Qwen frozen and connector LR `3e-5`.
4. After training, run D n10 and D n100; only escalate to D n1000 if n100 is
   competitive with the non-mirror branch.

Current decision:

```text
mirror training status: 200-step probe passed; approved only as a guarded exploratory long-train branch, not as the new baseline
```
