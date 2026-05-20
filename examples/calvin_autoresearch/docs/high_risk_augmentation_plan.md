# High-Risk CALVIN Augmentation Plan

Last updated: 2026-05-19

## Implementation Status

Implemented on 2026-05-19:

- Canonical mapping refinement for switch up/down:
  - `push/move/slide the switch down` -> `turn_off_lightbulb`
  - `push/move/slide the switch up` -> `turn_on_lightbulb`
- Language paraphrase augmentation, config-gated under `datasets.vla_data.language_augmentation`.
- Task-aware light image augmentation, config-gated under `datasets.vla_data.image_augmentation`.
- Camera-specific image profiles:
  - static camera: photometric plus very small crop/translation.
  - wrist camera: photometric only, no crop/translation.
- On/off balance diagnostics in `check_calvin_task_sampling.py`.
- Preview tools:
  - `examples/calvin_autoresearch/scripts/preview_calvin_language_aug.py`
  - `examples/calvin_autoresearch/scripts/preview_calvin_image_aug.py`

New experiment configs:

```text
examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang.yaml
examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang_taskaug.yaml
```

New training wrappers:

```text
examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_lang_h200.sh
examples/calvin_autoresearch/scripts/run_train_abc_state_connector_balanced_lang_taskaug_h200.sh
```

Still deliberately not implemented:

- Left/right horizontal mirror augmentation.
- Any action or proprio/state transform.
- Scene-state oracle input to the policy.

## Safety Snapshot

Before detailed left/right mirror planning, the current working state was saved at:

```text
safety_snapshots/left_right_mirror_plan_20260519_111307
```

The detailed mirror plan is maintained in:

```text
examples/calvin_autoresearch/docs/left_right_mirror_plan.md
```

Before planning the next high-risk changes, the current working state was saved at:

```text
safety_snapshots/high_risk_aug_20260519_093017
```

It contains:

- `base_commit.txt`: current git base commit.
- `status_short.txt`: working tree status at snapshot time.
- `tracked_changes.patch`: binary-capable patch for tracked-file changes.
- `untracked_files.txt`: untracked-file list.
- `untracked_files.tar.gz`: archive of untracked files at snapshot time.

Rollback principle: do not use destructive git rollback commands casually. If a later high-risk branch needs to be abandoned, first save a new snapshot of the current state, then restore from this snapshot deliberately.

## Pretraining vs Post-Training

Not all augmentation types should be treated the same.

Safe enough for pretraining, after a short probe:

- Hard-task balanced sampling. It only changes sampling frequency.
- Conservative language paraphrase, if canonical task id and direction are unchanged.
- Very light photometric augmentation, if preview confirms small affordances remain visible.

Better as post-training or short fine-tuning first:

- Task-aware crop/translation for drawer/slider/switch affordances.
- Stronger photometric augmentation.
- On/off balancing that depends on inferred visual state rather than task text.

Post-training only until validated:

- Left/right horizontal mirror augmentation.
- Any augmentation that changes action values.
- Any augmentation that changes state/proprio values.

Reason: the current best line uses `include_state=true`. If an image is mirrored, the image, language, action, and proprio state must all remain physically consistent. A wrong sign or wrong mirror center can poison the supervision.

## Current Repo Facts

Training data path:

```text
playground/Datasets/calvin_lerobot/calvin_abc_train_v3.0
```

Current LeRobot parquet columns:

```text
state, actions, timestamp, frame_index, episode_index, index, task_index
```

The current LeRobot training path does not expose `scene_obs`. Official raw CALVIN `.npz` data under `/public/inspire_shared` does have `scene_obs`, but using it in training would require a separate conversion or an auxiliary label pipeline.

Current state/action layout:

```text
state:  x, y, z, roll, pitch, yaw, pad, gripper
action: x, y, z, roll, pitch, yaw, gripper
```

Left/right action sign diagnostic on 120 episodes per task showed:

```text
move_slider_left   action.x sum mean ~= -5.87
move_slider_right  action.x sum mean ~= +4.52
push_*_left        action.x sum mean ~= -5.73 to -6.38
push_*_right       action.x sum mean ~= +5.84 to +6.29
```

So left/right is primarily `action.x` in this dataset. `action.yaw` also shifts with direction, but it is not the primary translation dimension and should not be flipped without an explicit distribution test.

## 1. Left/Right Horizontal Mirror

Status: feasible in principle, high risk in the current state-conditioned model.

Required synchronized changes:

- Flip `video.primary_image` horizontally.
- Decide whether to flip `video.wrist_image`; wrist camera may not have the same semantic left/right relationship as the static camera.
- Swap language:
  - `left` -> temporary token -> `right`
  - `right` -> `left`
- Swap canonical task ids:
  - `move_slider_left` <-> `move_slider_right`
  - `push_{color}_block_left` <-> `push_{color}_block_right`
- Transform action:
  - at minimum `action.x *= -1`.
  - do not flip `action.yaw` until a candidate-transform distribution check says it improves alignment.
- Transform state/proprio:
  - because `include_state=true`, `state.x` cannot be left unchanged.
  - naive `state.x *= -1` is likely wrong because state is absolute table/world coordinate.
  - safer candidate is `state.x = 2 * x_center - state.x`, where `x_center` must be estimated from dataset symmetry or CALVIN env constants.

Required diagnostic before any training:

- Add `check_lr_flip_sign_convention.py`.
- Compare real left/right trajectory distributions against candidate mirrored transforms:
  - candidate A: flip `action.x` only.
  - candidate B: flip `action.x` and `action.yaw`.
  - candidate C: flip `state.x` around estimated center and `action.x`.
- Report per-dimension mean/median/std and distribution distance.
- Require the transformed left distribution to match real right distribution better than the untransformed left distribution.

Recommendation:

- Do not put mirror into the next long train.
- Implement diagnostics first.
- If validated, try a 200-step probe and `NUM_SEQUENCES=100` eval.
- For the current `include_state=true` line, mirror should remain disabled until `state.x` transform is validated.

## 2. Light/LED State Balancing

Status: partially implemented now, full oracle-state balancing requires extra conversion.

Current limitation:

- LeRobot ABC training data only has robot state, action, and task index.
- It does not include `scene_obs`, so it cannot directly tell whether LED/lightbulb is currently on/off except through task language.

Low-risk immediate version:

- Improve canonical text mapping:
  - `push the switch downwards` -> likely `turn_off_lightbulb`
  - `push the switch upwards` -> likely `turn_on_lightbulb`
  - `slide/move/push the switch down` -> likely `turn_off_lightbulb`
  - `slide/move/push the switch up` -> likely `turn_on_lightbulb`
- Add an on/off balance report:
  - `turn_on_lightbulb` vs `turn_off_lightbulb`
  - `turn_on_led` vs `turn_off_led`
- Add optional sampler targets for on/off task pairs instead of only hard failures.

Implemented low-risk version:

- Switch up/down mapping is now part of `canonicalize_calvin_task()`.
- `check_calvin_task_sampling.py` now reports on/off pairs.
- Current ABC diagnostic:
  - lightbulb count: `on=525`, `off=525`
  - LED count: `on=525`, `off=526`
  - current hard-task sampler intentionally oversamples `turn_off_lightbulb` and `turn_off_led` because they dominate baseline failures.

Higher-risk version:

- Read official raw ABC `.npz` and use `scene_obs` to derive oracle labels.
- Either convert those labels into LeRobot metadata or use them only for diagnostics/auxiliary heads.
- Do not feed `scene_obs` as final policy input unless rules explicitly allow it.

Recommendation:

- First fix text mapping and sampling reports.
- Use oracle `scene_obs` only for analysis or auxiliary-label generation, not as a policy input.

## 3. Drawer/Slider Contact-Preserving Crop

Status: implemented as a conservative task-aware profile, medium risk for long training.

Current implementation:

- `balanced_aug` already applies small photometric jitter plus small crop/translation.
- It does not know the drawer handle, slider handle, switch, or LED location.
- It relies on small perturbation limits to avoid removing small affordances.

Recommended next refinement:

- Add task-aware crop profiles:
  - drawer/slider/switch tasks: smaller max translation and crop scale closer to 1.0.
  - block push tasks: slightly more tolerant translation.
- Optionally disable crop/translation for `video.wrist_image`; keep photometric jitter only on wrist.
- Add preview requirements:
  - save before/after static and wrist frames for each hard task.
  - record mean pixel diff.
  - manually inspect that handle/switch/LED remains visible.

Implemented refinement:

- `task_profiles` reduce hard affordance tasks to `max_translate_ratio: 0.02` and `scale_range: [0.98, 1.00]`.
- `camera_profiles` disable crop/translation for `video.wrist_image`.
- Preview records include the resolved profile per camera plus mean pixel difference.

Do not implement yet:

- Learned or heuristic bounding boxes unless we add a reliable detector/label source.
- Strong random crop.

Recommendation:

- This is a reasonable post-training fine-tune after balanced-only.
- Keep it short first: `2k~5k` steps, then `D n100/n200` eval.

## 4. Language Paraphrase

Status: implemented and relatively low risk.

Why it fits the repo:

- Language comes from `annotation.human.action.task_description`.
- We already have canonical task mapping.
- Replacing the task string before tokenization does not require changing images, actions, or state.

Implementation plan:

- Add `language_augmentation` config:

```yaml
datasets:
  vla_data:
    language_augmentation:
      enabled: true
      apply_to: hard_tasks
      probability: 0.3
      paraphrases:
        turn_off_lightbulb:
          - switch off the light bulb
          - turn the light off
          - deactivate the bulb
        close_drawer:
          - close the drawer
          - push the drawer shut
        move_slider_left:
          - slide the door to the left
          - move the sliding door left
```

Rules:

- Do not paraphrase to a sentence that changes direction or object.
- For left/right tasks, templates must be tied to canonical id.
- Keep canonical task id unchanged.
- Save a language preview report before training.

Recommendation:

- Good candidate for both pretraining and post-training.
- Implement before mirror augmentation.
- Test with a 200-step probe, then compare `D n100/n200`.

Implemented details:

- Applied before image augmentation and before tokenization.
- Uses deterministic per-sample RNG seeded from epoch, dataset index, trajectory id, step, and augmentation name.
- Default probability is `0.3`; preview script can override to `1.0`.
- State/action tensors are not changed.

## Proposed Execution Order

Stage A: Mapping and Reports

- Improve canonical text mapping for switch up/down.
- Add reports for:
  - canonical task counts.
  - on/off balance.
  - left/right action sign convention.
- No training behavior change except better task-balanced sampling.

Validation:

- `check_calvin_task_sampling.py`.
- new `check_lr_flip_sign_convention.py`.
- dataset sample check.

Stage B: Language Paraphrase

- Add config-gated `language_augmentation`. Completed.
- Add preview script that prints original -> augmented examples. Completed.
- Run 200-step probe. Pending GPU run.

Validation:

- no dataloader errors.
- no changed action/state tensors.
- `D n100/n200` after short fine-tune.

Stage C: Contact-Preserving Crop Profiles

- Refine existing `image_augmentation`. Completed.
- Add per-task/camera controls. Completed.
- Run preview. Completed.
- Run 200-step probe. Pending GPU run.

Validation:

- visual preview inspection.
- speed parity with current `balanced_aug`.
- `D n100/n200`.

Stage D: Left/Right Mirror

- Diagnostics only first.
- Do not train until action and state transforms are validated.
- If validation passes, use a tiny probe, then short post-training fine-tune.

Validation:

- left/right candidate transform report.
- preview images and language swaps.
- assert state/action dimensions transformed as intended.
- 200-step probe.
- `D n100/n200`.

## Current Recommendation

For the next serious run after the current long train:

1. Evaluate the current state+connector checkpoint.
2. Continue with `balanced-only` first.
3. Add language paraphrase or contact-preserving crop as short post-training experiments.
4. Keep mirror augmentation out of long training until state/action transform diagnostics pass.
