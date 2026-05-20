# WMH CALVIN ABC->D Exploration Report

Last updated: 2026-05-20
Owner: WMH
Repo scope: `examples/calvin_autoresearch`
Target: CALVIN ABC training -> CALVIN D closed-loop evaluation

## 1. Problem Setting And Compliance Boundary

The target task is to build and improve a compliant StarVLA baseline for CALVIN ABC -> D.

Hard constraints used throughout this exploration:

- Training data for this line stays CALVIN ABC-only.
- CALVIN D is used for closed-loop evaluation only.
- Do not initialize from upstream action-trained checkpoints such as LIBERO, RoboTwin, RoboCasa, Behavior, or CALVIN-D action checkpoints.
- Base VLM checkpoints are allowed. The main base VLM is `Qwen3-VL-4B-Instruct-Action`.
- Checkpoints trained by us on CALVIN ABC are allowed as continuation sources.

Main shared paths:

```text
private repo:
/inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA

shared project root:
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin

shared base model:
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/models/base/Qwen3-VL-4B-Instruct-Action

shared ABC LeRobot data:
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/datasets/calvin_lerobot

official CALVIN D eval data:
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d
```

## 2. Evaluation Metrics Used

The primary metric is CALVIN chained success on D:

- `avg_seq_len`: average number of consecutive subtasks completed in each 5-task sequence.
- `chain_sr@k`: fraction of sequences completing at least `k` consecutive instructions.
- `per_atomic_task`: success rate per atomic CALVIN task.
- `failure_position`: where each chain first fails.
- `conditional_success`: success probability conditioned on reaching position `k`.
- `action_stats`: action magnitude, saturation, jitter, gripper switch rate.
- `near_miss`: whether failed rollout accidentally achieved related or other tasks.

For quick exploration, `n100`/`n300` evaluations were used. For stable baselines, `n1000` was used when time allowed.

## 3. Main Result Table

All rows below use CALVIN D evaluation unless noted.

| Branch | Checkpoint / Eval | N | Avg Seq Len | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| WMH state8+connector 8k | `eval_8k_state8_connector_steps8000_fast_w4x8_d_n1000_0519_143010` | 1000 | 1.086 | 53.2% | 27.9% | 15.2% | 8.0% | 4.3% | Stable baseline |
| WMH base8k n300 | `eval_compare_d_n300_0520_031701/base8k` | 300 | 1.050 | 54.0% | 25.7% | 13.7% | 8.0% | 3.7% | Smaller eval, consistent with n1000 |
| WMH LoRA2000 | `eval_compare_d_n300_0520_031356/lora2000` | 300 | 1.630 | 64.0% | 43.7% | 27.3% | 18.3% | 9.7% | Clear improvement over base8k |
| WMH hardv2 aug | `eval_compare_d_n300_0520_031701/aug_hardv2` | 300 | 1.847 | 72.0% | 51.0% | 29.3% | 20.3% | 12.0% | Best verified WMH branch so far |
| WMH mirror hardv2 | manually aggregated from `eval_compare_d_n300_0520_033750/mirror_hardv2/worker_*` | 300 | 1.753 | 72.7% | 47.7% | 29.0% | 16.3% | 9.7% | Helpful for first-step/generalization, not better than non-mirror hardv2 |
| WMH MoE-Adaptive | `eval_moe_adaptive_d_n300_0520_043745` | 300 | 1.397 | 67.7% | 38.0% | 19.0% | 10.3% | 4.7% | Mixed; improves some light tasks but hurts drawer |
| GTY MoE95k | `eval_abc_augmented_moe_GTY_0519_092147_abc_to_d_n1000_0519_145552/results.json` | 100 | 1.910 | 76.0% | 54.0% | 32.0% | 17.0% | 12.0% | Strong external team branch; only n100 so not fully comparable |

Current running branches not yet evaluated:

- `abc_moe95k_lora_aug_3h_bs96_0520_045012`: GTY MoE95k-derived branch, fresh LoRA, no mirror, `BATCH_SIZE=96`, `NUM_PROCESSES=8`.
- `abc_moe95k_lora_mirror_3h_bs96_0520_045300`: same with left/right mirror augmentation.

Both have already produced at least `steps_3000_pytorch_model.pt` and are still training at roughly `1.5-1.6s/step`.

## 4. Candidate Technical Routes And Main Route Selection

The exploration did not follow a single predetermined architecture.  We compared several candidate routes under the same compliance boundary: ABC-only training, D-only evaluation, and no upstream action-trained initialization.  The selection criterion was not only `avg_seq_len`; we also considered whether the route directly addressed observed failure patterns, whether it was stable under H200 training, and whether it could be combined with the rest of the system without invalidating previous checkpoints.

| Candidate Route | Core Change | Evidence | Advantage | Risk / Limitation | Decision |
| --- | --- | --- | --- | --- | --- |
| A. Frozen-Qwen + GR00T action-head scaling | Keep Qwen frozen, train StarVLA QwenGR00T action head on ABC | Baseline and state8+connector stayed around `avg_seq_len=1.05-1.09`, `SR@5=3.7-4.3%` | Clean compliant baseline; simple and reproducible | Does not fix hard tasks or long-chain failure | Keep as reference only |
| B. State/proprio + connector | Add 8-D robot state and train the interface/action head | State8+connector n1000: `avg_seq_len=1.086`, `SR@5=4.3%` | Theoretically important for robot pose and gripper state | Current result does not prove state is used; needs zero/shuffle sanity tests | Keep as medium-term support, not current main route |
| C. Hard-task balanced data + controlled language/image augmentation | Oversample hard atomic tasks; add canonical mapping, paraphrases, and task-aware light visual augmentation | hardv2 aug n300: `avg_seq_len=1.847`, `SR@1=72.0%`, `SR@5=12.0%` | Best verified WMH branch; directly targets failure mass | Gains are coupled across sampler/paraphrase/image aug; needs further ablations | Verified main route for WMH |
| D. Left/right mirror augmentation | Mirror images, swap left/right language, transform action axes | mirror hardv2 n300: `avg_seq_len=1.753`, `SR@1=72.7%`, `SR@5=9.7%` | Addresses left/right asymmetry without extra data | Slightly worse than non-mirror hardv2; possible wrist/action sign inconsistency | Keep as separate branch, not default |
| E. Qwen LoRA | Add small LoRA on late Qwen attention layers while keeping base Qwen mostly insulated | LoRA2000 n300: `avg_seq_len=1.630`, `SR@5=9.7%` | Improves grounding with limited trainable params | LoRA alone does not beat hardv2 data route | Combine with stronger data/head, do not use alone as final route |
| F. MoE / adaptive action head | Use specialized action-head capacity instead of a single GR00T DiT head | MoE-Adaptive n300: `avg_seq_len=1.397`, but light tasks very strong; GTY MoE95k n100: `avg_seq_len=1.91` | High upside for task-specific motion modes | Adaptive branch regressed drawer badly; GTY n100 not directly comparable | Promising candidate, requires n300/n1000 verification |
| G. GTY MoE95k + fresh LoRA + hardv2 data | Start from compliant team ABC MoE checkpoint, add new LoRA, train on WMH augmented ABC | Training currently running; n300 eval pending | Combines the three strongest signals: hard-task data, LoRA, MoE head | Can disturb strong MoE policy; mirror variant may add geometry noise | Current highest-upside experimental route |

### 4.1 Final Route Choice

The current verified main route is **hard-task balanced ABC training with controlled language and task-aware image augmentation**.  This is the strongest WMH result that already has comparable D n300 evidence.  It improves both first-step success and long-chain success while staying fully inside the ABC-only training rule.

The current highest-upside candidate route is **GTY MoE95k + fresh Qwen LoRA + WMH hardv2 augmented ABC data**.  This is not yet the verified final route because its D n300/n1000 results are still pending.  It is worth running because it composes the strongest observed factors:

- hard-task data distribution fixes the largest observed failure mass;
- LoRA improves representation adaptation beyond connector-only training;
- GTY MoE95k provides a stronger ABC-trained action-head initialization than the plain GR00T branch.

The route we are **not** prioritizing is simply training the frozen-Qwen GR00T baseline longer.  The detailed metrics show that the dominant failures are task/affordance/contact failures, not random action noise or pure undertraining.  More steps on the same distribution are unlikely to be the most efficient use of limited GPU time.

## 5. Failure Pattern Analysis

### 5.1 Chain-Level Failure Pattern

The first major failure pattern is early-chain collapse.  In base8k n300, `138/300` sequences fail at the first instruction, so nearly half the evaluation never tests later recovery or long-horizon composition.  hardv2 augmentation reduces first-position failures to `84/300`, which explains most of its aggregate gain.

| Branch | Fail @1 | Fail @2 | Fail @3 | Fail @4 | Fail @5 | Completed All 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base8k n300 | 138 | 85 | 36 | 17 | 13 | 11 |
| hardv2 aug n300 | 84 | 63 | 65 | 27 | 25 | 36 |
| LoRA2000 n300 | 108 | 61 | 49 | 27 | 26 | 29 |
| MoE-Adaptive n300 | 97 | 89 | 57 | 26 | 17 | 14 |

Interpretation:

- `SR@1` is the most sensitive indicator for whether the model understands single atomic tasks in D.
- hardv2 improves `SR@1` from `54.0%` to `72.0%`; this is a direct sign that hard-task coverage matters.
- Later positions still remain weak.  Even hardv2 completes only `36/300` full five-task chains, so long-horizon robustness is still unsolved.

### 5.2 Task-Level Failure Pattern

Failures are not uniformly distributed.  They cluster around slider, light/LED, drawer, and right/left directional block manipulation.  The most important observation is that each route fixes a different subset, so aggregate score alone is misleading.

| Failure Group | Evidence | Technical Meaning | Route Response |
| --- | --- | --- | --- |
| Slider left/right | base8k top failures include `move_slider_right=30`, `move_slider_left=28`; hardv2 still has `move_slider_left=36`, `move_slider_right=34` failures | Small contact target plus directional ambiguity; not solved by more generic training | Keep left/right-specific diagnostics; mirror only as a controlled branch |
| Lightbulb on/off | base8k `turn_off_lightbulb=19%`, `turn_on_lightbulb=17%`; hardv2 improves to `39%/41%`; MoE-Adaptive reaches `96%/93%` | Light tasks need state/affordance understanding and a distinct motion primitive | Combine hard-task data with stronger action head; do not rely on GR00T-only head |
| Drawer | hardv2 fixes `close_drawer=100%`, `open_drawer=99%`; MoE-Adaptive collapses to `close_drawer=0%`, `open_drawer=56%` | Architecture changes can help one skill while destroying another; drawer is a regression detector | Any MoE/LoRA branch must be checked per-task, not only by average score |
| Directional push/right tasks | hardv2 improves `push_*_right`, but not uniformly; mirror is mixed | Geometry transform and language/action sign must be exact | Keep preview rendering and sign-convention tests for mirror |
| Stack/lift/place tasks | Frequent timeout failures remain across branches | Gripper timing and contact sequencing remain weak | Consider gripper head/hysteresis and action-history conditioning later |

### 5.3 Timeout And Near-Miss Pattern

For the detailed n300 metrics, failed subtasks almost always fail by reaching the evaluation horizon (`failure_step=360`).  This indicates that most failures are not immediate crashes; the policy spends the full rollout attempting something that does not satisfy the predicate.  This pattern is consistent with contact/affordance errors or wrong subtask grounding.

Near-miss rates also support this interpretation:

| Branch | Failed Subtasks | Any-Task Near Miss | Related-Task Near Miss |
| --- | ---: | ---: | ---: |
| base8k | 289 | 10.0% | 4.8% |
| hardv2 aug | 264 | 11.4% | 5.3% |
| LoRA2000 | 271 | 11.8% | 5.5% |
| MoE-Adaptive | 286 | 22.0% | 9.1% |

MoE-Adaptive has a much higher near-miss rate, which suggests that it often manipulates the scene but does not complete the requested predicate.  This is why it is promising for motion capacity but risky as a final route without task-level balancing.

### 5.4 Action-Diagnostic Pattern

Action statistics do not explain the main improvements.  base8k and hardv2 have similar jitter (`0.153` vs `0.151`) and similar gripper switch rates (`2.04%` vs `2.16%`), yet hardv2 is much stronger.  LoRA is slightly smoother, but its gain is smaller than hardv2.  MoE-Adaptive is more aggressive (`jitter=0.181`) and improves light tasks while regressing drawer.

Therefore, the primary failure is not simply "actions are too noisy."  The stronger explanation is:

1. The model under-recognizes a small set of D-critical hard tasks.
2. Contact affordances such as slider handle, drawer handle, and light switch are brittle.
3. Some action-head architectures can express useful task-specific motions but may over-specialize.
4. Long chains compound these atomic failures; fixing `SR@1` is necessary but not sufficient.

### 5.5 How Failure Pattern Drives The Main Route

The chosen main route follows directly from these patterns:

- Because failures are concentrated in a small set of tasks, use hard-task balanced sampling.
- Because hard tasks are language-templated and visually small, use controlled paraphrases and task-aware image augmentation rather than strong generic augmentation.
- Because representation adaptation helps but is not sufficient alone, add LoRA only after establishing a stronger data distribution.
- Because MoE can solve some hard motion modes but can regress others, use MoE as a high-upside candidate and require per-task n300 verification before treating it as the final route.

This gives a defensible route hierarchy:

1. **Verified WMH main route:** hardv2 augmented ABC training.
2. **Current high-upside candidate:** GTY MoE95k + fresh LoRA + hardv2 ABC data.
3. **Separate diagnostic branch:** left/right mirror, used only if it improves directional task subsets.
4. **Deferred support work:** proprio/state sanity checks, gripper head, auxiliary state/success prediction.

## 6. Exploration Path

### 6.1 Baseline: Qwen3-VL + QwenGR00T Action Head

Direction selected:

- Use `Qwen3-VL-4B-Instruct-Action` as the base VLM.
- Use StarVLA QwenGR00T action head.
- Train on CALVIN ABC only.
- Evaluate on official CALVIN D.

Why:

- It satisfies the rule that base models are allowed while upstream action-trained checkpoints are not.
- It gives a clean reference point before adding state, adapters, LoRA, or data augmentation.
- QwenGR00T already matches the StarVLA code path and action chunking interface.

Training:

```text
abc_pretrain_qwen3vl_gr00t_headonly_h200_60k_0518_163437
```

Observation:

- The first working closed-loop D eval pipeline was established from this line.
- Performance was low, especially for long chains and hard first subtasks.
- This established that pipeline correctness alone was not enough; model/data changes were needed.

Improvement direction:

- Keep this branch only as a compliance and infrastructure baseline.
- Do not spend more time scaling the frozen/action-only baseline.

### 6.2 Evaluation Infrastructure

Direction selected:

- Build one-click CALVIN D eval around StarVLA websocket policy server.
- Add parallel eval over multiple worker/server pairs.
- Add metrics beyond average chain length:
  - conditional success
  - failure position
  - per-atomic-task success
  - action magnitude/saturation/jitter
  - near-miss rate

Why:

- Early eval was too slow and too sparse to guide design.
- Hard-task failures needed task-level evidence, not only aggregate success.

Effect:

- Formal n1000 eval became feasible for stable checkpoints.
- n300 eval became practical for candidate ranking.
- The added diagnostics revealed that improvement was mostly from first-step success and hard task handling, not from smoother actions alone.

Improvement direction:

- Keep n300 for candidate ranking.
- Run n1000 only for finalists.
- Add automatic report comparison script for `base -> candidate` diffs.

### 6.3 State / Proprio Input

Direction selected:

- Enable 8-D CALVIN proprio/state:
  `state.x, state.y, state.z, state.roll, state.pitch, state.yaw, state.pad, state.gripper`.
- Add eval support for sending state via `CALVIN_SEND_STATE=1`.
- Train state-aware branch together with connector changes.

Why:

- RGB + language alone forces the model to infer robot pose, gripper state, drawer/slider/light state, and current subtask progress from a single frame.
- Strong VLA systems usually include proprio and sometimes wrist/state history.

Observed effect:

- State-aware smoke and small eval passed; model metadata correctly reported `model_state_dim=8`.
- The 8k state+connector n1000 result was stable but weak:
  - `avg_seq_len=1.086`
  - `SR@1=53.2%`
  - `SR@5=4.3%`
- This is close to the n300 base8k result and not a meaningful improvement by itself.

Interpretation:

- The state path is wired, but current training did not prove that proprio is semantically used.
- Possible causes:
  - action head already depends strongly on visual/token features;
  - state normalization or embedding scale may be weak;
  - no task-specific objective pressures the model to use state;
  - 8k steps may be too short for this branch.

Planned verification:

- Eval with state zeroed.
- Eval with state shuffled across sequences.
- Train `state_projector + action head` only.
- Train connector without state.
- Compare gradient norms for `state_projector`, `qwen_vl_interface`, and `action_model`.

Improvement direction:

- Keep state as a medium-term path but do not treat it as solved.
- Consider short proprio history instead of single-frame proprio.
- Add explicit auxiliary state prediction/success prediction before relying on state for final score.

### 6.4 Connector / Interface Training

Direction selected:

- Stop freezing the whole `qwen_vl_interface`.
- Keep Qwen backbone frozen.
- Train connector/interface plus action head.

Why:

- The VLM backbone may provide useful visual-language features, but the bridge from Qwen features to action tokens must adapt to CALVIN.
- Full Qwen fine-tuning is expensive and riskier; connector training is a conservative middle ground.

Observed effect:

- Connector + state 8k alone did not move metrics strongly.
- However, later LoRA and augmentation branches indicate that representation adaptation is important.

Interpretation:

- Connector training alone is not enough.
- It likely needs better data distribution, hard-task sampling, and/or LoRA.

Improvement direction:

- Keep connector trainable.
- Pair connector with either hard-task augmentation or LoRA.
- Track connector gradient norms and action-head gradients in logs.

### 6.5 Hard-Task Balanced Sampling

Direction selected:

- Oversample hard tasks identified from failure analysis:
  - `open_drawer`
  - `close_drawer`
  - `move_slider_left`
  - `turn_off_led`
  - `turn_off_lightbulb`
  - `turn_on_lightbulb`
  - `push_*_right`
- Implement task-balanced sampling against canonical task labels.

Why:

- The main failure mass came from a small set of hard atomic tasks.
- Plain ABC sampling underexposes these tasks relative to their impact on D-chain failure.

Effect:

- Combined hardv2 augmentation gave the strongest verified WMH result:
  - base8k n300: `avg_seq_len=1.05`, `SR@1=54.0%`, `SR@5=3.7%`
  - hardv2 aug n300: `avg_seq_len=1.847`, `SR@1=72.0%`, `SR@5=12.0%`
- Hard task improvements:

| Task | base8k | hardv2 aug | LoRA2000 | MoE-Adaptive |
| --- | ---: | ---: | ---: | ---: |
| `close_drawer` | 85% | 100% | 100% | 0% |
| `open_drawer` | 60% | 99% | 92% | 56% |
| `move_slider_left` | 18% | 16% | 28% | 13% |
| `move_slider_right` | 38% | 52% | 37% | 90% |
| `turn_off_led` | 83% | 100% | 88% | 85% |
| `turn_off_lightbulb` | 19% | 39% | 44% | 96% |
| `turn_on_lightbulb` | 17% | 41% | 30% | 93% |
| `push_blue_block_right` | 33% | 45% | 50% | 17% |
| `push_pink_block_right` | 27% | 47% | 50% | 25% |
| `push_red_block_right` | 17% | 40% | 31% | 14% |

Interpretation:

- Hard-task balance is a high-value direction.
- It especially helped drawer, LED/light, and right-push tasks.
- `move_slider_left` remains hard and needs either better left/right symmetry handling or more targeted data.

Improvement direction:

- Keep hard-task sampler enabled.
- Tune oversampling weights using per-task D failure contribution.
- Add task-specific eval subsets so one weak task does not require full n300 each time.

### 6.6 Language Paraphrase And Task-Aware Image Augmentation

Direction selected:

- Add canonical task mapping.
- Add controlled language paraphrases for hard tasks.
- Add task-aware light visual augmentation rather than strong generic perturbations.
- Keep canonical task label unchanged.

Why:

- CALVIN language is templated; hard tasks benefit from broader language grounding.
- Strong random crop/color jitter can damage small affordances such as drawer handles and switches.

Effect:

- The best verified WMH branch (`hardv2 aug`) includes this path together with hard-task balancing.
- Compared with LoRA2000, hardv2 aug had better first-step success and similar/better long-chain success:
  - hardv2 aug: `SR@1=72.0%`, `SR@5=12.0%`
  - LoRA2000: `SR@1=64.0%`, `SR@5=9.7%`

Interpretation:

- Data distribution and task-aware augmentation gave a stronger signal than LoRA alone on the WMH state+connector base.
- However, this result is coupled with hard-task sampling; separate ablations should be used before attributing gains only to paraphrase/image augmentation.

Improvement direction:

- Keep conservative image augmentation.
- Generate preview sheets for any future geometry-affecting augmentation.
- Add automatic task-label/paraphrase coverage report.

### 6.7 Left/Right Mirror Augmentation

Direction selected:

- Mirror primary and wrist images.
- Swap `left`/`right` in language.
- Apply action sign transforms for left/right axes.
- Restrict to left/right tasks where transform is defined.
- Generate visualization samples for manual inspection.

Why:

- `move_slider_left` and `push_*_right` showed persistent asymmetry.
- Mirroring can increase geometric coverage without downloading non-ABC datasets.

Effect:

- Manually aggregated n300 mirror hardv2:
  - `avg_seq_len=1.753`
  - `SR@1=72.7%`
  - `SR@5=9.7%`
- This is better than base8k, but slightly worse than non-mirror hardv2 (`avg_seq_len=1.847`, `SR@5=12.0%`).

Interpretation:

- Mirror is not obviously harmful, but it is not yet a clear win.
- The likely risk is imperfect action/sign convention or view consistency, especially for wrist camera and robot-centric contact geometry.
- It may help first-step robustness while slightly hurting later-chain consistency.

Improvement direction:

- Keep mirror as a separate branch, not default.
- Validate per-task mirror benefits, especially `move_slider_left/right` and `push_*_left/right`.
- Add a sign-convention unit test using oracle action labels and rendered before/after samples.
- Consider lower mirror probability instead of always-on oversampling.

### 6.8 LoRA Exploration

Direction selected:

- Add Qwen LoRA to the last 4 language layers.
- Target `q_proj,k_proj,v_proj,o_proj`.
- Keep rank small (`rank=8`, `alpha=16`) and preserve base Qwen backbone.

Why:

- Connector-only adaptation had limited effect.
- Full Qwen fine-tuning is too expensive and risks destroying base VLM semantics.
- LoRA is a controlled way to adapt grounding and late language/vision-action coupling.

Effect:

- LoRA2000 n300:
  - `avg_seq_len=1.63`
  - `SR@1=64.0%`
  - `SR@5=9.7%`
- This is clearly above base8k and state+connector, but below hardv2 aug.

Interpretation:

- LoRA is useful.
- LoRA alone is not enough to beat task-balanced augmentation.
- The best next hypothesis is combining LoRA with MoE/head improvements and hard-task data.

Improvement direction:

- Continue LoRA only from stronger checkpoints.
- Keep LR modest (`5e-6` for LoRA).
- Evaluate LoRA+hardv2 and LoRA+MoE separately.
- Avoid increasing LoRA rank until data-side improvements saturate.

### 6.9 MoE And Adaptive Action Head

Direction selected:

- Evaluate/continue team MoE branch.
- Add an adaptive MoE wrapper branch.
- Explore whether action-head architecture can solve hard tasks better than GR00T DiT alone.

Why:

- GTY MoE95k showed strong n100 D result.
- Certain tasks may need specialized motion modes; MoE may express this better than a single head.

Effect:

- GTY MoE95k n100:
  - `avg_seq_len=1.91`
  - `SR@1=76.0%`
  - `SR@5=12.0%`
- WMH MoE-Adaptive n300:
  - `avg_seq_len=1.397`
  - `SR@1=67.7%`
  - `SR@5=4.7%`

Task-level behavior:

- MoE-Adaptive strongly improves `turn_off_lightbulb` and `turn_on_lightbulb`.
- It severely hurts `close_drawer`.
- It improves `move_slider_right` but not `move_slider_left`.

Interpretation:

- MoE architecture has signal, but adaptive branch is unstable/task-biased.
- GTY MoE is promising but needs comparable n300/n1000 evaluation.
- Architecture changes should be paired with hard-task data, not judged by aggregate n100 alone.

Improvement direction:

- Use GTY MoE95k as a strong action-head initialization, since it is ABC-trained by the team.
- Add fresh LoRA on top, then train with WMH hard-task augmented data.
- Run n300 comparison against GTY MoE95k and WMH hardv2.

### 6.10 Current MoE95k + Fresh LoRA + Augmented ABC Training

Direction selected:

- Start from GTY MoE95k ABC checkpoint.
- Add fresh Qwen LoRA.
- Train two variants:
  - no mirror: `abc_moe95k_lora_aug_3h_bs96_0520_045012`
  - mirror: `abc_moe95k_lora_mirror_3h_bs96_0520_045300`
- Use ABC augmented data only.
- Use aggressive H200 utilization:
  - `NUM_PROCESSES=8`
  - `BATCH_SIZE=96`
  - `DATALOADER_NUM_WORKERS=12`
  - `SAVE_INTERVAL=1000`

Why:

- The strongest signals so far are:
  - hard-task augmentation,
  - LoRA,
  - GTY MoE action head.
- Combining these is the highest-upside remaining experiment under the time budget.

Current status:

- Both branches are training normally.
- Both have produced at least `steps_3000_pytorch_model.pt`.
- No D eval has been run yet for these 3h branches.

Risks:

- Large batch may reduce optimization noise and overfit to hard-task oversampling.
- Starting from MoE95k and adding LoRA may disturb a strong existing policy.
- Mirror branch may add sign/view inconsistency.

Required evaluation:

- After training window ends, evaluate:
  - no mirror latest checkpoint, D n300;
  - mirror latest checkpoint, D n300;
  - if one wins, run D n1000.
- Compare to:
  - WMH hardv2 aug n300;
  - WMH LoRA2000 n300;
  - GTY MoE95k n100/n300 if available.

## 7. Action Diagnostics

Across verified branches, action saturation is low for continuous dimensions but gripper is always binarized to `1.0` in postprocessed stats. Useful observed values:

| Branch | Jitter Mean L2 | Gripper Switch Rate | Interpretation |
| --- | ---: | ---: | --- |
| base8k | 0.153 | 2.04% | Baseline motion scale |
| hardv2 aug | 0.151 | 2.16% | Similar smoothness; gains are not from smoother action alone |
| LoRA2000 | 0.143 | 1.89% | Slightly smoother; better chain success |
| MoE-Adaptive | 0.181 | 1.95% | More aggressive motion; task-specific gains but drawer regression |

Interpretation:

- The best performance improvement did not come from reducing action jitter.
- The bottleneck is more likely task grounding, hard-task data coverage, and contact/affordance execution.
- Separate gripper head remains plausible, but not yet implemented/evaluated.

## 8. What Worked

1. Parallel eval and richer metrics were essential.
2. Hard-task balanced sampling plus controlled language/image augmentation produced the strongest verified WMH result.
3. LoRA improved over state+connector baseline.
4. GTY MoE appears promising and should be evaluated at larger N.
5. Aggressive 8-GPU batch-96 training is stable so far on H200.

## 9. What Did Not Clearly Work

1. State/proprio + connector alone did not produce measurable D improvement.
2. Mirror augmentation is not a verified win yet.
3. MoE-Adaptive improved light tasks but regressed drawer tasks and aggregate score.
4. Training longer without changing data/model direction is unlikely to be the best use of GPU time.

## 10. Recommended Next Steps

Immediate:

1. Finish current `MoE95k + LoRA + aug` and `MoE95k + LoRA + mirror` training.
2. Run D n300 for both latest checkpoints.
3. If either beats hardv2 aug, run D n1000.
4. Preserve all final configs, logs, and checkpoint paths.

Near term:

1. Run a comparable n300/n1000 eval for GTY MoE95k.
2. Run state zero/shuffle eval on state-aware checkpoints to verify proprio usage.
3. Evaluate mirror only on left/right task subsets.
4. Tune hard-task sampler weights using n300 per-task failures.
5. Add a separate gripper-head or gripper hysteresis ablation.

Longer term:

1. Add lightweight auxiliary prediction heads:
   - drawer open/closed;
   - slider left/right;
   - light/LED on/off;
   - success/value prediction.
2. Add short observation/state history.
3. Consider wrist-specific augmentation and camera-consistent geometry transforms.
4. Use n1000 only for finalist branches, not for every probe.

## 11. Submission Summary

The route selection is driven by the failure pattern.  The baseline mainly fails on a small set of hard atomic tasks, and most failures time out at the horizon rather than crashing immediately.  This points to grounding/contact/affordance errors rather than simple action-noise problems.

The most reliable verified WMH route is therefore hard-task balanced ABC training with language and task-aware image augmentation:

```text
base8k n300:       avg_seq_len 1.05, SR@5 3.7%
hardv2 aug n300:   avg_seq_len 1.85, SR@5 12.0%
```

LoRA is also useful, but it is not the best standalone route:

```text
LoRA2000 n300:     avg_seq_len 1.63, SR@5 9.7%
```

MoE action heads are promising, especially GTY MoE95k, but they must be judged with per-task metrics because MoE-Adaptive improved light tasks while regressing drawer tasks.  The current highest-upside candidate combines GTY MoE95k, WMH hard-task augmented ABC data, and fresh Qwen LoRA, with and without mirror augmentation.  It should become the main route only if D n300/n1000 verifies that it beats hardv2 without creating new task regressions.
