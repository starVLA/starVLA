# RoboDojo

StarVLA supports [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)
training and, through
[XPolicyLab](https://github.com/JinhuiYE/XPolicyLab/tree/fix/starvla-hf-robodojo-eval).
This directory provides the StarVLA-side data registration, reproducible
training recipes, and a thin launcher for evaluating the released Hugging Face
checkpoints. RoboDojo remains the source of the simulator, tasks, assets, and
official scoring; XPolicyLab maintains the policy adapter and evaluation
runtime.

The released policies share the same observation/action contract:

| Item | Value |
|---|---|
| Base VLM | `Qwen/Qwen3-VL-4B-Instruct` |
| Training data | RoboDojo LeRobot v2.1, 3,500 episodes, 35 training tasks |
| RGB observations | Head, left wrist, right wrist; resized to 224 x 224 |
| Robot state | Raw 14D ARX X5 absolute-joint state |
| Action | 14D absolute joint position (`abs_qpos`) |
| Normalization | Saved `arx_x5` q99 statistics, including continuous grippers |
| Predicted action horizon | 50 |
| Evaluation replanning interval | Execute 16 actions, then request a new chunk |

## Training

### Download the official dataset

The recipes read RoboDojo's official 64 GB LeRobot v2.1 export directly. No
StarVLA data conversion is required.

```bash
cd /absolute/path/to/RoboDojo

ROBO_DOJO_DATA_ROOT=/absolute/path/to/shared-datasets \
  bash scripts/RoboDojo/download_data.sh huggingface lerobot_v2.1
```

The downloader creates:

```text
/absolute/path/to/shared-datasets/RoboDojo_lerobot_v21_video
```

The checked-in
[`modality.json`](train_files/modality.json) matches the metadata shipped with
that physical dataset. The data registry selects the three RGB streams used by
the released policies and ignores additional cameras.

### Select a recipe

| Variant | Action head | Recipe | Released step |
|---|---|---|---:|
| QwenOFT | MLP with L1 action regression | `starvla_robodojo_v21_qwenoft_h50_q99.yaml` | 100,000 |
| QwenGR00T | 16-layer DiT-B flow-matching head | `starvla_robodojo_v21_qwengroot_h50_q99.yaml` | 130,000 |
| QwenPI_v3 | 36-layer LayerwiseFM head | `starvla_robodojo_v21_qwenpi_v3_h50_q99.yaml` | 100,000 |

All recipes train the VLM, VLM interface, and action head end to end. Shared
optimization settings are a per-GPU batch size of 16, AdamW with betas
`(0.9, 0.95)`, VLM learning rate `1e-5`, action-model learning rate
`1e-4`, 5,000 warmup steps, and a cosine schedule.

### Launch training

Run from the StarVLA repository root. For example, to train QwenPI_v3:

```bash
config_yaml=examples/simBenchmarks/RoboDojo/train_files/starvla_robodojo_v21_qwenpi_v3_h50_q99.yaml \
robodojo_data_root=/absolute/path/to/shared-datasets \
bash examples/simBenchmarks/RoboDojo/train_files/run_robodojo_train.sh
```

The benchmark registry is auto-discovered from
[`train_files/data_registry/data_config.py`](train_files/data_registry/data_config.py).
It registers the 35-task mixture as `robodojo_v21_all_h50_q99`, with a
50-action chunk and q99 transforms for all state/action dimensions.


## Evaluation

### Install RoboDojo and XPolicyLab

Use separate policy and simulator environments:

```bash
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git
cd RoboDojo

git clone --single-branch \
  --branch fix/starvla-hf-robodojo-eval \
  https://github.com/JinhuiYE/XPolicyLab.git

bash scripts/init_assets.sh

cd XPolicyLab/policy/starVLA
bash install.sh
```

Export the checkout and environment paths:

```bash
export ROBODOJO_PATH=/absolute/path/to/RoboDojo
export STARVLA_ENV_PATH=/absolute/path/to/starvla-policy-env
export ROBODOJO_ENV_PATH=/absolute/path/to/robodojo-simulator-env
```

XPolicyLab must be located at `$ROBODOJO_PATH/XPolicyLab`. Absolute environment
prefixes are recommended for non-interactive cluster jobs.

### Evaluate a released checkpoint

The StarVLA launcher delegates checkpoint download, hash verification, policy
serving, simulator startup, and result collection to XPolicyLab:

```bash
ROBODOJO_PATH=/absolute/path/to/RoboDojo \
bash examples/simBenchmarks/RoboDojo/eval_files/start_eval.sh \
  <oft|groot|pi_v3> <task> <seed> <policy_gpu> <sim_gpu> \
  "$STARVLA_ENV_PATH" "$ROBODOJO_ENV_PATH" \
  <episode_count|native>
```

## Released checkpoint results

The tables below reproduce the results published in the three Hugging Face
model cards. The protocol contains 42 evaluation tasks, 50 episodes per task,
and 2,100 episodes per policy. Values are **success rate (%) / score**; higher
is better for both. The policies train on 35 tasks, while the complete
evaluation includes held-out and open tasks.

### Overall and category summary

| Policy | Average | Generalization | Precision | Long-Horizon | Memory | Open |
|---|---:|---:|---:|---:|---:|---:|
| QwenOFT | 4.86 / 8.01 | 4.33 / 6.42 | 11.75 / 17.54 | 5.50 / 12.95 | 1.67 / 1.77 | 0.50 / 0.60 |
| QwenGR00T | 3.81 / 7.35 | 3.50 / 6.52 | 5.75 / 10.09 | 6.50 / 15.46 | 3.33 / 4.37 | 0.00 / 0.00 |
| **QwenPI_v3** | **6.19 / 9.60** | **4.17 / 7.28** | **14.00 / 19.06** | **10.00 / 17.84** | **2.00 / 2.32** | **0.75 / 0.88** |
| **VLAct** | **9.33 / 12.96** | **11.67 / 16.33** | **15.17 / 20.57** | **13.67 / 20.12** | 0.56 / 0.66 | **2.25 / 2.37** |

### Per-task results

| Task | QwenOFT | QwenGR00T | QwenPI_v3 | VLAct (official) |
|---|---:|---:|---:|---:|
| **Generalization average** | **4.33 / 6.42** | **3.50 / 6.52** | **4.17 / 7.28** | **11.67 / 16.33** |
| `stack_bowls` | 18.00 / 21.00 | 10.00 / 14.80 | 14.00 / 16.70 | 42.67 / 46.87 |
| `push_T` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `pack_objects_into_box` | 0.00 / 3.10 | 0.00 / 7.80 | 0.00 / 6.80 | 0.00 / 10.33 |
| `fold_clothes` | 10.00 / 12.80 | 8.00 / 12.40 | 2.00 / 9.60 | 22.67 / 28.53 |
| `hang_mugs` | 0.00 / 3.60 | 0.00 / 3.00 | 0.00 / 3.50 | 0.00 / 7.53 |
| `sweep_blocks` | 0.00 / 0.00 | 0.00 / 0.00 | 2.00 / 2.00 | 0.00 / 0.00 |
| `pour_liquid_into_cup` | 14.00 / 14.00 | 14.00 / 14.00 | 12.00 / 12.00 | 30.67 / 30.67 |
| `make_toast` | 0.00 / 1.00 | 2.00 / 5.00 | 2.00 / 5.00 | 1.33 / 6.33 |
| `arrange_largest_number` | 0.00 / 1.90 | 2.00 / 4.10 | 2.00 / 5.70 | 0.00 / 4.53 |
| `sort_nesting_dolls_by_size` | 0.00 / 0.00 | 4.00 / 4.00 | 6.00 / 6.00 | 16.00 / 16.00 |
| `store_laptop_and_headphones` | 4.00 / 11.20 | 0.00 / 7.20 | 2.00 / 8.40 | 1.33 / 12.80 |
| `stack_blocks` | 6.00 / 8.40 | 2.00 / 5.90 | 8.00 / 11.60 | 25.33 / 32.33 |
| **Precision average** | **11.75 / 17.54** | **5.75 / 10.09** | **14.00 / 19.06** | **15.17 / 20.57** |
| `fasten_screws` | 4.00 / 8.00 | 0.00 / 2.00 | 0.00 / 6.00 | 0.00 / 4.80 |
| `plug_in_charger` | 6.00 / 6.00 | 2.00 / 2.00 | 4.00 / 4.00 | 2.00 / 2.00 |
| `insert_tubes` | 40.00 / 51.60 | 28.00 / 40.40 | 44.00 / 56.80 | 48.67 / 59.87 |
| `pour_balls_into_vase` | 8.00 / 8.00 | 0.00 / 0.00 | 2.00 / 2.00 | 17.33 / 17.33 |
| `play_Xylophone` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `deposit_coin` | 0.00 / 3.20 | 2.00 / 5.60 | 6.00 / 7.60 | 4.00 / 8.40 |
| `insert_key` | 0.00 / 12.90 | 0.00 / 9.90 | 0.00 / 11.10 | 0.00 / 13.70 |
| `build_tower` | 36.00 / 50.60 | 14.00 / 20.80 | 56.00 / 65.00 | 49.33 / 58.47 |
| **Long-Horizon average** | **5.50 / 12.95** | **6.50 / 15.46** | **10.00 / 17.84** | **13.67 / 20.12** |
| `put_bottles_into_dustbin` | 22.00 / 40.90 | 26.00 / 44.40 | 64.00 / 73.60 | 59.33 / 70.13 |
| `fill_pen_holder` | 4.00 / 11.70 | 6.00 / 14.40 | 8.00 / 23.00 | 1.33 / 13.10 |
| `classify_objects` | 2.00 / 5.50 | 6.00 / 11.50 | 0.00 / 7.50 | 1.33 / 7.33 |
| `play_tic_tac_toe` | 0.00 / 12.40 | 2.00 / 16.40 | 0.00 / 6.80 | 0.00 / 4.93 |
| `fill_egg_holder` | 0.00 / 0.60 | 0.00 / 0.00 | 0.00 / 0.80 | 0.00 / 0.93 |
| `organize_table` | 0.00 / 16.50 | 0.00 / 25.00 | 4.00 / 27.00 | 3.33 / 20.50 |
| `make_kong` | 16.00 / 16.00 | 12.00 / 12.00 | 4.00 / 4.00 | 44.00 / 44.00 |
| `play_stacking_toy` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| **Memory average** | **1.67 / 1.77** | **3.33 / 4.37** | **2.00 / 2.32** | **0.56 / 0.66** |
| `cover_blocks` | 0.00 / 0.60 | 6.00 / 12.10 | 0.00 / 1.50 | 0.00 / 0.03 |
| `match_and_pick_from_conveyor` | 10.00 / 10.00 | 14.00 / 14.00 | 12.00 / 12.00 | 3.33 / 3.33 |
| `swap_blocks` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `swap_T` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `press_by_number` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `imitate_sorting_sequence` | 0.00 / 0.00 | 0.00 / 0.10 | 0.00 / 0.40 | 0.00 / 0.57 |
| **Open average** | **0.50 / 0.60** | **0.00 / 0.00** | **0.75 / 0.88** | **2.25 / 2.37** |
| `align_blocks` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `general_pickup` | 4.00 / 4.00 | 0.00 / 0.00 | 6.00 / 6.00 | 18.00 / 18.00 |
| `stack_blocks_by_language` | 0.00 / 0.80 | 0.00 / 0.00 | 0.00 / 0.80 | 0.00 / 0.80 |
| `solve_equation` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `classify_objects_by_language` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.20 | 0.00 / 0.13 |
| `pick_from_conveyor_by_image` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `store_tools_in_toolbox` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `pour_by_language` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

## Comparison with public RoboDojo policies

> **Research note (2026-08-09).** This comparison covers the released StarVLA
> QwenPI_v3 recipe and the public XPolicyLab implementations of
> Xiaomi-Robotics-1, Hy-Embodied-0.5-VLA, and Spatial Forcing as available on
> that date. Only parameters visible in the released code and configurations
> are included.

Here, **robot-policy pretraining** means initialization from a checkpoint that
already predicts robot actions. Initializing only the vision-language backbone
does not count.

| Policy / model | Initialization and action model | RGB input | State input | Action output | Chunk / replanning | Additional input or supervision |
|---|---|---|---|---|---|---|
| **StarVLA QwenPI_v3** | `Qwen3-VL-4B-Instruct` + 36-layer LayerwiseFM; action modules start from scratch | Current head + two wrist frames, 224 x 224 | 14D absolute joint position | 50 x 14 absolute joint position | Predict 50; execute 16 | Current observation only; no separate temporal or geometry objective |
| [Xiaomi-Robotics-1](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Xiaomi_Robotics_1) | Pretrained `Xiaomi-Robotics-1-5B` robot policy | Current head + two wrist frames; resize/crop to 320 x 256 | Dual-arm joint position and grippers packed into 60D; unused slots are zero | 30 x 60 relative end-effector deltas; active slots encode two poses and grippers | Predict 30; current deployment executes 10 | Current observation only |
| [Hy-Embodied-0.5-VLA](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Hy_Embodied_05_VLA) | Pretrained `Hy-Embodied-0.5-VLA-UMI` robot policy with video encoder | Head + two wrist streams, including history frames | 16D dual-arm end-effector pose and grippers, converted to the model's pose representation | Relative dual-arm end-effector pose and gripper actions in the UMI frame | Current deployment executes 10 | Six history frames sampled at 20-step intervals |
| [Spatial Forcing](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Spatial_Forcing) | Pretrained `pi05_base` robot policy with a VGGT-1B alignment branch | Current head + two wrist frames | Packed dual-arm joint state | Absolute joint action | Public adapter returns the predicted chunk | VGGT feature alignment loss (`0.2`); no explicit temporal history |

### Observed gaps

- **The camera contract is not the main differentiator.** All four public
  integrations use the same head-and-two-wrist RGB layout. The larger
  differences are how state and actions are represented and what the policy was
  trained on before RoboDojo.
- **Robot-policy pretraining is the clearest shared difference.** Every compared
  higher-scoring recipe starts from a robot/action checkpoint. QwenPI_v3 starts
  from a general-purpose VLM and learns its action-policy modules only from the
  35-task RoboDojo mixture. This is a correlation in the available releases,
  not proof that pretraining alone causes the score gap.
- **The output interface differs substantially.** QwenPI_v3 and Spatial Forcing
  emit absolute joint targets. Xiaomi-Robotics-1 and Hy-Embodied emit relative
  end-effector targets and require coordinate-frame transforms. The released
  results do not isolate this variable, so they cannot establish that one
  action representation is intrinsically better.
- **Some competitors add information or supervision absent from this StarVLA
  baseline.** Hy-Embodied supplies explicit temporal history, while Spatial
  Forcing adds VGGT-based spatial alignment. QwenPI_v3 uses the current RGB and
  joint state only, without a separate memory input or geometry-alignment
  objective.

These are differences observable in the released artifacts. They should not be
read as claims about unpublished model internals or as isolated explanations of
leaderboard performance.

### Suggested controlled comparisons

1. **Test robot-policy pretraining first.** Initialize QwenPI_v3 from an
   OXE/generalist robot checkpoint while keeping the current three images, 14D
   state, absolute-joint target, training mixture, and evaluation cadence
   unchanged. This isolates the largest shared configuration difference.
2. **Then isolate action representation.** Compare absolute and relative joint
   targets with the same initialization and data before introducing an
   end-effector interface, which also changes kinematics and coordinate-frame
   handling. Report normalization and gripper treatment with each result.
3. **Evaluate history only as a memory ablation.** Add a fixed observation
   history to the otherwise unchanged policy and report the Memory tasks
   separately. Hy-Embodied provides a concrete reference of six frames at
   20-step intervals; this should be treated as a comparison setting, not an
   assumed optimum for StarVLA.
4. **Keep geometry supervision as a separate ablation.** Spatial Forcing mixes
   `pi05_base` pretraining with VGGT alignment, so matching its alignment loss
   while also changing initialization would not identify which change matters.

## References

- [RoboDojo documentation](https://robodojo-benchmark.com/doc/)
- [RoboDojo leaderboard](https://robodojo-benchmark.com/leaderboard)
- [XPolicyLab StarVLA integration](https://github.com/JinhuiYE/XPolicyLab/tree/fix/starvla-hf-robodojo-eval/policy/starVLA)
- [QwenOFT RoboDojo checkpoint](https://huggingface.co/StarVLA/Qwen3vl4b-OFT-RoboDojo)
- [QwenGR00T RoboDojo checkpoint](https://huggingface.co/StarVLA/Qwen3vl4b-GR00T-RoboDojo)
- [QwenPI_v3 RoboDojo checkpoint](https://huggingface.co/StarVLA/StarVLA-Qwen3vl4b-PIv3-RoboDojo)
