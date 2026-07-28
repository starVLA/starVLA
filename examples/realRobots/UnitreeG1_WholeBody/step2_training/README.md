# Step 2: Training

This step trains StarVLA on the local G1 dataset collected with
GR00T-WholeBodyControl. In the upstream NVlabs VLA workflow, this stage fine-tunes
Isaac-GR00T. In this StarVLA example, this stage trains StarVLA instead.

Training should not depend on Unitree SDK or the teleop runtime. At this point the only required input is the LeRobot-format dataset plus StarVLA data registry files.

For this workspace, the local LeRobot root is:

```text
playground/Datasets/UnitreeG1_WholeBody/lerobot
```

## Expected Files

Recommended local structure:

```text
examples/realRobots/UnitreeG1_WholeBody/step2_training/
  README.md
  train_files/
    data_registry/
      data_config.py
    modality.json
    starvla_cotrain_g1_wholebody.yaml
    run_g1_train.sh
```

These files are now implemented in this scaffold. The single-GPU walk-through
uses:

```text
examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/
  data_registry/data_config.py
  modality.json
  starvla_qwenoft_g1_sonic.yaml
  run_starvla_qwenoft_g1_sonic_train.sh
```

Follow the patterns in:

- `examples/realRobots/Franka/train_files/`
- `examples/realRobots/RoboChallenge_table30v2/train_files/`
- `docs/agent_skills/integrate-starvla-dataset/assets/templates/`

## Data Registry Shape

For the GR00T-WholeBodyControl / SONIC example version, prefer explicit naming:

```python
class UnitreeG1SonicConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_leg", "state.right_leg", "state.waist",
        "state.left_arm", "state.left_hand", "state.right_arm",
        "state.right_hand", "state.left_wrist_pos", "state.left_wrist_abs_quat",
        "state.right_wrist_pos", "state.right_wrist_abs_quat",
        "state.root_orientation", "state.projected_gravity",
        "state.cpp_rotation_offset", "state.init_base_quat",
    ]
    action_keys = ["action.motion_token", "action.left_hand_joints", "action.right_hand_joints"]
    language_keys = ["annotation.human.action.task_description"]

    # Example only. Use real dimensions from the dataset.
    action_key_dims = {
        "action.motion_token": 64,
        "action.left_hand_joints": 7,
        "action.right_hand_joints": 7,
    }
    state_key_dims = {
        "state.left_leg": 6,
        "state.right_leg": 6,
        "state.waist": 3,
        "state.left_arm": 7,
        "state.left_hand": 7,
        "state.right_arm": 7,
        "state.right_hand": 7,
        "state.left_wrist_pos": 3,
        "state.left_wrist_abs_quat": 4,
        "state.right_wrist_pos": 3,
        "state.right_wrist_abs_quat": 4,
        "state.root_orientation": 4,
        "state.projected_gravity": 3,
        "state.cpp_rotation_offset": 4,
        "state.init_base_quat": 4,
    }
```

Do not copy these dimensions blindly. Use the dataset and controller action contract as the source of truth. If the user chooses a different controller route, define different `action_keys` and `action_key_dims`.

For the primary local walk-through route, the expected action family is:

```text
64D SONIC motion latent + 7D left hand + 7D right hand = 78D
```

That action representation is what the downstream SONIC/controller side should know how to consume.

## Training Config Responsibilities

The YAML should make these values easy to audit:

- dataset path
- mixture name
- action dimension
- state dimension
- normalization modes, with `q99` as the default for continuous state/action keys
- action horizon
- image size
- backbone / action head
- output directory
- batch size and training steps

For the primary G1/SONIC route, the data registry should use `q99` for continuous keys:

```python
StateActionTransform(
    apply_to=self.state_keys,
    normalization_modes={k: "q99" for k in self.state_keys},
)
StateActionTransform(
    apply_to=self.action_keys,
    normalization_modes={
        "action.sonic_latent": "q99",
        "action.left_hand": "q99",
        "action.right_hand": "q99",
    },
)
```

Use `binary` only for actual binary gripper/open-close commands. Do not use `min_max` as the default for this example unless you are intentionally matching an existing checkpoint or external controller contract.

## Smoke Tests Before Full Training

Run the same style of checks used by other StarVLA examples:

```bash
python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/starvla_qwenoft_g1_sonic.yaml

python starVLA/model/framework/VLM4A/QwenOFT.py \
  --config_yaml examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/starvla_qwenoft_g1_sonic.yaml
```

Only start full training after both pass.

## Verified Local Run

This workspace has been verified with a 1-step smoke training run:

```bash
cd starVLA
WANDB_MODE=disabled BATCH=1 MAX_STEPS=1 SAVE_EVERY=1 EVAL_EVERY=1 LOG_EVERY=1 \
  bash examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/run_starvla_qwenoft_g1_sonic_train.sh
```

Observed result:

```text
results/Checkpoints/starvla_qwenoft_g1_sonic_smoke/
  checkpoints/steps_1/
  config.full.yaml
  config.yaml
  dataset_statistics.json
```

The run completed successfully on a single RTX 4090 and reached the first checkpoint.

## Training Output

Expected output:

```text
playground/Checkpoints/<g1_run_name>/
  dataset_statistics.json
  checkpoints/
    steps_<N>_pytorch_model.pt
  config.yaml
```

Keep `dataset_statistics.json` with the checkpoint. Deployment needs it for action unnormalization.

## Handoff to Step 3

Move to [step3_deployment](../step3_deployment/README.md) when:

- dataloader smoke test passes,
- model forward smoke test passes,
- a checkpoint exists,
- `dataset_statistics.json` exists beside the checkpoint,
- action/state keys match the intended deployment adapter.
