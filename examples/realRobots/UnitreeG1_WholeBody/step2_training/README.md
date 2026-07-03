# Step 2: Training

This step trains StarVLA on the G1 dataset collected with GR00T-WholeBodyControl. In the upstream NVlabs VLA workflow, this stage fine-tunes Isaac-GR00T. In this StarVLA example, this stage trains StarVLA instead.

Training should not depend on Unitree SDK or the teleop runtime. At this point the only required input is the LeRobot-format dataset plus StarVLA data registry files.

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

These files are not implemented yet in this scaffold. When implementing them, follow the patterns in:

- `examples/realRobots/Franka/train_files/`
- `examples/realRobots/RoboChallenge_table30v2/train_files/`
- `docs/agent_skills/integrate-starvla-dataset/assets/templates/`

## Data Registry Shape

For the GR00T-WholeBodyControl / SONIC example version, prefer explicit naming:

```python
class UnitreeG1SonicConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.<camera_0>", "video.<camera_1>"]
    state_keys = ["state.<robot_state_group>"]
    action_keys = ["action.sonic_latent", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.action.task_description"]

    # Example only. Use real dimensions from the dataset.
    action_key_dims = {
        "action.sonic_latent": 64,
        "action.left_hand": 7,
        "action.right_hand": 7,
    }
```

Do not copy these dimensions blindly. Use the dataset and controller action contract as the source of truth. If the user chooses a different controller route, define different `action_keys` and `action_key_dims`.

For the primary NVlabs-style route, the expected action family is:

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
  --config_yaml examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/starvla_cotrain_g1_wholebody.yaml

python starVLA/model/framework/VLM4A/QwenOFT.py \
  --config_yaml examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/starvla_cotrain_g1_wholebody.yaml
```

Only start full training after both pass.

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
