# Step 0: Teleop Data Collection

This step is about collecting usable G1 demonstrations. In this example, use `NVlabs/GR00T-WholeBodyControl` as the concrete data-collection path, then hand the exported dataset to StarVLA.

StarVLA does not own the teleop stack. Here the teleop, camera server, data exporter, and G1 controller side come from GR00T-WholeBodyControl.

Other routes are valid, but they are alternatives to this example:

- LeRobot's official Unitree G1 workflow,
- the user's existing Unitree G1 infra,
- Unitree SDK-based collection,
- Unitree `xr_teleoperate`,
- Unitree `unitree_lerobot`,
- lab-specific VR / XR / motion-capture teleop,
- XRoboToolkit-style teleop,
- custom keyboard / gamepad / SpaceMouse collection,
- NVIDIA GR00T-WholeBodyControl.

## Primary Example: GR00T-WholeBodyControl

Use these upstream docs as the reference for environment setup and hardware safety:

- PICO VR whole-body teleop: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html
- Data collection for VLA: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html
- VLA workflow: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html
- Repository: https://github.com/NVlabs/GR00T-WholeBodyControl

The upstream teleop workflow has three pieces:

```text
MuJoCo sim or real G1
  + C++ deploy / SONIC controller
  + PICO teleop streamer
  + optional data exporter
```

For StarVLA, this step's job is to produce the same kind of dataset that the NVlabs VLA workflow would fine-tune on.

## Step A: Bring Up Teleop in Simulation

Start in simulation before touching the real robot.

Terminal 1, MuJoCo simulation loop from the StarVLA repo root:

```bash
cd examples/realRobots/UnitreeG1_WholeBody/sdk_tools/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

Terminal 2, SONIC deploy from the StarVLA repo root:

```bash
cd examples/realRobots/UnitreeG1_WholeBody/sdk_tools/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager sim
```

Terminal 3, PICO teleop streamer from the StarVLA repo root:

```bash
cd examples/realRobots/UnitreeG1_WholeBody/sdk_tools/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
  --vis_vr3pt --vis_smpl
```

Validation before continuing:

- the visualizer opens,
- the simulated G1 can be controlled,
- mode switching is understood,
- emergency stop behavior is tested,
- calibration pose is repeatable.

## Step B: Collect VLA Demonstrations

Use GR00T-WholeBodyControl's data collection launcher as the first concrete collection path:

```bash
cd examples/realRobots/UnitreeG1_WholeBody/sdk_tools/GR00T-WholeBodyControl
python gear_sonic/scripts/launch_data_collection.py \
  --camera-host 192.168.123.164 \
  --task-prompt "pick up the soda can and place it in the bin"
```

Expected upstream output is a LeRobot v2.1-style dataset directory, for example:

```text
outputs/2026-04-03-14-30-00-G1-robot01/
  data/
    train-00000.parquet
  videos/
    observation.images.ego_view/
      episode_000000.mp4
  meta/
    info.json
    modality.json
    episodes.jsonl
    tasks.jsonl
```

Move or symlink the collected dataset into the StarVLA workspace convention:

```text
playground/Datasets/UnitreeG1_WholeBody/lerobot/<task_name>/
```

## Step C: Optional Real-Robot Collection

Only move to the physical G1 after simulation teleop is stable.

The upstream real-robot path uses two main terminals:

Terminal 1, C++ deployment from the StarVLA repo root:

```bash
cd examples/realRobots/UnitreeG1_WholeBody/sdk_tools/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager real
```

Terminal 2, PICO teleop streamer from the StarVLA repo root:

```bash
cd examples/realRobots/UnitreeG1_WholeBody/sdk_tools/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

If the teleop streamer runs offboard, set the ZMQ host consistently with the upstream instructions. Keep a safety operator ready and verify emergency stop before recording.

## Alternative Public Paths

### LeRobot Unitree G1

Docs: https://huggingface.co/docs/lerobot/unitree_g1

This path is useful when the user wants LeRobot-native collection and training conventions. The docs cover:

- Unitree SDK setup,
- LeRobot installation with the G1 extra,
- simulation teleoperation,
- physical robot connection,
- robot server and camera setup,
- loco-manipulation recording with `lerobot-record`,
- training and RTC inference examples.

For StarVLA, the most important output is a LeRobot dataset with clear camera/state/action keys.

### Unitree XR Teleoperate

Repo: https://github.com/unitreerobotics/xr_teleoperate

This path is useful when the user wants XR-device teleoperation. The repo supports G1 29 DoF and 23 DoF variants, supports devices such as Apple Vision Pro, PICO, and Quest, and has recording mode via launch flags such as `--record`.

For StarVLA, this route usually needs a conversion step after collection.

### Unitree LeRobot Tools

Repo: https://github.com/unitreerobotics/unitree_lerobot

This path is useful when working with Unitree-recorded JSON datasets or Unitree public examples. The repo includes data conversion, dataset visualization/editing, and G1/Dex real-world evaluation utilities.

For StarVLA, this can be used as a bridge from Unitree raw data into LeRobot.

## Purpose of This Layer

This layer answers:

- How do we safely control or teleoperate G1?
- How do we record demonstrations?
- How do we know which episodes are successful?
- What camera/state/action streams will StarVLA see later?

This layer should not answer:

- How does StarVLA train?
- How does the VLA model normalize actions?
- How does the policy server run?

## What This Step Should Produce

At the end of this step, you should have raw or already-exported demonstrations:

```text
playground/Datasets/UnitreeG1_WholeBody/
  raw/
    <task_name>/
      episodes/
      metadata/
```

or, if the external stack already exports LeRobot v2.1:

```text
playground/Datasets/UnitreeG1_WholeBody/
  lerobot/
    <task_name>/
      data/
      videos/
      meta/
```

## Minimum Data Contract

Every collected episode should include:

- episode id
- task prompt
- success / failure / discarded flag
- synchronized camera frames
- synchronized robot state
- synchronized action or teleop command
- timestamps for every stream
- control frequency
- camera names and order
- state/action schema version

## Practical G1 Notes

For first StarVLA integration, prefer a constrained setup:

- Start in simulation or replay first.
- Use a fixed task prompt.
- Use one or two camera views before adding all cameras.
- Freeze locomotion or lower-body control unless your controller stack already handles it safely.
- Prefer a controller-level latent or grouped action interface over direct motor command prediction.

If using GR00T-WholeBodyControl / SONIC, keep their teleop and WBC process outside StarVLA. If using another stack, keep that stack outside StarVLA too. StarVLA only needs the exported dataset and the action semantics.

## Safety Gate

Before collecting on the real robot:

- Teleop works smoothly in simulation.
- Emergency stop is tested.
- A safety operator is present.
- Calibration pose and mode switching are understood.
- No StarVLA model is in the loop yet.

## Handoff to Step 1

Move to [step1_data_conversion](../step1_data_conversion/README.md) when you can answer:

- Where is the dataset?
- Is it already LeRobot v2.1?
- What are the exact camera keys?
- What are the exact state keys and dimensions?
- What are the exact action keys and dimensions?
- What is the embodiment/action route: SONIC latent or direct grouped control?
