# Unitree G1 WholeBody with StarVLA

This directory is a workflow scaffold for running StarVLA on Unitree G1-style whole-body embodiments.

The purpose is to help users understand how to connect StarVLA to a real G1 workflow. This directory is intentionally written as a concrete example, not only a link collection.

The main example follows the shape of NVIDIA's `NVlabs/GR00T-WholeBodyControl` workflow:

```text
GR00T-WholeBodyControl teleop / data collection
  -> LeRobot v2.1 dataset
  -> StarVLA training
  -> StarVLA WebSocket policy server
  -> GR00T-WholeBodyControl / SONIC / user G1 controller side
```

StarVLA does not re-implement Unitree SDK, SONIC WBC, PICO teleop, camera servers, or third-party deployment stacks. It documents how to use their outputs and where StarVLA plugs in.

The common path is:

```text
teleop / data collection
  -> LeRobot-format dataset
  -> StarVLA training
  -> StarVLA policy server
  -> G1 whole-body controller / user controller
```

Many implementations are valid. Users may use their own Unitree infra, lab teleop stack, XR/VR toolchain, vendor SDK wrappers, or third-party projects. In this folder, however, the step READMEs are written around the GR00T-WholeBodyControl route so users have a complete worked path.

Useful public paths include:

- LeRobot Unitree G1 docs: https://huggingface.co/docs/lerobot/unitree_g1
- Unitree XR teleoperation: https://github.com/unitreerobotics/xr_teleoperate
- Unitree LeRobot tools: https://github.com/unitreerobotics/unitree_lerobot
- NVIDIA GR00T-WholeBodyControl: https://github.com/NVlabs/GR00T-WholeBodyControl

We use GR00T-WholeBodyControl as the primary reference path because it already demonstrates G1 PICO VR teleop, data collection, LeRobot v2.1 export, and SONIC whole-body control:

- Project: https://github.com/NVlabs/GR00T-WholeBodyControl
- PICO VR whole-body teleop: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html
- Data collection: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html
- VLA workflow: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html
- VLA inference: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html

This is the example route for this directory, not a global StarVLA dependency requirement. LeRobot's G1 path or Unitree's XR teleop path are equally reasonable if they match the user's hardware and data needs. StarVLA should sit at the model and data boundary: consume a standardized dataset, train a VLA policy, serve actions, and let the user-owned robot infra execute them safely.

## Collaboration Style

When extending this example, describe the purpose of each layer before writing code:

- What problem does this layer solve?
- Which parts are owned by StarVLA?
- Which parts are owned by the user's robot infra or a third-party repo?
- What artifact does this step produce for the next step?
- What is the simplest safe way to validate it?

For the first full example, follow the GR00T-WholeBodyControl-style route. Other users can replace that route with LeRobot G1, Unitree XR teleop, Unitree LeRobot conversion tools, or their own teleop/control stacks as long as the same StarVLA boundary is preserved.

## How This Example Maps NVlabs to StarVLA

NVlabs workflow:

```text
Collect with GR00T-WholeBodyControl
  -> fine-tune Isaac-GR00T
  -> run Isaac-GR00T PolicyServer
  -> launch GR00T-WholeBodyControl inference
```

StarVLA workflow in this folder:

```text
Collect with GR00T-WholeBodyControl
  -> train StarVLA on the exported LeRobot dataset
  -> run StarVLA WebSocket policy server
  -> adapt GR00T-WholeBodyControl / SONIC controller side to consume StarVLA actions
```

The important replacement is the model layer. We reuse the external teleop, camera, data collection, and controller-side ideas; StarVLA owns training and policy serving.

## Viable Upstream Routes

### LeRobot Unitree G1 Route

Use this when the user wants the shortest path through the LeRobot ecosystem. The official LeRobot G1 docs cover Unitree SDK setup, simulation teleoperation, physical robot connection, recording with `lerobot-record`, training, and real-time chunking inference.

StarVLA handoff:

```text
LeRobot G1 dataset
  -> StarVLA data_registry + modality
  -> StarVLA training
  -> StarVLA policy server
```

### Unitree XR Teleoperate Route

Use this when the user wants XR-device teleoperation with Unitree's own tooling. The `xr_teleoperate` project supports devices such as Apple Vision Pro, PICO, and Quest, and supports G1 23/29 DoF plus multiple end-effectors.

StarVLA handoff:

```text
xr_teleoperate recorded data
  -> Unitree / custom conversion to LeRobot
  -> StarVLA data_registry + modality
  -> StarVLA training
```

### Unitree LeRobot Tools Route

Use this when data is already in Unitree JSON format or collected by Unitree tooling. The Unitree LeRobot repo includes data conversion, dataset visualization/editing, and real-robot evaluation utilities.

StarVLA handoff:

```text
Unitree JSON or Unitree dataset
  -> unitree_lerobot / custom conversion
  -> LeRobot dataset
  -> StarVLA training and deployment
```

### GR00T-WholeBodyControl / SONIC Route

Use this when the user wants a full G1 whole-body controller stack with SONIC-style latent actions. This is a strong example for whole-body deployment because the high-frequency balance and motion generation stay in the controller stack.

## Directory Flow

```text
examples/realRobots/UnitreeG1_WholeBody/
  README.md
  DISCUSSION.md

  step0_teleop_data_collection/
    README.md

  step1_data_conversion/
    README.md

  step2_training/
    README.md

  step3_deployment/
    README.md

  sdk_tools/
    README.md
```

## Recommended First Target

Do not start with full G1 whole-body autonomy. Start with a narrow, testable path:

1. Use an external G1 teleop/control stack to collect demonstrations.
2. Export or convert the demonstrations to LeRobot v2.1.
3. Define a StarVLA data registry for one G1 embodiment tag.
4. Train a small StarVLA checkpoint.
5. Run replay and mock-controller dry-runs before touching the real robot.
6. Deploy through StarVLA WebSocket policy server into the user's G1 controller adapter.

## Action-Space Choices

There are two practical action-space routes:

### Route A: SONIC-style latent actions

NVIDIA's VLA workflow describes a compact 78D action space: 64D motion token plus 7D left hand joints and 7D right hand joints. In this route, StarVLA predicts the same high-level latent action representation, and SONIC or the user's whole-body controller decodes it into high-frequency robot commands.

This is a good example route if users already rely on GR00T-WholeBodyControl / SONIC for G1, because it keeps balance and high-frequency control in a controller stack designed for that job.

### Route B: Direct grouped control

StarVLA predicts grouped robot commands such as:

```text
base + torso + left_arm + right_arm + left_hand + right_hand
```

This route gives more direct control, but it requires the user to own more controller logic, safety checks, latency handling, and train/test consistency validation.

### Route C: User-defined controller actions

Some teams will already have a production G1 controller API. In that case, StarVLA can predict the team's existing action representation. The requirement is that the representation be documented, logged during teleop, converted into LeRobot, and replayable in dry-run before real execution.

## StarVLA Boundary

StarVLA should standardize these pieces:

- LeRobot dataset layout and `meta/modality.json`.
- `train_files/data_registry/data_config.py`.
- StarVLA training config.
- Policy server request / response.
- Replay and dry-run validation.
- Thin adapter from StarVLA action chunk to the user's G1 controller API.

StarVLA should not own these pieces:

- PICO / XR / VR device setup.
- Unitree SDK installation.
- Low-level balance control.
- Real-time whole-body control loop.
- Robot network setup.
- Emergency stop implementation.

Those pieces should live in the user's G1 infra or in referenced third-party repositories. This example documents how to connect to them, not how to own all of them.

## Start Here

Follow the folders in order:

1. [step0_teleop_data_collection](step0_teleop_data_collection/README.md)
2. [step1_data_conversion](step1_data_conversion/README.md)
3. [step2_training](step2_training/README.md)
4. [step3_deployment](step3_deployment/README.md)
5. [sdk_tools](sdk_tools/README.md)

Each step should leave behind a small, checkable artifact before moving to the next step.
