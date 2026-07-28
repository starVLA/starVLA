# SDK Tools

This folder is for helper scripts and notes around the G1 integration. Its purpose is to make collaboration easier: users can keep third-party setup notes, visualization scripts, inspection tools, mock adapters, and one-off debugging tools here without mixing them into StarVLA model code.

Do not put core StarVLA model code here. Keep this folder practical and integration-facing.

## Good Candidates

```text
sdk_tools/
  README.md
  clone_third_party.md
  visualize_lerobot_episode.py
  inspect_dataset_schema.py
  check_camera_stream.py
  check_action_stats.py
  mock_g1_state_publisher.py
  mock_g1_action_consumer.py
  plot_action_chunks.py
  compare_train_deploy_obs.py
```

## Third-Party Repositories

It is acceptable to point users to third-party repositories instead of copying them into StarVLA. For this G1 whole-body example, treat GR00T-WholeBodyControl as the primary external repo.

Primary setup note, if the external repo has not been cloned yet:

```bash
mkdir -p ~/playground/Code
cd ~/playground/Code
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
```

Expose the external repo through this folder with a symlink:

```bash
cd <STARVLA_ROOT>
ln -s /path/to/GR00T-WholeBodyControl \
  examples/realRobots/UnitreeG1_WholeBody/sdk_tools/GR00T-WholeBodyControl
```

After that, use this path as the stable entry point from the StarVLA repo root:

```bash
cd examples/realRobots/UnitreeG1_WholeBody/sdk_tools/GR00T-WholeBodyControl
```

If the terminal starts from the parent directory that contains `starVLA`, this equivalent form is also fine:

```bash
cd starVLA/examples/realRobots/UnitreeG1_WholeBody/sdk_tools/GR00T-WholeBodyControl
```

Optional alternatives to document when needed:

```bash
cd ~/playground/Code
git clone https://github.com/unitreerobotics/xr_teleoperate.git
git clone https://github.com/unitreerobotics/unitree_lerobot.git
```

Users should follow the GR00T-WholeBodyControl installation docs for the primary example path. Other labs can document their own clone/install commands here if they replace the upstream teleop/control route.

Useful public options:

- LeRobot Unitree G1 docs: https://huggingface.co/docs/lerobot/unitree_g1
- Unitree XR teleoperate: https://github.com/unitreerobotics/xr_teleoperate
- Unitree LeRobot tools: https://github.com/unitreerobotics/unitree_lerobot
- GR00T-WholeBodyControl: https://github.com/NVlabs/GR00T-WholeBodyControl
- SonicStar SONIC/G1 reference: https://github.com/BlackOtters/SonicStar

## SonicStar WBC Deployment Reference

This example can use SonicStar as a reference for the SONIC / WBC side of the
Unitree G1 stack. We keep that code as a local SDK reference instead of
tracking it in this repository, because the full tree can contain large runtime
assets, generated files, model weights, meshes, and machine-specific setup.

Local paths used by this workspace:

```text
examples/realRobots/UnitreeG1_WholeBody/sdk_tools/SonicStar/
examples/realRobots/UnitreeG1_WholeBody/sdk_tools/sonicstar_wbc_deployment_reference/
```

Both paths are intentionally git-ignored. The first can hold a fuller local
SonicStar mirror. The second holds a curated deployment-facing subset for quick
inspection.

To recreate the local reference from a fresh checkout:

```bash
cd <STARVLA_ROOT>
mkdir -p bar
git clone https://github.com/BlackOtters/SonicStar.git bar/SonicStar

mkdir -p examples/realRobots/UnitreeG1_WholeBody/sdk_tools/SonicStar
rsync -a bar/SonicStar/wbc/ \
  examples/realRobots/UnitreeG1_WholeBody/sdk_tools/SonicStar/wbc/
```

Create the curated deployment subset:

```bash
cd <STARVLA_ROOT>
mkdir -p examples/realRobots/UnitreeG1_WholeBody/sdk_tools/sonicstar_wbc_deployment_reference

rsync -a \
  --exclude 'thirdparty/' \
  --exclude 'policy/release/*.onnx' \
  --exclude 'g1/meshes/' \
  --exclude 'g1/images/' \
  --exclude '*.so' \
  --exclude '*.a' \
  --exclude '*.onnx' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.pkl' \
  --exclude '*.mp4' \
  --exclude '*.gif' \
  --exclude '*.usd' \
  examples/realRobots/UnitreeG1_WholeBody/sdk_tools/SonicStar/wbc/gear_sonic_deploy/ \
  examples/realRobots/UnitreeG1_WholeBody/sdk_tools/sonicstar_wbc_deployment_reference/gear_sonic_deploy/

rsync -a \
  --exclude 'data/assets/' \
  --exclude 'data/robots/' \
  --exclude 'data/robot_model/model_data/' \
  --exclude 'data/human/*.pkl' \
  --exclude '*.onnx' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.pkl' \
  --exclude '*.mp4' \
  --exclude '*.gif' \
  --exclude '*.usd' \
  --exclude '*.STL' \
  --exclude '*.stl' \
  examples/realRobots/UnitreeG1_WholeBody/sdk_tools/SonicStar/wbc/gear_sonic/ \
  examples/realRobots/UnitreeG1_WholeBody/sdk_tools/sonicstar_wbc_deployment_reference/gear_sonic/
```

The useful entry points in the curated copy are:

```text
sonicstar_wbc_deployment_reference/gear_sonic_deploy/deploy.sh
sonicstar_wbc_deployment_reference/gear_sonic_deploy/src/
sonicstar_wbc_deployment_reference/gear_sonic_deploy/g1/*.xml
sonicstar_wbc_deployment_reference/gear_sonic/scripts/run_sim_loop.py
sonicstar_wbc_deployment_reference/gear_sonic/scripts/run_camera_viewer.py
sonicstar_wbc_deployment_reference/gear_sonic/scripts/run_data_exporter.py
sonicstar_wbc_deployment_reference/gear_sonic/scripts/run_vla_inference.py
sonicstar_wbc_deployment_reference/gear_sonic/scripts/send_keyboard_cmd.py
```

The tracked StarVLA-side deployment entry points remain:

```text
examples/realRobots/UnitreeG1_WholeBody/step3_deployment/run_policy_server.sh
examples/realRobots/UnitreeG1_WholeBody/step3_deployment/run_starvla_eval.sh
examples/realRobots/UnitreeG1_WholeBody/step3_deployment/eval_files/model2unitree_g1_interface.py
```

The active training entry point is:

```text
examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/run_starvla_qwenoft_g1_sonic_train.sh
```

Important difference from SonicStar: the SonicStar training reference uses a
QwenGR00T/DiT route with longer action horizon. This example currently uses a
QwenOFT/MLP route and a 78D SONIC action contract:

```text
64D motion token + 7D left hand + 7D right hand
```

Do not mix checkpoints, normalization stats, and deployment adapters across
these routes without updating the dataset registry and action contract
together.

For the primary GR00T-WholeBodyControl path, the external repo covers:

- PICO teleop setup,
- camera server,
- SONIC / WBC deployment,
- Unitree G1 network setup,
- real robot safety procedures.

The optional Unitree / LeRobot paths cover XR teleop, LeRobot G1 dataset recording, and Unitree JSON to LeRobot conversion. StarVLA should document how to interoperate with whichever outputs the chosen path produces.

## Visualization Tools

Useful tools to add here:

- LeRobot episode viewer.
- Multi-camera frame checker.
- State/action dimension printer.
- Action histogram and min/max checker.
- Train-vs-deploy observation diff.
- Policy server latency logger.
- Controller command dry-run viewer.

## Checks That Matter

The most important G1 bugs are semantic mismatches, not Python exceptions. Tools in this folder should make these visible:

- camera order changed,
- RGB/BGR swapped,
- resize/crop mismatch,
- state joint order mismatch,
- action group order mismatch,
- SONIC latent action dimension mismatch,
- hand command sign flipped,
- stale frames,
- action chunk executed at the wrong frequency.

## Rule

Every helper should print:

```text
input path / source
camera keys
state keys and shape
action keys and shape
language key
episode count
sample timestamps
```

This makes the tools useful for both humans and code agents.
