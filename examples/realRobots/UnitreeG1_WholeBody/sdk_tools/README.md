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

Primary setup note:

```bash
mkdir -p ~/playground/Code
cd ~/playground/Code
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
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
