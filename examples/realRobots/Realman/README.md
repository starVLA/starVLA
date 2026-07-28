# Realman RM-75 VM4A Example

This example trains the VM4A ACT and Diffusion Policy variants for a Realman RM-75 setup with an 8-DoF control vector: seven arm joints plus one gripper command. It is adapted from a real-robot recipe validated on a private in-house LeRobot dataset. The dataset is not distributed with StarVLA; substitute your own compatible LeRobot dataset before running the scripts.

The example assumes two RGB streams named `cam0_rgb` and `cam1_rgb`, an eight-dimensional state, and an eight-dimensional action. The first seven action dimensions are joint deltas and the last is an absolute gripper target. Adjust the YAML and registry together if your schema differs.

Before training:

1. Replace `<your_dataset>` in `train_files/data_registry/data_config.py` with the directory name of your LeRobot dataset.
2. Set `DATA_ROOT_DIR` to the parent of that directory, and optionally set `WANDB_ENTITY`, `NUM_PROCESSES`, `BATCH_SIZE`, and `MAX_STEPS`.
3. Activate a StarVLA environment with the dependencies required by the selected policy.

Run either recipe from the repository root:

```bash
DATA_ROOT_DIR=/path/to/lerobot_datasets \
  bash examples/realRobots/Realman/train_files/train_realman_act.sh

DATA_ROOT_DIR=/path/to/lerobot_datasets \
  bash examples/realRobots/Realman/train_files/train_realman_dp.sh
```

The registry is discovered automatically because it lives under `train_files/data_registry/`. For the policy architecture, normalization bridge, and Diffusion Policy EMA checkpoint behavior, see [the VM4A guide](../../../docs/VM4A.md).

