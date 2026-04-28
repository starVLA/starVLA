# RoboCasa365 (PandaOmron) walk-through

End-to-end example for training and evaluating starVLA on the upstream
[RoboCasa](https://github.com/robocasa/robocasa) benchmark (single-arm Franka
PandaOmron mobile robot, 365 simulated kitchen tasks). This walk-through covers:

1. Environment install (`robocasa365` conda env)
2. Data download (one task: `OpenDrawer`, target/human, already in LeRobot v2.1)
3. Training (Qwen3VL-OFT, 100 steps, all visible GPUs)
4. Evaluation (websocket policy server + gym sim client)

> The Nvidia GR1 fork lives under [`examples/Robocasa_tabletop`](../Robocasa_tabletop/README.md). This folder targets the **official** robocasa repo at the version released for the 365-task benchmark. They are intentionally separate.

---

## 1. Environment

We isolate the simulator from the trainer with two conda envs.

```bash
# trainer env (already provided by the repo)
conda activate starVLA

# fresh sim env for upstream robocasa
conda create -n robocasa365 python=3.11 -y
conda activate robocasa365

# clone upstream side-by-side under playground/Code/
mkdir -p playground/Code && cd playground/Code
git clone https://github.com/ARISE-Initiative/robosuite.git
git clone https://github.com/robocasa/robocasa.git robocasa365
pip install -e robosuite -e robocasa365
pip install lerobot mujoco

# write dataset/asset paths
python robocasa365/robocasa/scripts/setup_macros.py
# edit playground/Code/robocasa365/robocasa/macros_private.py and set
# DATASET_BASE_PATH = "<repo>/playground/Datasets/robocasa365"
```

## 2. Data

```bash
conda activate robocasa365

# (a) ~10 GB of textures / objects — needed to render the kitchens
python -m robocasa.scripts.download_kitchen_assets

# (b) Per-task LeRobot v2.1 datasets (no HDF5 conversion needed; box ships .tar)
python -m robocasa.scripts.download_datasets \
    --tasks OpenDrawer \
    --split target \
    --source human
# -> playground/Datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot/
```

The dataset registry [`train_files/data_registry/data_config.py`](train_files/data_registry/data_config.py) is auto-discovered by
`starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge`. It exposes:

| mixture name                              | tasks                       |
| ----------------------------------------- | --------------------------- |
| `robocasa365_open_drawer_target_human`    | OpenDrawer (atomic, target) |
| `robocasa365_atomic_target_human_all`     | extend manually as you download more atomic tasks |

Modalities (matches the dataset's `meta/modality.json`):

* state 16-d: `base_position(3) + base_rotation(4) + eef_pos_rel(3) + eef_rot_rel(4) + gripper_qpos(2)`
* action 12-d: `eef_pos(3) + eef_rot(3) + gripper_close(1) + base_motion(4) + control_mode(1)`
* video: `robot0_agentview_left` (256 × 256 → resized to 224 × 224 in the loader)

## 3. Train (100-step walk-through)

```bash
conda activate starVLA
bash examples/Robocasa_365/train_files/run_robocasa365.sh
# overrides: NUM_GPUS=4 bash ...
```

The YAML at [`train_files/starvla_qwenoft_robocasa365.yaml`](train_files/starvla_qwenoft_robocasa365.yaml)
configures a `QwenOFT` framework (Qwen3-VL-4B + MLP regression head, L1 loss),
`action_dim=12`, `action_horizon=16`, batch size 4 / GPU. After 100 steps the
checkpoint is at:

```
playground/Checkpoints/robocasa365_qwenoft_OpenDrawer_100step/checkpoints/steps_100_pytorch_model.pt
```

For real training, raise `--trainer.max_train_steps` (e.g. 100k–1M), enable
`wandb` (drop the `WANDB_MODE=disabled` line) and add more tasks to the mixture.

## 4. Evaluate

Two terminals; the script is the same wrapper for both.

```bash
# terminal 1 (trainer env, GPU)
conda activate starVLA
bash examples/Robocasa_365/eval_files/run_eval.sh server

# terminal 2 (sim env, MuJoCo)
conda activate robocasa365
bash examples/Robocasa_365/eval_files/run_eval.sh client
```

Or launch directly with `tyro` flags (note the `--args.` prefix and dashes):

```bash
conda activate robocasa365
python -m examples.Robocasa_365.eval_files.simulation_env \
  --args.pretrained-path ./playground/Checkpoints/robocasa365_qwenoft_OpenDrawer_100step/checkpoints/steps_100_pytorch_model.pt \
  --args.env-name robocasa/OpenDrawer \
  --args.port 5678 \
  --args.n-episodes 2 \
  --args.n-envs 1 \
  --args.max-episode-steps 200 \
  --args.n-action-steps 8
```

The client writes per-env JSON results next to the checkpoint:

```
<ckpt>.eval/robocasa_OpenDrawer.json   # {"env": ..., "success_rate": 0.0, "successes": [...]}
```

Videos go to `results/robocasa365_eval_test/videos/` by default.

## 5. Task list & scoring

RoboCasa 365 is split into:

| group     | count | description                                                |
| --------- | ----- | ---------------------------------------------------------- |
| atomic    | ~24   | primitive skills (e.g. `OpenDrawer`, `CloseDoor`, `PnPCounterToCab`) |
| composite | ~341  | multi-step kitchen tasks built on top of atomic skills    |

Each task can be downloaded with two orthogonal flags:

* `--split {target, source}` — `target` are the curated 365 tasks; `source` is auxiliary data
* `--source {human, mg, im}` — human teleop / MimicGen / improvement (varies per task)

The official benchmark reports **per-task success rate** averaged over **50
rollouts** with **`max_episode_steps=500`** (atomic) or 1000+ (composite).
`run_eval.sh` defaults to 5 rollouts so the walk-through finishes quickly; raise
`N_EPISODES=50` to match the leaderboard protocol.

### Walk-through results (this README)

100-step training on a single H800 pair (~12 s wall-clock; final action L1 ≈ 0.21):

| task        | split / source | episodes | success | notes                                |
| ----------- | -------------- | -------: | ------: | ------------------------------------ |
| OpenDrawer  | target / human |        2 |   0 / 2 | 100-step Qwen3VL-OFT smoke test (sanity check). |

(Train more, then update this table.)

## 6. References

* RoboCasa paper / repo: <https://robocasa.ai>
* Upstream code: <https://github.com/robocasa/robocasa>
* Dataset converter (HDF5 → LeRobot v2.1, only needed if you start from raw demos):
  `python -m robocasa.scripts.dataset_scripts.convert_hdf5_lerobot --raw_dataset_path <demos.hdf5>`
