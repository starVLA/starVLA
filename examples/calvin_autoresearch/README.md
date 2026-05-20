# CALVIN AutoResearch Smoke Baseline

This directory is the safe entry point for the current WMH CALVIN baseline and
the final image submission workflow:

- VLM backbone: `Qwen3-VL-4B-Instruct-Action` as a base model asset.
- Action head: StarVLA `QwenGR00T`, initialized from config, not from an upstream action-trained checkpoint.
- Imitation data: CALVIN ABC LeRobot v3.0 mixture `calvin_abc_train_v3.0`.
- Closed-loop eval: official CALVIN task D with a checkpoint produced by this WMH run.

Do not use `examples/calvin/eval_files/run_policy_server.sh` or
`examples/calvin/eval_files/eval_calvin.sh` for this baseline. Those upstream
scripts still contain trained checkpoint defaults and are intentionally left
unchanged until WMH-trained checkpoints replace them.

## Final Submission Entrypoints

Run these from a shell that has the project and public data mounts visible.

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh
```

Prepare stable public links and the manifest used by the final image/test
harness:

```bash
bash examples/calvin_autoresearch/scripts/prepare_submission_env.sh
```

If this shell cannot write `/public/seven`, run the same command in a login or
GPU shell with public write permission. The key link it creates is:

```bash
mkdir -p /inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/checkpoints/wmh_trained
ln -sfnT \
  /inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_pretrain_qwen3vl_gr00t_headonly_h200_60k_0518_163437/checkpoints/steps_60000_pytorch_model.pt \
  /inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/checkpoints/wmh_trained/best_abc_to_d_steps_60000_pytorch_model.pt
```

Reproduce CALVIN ABC training on a 3-GPU H200 allocation:

```bash
GPU_IDS=0,1,2 NUM_PROCESSES=3 MAX_TRAIN_STEPS=60000 \
  bash examples/calvin_autoresearch/scripts/run_train_abc_h200_oneclick.sh
```

Run one-command ABC->D closed-loop evaluation. Use `NUM_SEQUENCES=10` for a
short smoke pass and `NUM_SEQUENCES=1000` for the formal pass.
The default evaluator disables CALVIN EGL rendering for portability; set
`CALVIN_USE_EGL=1` only if the GPU image has a working EGL stack.

```bash
NUM_SEQUENCES=10 GPU_ID=0 PORT=5694 \
  bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_oneclick.sh
```

To use multiple H200s, run the sharded parallel evaluator. It starts one or more
policy servers per GPU, assigns non-overlapping sequence indices, and writes an
aggregated `results.json`. New runs also write `metrics.json` and
`metrics_sequences_epoch_0.jsonl` with conditional success, failure position,
failure step, per-atomic-task success, exact task-chain success, near-miss
flags, and action magnitude/saturation/jitter statistics.

```bash
TOTAL_SEQUENCES=1000 GPU_IDS=0,1,2,3 BASE_PORT=5800 \
  bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh
```

For a conservative run that uses exactly one worker on each GPU:

```bash
TOTAL_SEQUENCES=1000 GPU_IDS=0,1,2,3 WORKERS_PER_GPU=1 BASE_PORT=5800 \
  bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh
```

If all workers finished but the parent process did not aggregate, finalize the
directory manually:

```bash
EVAL_DIR=/path/to/eval_abc_to_d_parallel_n1000_xxxx \
  bash examples/calvin_autoresearch/scripts/finalize_parallel_eval_dir.sh
```

Print the detailed metric summary:

```bash
python examples/calvin_autoresearch/scripts/summarize_eval_metrics.py \
  /path/to/eval_abc_to_d_parallel_n1000_xxxx/metrics.json
```

To compare D closed-loop task success with an ABC closed-loop eval, pass both
metrics files. Without `--abc-metrics`, the script reports ABC LeRobot training
task distribution only, not ABC success rate.

```bash
python examples/calvin_autoresearch/scripts/compare_abc_d_task_success.py \
  --d-metrics /path/to/d_eval/metrics.json \
  --abc-metrics /path/to/abc_eval/metrics.json \
  --out /path/to/abc_vs_d_task_success.json
```

The current best checkpoint is linked at:

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/checkpoints/wmh_trained/best_abc_to_d_steps_60000_pytorch_model.pt
```

## Preflight

```bash
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh
bash examples/calvin_autoresearch/scripts/verify_assets.sh
```

Use `STRICT_ASSETS=1` when the base model and datasets should already be
present:

```bash
STRICT_ASSETS=1 bash examples/calvin_autoresearch/scripts/verify_assets.sh
```

Expected default asset locations:

```text
playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
playground/Datasets/calvin_lerobot/calvin_abc_train_v3.0
playground/Datasets/calvin_original/task_D_D
```

The full asset list is tracked in `examples/calvin_autoresearch/configs/assets.yaml`.

Download or link the base model and LeRobot assets:

```bash
bash examples/calvin_autoresearch/scripts/setup_assets.sh
```

The official CALVIN D dataset is already reused from:

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d
```

`prepare_submission_env.sh` links it into both the local project layout and the
shared `/public/seven` layout. Verify the formal D split with:

```bash
CHECK_ORIGINAL_D=1 STRICT_ASSETS=1 bash examples/calvin_autoresearch/scripts/verify_assets.sh
```

## Train One Legal Smoke Checkpoint

```bash
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh
STRICT_ASSETS=1 MAX_TRAIN_STEPS=1 SAVE_INTERVAL=1 NUM_PROCESSES=1 \
  bash examples/calvin_autoresearch/scripts/run_train_smoke.sh
```

This writes a new WMH checkpoint under:

```text
results/Checkpoints/baseline_qwen3vl_action_gr00t_calvin_abc_smoke/checkpoints/
```

## Train CALVIN ABC On H200

Run this in a GPU session. It freezes the Qwen VLM interface and trains the
randomly initialized QwenGR00T / DiT-B action head on CALVIN ABC only.

```bash
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh
GPU_IDS=0,1,2 NUM_PROCESSES=3 BATCH_SIZE=16 MAX_TRAIN_STEPS=20000 SAVE_INTERVAL=5000 \
  DATALOADER_NUM_WORKERS=8 DATALOADER_PREFETCH_FACTOR=2 \
  bash examples/calvin_autoresearch/scripts/run_train_abc_pretrain_h200.sh
```

By default, checkpoints are written under:

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/
```

This launcher is hard-gated to `calvin_abc_train_v3.0`; it refuses CALVIN D or
ABCD-D dataset names.

## Serve And Evaluate D Smoke

Start the policy server with a newly trained WMH checkpoint:

```bash
CKPT=results/Checkpoints/baseline_qwen3vl_action_gr00t_calvin_abc_smoke/checkpoints/steps_1_pytorch_model.pt \
  bash examples/calvin_autoresearch/scripts/run_policy_server.sh
```

In a CALVIN-capable environment, run one D sequence:

```bash
CALVIN_PYTHON=/path/to/calvin/env/bin/python \
CALVIN_D_DATASET=/path/to/calvin/task_D_D \
CALVIN_CONFIG_PATH=/path/to/calvin/calvin_models/conf \
CKPT=results/Checkpoints/baseline_qwen3vl_action_gr00t_calvin_abc_smoke/checkpoints/steps_1_pytorch_model.pt \
NUM_SEQUENCES=1 \
  bash examples/calvin_autoresearch/scripts/run_eval_d_smoke.sh
```

## Formal ABC To D Eval

For normal use, prefer the one-command evaluator above. The explicit two-step
form is useful for debugging server startup separately:

```bash
RUN_ID=abc_pretrain_qwen3vl_gr00t_headonly_h200_60k_0518_163437
CKPT=/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/${RUN_ID}/checkpoints/steps_60000_pytorch_model.pt
CALVIN_D_DATASET=/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/datasets/calvin_original/task_D_D
CALVIN_CONFIG_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/four/calvin/calvin_models/conf
CALVIN_PYTHON=/inspire/qb-ilm2/project/26summer-camp-10/public/four/miniconda3/envs/calvin_venv/bin/python
EVAL_LOG_DIR=/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/reports/eval_abc_to_d_formal_n10

GPU_ID=0 PORT=5694 CKPT="${CKPT}" \
  bash examples/calvin_autoresearch/scripts/run_policy_server.sh

CALVIN_PYTHON="${CALVIN_PYTHON}" \
CALVIN_D_DATASET="${CALVIN_D_DATASET}" \
CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH}" \
CKPT="${CKPT}" \
PORT=5694 \
NUM_SEQUENCES=10 \
EVAL_LOG_DIR="${EVAL_LOG_DIR}" \
  bash examples/calvin_autoresearch/scripts/run_eval_d_formal.sh
```
