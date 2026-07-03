# OpenPI PI0/PI05 LIBERO Training

This folder contains StarVLA configs for OpenPI-style PI0 and PI05 on LIBERO.
Use the 8-GPU configs for full training, and `pi05_libero_local.yaml` for a
single-machine smoke run.

Converted OpenPI-to-StarVLA checkpoints are available at:
`https://huggingface.co/tenstep/pi_model_starvla`

## Local Layout

The default files assume this repo-local layout:

```text
data/starvla_lerobot_root/libero
paligemma_tokenizer.model
openpi_converted_protocol/pi0_base_starvla/fp32/model.safetensors
openpi_converted_protocol/pi05_base_starvla/fp32/model.safetensors
outputs/starvla
```

Override any of these with env vars in the shell scripts if your layout differs.

## Files

- `pi0_libero_8gpu.yaml`: PI0 full training config.
- `pi05_libero_8gpu.yaml`: PI05 full training config.
- `pi05_libero_local.yaml`: PI05 local smoke/debug config.
- `run_pi0_libero_8gpu.sh`: PI0 launcher.
- `run_pi05_libero_8gpu.sh`: PI05 launcher.
- `pi0_05_to_starvla.py`: convert OpenPI checkpoints to StarVLA format.

## Convert Base Checkpoint

If you already want ready-to-use converted checkpoints, download them from
`https://huggingface.co/tenstep/pi_model_starvla` and place them under
`openpi_converted_protocol/`.

```bash
python examples/simBenchmarks/LIBERO/train_files/openpi/pi0_05_to_starvla.py \
  --model PI05 \
  --checkpoint "${OPENPI_BASE_CHECKPOINT}" \
  --output-dir openpi_converted_protocol/pi05_base_starvla \
  --tokenizer paligemma_tokenizer.model \
  --variants fp32 bfloat16
```

Use `PI0` and `pi0_base_starvla` for PI0.

## Train

```bash
bash examples/simBenchmarks/LIBERO/train_files/openpi/run_pi0_libero_8gpu.sh
bash examples/simBenchmarks/LIBERO/train_files/openpi/run_pi05_libero_8gpu.sh
```

Common overrides:

```bash
DATA_ROOT=/path/to/libero \
TOKENIZER_MODEL=/path/to/paligemma_tokenizer.model \
PRETRAINED_CHECKPOINT=/path/to/model.safetensors \
WANDB_ENTITY=your_wandb_entity \
WANDB_MODE=disabled \
bash examples/simBenchmarks/LIBERO/train_files/openpi/run_pi05_libero_8gpu.sh
```

For a local smoke run, start from `pi05_libero_local.yaml` and lower or raise
`per_device_batch_size`, `num_workers`, and `max_train_steps` as needed.

## Reproduction Results

| Method | Setting | Spatial | Object | Goal | LIBERO-10 | Avg |
|---|---|---:|---:|---:|---:|---:|
| OpenPI Repo (official) | mixed | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |
| OpenPI JAX re-run eval | mixed | 98.4 | 99.2 | 98.8 | 91.2 | 96.90 |
| OpenPI Torch re-run eval | FP32 | 98.2 | 98.6 | 97.4 | 91.2 | 96.35 |
| OpenPI Torch re-run eval | BF16 | 98.0 | 99.0 | 98.4 | 93.4 | 97.20 |
| StarVLA Torch reproduced eval | FP32-Eval1 | 98.0 | 98.4 | 97.2 | 92.8 | 96.60 |
| StarVLA Torch reproduced eval | FP32-Eval2 | 98.0 | 98.4 | 98.6 | 92.6 | 96.90 |
| StarVLA Torch reproduced eval | FP32-Eval3 | 98.6 | 99.4 | 98.6 | 92.6 | 97.30 |
| StarVLA Torch reproduced eval | BF16 | 98.8 | 98.2 | 98.0 | 91.6 | 96.65 |
| StarVLA Torch reproduced train + eval | mixed precision + DeepSpeed | 99.0 | 98.4 | 96.4 | TBD | TBD |

Notes:
- `re-run eval` results can fluctuate slightly because LIBERO evaluation is not bit-for-bit deterministic across runs.
- StarVLA training uses mixed precision with DeepSpeed. This is different from the OpenPI Torch `BF16` setting, which refers to a pure Torch BF16 run.

## Notes

- `include_state=true` is required for PI0 and PI05.
- PI0 uses `action_mode: delta`.
- PI05 uses `action_mode: abs` because the downloaded LeRobot LIBERO actions are
  already delta EEF actions.
- Eval entrypoints live under `examples/simBenchmarks/LIBERO/eval_files/openpi`.
