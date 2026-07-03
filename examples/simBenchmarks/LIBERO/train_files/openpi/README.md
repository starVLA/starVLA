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

## Notes

- `include_state=true` is required for PI0 and PI05.
- PI0 uses `action_mode: delta`.
- PI05 uses `action_mode: abs` because the downloaded LeRobot LIBERO actions are
  already delta EEF actions.
- Eval entrypoints live under `examples/simBenchmarks/LIBERO/eval_files/openpi`.
