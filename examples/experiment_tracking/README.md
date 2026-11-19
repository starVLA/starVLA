# SwanLab experiment tracking

`starVLA/training/train_starvla.py` supports SwanLab as an optional experiment
logger. Existing configurations continue to use W&B by default.

Install the optional dependency in your training environment:

```bash
pip install -e '.[swanlab]'
swanlab login
```

For unattended runs, supply `SWANLAB_API_KEY` through the environment instead of
storing it in a YAML file. See the [SwanLab environment variable documentation](https://docs.swanlab.cn/en/api/environment-variable.html).

Copy the `tracking` section from [swanlab.yaml](swanlab.yaml) into an existing
training configuration, or use CLI overrides on your usual launch command:

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --tracking.backend swanlab \
  --tracking.project starVLA
```

The experiment name is `run_id`. Logs are saved under `<output_dir>/swanlog`, and
the resolved training configuration is attached to the SwanLab run. Set
`tracking.workspace` to an organization username when needed, and
`tracking.mode` to `offline` for local logging without login or network access.
Available modes follow [swanlab.init](https://docs.swanlab.cn/en/api/py-init.html).

Only the main process initializes, logs, and finishes the experiment. Tracking
initialization and logging errors are handled on a best-effort basis, preserving
the trainer's existing rank synchronization and W&B behavior. `WANDB_MODE` and
`WANDB_DISABLED` only apply when the selected backend is W&B. The VLM-only and
co-training entry points retain their existing W&B integration.

Run the focused tests from the repository root (only OmegaConf is required;
the tracking SDKs are mocked):

```bash
python -m unittest discover -s examples/experiment_tracking -p 'test_*.py' -v
```
