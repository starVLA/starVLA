# StarVLA PI0/PI05 LIBERO Eval

This folder keeps the minimal LIBERO eval path for OpenPI-style `PI0` / `PI05`
inside the StarVLA codebase.

Converted OpenPI-to-StarVLA checkpoints are available at:
`https://huggingface.co/tenstep/pi_model_starvla`

Default model loading uses converted StarVLA checkpoints under
`OPENPI_CONVERTED_ROOT`:

```text
openpi_converted_protocol
```

Action normalization stats are read from the converted directory when present,
otherwise the code falls back to the original OpenPI assets directory.

Retained entry points:
- `server_starvla_openpi.py`: StarVLA model server with OpenPI-style action semantics.
- `eval_starvla_openpi_client.py`: host-side LIBERO rollout client against that server.
- `run_eval_pi0_libero.sh` / `run_eval_pi05_libero.sh`: host-side smoke wrappers.
- `run_servers.sh`: minimal multi-server launcher.

## Smoke Tests

Start the model server in apptainer:

```bash
cd /path/to/starvla_lab
appstart
source ~/.bashrc
uvact starvla
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OPENPI_CONVERTED_ROOT=/path/to/openpi_converted_protocol
export OPENPI_ASSETS_ROOT=/path/to/openpi_libero/torch

MODEL=PI05 MODEL_SOURCE=openpi GPU_LIST=0 PORT_BASE=18000 \
  bash examples/simBenchmarks/LIBERO/eval_files/openpi/run_servers.sh
```

Run the LIBERO client on the host:

```bash
cd /path/to/starvla_lab
source ~/.bashrc
conda activate py310
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LIBERO_HOME=/path/to/libero

MODEL_SOURCE=openpi PORT=18000 RUN_ID=pi05_converted_smoke \
  bash examples/simBenchmarks/LIBERO/eval_files/openpi/run_eval_pi05_libero.sh
```

For a StarVLA-trained PI05 checkpoint, pass the checkpoint on the server and switch
both sides to `MODEL_SOURCE=starvla`:

```bash
CHECKPOINT=/path/to/starvla_pi05_libero_multitask/checkpoints/steps_60000_pytorch_model.pt \
MODEL=PI05 MODEL_SOURCE=starvla GPU_LIST=0 PORT_BASE=18000 \
  bash examples/simBenchmarks/LIBERO/eval_files/openpi/run_servers.sh

MODEL_SOURCE=starvla PORT=18000 RUN_ID=pi05_starvla_trained_smoke \
  bash examples/simBenchmarks/LIBERO/eval_files/openpi/run_eval_pi05_libero.sh
```

For converted PI0:

```bash
MODEL=PI0 MODEL_SOURCE=openpi GPU_LIST=0 PORT_BASE=18000 \
  bash examples/simBenchmarks/LIBERO/eval_files/openpi/run_servers.sh

MODEL_SOURCE=openpi PORT=18000 RUN_ID=pi0_converted_smoke \
  bash examples/simBenchmarks/LIBERO/eval_files/openpi/run_eval_pi0_libero.sh
```

## Default Checkpoints

```text
${OPENPI_CONVERTED_ROOT}/pi0_libero_starvla/fp32/model.safetensors
${OPENPI_CONVERTED_ROOT}/pi05_libero_starvla/fp32/model.safetensors
```
