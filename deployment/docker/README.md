# Docker images

This directory contains Docker images for serving, training, and robocasa
tabletop evaluation.

| Image | Dockerfile | What it runs |
| --- | --- | --- |
| `starvla-policy-server` | `Dockerfile.server` | The websocket policy server (`deployment/model_server/server_policy.py`) |
| `starvla-train` | `Dockerfile.train` | Training environments that need CUDA build tools, DeepSpeed, and flash-attn |
| `starvla-robocasa-eval` | `Dockerfile.robocasa` | The robocasa-gr1-tabletop simulator and starVLA eval client |

The serving image uses `python:3.10-slim` and does not install a CUDA toolkit.
The VLM interfaces run inference with sdpa, and torch's pip wheels bundle the
CUDA 12.4 runtime. The host only needs an NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

All images accept `--build-arg PIP_INDEX_URL=...` for pip mirrors and honor the
standard `HTTP_PROXY`/`HTTPS_PROXY` build args on restricted networks. Large
runtime assets, checkpoints, datasets, and outputs are mounted instead of baked
into images.

## Build

From the repository root:

```bash
docker build -f deployment/docker/Dockerfile.server -t starvla-policy-server .
docker build -f deployment/docker/Dockerfile.train -t starvla-train .
docker build -f deployment/docker/Dockerfile.robocasa -t starvla-robocasa-eval .
```

Behind a slow PyPI route, add:

```bash
  --build-arg PIP_INDEX_URL=https://your-mirror/simple
```

## Serve a checkpoint

The server needs two mounts:

| Mount target | Content |
| --- | --- |
| `/models` | A checkpoint snapshot directory: `config.yaml`, `dataset_statistics.json`, `checkpoints/steps_XXXXX_pytorch_model.pt` (the layout of the released HF repos, e.g. [StarVLA/Qwen3-VL-OFT-Robocasa](https://huggingface.co/StarVLA/Qwen3-VL-OFT-Robocasa)) |
| `/workspace/starVLA/playground/Pretrained_models` | The base VLM referenced by the checkpoint's `framework.qwenvl.base_vlm` (e.g. `Qwen3-VL-4B-Instruct`) |

```bash
docker run --gpus all -p 5678:5678 \
  -v /abs/path/to/Qwen3-VL-OFT-Robocasa:/models:ro \
  -v /abs/path/to/Pretrained_models:/workspace/starVLA/playground/Pretrained_models:ro \
  starvla-policy-server \
  --ckpt_path /models/checkpoints/steps_90000_pytorch_model.pt \
  --port 5678 --use_bf16 --idle_timeout -1
```

`--idle_timeout -1` keeps the server alive indefinitely; the default (1800 s)
shuts it down after 30 idle minutes.

Evaluation clients (LIBERO, Robocasa_tabletop, SimplerEnv, ...) connect to the
published port exactly as described in each benchmark's README. This Docker
setup can run only the policy server, or it can also run the robocasa eval
client through the compose `eval` profile.

## Train

The training image is a reusable environment rather than a one-shot entrypoint.
Mount the project data/output directories and launch the benchmark script you
need:

```bash
docker run --gpus all -it --shm-size 16g \
  -v /abs/path/to/playground:/workspace/starVLA/playground \
  -v /abs/path/to/results:/workspace/starVLA/results \
  starvla-train \
  bash examples/Robocasa_tabletop/train_files/run_robocasa.sh
```

The default `FLASH_ATTN_WHEEL` build arg matches the torch pin used here
(torch 2.6 / CUDA 12 / Python 3.10). Override it if you change torch or Python.

## Robocasa tabletop evaluation

The robocasa image installs `robosuite` from source at `v1.5.1`. The PyPI wheel
does not ship `robosuite/examples`, where the GR1 whole-body mink IK controller
lives, so this is intentionally pinned in the Dockerfile.

Download the simulation assets once on the host (they are large and should not
be baked into the image):

```bash
docker run --rm \
  -v /abs/path/to/assets:/workspace/robocasa-gr1-tabletop-tasks/robocasa/models/assets \
  starvla-robocasa-eval \
  python /workspace/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_tabletop_assets.py -y
```

## Docker compose

The compose file starts the policy server by default:

```bash
export MODEL_DIR=/abs/path/to/Qwen3-VL-OFT-Robocasa
export PRETRAINED_DIR=/abs/path/to/Pretrained_models
docker compose -f deployment/docker/docker-compose.yml up
```

`CKPT_FILE` is overridable via environment variable when the checkpoint file name
differs from `checkpoints/steps_90000_pytorch_model.pt`.

Run the server plus robocasa eval client with the `eval` profile:

```bash
export MODEL_DIR=/abs/path/to/Qwen3-VL-OFT-Robocasa
export PRETRAINED_DIR=/abs/path/to/Pretrained_models
export ROBOCASA_ASSETS_DIR=/abs/path/to/assets
docker compose -f deployment/docker/docker-compose.yml --profile eval up
```

The compose eval command defaults to the released
`StarVLA/Qwen3-VL-OFT-Robocasa` input contract (`--args.no_send_state`). For
state-conditioned checkpoints such as Qwen3VL-GR00T, remove that flag or run an
equivalent command that keeps the default `send_state=True`.

`ROBOCASA_ASSETS_DIR` defaults to `./robocasa_assets` only so the compose file
can render without the eval profile's assets present. Set it explicitly before
running real robocasa evaluations.

Useful robocasa overrides:

| Variable | Default |
| --- | --- |
| `ROBOCASA_ENV_NAME` | `gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env` |
| `ROBOCASA_N_EPISODES` | `50` |
| `ROBOCASA_N_ENVS` | `1` |
| `ROBOCASA_MAX_EPISODE_STEPS` | `720` |
| `ROBOCASA_N_ACTION_STEPS` | `12` |
| `UNNORM_KEY` | `gr1` |
| `EVAL_OUT_DIR` | `./eval_out` |

Run an interactive training container with the `train` profile:

```bash
export PLAYGROUND_DIR=/abs/path/to/playground
export RESULTS_DIR=/abs/path/to/results
docker compose -f deployment/docker/docker-compose.yml --profile train run --rm train
```
