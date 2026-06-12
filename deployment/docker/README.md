# Docker policy server

This directory contains a Docker image and compose example for the starVLA
websocket policy server (`deployment/model_server/server_policy.py`).

The image uses `python:3.10-slim` and does not install a CUDA toolkit. The VLM
interfaces run inference with sdpa, and torch's pip wheels bundle the CUDA 12.4
runtime. The host only needs an NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

The image accepts `--build-arg PIP_INDEX_URL=...` for pip mirrors and honors the
standard `HTTP_PROXY`/`HTTPS_PROXY` build args on restricted networks.

## Build

From the repository root:

```bash
docker build -f deployment/docker/Dockerfile.server -t starvla-policy-server .
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
setup only containerizes the policy server; simulators can keep running on the
host or in their own environments.

## Docker compose

The compose file builds the same server image and mounts the checkpoint and base
VLM directories:

```bash
export MODEL_DIR=/abs/path/to/Qwen3-VL-OFT-Robocasa
export PRETRAINED_DIR=/abs/path/to/Pretrained_models
docker compose -f deployment/docker/docker-compose.yml up
```

`CKPT_FILE` is overridable via environment variable when the checkpoint file name
differs from `checkpoints/steps_90000_pytorch_model.pt`.
