# WM4A: World Model for Action

WM4A repurposes **pretrained video-generation world models** as visual encoders for
robot action prediction. Instead of using a Vision-Language Model (VLM) backbone,
WM4A feeds images through a **Diffusion Transformer (DiT)** — the same architecture
used for video prediction — and attaches a lightweight action head on top of its
intermediate representations.

## Motivation

Video prediction models learn rich spatiotemporal representations of physics,
object permanence, and dynamics. WM4A exploits these representations directly:

```
VLM approach:  Image → VLM encoder → language-aligned features → Action Head
WM4A approach: Image → Video DiT encoder → physics-aligned features → Action Head
```

## Supported Backbones

| Backbone | Params | Layers | Hidden Dim | Source |
|----------|--------|--------|------------|--------|
| **Cosmos-Predict2-2B** | 2B | 28 | 2048 | NVIDIA Cosmos |
| **Wan2.2-T2V** | 5B | 30 | 3072 | Alibaba Wan |

## Architecture

```
Input: {images: [PIL], instruction: str, actions: [T, action_dim]}
  │
  ├─ Text Encoder (T5 / UMT5)
  │   instruction → [B, L_text, text_hidden]
  │
  ├─ VAE Encoder
  │   images → latents [B, 16, T_latent, H/8, W/8]
  │
  └─ DiT Transformer (frozen or fine-tuned)
      latents + text_embeds → hidden_states [B, N_tokens, hidden_dim]
      │
      └─ Action Head (one of three variants below)
          → predicted actions [B, chunk_len, action_dim]
```

## Action Head Variants

Each backbone can be paired with three action head types:

| Variant | Action Head | What it uses | Speed | Frameworks |
|---------|------------|--------------|-------|------------|
| **OFT** | MLP regression | Last layer only | Fastest | `CosmoPredict2OFT`, `WanOFT` |
| **GR00T** | Flow-matching diffusion (single-layer) | Last layer only | Medium | `CosmoPredict2GR00T`, `WanGR00T` |
| **PI** | Layer-wise cross-attention DiT | All transformer layers | Slowest | `CosmoPredict2PI`, `WanPI` |

This gives **7 framework combinations** (including a generic `WM4A_OFT`).

## Quick Start: Inspect the Data Flow

Run the built-in demo to see the full forward pass (training + inference):

```bash
python starVLA/model/framework/WM4A/CosmoPredict2GR00T.py
```

This script:
1. Loads a config and instantiates the `CosmoPredict2_GR00T` model
2. Creates a synthetic batch with multi-view images and random actions
3. Runs the **training forward** and prints the action loss
4. Runs the **inference forward** and prints the predicted actions

> **Note:** You need the Cosmos-Predict2-2B weights downloaded locally.
> Set the path in the script or download via:
> ```bash
> huggingface-cli download nvidia/Cosmos-Predict2-2B-Video2World \
>     --local-dir ./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World
> ```

## Training

### Recommended: LIBERO with CosmoPredict2OFT

The simplest way to train a WM4A model is with the OFT (MLP) action head on
the LIBERO benchmark:

```bash
# From the starVLA project root
bash examples/LIBERO/train_files/run_libero_train.sh
```

Before running, edit the script to set your local paths:

```bash
Framework_name=CosmoPredict2OFT          # or CosmoPredict2GR00T, WanOFT, etc.
base_wm=nvidia/Cosmos-Predict2-2B-Video2World>  # still needed for tokenizer/processor
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=<path_to_LIBERO_dataset>
data_mix=libero_all                       # or libero_goal
```

Key training arguments:

| Argument | Description | Recommended |
|----------|-------------|-------------|
| `--framework.name` | Framework class name | `CosmoPredict2OFT` |
| `--framework.action_model.future_action_window_size` | Action prediction horizon | 7 |
| `--datasets.vla_data.per_device_batch_size` | Per-GPU batch size | 16 (OFT) / 8 (GR00T) |
| `--trainer.max_train_steps` | Total training steps | 80000 |
| `--trainer.save_interval` | Checkpoint save frequency | 10000 |

### Switching Backbones

To use Wan instead of Cosmos, simply change the framework name:

```bash
Framework_name=WanOFT       # or WanGR00T, WanPI
```

The config YAML remains the same — the framework class handles backbone loading
automatically based on its registered name.

## Code Structure

```
starVLA/model/
├── framework/
│   └── WM4A/
│       ├── CosmoPredict2GR00T.py   # Cosmos + flow-matching head
│       ├── CosmoPredict2OFT.py     # Cosmos + MLP head
│       ├── CosmoPredict2PI.py      # Cosmos + layer-wise cross-DiT head
│       ├── WanGR00T.py             # Wan + flow-matching head
│       ├── WanOFT.py               # Wan + MLP head
│       ├── WanPI.py                # Wan + layer-wise cross-DiT head
│       └── WM4A_OFT.py            # Generic WM backend + MLP head
└── modules/
    ├── world_model/
    │   └── CosmoPredict2.py        # Cosmos backbone wrapper (VAE + T5 + DiT)
    └── action_model/
        ├── MLP_ActionHeader.py             # OFT action head
        ├── flow_matching_head/             # GR00T action head (single-layer FM)
        └── LayerwiseFM_ActionHeader.py     # PI action head (layer-wise FM)
```

## Key Implementation Details

- **Precision**: DiT forward runs in `bfloat16` for speed; action head runs in
  `float32` for numerical stability.
- **Feature extraction**: Uses PyTorch hooks on selected DiT blocks to capture
  intermediate hidden states without modifying the backbone code.
- **Timestep trick**: Sets `timestep=0` during feature extraction — this extracts
  representations at the near-clean noise level rather than denoising.
- **Shape reshaping**: DiT outputs `[B, C, T, H, W]` video tensors, which are
  reshaped to `[B, T*H*W, C]` token sequences for the action head.
- **Framework registry**: All WM4A classes register themselves via
  `@FRAMEWORK_REGISTRY.register("FrameworkName")` and are auto-discovered at
  import time — no manual imports needed.
