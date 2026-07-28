# VM4A: VisuoMotor for Action

![Hugging Face checkpoint: TBD](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-checkpoint%20TBD-lightgrey)

## Overview

VM4A is StarVLA's family of lightweight visuomotor action policies. Unlike the VLM4A and WM4A families, VM4A does not use a vision-language model or a world-model backbone. It maps camera observations and proprioceptive state directly to a chunk of robot actions. ACT and Diffusion Policy use ImageNet-pretrained ResNet-18 visual encoders by default, while still supporting from-scratch training. The initial family contains these two complementary baselines.

These implementations use the same `baseframework` interface as the rest of StarVLA. Training returns an `action_loss`, while inference returns `normalized_actions`. Consequently, VM4A can be selected through configuration without adding a policy-specific trainer or dataloader branch.

## Motivation

Large pretrained multimodal backbones are valuable, but they make it difficult to distinguish their benefits from those of the data pipeline, action representation, and optimization setup. VM4A provides apples-to-apples visuomotor baselines under StarVLA's existing LeRobot data pipeline, normalization statistics, trainer, checkpoint format, and deployment interface. By default, only the compact ResNet-18 visual encoder starts from ImageNet weights, and that pretraining can be disabled. This makes ACT and Diffusion Policy useful controls for experiments that compare model families on the same observations and action chunks.

## Supported Variants

| Framework name | Policy | Visual encoder | Action generation |
| --- | --- | --- | --- |
| `ACT` | LeRobot `ACTPolicy` wrapper | ResNet-18, ImageNet-pretrained by default | Transformer action chunk |
| `DiffusionPolicy` | Trimmed `real-stanford/diffusion_policy` image-policy subset | ResNet-18, ImageNet-pretrained by default | Conditional 1D U-Net with a DDPM scheduler |

Both variants are real-robot validated (see PR description). Public checkpoints are TBD per maintainer discussion; no benchmark result is claimed here.

## Architecture

For every sample, StarVLA supplies one or more RGB camera views, the robot's proprioceptive state, and, during training, a future action sequence. VM4A converts these fields to each underlying policy's expected tensor layout:

```text
RGB camera views + proprioceptive state
                  |
                  v
        ACT or Diffusion Policy
                  |
                  v
          normalized action chunk
```

ACT consumes the current observation and predicts a fixed-length chunk. Diffusion Policy can consume multiple observation steps, encodes each configured camera with ResNet-18, and denoises a horizon of actions before returning the configured execution slice. Camera order follows `framework.image_keys`; action and state dimensions must match the dataset registry.

## Quick Start

Select a variant in the training YAML:

```yaml
framework:
  name: ACT  # or DiffusionPolicy
  action_dim: 8
  state_dim: 8
  image_keys: [cam0_rgb, cam1_rgb]
```

No manual import is required. `build_framework` scans framework subpackages, imports the public modules under `VM4A/`, and resolves the registered name. The underscore-prefixed vendored package is skipped by discovery.

## Training

The Realman example provides complete ACT and Diffusion Policy recipes. Replace its dataset and environment placeholders, then run, for example:

```bash
bash examples/realRobots/Realman/train_files/train_realman_act.sh
# or
bash examples/realRobots/Realman/train_files/train_realman_dp.sh
```

Important framework knobs are:

| Knob | Applies to | Meaning |
| --- | --- | --- |
| `chunk_size` | ACT | Number of future actions predicted and executed as a chunk |
| `pretrained_backbone_weights` | ACT | Defaults to ImageNet-pretrained `ResNet18_Weights.IMAGENET1K_V1`; use `null` to train from scratch or avoid a download |
| `horizon` | Diffusion Policy | Full action trajectory length modeled by diffusion |
| `pretrained_backbone` | Diffusion Policy | Defaults to `true` for ImageNet-pretrained ResNet-18; use `false` to train from scratch or avoid a download |
| `n_obs_steps` | Both | Number of observation steps expected by the policy |
| `n_action_steps` | Diffusion Policy | Number of predicted steps returned for execution |
| `num_train_timesteps` | Diffusion Policy | DDPM training schedule length |
| `num_inference_steps` | Diffusion Policy | Optional denoising-step override at inference |
| `image_size`, `image_keys` | Both | Input resolution and ordered camera names |

The dataset registry's action window must start at the sampled observation and cover the selected
policy horizon; Diffusion Policy trains on the immediate `horizon`-step prefix of that window (the
Realman example uses a DP-specific mixture whose window equals the horizon exactly). Keep
action/state dimensions and camera keys synchronized across the registry and YAML.

## Code Structure

```text
starVLA/model/framework/VM4A/
├── ACT.py                         # LeRobot ACT adapter
├── DiffusionPolicy.py             # Diffusion Policy adapter and EMA persistence
├── __init__.py
└── _dp_vendor/
    ├── LICENSE                    # Upstream MIT license
    └── diffusion_policy/          # Trimmed non-hybrid image-policy subset
```

## Key Implementation Details

- **Single normalization owner:** StarVLA's `StateActionTransform` and `PolicyNormProcessor` own dataset normalization. VM4A installs identity normalizers inside ACT and Diffusion Policy to prevent double normalization. Outputs remain in StarVLA's normalized action space for the shared deployment path to unnormalize.
- **Persistent Diffusion Policy EMA:** Upstream `EMAModel` is a plain class, not an `nn.Module`, so its `averaged_model` would be absent from a normal `state_dict`. VM4A saves true EMA weights under `ema_averaged.*`. When loading an older checkpoint without those keys, it reseeds EMA from the loaded raw action model rather than leaving random initialization in the inference path.
- **Self-contained vendoring:** `DiffusionPolicy.py` adds its local `_dp_vendor` directory to `sys.path`. This pins the required upstream implementation and carries only the non-hybrid image-policy subset, without requiring a separate Diffusion Policy checkout or a global path modification.
