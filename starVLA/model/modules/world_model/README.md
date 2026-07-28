# World Model Module (`starVLA/model/modules/world_model/`)

## Overview

The World Model module wraps **world model** backends used for action prediction.
Unlike VLM backends, these models focus on physical-world modeling during pretraining
(e.g., video prediction, spatial reasoning), providing stronger physical-prior understanding.

## Why Does Cosmos-Reason2 Use the Qwen3-VL Architecture?

Cosmos-Reason2 is NVIDIA's model fine-tuned for physical reasoning on top of the Qwen3-VL architecture.
It shares the same `Qwen3VLForConditionalGeneration` and `Qwen3VLProcessor` under the hood,
making its interface fully compatible with the VLM wrappers. The differences are:

- **Different pretraining objective**: Cosmos focuses on physical reasoning and causal understanding
- **Different representation semantics**: hidden_states encode richer spatio-temporal dynamics
- **Use case**: better suited for manipulation tasks requiring physical understanding

## Interface Contract

Fully aligned with the VLM module (enabling transparent swap at the framework layer):

| Method | Signature | Purpose |
|------|------|------|
| `__init__` | `(config)` | Load pretrained model + processor |
| `forward` | `(**kwargs) -> CausalLMOutputWithPast` | Forward pass |
| `generate` | `(**kwargs)` | Autoregressive generation |
| `build_qwenvl_inputs` | `(images, instructions)` | Build model inputs |

## Factory Function

```python
from starVLA.model.modules.world_model import get_world_model
wm = get_world_model(config)  # Dispatches on model name
```

## Existing Implementations

| File | Class | Supported Models | Architecture |
|------|------|-----------|------|
| `CosmosReason2.py` | `_CosmosReason2_Interface` | nvidia/Cosmos-Reason2-2B | Qwen3-VL (autoregressive LM), hidden=2048 |
| `CosmoPredict2.py` | `_CosmoPredict2_Interface` | nvidia/Cosmos-Predict2-2B | DiT (Diffusion Transformer), hidden=4096 |

## Usage in Framework Layer

The World Model is consumed by frameworks under `framework/WM4A/`:

```python
# WM4A/CosmosGR00T.py
from starVLA.model.modules.world_model import get_world_model

class Cosmos_GR00T(baseframework):
    def __init__(self, config):
        self.qwen_vl_interface = get_world_model(config=self.config)
        # Subsequent usage is identical to VLM4A frameworks
```

## Architecture Overview

```
starVLA/model/
├── modules/
│   ├── vlm/                # VLM backends (Qwen2.5-VL, Qwen3-VL, ...)
│   ├── world_model/        # World model backends (Cosmos-Reason2, ...)
│   ├── action_model/       # Action heads (DiT, FAST, L1Regression, ...)
│   ├── dino_model/         # DINO spatial encoder
│   └── projector/          # QFormer and other projection modules
├── framework/
│   ├── base_framework.py   # Base class + build_framework() + auto-import
│   ├── share_tools.py      # Config merge utilities
│   ├── VLM4A/              # VLM-for-Action frameworks
│   │   ├── QwenGR00T.py    # VLM + DiT flow-matching
│   │   ├── QwenFast.py     # VLM + FAST discrete token
│   │   ├── QwenAdapter.py  # VLM + Query adapter
│   │   ├── QwenPI.py       # VLM + Layer-wise FM
│   │   ├── QwenDual.py     # VLM + DINO + DiT
│   │   ├── LangForce.py    # VLM + Dual-branch DiT
│   │   ├── ABot_M0.py      # VLM + VGGT + DiT
│   │   └── M1.py           # VLM + DINO + QFormer + DiT
│   └── WM4A/               # World-Model-for-Action frameworks
│       ├── CosmosGR00T.py           # CosmosReason2 + Flowmatching
│       └── CosmoPredict2GR00T.py    # CosmosPredict2 (DiT) + Flowmatching
└── tools.py                # Registry, normalization utilities
```
