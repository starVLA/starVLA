# VLM Module (`starVLA/model/modules/vlm/`)

## Overview

The VLM module wraps various **Vision-Language Model** backends (Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, Florence-2),
providing a unified interface to the VLA frameworks in `framework/VLM4A/`.

## Interface Contract

Every VLM wrapper inherits `nn.Module` and implements:

| Method | Signature | Purpose |
|------|------|------|
| `__init__` | `(config)` | Load pretrained model + processor |
| `forward` | `(**kwargs) -> CausalLMOutputWithPast` | Forward pass (train / inference) |
| `generate` | `(**kwargs)` | Autoregressive generation |
| `build_qwenvl_inputs` | `(images, instructions, solutions=None)` | Build model inputs |

## Factory Function

```python
from starVLA.model.modules.vlm import get_vlm_model
vlm = get_vlm_model(config)  # Dispatches on config.framework.qwenvl.base_vlm
```

## Existing Implementations

| File | Class | Supported Models |
|------|------|-----------|
| `QWen2_5.py` | `_QWen_VL_Interface` | Qwen2.5-VL series |
| `QWen3.py` | `_QWen3_VL_Interface` | Qwen3-VL series |
| `QWen3_5.py` | `_QWen3_5_VL_Interface` | Qwen3.5-VL series |
| `Florence2.py` | `_Florence_Interface` | Florence-2 series |
| `Gemma4.py` | `_Gemma4_VL_Interface` | Gemma-4 (E2B, etc.) |
| `Molmo2.py` | `_Molmo2_VL_Interface` | AllenAI Molmo2 (4B / 8B / O-7B / VideoPoint-4B) |
| `MiniCPM_V.py` | `_MiniCPM_VL_Interface` | OpenBMB MiniCPM-V 4.6 |

## Speedups Applied by the Factory

`get_vlm_model()` applies a fixed list of speedups (`BACKBONE_SPEEDUPS` in `__init__.py`) to the interface it builds.
All of them leave the results **bitwise identical**; each one checks the model and skips itself where it does not
apply, so there is nothing to configure and nothing to add when writing a new wrapper.

| File | Function | What it does |
|------|------|-----------|
| `monkey_patch.py` | `patch_attention_mask` | Causal + padding mask from two tensor ops instead of nested `torch.vmap` |
| `monkey_patch.py` | `batch_vision_attention` | One attention call per vision layer instead of one per image (Qwen towers, SDPA) |
| `monkey_patch.py` | `cache_vision_positions` | Vision position tables built once per grid size (Qwen towers) |
| `starVLA/dataloader/worker_prep.py` | `prepare_inputs_in_workers` | `build_qwenvl_inputs` runs in the data-loader workers |

A framework that edits the prompt before calling `build_qwenvl_inputs` registers the edit with `@prompt` in
`starVLA/dataloader/worker_prep.py`; without it the inputs are simply built in the training process as before.
Tests: `tests/test_speedups.py`, `tests/test_speedups_frameworks.py`.

## Relationship with World Model

Cosmos-Reason2 has been migrated to `starVLA/model/modules/world_model/`.
If `config.framework.qwenvl.base_vlm` contains `cosmos-reason2`,
`get_vlm_model()` automatically delegates to `get_world_model()` for backward compatibility.

## Data Flow

```
Framework.__init__()
  └─> get_vlm_model(config) -> VLM wrapper instance
       └─> self.qwen_vl_interface

Framework.forward(examples)
  └─> self.qwen_vl_interface.build_qwenvl_inputs(images, instructions)
       └─> BatchFeature {input_ids, attention_mask, pixel_values, ...}
  └─> self.qwen_vl_interface(**inputs, output_hidden_states=True)
       └─> hidden_states[-1] : [B, L, H]
  └─> action_model(hidden_states, actions) -> loss
```
