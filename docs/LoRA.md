# VLM LoRA Training

LoRA support fine-tunes the VLM backbone through PEFT while keeping the StarVLA
action model, optimizer, checkpoint, and `from_pretrained()` flows intact. The
main training entrypoint is `starVLA/training/train_starvla.py`.

## Support Scope

Formal support:

- All Qwen-series StarVLA VLM backbones that are mounted at
  `model.qwen_vl_interface.model`.
- All benches that launch `starVLA/training/train_starvla.py`, including
  Robotwin, LIBERO, SimplerEnv/OXE, DOMINO, Franka, Calvin, Robocasa,
  Robocasa365, RoboChallenge Table30v2, VLA-Arena, Gemma4, MiniCPM, and
  NeuralVLA launchers.

Experimental support:

- Non-Qwen VLMs whose StarVLA framework exposes a PEFT-compatible backbone
  module. The default module path is still `qwen_vl_interface.model` because
  current StarVLA VLM frameworks reuse that attribute name for the backbone
  interface.
- For non-Qwen models, start with `target_modules: all-linear` or explicitly
  set the target module names after inspecting the backbone.

Not included in this VLM LoRA path:

- WM4A internal world-model LoRA for DiT/T5/VAE. DiT LoRA can be useful as a
  separate follow-up, T5 LoRA is workload-dependent, and VAE LoRA is uncommon
  for this training target.

## Configuration

Use `framework.vlm.lora` as the canonical config path:

```yaml
framework:
  name: QwenOFT
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct
    attn_implementation: flash_attention_2
  vlm:
    lora:
      enabled: true
      r: 16
      alpha: 32
      dropout: 0.05
      bias: none
      target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
      adapter_path: null
      adapter_name: default
      is_trainable: true
      save_adapter_only: false
      module_path: qwen_vl_interface.model
      adapter_dir_name: vlm_lora_adapter
      task_type: CAUSAL_LM
      modules_to_save: null
```

`framework.qwenvl.lora` remains load-compatible for older local configs and
checkpoints, but new configs should use `framework.vlm.lora`.

For non-Qwen experiments:

```bash
export VLM_LORA_ENABLED=true
export VLM_LORA_TARGET_MODULES=all-linear
export VLM_LORA_R=8
export VLM_LORA_ALPHA=16
```

If a framework stores the backbone somewhere else, override:

```bash
export VLM_LORA_MODULE_PATH=some_interface.model
```

## Launchers

Direct `train_starvla.py` launchers source:

```bash
examples/common/vlm_lora_args.sh
```

Recommended environment variables:

```bash
export VLM_LORA_ENABLED=true
export VLM_LORA_R=16
export VLM_LORA_ALPHA=32
export VLM_LORA_DROPOUT=0.05
export VLM_LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export VLM_LORA_LR=2.0e-4
```

The helper also accepts the previous `LORA_*` names for compatibility. It maps
LoRA LR to `trainer.learning_rate.qwen_vl_interface` by default; in current
StarVLA this is the VLM interface module name, not a restriction to Qwen models.
Override it with `VLM_LORA_LR_MODULE` if a framework uses a different module
name.

Small Robotwin smoke example:

```bash
cd /home/hfang/code/starvla-lora
export PYTHONPATH="$PWD"
export WANDB_MODE=disabled

export NUM_PROCESSES=1
export PER_DEVICE_BATCH_SIZE=1
export MAX_TRAIN_STEPS=1
export SAVE_INTERVAL=999999
export EVAL_INTERVAL=999999
export LOGGING_FREQUENCY=1

export VLM_LORA_ENABLED=true
export VLM_LORA_R=2
export VLM_LORA_ALPHA=4
export VLM_LORA_DROPOUT=0.0
export VLM_LORA_TARGET_MODULES=q_proj,v_proj
export VLM_LORA_LR=2.0e-4

bash examples/Robotwin/train_files/run_robotwin_train.sh
```

The command still needs valid local model and dataset paths. On Ascend/NPU, a
backend-specific PEFT/transformers/torch_npu failure is not treated as a LoRA
migration failure if compile, config, and local module-path smoke checks pass.

## Optional ZeRO-1 Config

LoRA trains far fewer parameters than full fine-tuning. If ZeRO-2 communication
overhead dominates on a local cluster or Ascend/NPU setup, try the migrated
LoRA-oriented ZeRO-1 config:

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_lora_zero1.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml <bench_config.yaml> \
  --framework.vlm.lora.enabled true
```

Keep the default ZeRO-2 config when the model/head memory footprint requires
more optimizer-state partitioning. Use ZeRO-1 as a communication-light first
try for adapter-only or small-rank LoRA runs.

## Checkpoints

Full model checkpoints are saved by default. A sibling PEFT adapter directory is
also saved:

- `steps_N_pytorch_model.pt` -> `steps_N_vlm_lora_adapter/`
- `steps_N_model.safetensors` -> `steps_N_vlm_lora_adapter/`
- `final_model/pytorch_model.pt` -> `final_model/vlm_lora_adapter/`
- `final_model/model.safetensors` -> `final_model/vlm_lora_adapter/`

Set `save_adapter_only: true` to skip full StarVLA model weights and write only
the PEFT adapter directories.

The loader prefers `vlm_lora_adapter` and falls back to legacy
`qwen_lora_adapter` directories.

`trainer.is_resume=true` can discover `steps_N_vlm_lora_adapter/` adapter-only
checkpoints and restore the adapter with the correct step count. Exact recovery
of action-head weights and optimizer state still requires full checkpoint files;
use adapter-only saves for compact adapter export or continuation from a fixed
base checkpoint, not for bitwise-identical training resume.

## Local Smoke Checks

Config-only check:

```bash
python - <<'PY'
from omegaconf import OmegaConf
from starVLA.model.modules.vlm.lora_utils import get_lora_settings

cfg = OmegaConf.create({"framework": {"vlm": {"lora": {
    "enabled": True,
    "r": 2,
    "alpha": 4,
    "target_modules": "all-linear",
}}}})
settings = get_lora_settings(cfg)
assert settings.enabled is True
assert settings.target_modules == "all-linear"
assert settings.module_path == "qwen_vl_interface.model"
print("vlm_lora_config_ok")
PY
```

Tiny PEFT check without downloading Qwen:

```bash
python - <<'PY'
from tempfile import TemporaryDirectory

import torch
from omegaconf import OmegaConf
from transformers import GPT2Config, GPT2LMHeadModel

from starVLA.model.modules.vlm.lora_utils import (
    apply_lora_to_vlm_backbone,
    save_vlm_lora_adapter,
)

class TinyInterface(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GPT2LMHeadModel(
            GPT2Config(n_layer=1, n_head=1, n_embd=16, vocab_size=32, n_positions=8)
        )

class TinyVLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_vl_interface = TinyInterface()

cfg = OmegaConf.create({"framework": {"vlm": {"lora": {
    "enabled": True,
    "r": 2,
    "alpha": 4,
    "dropout": 0.0,
    "bias": "none",
    "target_modules": ["c_attn"],
    "adapter_name": "smoke",
}}}})

model, settings = apply_lora_to_vlm_backbone(TinyVLA(), cfg)
trainable = [name for name, param in model.named_parameters() if param.requires_grad]
assert trainable and all("lora_" in name for name in trainable), trainable[:10]
with TemporaryDirectory() as output_dir:
    save_vlm_lora_adapter(
        model,
        output_dir,
        adapter_name=settings.adapter_name,
        module_path=settings.module_path,
    )
print("vlm_lora_peft_ok", len(trainable))
PY
```

## Tuning Notes

- Start with `r=8` or `r=16`; increase to `r=32` when the task underfits and
  there is enough data.
- Use `alpha = 2 * r` as a first pass.
- Use `dropout=0.05` for small real-robot datasets; use `0.0` for smoke tests.
- For lower memory on Qwen, start with `q_proj,v_proj`.
- For stronger Qwen adaptation, use
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- For non-Qwen experiments, prefer `all-linear` first, then narrow the target
  modules after checking trainable parameter count and memory.
- If `trainer.freeze_modules=qwen_vl_interface`, the trainer strips that freeze
  pattern before rebuilding the optimizer so PEFT adapter parameters remain
  trainable. PEFT keeps the base VLM weights frozen.
