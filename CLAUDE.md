# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

StarVLA is a "Lego-like" research codebase for Vision-Language-Action (VLA) models. The core design principle is **high cohesion / low coupling**: each axis — backbone VLM, action head, dataloader, trainer, benchmark — is independently swappable. A new framework variant usually reduces to swapping the action head while reusing the rest.

Active development happens on the `starVLA_dev` branch (may be unstable); `starVLA` is the stable release branch. Branches are rebases of each other, not long-lived forks.

## Environment & common commands

Python 3.10. Install: `pip install -r requirements.txt && pip install flash-attn --no-build-isolation && pip install -e .`

```bash
# Lint (note: full-repo check is expected to fail due to historical backlog — only check files you touched)
make check        # black --check . && ruff check --show-source .
make autoformat   # black . && ruff check --fix-only --show-fixes .
make clean        # remove pyc/__pycache__

# Smoke-test a single framework module (each framework file runs standalone on fake data)
python starVLA/model/framework/VLM4A/QwenGR00T.py --config_yaml examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml

# Smoke-test a dataloader
python starVLA/dataloader/lerobot_datasets.py --config_yaml examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml

# Run a single test
python -m pytest tests/test_config_overrides.py -q
python -m pytest tests/test_config_overrides.py::TestClassName::test_method -q
```

Formatting: black + ruff, line-length **121**, target py310. Ruff lints `A,B,E,F,I,RUF,W` (ignores `F722`); `__init__.py` ignores `E402,F401`.

### Launching training

All training goes through `accelerate launch` + a DeepSpeed config, with the YAML config as the single source of truth and CLI dotlist overrides on top. Each benchmark ships a `run_*.sh` wrapper; the canonical one is `examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh`:

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenGR00T            # any key=value overrides YAML (dotlist)
```

CLI args are merged via OmegaConf dotlist (`normalize_dotlist_args`), then `apply_config_compat` normalizes legacy keys to the current schema idempotently before `main(cfg)`.

### Serving / evaluating

Evaluation uses a **client-server split**: a policy server runs in the `starVLA` env; the simulator runs in a separate env and talks to it. Two server protocols:

```bash
python deployment/model_server/server_policy.py     --ckpt_path CKPT --port 10093 --use_bf16   # websockets
python deployment/model_server/server_policy_gr00t_zmq.py --ckpt_path CKPT --port 5555 --use_bf16   # GR00T ZMQ/msgpack
```

Per-benchmark eval clients live under `examples/simBenchmarks/<BENCH>/eval_files/` (e.g. `eval_libero.sh`).

## Architecture

### The framework registry is the spine of the codebase

`starVLA/model/framework/base_framework.py:build_framework(cfg)` is the **single entry point** for constructing a model. It looks up `cfg.framework.name` in `FRAMEWORK_REGISTRY` (defined in `starVLA/model/tools.py`). Every concrete framework file registers itself with a decorator:

```python
@FRAMEWORK_REGISTRY.register("QwenGR00T")
class QwenGR00T(...): ...
```

`base_framework._auto_import_framework_modules()` walks `starVLA/model/framework/{VLM4A,VM4A,WM4A,...}` and imports every module so their `@register` decorators fire. **To add a new framework: drop a file in the right package and register a name — nothing else needs to change.** The framework file is also the single external API surface (`forward()` / `predict_action()` operate on raw, model-agnostic inputs).

Three framework families, all sharing the same data/training interface:
- **VLM4A** (VLM for Action) — VLM backbones (Qwen/InternVL/Florence-2/Gemma/MiniCPM) + action heads (OFT/FAST/PI/GR00T dual-system). The primary family. e.g. `QwenOFT`, `QwenFast`, `QwenPI_v3`, `QwenGR00T`.
- **VM4A** (VisuoMotor for Action) — non-VLM visuomotor policies (ACT, Diffusion Policy) under `starVLA/model/framework/VM4A/_dp_vendor`.
- **WM4A** (World Model for Action) — video-generation DiT backbones (Cosmos-Predict2, Wan2.2) repurposed for action prediction. See `docs/WM4A.md`.

### Raw-dict data contract

Dataloaders (`starVLA/dataloader/`, dispatched by `cfg.datasets.<name>.dataset_py` → `build_dataloader`) return **raw, model-agnostic dicts only** — no tokenization or image encoding happens in the loader. A sample is roughly `{image: list[PIL]|ndarray, lang: str, action: ndarray[T,action_dim], state: ndarray[...]|None}`. Both `framework.forward()` and `framework.predict_action()` consume these raw inputs directly, keeping train/test boundaries identical. Datasets are LeRobot-format; per-dataset `meta/modality.json` declares the modality schema.

### Trainer layering

`starVLA/training/train_*.py` — one file per recipe, no big if/else chains:
- `train_starvla.py` — VLA SFT
- `train_starvla_cotrain.py` — VLA + VLM multimodal multi-objective co-training (two dataloaders: `vla_data` + `vlm_data`)
- `train_starvlm.py` — VLM-only training
- `train_starvln.py` — VLN (navigation)

Each: `OmegaConf.load(config_yaml)` → merge CLI dotlist → `apply_config_compat` → `build_dataloader` + `build_framework` → `TrainerUtils`-driven loop. Built on native PyTorch + Accelerate + DeepSpeed; the loop is deliberately explicit. Runtime state lives in dicts (config, processing info). NPU (Ascend) is supported via a silent `torch_npu` import that no-ops on GPU.

Shared trainer machinery lives in `starVLA/training/trainer_utils/`:
- `trainer_tools.py` — `TrainerUtils` (freezing, checkpoint load/save, param grouping), `build_param_lr_groups`, `setup_optimizer_and_scheduler`, `normalize_dotlist_args`.
- `config_tracker.py` — `AccessTrackedConfig`/`wrap_config` track which config keys are actually read (used to detect dead/unused config).
- `overwatch.py` — `initialize_overwatch` logging init.
- `monkey_patch.py`.

### Per-module freeze / learning-rate control

These are the two most-asked knobs (see README FAQ):
- **Freeze** via `--trainer.freeze_modules "qwen_vl_interface.model.model.visual,dino_encoder"` — a comma-separated name list / regex; impl in `TrainerUtils.freeze_backbones`.
- **Per-module LR** via `trainer.learning_rate: {base: 1e-5, qwen_vl_interface: 1e-5, action_model: 1e-4}` — name→value dict; impl in `build_param_lr_groups`. Tip: `print(model)` to find the module names.
- **Resume** via `trainer.pretrained_checkpoint: path.pt` + `reload_modules: "action_model"` (empty = full load). Optimizer state is intentionally **not** saved.

### Checkpoint & deployment format

Training saves `steps_XXXXX_pytorch_model.pt` under `results/Checkpoints/{run_id}/checkpoints/`. The checkpoint embeds the framework config + normalization stats, so the policy server can rebuild the model from a single `.pt` path. `deployment/model_server/policy_wrapper.py` + `base_framework` handle load; the ZMQ server additionally flattens/splits named state/action groups per the checkpoint's `DataConfig` state_keys/action_keys, so adding an embodiment needs only a registered DataConfig — no protocol code.

## Layout map

```
starVLA/
  config/{deepseeds,training}/        # DeepSpeed YAMLs + example training YAMLs
  dataloader/                          # build_dataloader + lerobot/vlm/gr00t/llavajson loaders
  model/
    tools.py                           # FRAMEWORK_REGISTRY, auto_get_trainable_modules, has_flash_attn
    framework/
      base_framework.py                # build_framework() — the single model entry point
      share_tools.py                   # dict_to_namespace, read_mode_config, apply_config_compat
      VLM4A/ VM4A/ WM4A/               # one file per framework variant, self-registering
    modules/{vlm,action_model,world_model,projector,dino_model}/   # reusable backbone/head components
  training/
    train_starvla.py / _cotrain.py / train_starvlm.py / train_starvln.py
    trainer_utils/{trainer_tools,config_tracker,overwatch,monkey_patch}.py
deployment/model_server/              # server_policy.py (ws), server_policy_gr00t_zmq.py (ZMQ)
examples/
  simBenchmarks/<BENCH>/train_files/   # config YAML + run_*.sh + data_registry/ per benchmark
  simBenchmarks/<BENCH>/eval_files/    # eval client + benchmark env interface
  modelExtensions/  realRobots/        # extra frameworks / real-robot deployment cases
tests/                                 # unittest; cover config overrides, ZMQ server, dist safety
```

## Conventions worth knowing

- **`**/bar/` is git-ignored** — put personal/experimental scripts under any `bar/` dir (e.g. `examples/simBenchmarks/LIBERO/train_files/bar/my_train.sh`) to keep them out of the repo without editing `.gitignore`.
- **`playground/`** holds symlinks to datasets (`playground/Datasets/`) and pretrained models (`playground/Pretrained_models/`); it's git-ignored and excluded from packaging. Base VLMs are downloaded here (e.g. `Qwen3-VL-4B-Instruct`).
- A framework's external API is `starVLA/model/framework/<family>/<name>.py`; it should mirror the architecture diagram from the paper and be runnable standalone for smoke tests.
- `--framework.qwenvl.base_vlm` selects the backbone VLM (kept as `qwenvl` for checkpoint compat with releases, even for non-Qwen VLMs like Florence-2).
- Adding a new module to the action model: `--framework.action_model.new_module <module>` just adds it to the global config — wiring it into the forward pass is the framework's own responsibility.

See `docs/starVLA_guideline.md` (end-to-end LIBERO walkthrough), `docs/faq.md`, `docs/model_zoo.md`, `docs/CONTRIBUTING.md`, `docs/PR_readme.md` (pre-submit checklist).
