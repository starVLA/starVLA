# Discrete Diffusion Action Head (MaskGIT-style)

A new framework for starVLA: `QwenDiscreteDiffusion`. The VLM backbone is the
same Qwen2.5-VL stack used by `QwenPI`, but the action head replaces continuous
flow-matching with **MaskGIT-style discrete diffusion** over a uniform action
binning. The head also exposes a **real-time chunking (RTC)** decode that
conditions on the previous chunk's un-executed actions as a known prefix,
enabling smooth closed-loop control under inference latency.

## What's added

| Component | Path |
|---|---|
| Framework | `starVLA/model/framework/VLM4A/QwenDiscreteDiffusion.py` |
| Action head | `starVLA/model/modules/action_model/LayerwiseDiscreteDiffusion_ActionHeader.py` |
| Binning + MaskGIT helpers | `starVLA/model/modules/action_model/discrete_diffusion/` |
| Sim config (RoboTwin, 14D) | `starVLA/config/training/starvla_train_discrete_diffusion.yaml` |
| Real config (FastUMI, 10D) | `starVLA/config/training/starvla_train_discrete_diffusion_real.yaml` |

### Architecture

`LayerwiseDiscreteDiffusionActionHead` mirrors the QwenPI flow-matching head:
the same DiT block stack, the same layer-wise cross-attention against the
Qwen-VL hidden states (`vl_embs_list`), the same future-token / state /
position-embedding inputs. Two things change:

1. **Tokens, not noise.** Each continuous action `a ∈ [low, high]^D` is encoded
   into discrete bins (`continuous_to_bins`, default `num_bins=256`). The head
   predicts logits over bins per (timestep, action-dim) position, and decodes
   back to bin centers (`ActionBinning.decode`). Two representations are
   supported:
   - `bin` — output `num_bins` logits per position, cross-entropy loss.
   - `bit` — output 8 sigmoid-bit logits per position, BCE loss; 8 bits
     simulate 256 bins. Useful when the vocab is large.
2. **MaskGIT decode replaces Euler integration.** Training masks a
   schedule-controlled fraction of positions and asks the model to fill them
   in (`apply_mask` + cross-entropy). Inference iteratively unmasks the
   highest-confidence positions over `num_inference_steps` rounds
   (`predict_action`). An auxiliary L1 loss on the decoded continuous
   prediction (`l1_loss_weight=0.1`) keeps bin centers calibrated.

### Real-time chunking decode

`predict_action_realtime(vl_embs_list, state, prev_action_chunk, inference_delay,
execution_horizon, …)` is the RTC entry point. The previous chunk's tail is
encoded into bins and pinned as a known prefix; only the trailing
`execution_horizon` positions are masked and resampled. The number of decode
steps scales with the masked fraction (or is held fixed via `fixed_steps=True`),
and `early_stop=True` exits as soon as every non-prefix slot is unmasked.

| Argument | Effect |
|---|---|
| `prev_action_chunk` | `(B, T, action_dim)` continuous actions from the previous prediction. |
| `inference_delay` | Fallback for `execution_horizon` if the latter is unset; sets prefix length under `hard_mask=True`. |
| `execution_horizon` | Number of trailing positions to regenerate. Defaults to `inference_delay`. |
| `hard_mask` | If `True`, prefix length is `inference_delay` (legacy). If `False` (default), prefix length is `action_horizon - execution_horizon`. |
| `fixed_steps` | Always run `num_inference_steps`, otherwise scale by mask fraction. |
| `choice_temperature` / `decode_temperature` | Top-k Gumbel temperature for which positions stay masked / per-step softmax temperature. |
| `early_stop` | Exit early once every non-prefix slot is unmasked. |

The flow-matching counterpart `LayerwiseFlowmatchingActionHead.predict_action_realtime`
exposes the same `prev_action_chunk` interface but uses ΠGDM (pseudo-inverse
guided diffusion) by default; pass `mode="simulated_delay"` for the naive
replace-and-denoise variant. They are interchangeable from the deployment
client's perspective (same `examples`, `prev_action_chunk_normalized`,
`inference_delay` API on the framework wrapper).

## Configs at a glance

Discrete-diffusion-specific keys under `framework.action_model` (see the YAMLs
for the full set):

```yaml
representation: bin           # "bin" or "bit"
num_bins: 256                 # 256 bins → 8-bit "bit" mode also works
action_low: -1.0
action_high: 1.0

num_inference_steps: 8        # MaskGIT decode rounds
train_mask_schedule: cosine   # "cosine" or "linear"
decode_schedule: cosine
no_mask_token_prob: 0.0       # leak some unmasked tokens during training
l1_loss_weight: 0.1           # auxiliary L1 on decoded continuous values
use_simple_max: false         # if true: single forward + argmax (no iteration)
```

`action_dim`, `state_dim`, `future_action_window_size`, and the DiT block
shape (`num_layers`, `attention_head_dim`, `num_attention_heads`) are inherited
from the same conventions as QwenPI; the head overrides DiT layer count and
hidden dim from `framework.qwenvl` at construction time so the action head
matches the VLM size automatically.

## Training

The training entrypoint is unchanged — `starVLA/training/train_starvla.py`.
Pick the discrete-diffusion config and set `framework.name=QwenDiscreteDiffusion`.

### Sim / RoboTwin (14D, dual-arm)

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_discrete_diffusion.yaml \
  --framework.name QwenDiscreteDiffusion \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --datasets.vla_data.data_root_dir playground/Datasets/RoboTwin \
  --datasets.vla_data.data_mix robotwin \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 5000 \
  --run_id qwendd_robotwin
```

### Real / FastUMI pick-and-place (10D, single-arm)

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_discrete_diffusion_real.yaml \
  --framework.name QwenDiscreteDiffusion \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix fastumi_pickandplace_real_0307 \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --run_id qwendd_fastumi_real
```

Either command can be wrapped into a slurm/launch script following the
template in `0130-train-libero.sh` (just swap `Framework_name` and
`config_yaml`). Override any YAML key from the CLI with the
`--<group>.<subgroup>.<key> <value>` syntax used above.

## Inference

In-process Python use:

```python
from starVLA.model.framework import build_framework
from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace

model_config, norm_stats = read_mode_config("path/to/checkpoint.pt")
config = dict_to_namespace(model_config)
config.trainer.pretrained_checkpoint = None
model = build_framework(cfg=config).cuda().eval()
model.load_state_dict(torch.load("path/to/checkpoint.pt", map_location="cpu"), strict=True)
model.norm_stats = norm_stats

# Single shot
out = model.predict_action(examples=[{"image": [...], "lang": "...", "state": ...}])
normalized_actions = out["normalized_actions"]   # (B, action_horizon, action_dim)

# RTC-aware (continuous prefix from the previous chunk)
out = model.predict_action_realtime(
    examples=[...],
    prev_action_chunk_normalized=prev_chunk,     # (B, action_horizon, action_dim)
    inference_delay=5,
    execution_horizon=5,
    hard_mask=False,
    fixed_steps=False,
    early_stop=False,
    decode_temperature=0.1,
    choice_temperature=0.1,
)
```

## Real-world deployment

Deployment scripts (server/client split with `inference_server.py`,
`closedloop_rtc_*.py`, `closedloop_sync.py`, `openloop_eval.py`) live in
the author's fork and are kept out of this PR to keep the surface minimal.
The framework wrapper's `predict_action_realtime(...)` is the integration
point — anything that calls it the way `predict_action(...)` is called can
drive a real-time RTC loop. The flow-matching counterpart
`LayerwiseFlowmatchingActionHead.predict_action_realtime` exposes the same
`prev_action_chunk` / `inference_delay` / `execution_horizon` interface, so
the two heads are interchangeable from the deployment client's perspective.

## File map

```
examples/modelExtensions/DiscreteDiffusion/
└── README.md                                # this file
starVLA/config/training/
├── starvla_train_discrete_diffusion.yaml      # sim (14D)
└── starvla_train_discrete_diffusion_real.yaml # real (10D)
starVLA/model/framework/VLM4A/
└── QwenDiscreteDiffusion.py                 # framework wrapper (forward + predict_action[_realtime])
starVLA/model/modules/action_model/
├── LayerwiseDiscreteDiffusion_ActionHeader.py
└── discrete_diffusion/
    ├── action_binning.py                    # continuous ↔ bin index, logits → indices
    ├── mask_git_schedule.py                 # train/decode schedules, top-k mask helpers
    └── __init__.py
```
