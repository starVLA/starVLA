# Baseline Snapshot: Qwen3-VL + GR00T Action Head on CALVIN ABC -> D

Snapshot date: 2026-05-19

This is the comparison point for the next compliant improvement runs. It uses only the allowed base VLM initialization and CALVIN ABC imitation data. It does not load upstream action-trained StarVLA/GR00T/OFT checkpoints.

## Training Run

Checkpoint:

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_pretrain_qwen3vl_gr00t_headonly_h200_60k_0518_163437/checkpoints/steps_60000_pytorch_model.pt
```

Run directory:

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_pretrain_qwen3vl_gr00t_headonly_h200_60k_0518_163437
```

Key config facts:

- Framework: `QwenGR00T`
- Base VLM: `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action`
- Action head: GR00T flow-matching DiT head
- Training data: `calvin_abc_train_v3.0`
- `include_state: false`
- `framework.action_model.state_dim: 7`
- `framework.action_model.action_dim: 7`
- `framework.action_model.action_horizon: 8`
- `trainer.freeze_modules: qwen_vl_interface`
- `trainer.max_train_steps: 60000`
- Launch: 3 GPU processes, per-device batch size 16, dataloader workers 8

## Formal D Eval

Formal eval directory:

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/reports/eval_abc_to_d_parallel_n1000_0519_053605
```

Debug GIF eval directory:

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/reports/eval_abc_to_d_debug_gif_n128_0519_071808
```

Summary command:

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh

python examples/calvin_autoresearch/scripts/summarize_eval_metrics.py \
  /inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/reports/eval_abc_to_d_parallel_n1000_0519_053605
```

## Metrics

- Sequences: 1000
- Average successful sequence length: 0.923
- Chain success:
  - 1/5: 50.7%
  - 2/5: 23.8%
  - 3/5: 11.0%
  - 4/5: 4.9%
  - 5/5: 1.9%
- Conditional success:
  - position 1: 507/1000, 50.7%
  - position 2: 238/507, 46.9%
  - position 3: 110/238, 46.2%
  - position 4: 49/110, 44.5%
  - position 5: 19/49, 38.8%
- Failure position:
  - first task: 493
  - second task: 269
  - third task: 128
  - fourth task: 61
  - fifth task: 30
  - complete chain: 19
- Near-miss:
  - any task: 180/981, 18.3%
  - related task: 73/981, 7.4%

## Worst Atomic Tasks

| Task | Success | Failure step mean | Related near-miss |
| --- | ---: | ---: | ---: |
| `turn_off_lightbulb` | 0/72, 0.0% | 360.0 | 1.4% |
| `close_drawer` | 6/83, 7.2% | 360.0 | 1.3% |
| `push_red_block_right` | 5/35, 14.3% | 360.0 | 3.3% |
| `turn_off_led` | 13/85, 15.3% | 360.0 | 0.0% |
| `move_slider_left` | 19/116, 16.4% | 360.0 | 0.0% |
| `turn_on_lightbulb` | 20/85, 23.5% | 360.0 | 1.5% |
| `push_blue_block_right` | 11/35, 31.4% | 360.0 | 4.2% |
| `open_drawer` | 64/177, 36.2% | 360.0 | 2.7% |
| `stack_block` | 22/58, 37.9% | 360.0 | 36.1% |
| `push_pink_block_right` | 16/39, 41.0% | 360.0 | 8.7% |

## Action Statistics

Raw model action:

| Dim | Mean abs | Max abs | Saturation |
| --- | ---: | ---: | ---: |
| x | 0.13523 | 0.98886 | 0.0% |
| y | 0.13011 | 0.89092 | 0.0% |
| z | 0.13123 | 0.93814 | 0.0% |
| roll | 0.073664 | 1.2256 | 0.0% |
| pitch | 0.099051 | 1.0421 | 0.0% |
| yaw | 0.14113 | 1.1247 | 0.3% |
| gripper | 1.0032 | 1.1129 | 67.3% |

Raw action jitter:

- Mean L2: 0.1395
- Max L2: 2.6749
- Gripper switch rate: 1.5%

After gripper binarization:

- Gripper saturation: 100.0%
- Mean jitter L2: 0.13776
- Max jitter L2: 2.6792
- Gripper switch rate: 1.5%

## Interpretation

The baseline has a usable first-step success rate, but long-chain reliability is weak. Hard failures concentrate on light/LED off, drawer close, slider-left, and right-push tasks. The gripper channel is frequently saturated and switches rarely, so gripper timing should be treated as a likely amplifier of manipulation failures. The next experiment should keep this snapshot fixed and compare against it with the same D eval scripts and metrics.
