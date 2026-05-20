# Public WMH Best Checkpoints

Use the public runtime and public WMH code copy. Set `MEMBER` to your own initials so outputs go to your own public member directory.

## Current Best Checkpoints

| name | checkpoint | n300 5/5 |
|---|---|---:|
| `aug_hardv2` | `/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_aug_hardv2_8000_0519_171848/checkpoints/steps_8000_pytorch_model.pt` | 12.0% |
| `mirror_hardv2` | `/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_mirror_hardv2_8000_0519_171848/checkpoints/steps_8000_pytorch_model.pt` | 9.7% |
| `lora2000` | `/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_lora_explore_ft2000_0519_210816/checkpoints/steps_2000_pytorch_model.pt` | 9.7% |
| `base8k` | `/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_state8_connector_8h200_bs96_8k_0519_083200/checkpoints/steps_8000_pytorch_model.pt` | 3.7% |

Recommended ranking for follow-up evaluation: `aug_hardv2`, then `mirror_hardv2`, then `lora2000`; keep `base8k` as baseline.

## Run With Wrapper

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA

MEMBER=GTY \
CANDIDATES="aug_hardv2 mirror_hardv2 lora2000 base8k" \
TOTAL_SEQUENCES=300 \
GPU_IDS=0,1,2,3 \
WORKERS_PER_GPU=1 \
BASE_PORT=7400 \
bash examples/calvin_autoresearch/scripts/run_public_wmh_best_eval.sh
```

## Public-Only Direct Run

Use this when the runner only has `public/seven` access.

```bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}
PUBLIC=/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin
source ${PUBLIC}/shared/runtime/starvla_env.sh
export PATH=${STARVLA_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}
export STARVLA_ROOT=${PUBLIC}/members/WMH/code/starVLA_ft_aug
export PYTHONPATH=${STARVLA_ROOT}:${PYTHONPATH:-}
cd ${STARVLA_ROOT}

MEMBER=GTY
CANDIDATE=aug_hardv2
CKPT=${PUBLIC}/members/WMH/runs/abc_aug_hardv2_8000_0519_171848/checkpoints/steps_8000_pytorch_model.pt
OUT=${PUBLIC}/members/${MEMBER}/reports/eval_${CANDIDATE}_d_n300_$(date +%m%d_%H%M%S)
LOG=${PUBLIC}/members/${MEMBER}/logs/eval_${CANDIDATE}_d_n300_$(date +%m%d_%H%M%S).log
mkdir -p "$(dirname "${LOG}")" "${OUT}"

CKPT=${CKPT} \
EVAL_LOG_DIR=${OUT} \
TOTAL_SEQUENCES=300 \
GPU_IDS=0,1,2,3 \
WORKERS_PER_GPU=1 \
BASE_PORT=7400 \
CALVIN_SEND_STATE=1 \
bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh > "${LOG}" 2>&1

${STARVLA_PYTHON} examples/calvin_autoresearch/scripts/summarize_eval_metrics.py "${OUT}/metrics.json"
```

Outputs:

- reports: `/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/$MEMBER/reports/eval_wmh_best_d_n300_*`
- logs: `/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/$MEMBER/logs/eval_wmh_best_d_n300_*`

Use `WORKERS_PER_GPU=1` first. On an idle H200 node, `WORKERS_PER_GPU=2` is faster but has caused process kills when the node is busy.

## Quick Check

```bash
MEMBER=GTY CANDIDATES="aug_hardv2" TOTAL_SEQUENCES=8 GPU_IDS=0,1 WORKERS_PER_GPU=1 DRY_RUN=1 \
bash examples/calvin_autoresearch/scripts/run_public_wmh_best_eval.sh
```
