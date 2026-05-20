# Asset Status

Current non-violating assets for the CALVIN smoke baseline:

| Asset | Local path | Source | Status |
| --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct-Action base VLM | `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action` | `StarVLA/Qwen3-VL-4B-Instruct-Action` | downloaded, symlinked from `/tmp/starvla_hf_cache` |
| CALVIN ABC LeRobot train data | `playground/Datasets/calvin_lerobot/calvin_abc_train_v3.0` | shared `/inspire/.../calvin_task_ABC_D` | linked |
| CALVIN original D data | `playground/Datasets/calvin_original/task_D_D` | shared `/inspire/.../task_D_D` | linked |
| CALVIN D-D LeRobot data | `playground/Datasets/calvin_lerobot/calvin_task_D_D_v3.0` | `fywang/calvin-task-D-D-lerobot` | optional, not required for current smoke |

The Qwen asset is treated as an allowed base VLM. No LIBERO, Robotwin,
Robocasa, Behavior, SimplerEnv, or upstream CALVIN action-trained policy
checkpoint is preloaded by these scripts.

The model path currently points to `/tmp/starvla_hf_cache` because direct HF
`--local-dir` writes into the GPFS project path failed with a read-only
filesystem error for large model directories. If the image export does not
include `/tmp`, re-run:

```bash
source /inspire/qb-ilm2/project/26summer-camp-10/26220172/starvla_env.sh
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  HF_HOME=/tmp/starvla_hf_home HF_HUB_CACHE=/tmp/starvla_hf_cache \
  hf download StarVLA/Qwen3-VL-4B-Instruct-Action \
  --repo-type model --cache-dir /tmp/starvla_hf_cache --max-workers 8
ln -sfn /tmp/starvla_hf_cache/models--StarVLA--Qwen3-VL-4B-Instruct-Action/snapshots/ada21835b2a13ec0456d02f0b630e7ab91a43ef3 \
  /inspire/qb-ilm2/project/26summer-camp-10/26220172/models/Qwen3-VL-4B-Instruct-Action
```
