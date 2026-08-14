#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

TOKEN_CACHE="playground/Checkpoints/var_stage1_metaworld_mt50_e32_aeinit_productvq_g16_s1_2_4_8/stage2_token_cache_full.pt"
LA_CKPT="/root/nas/feihong/starVLA/Checkpoints/starvla_metaworld_qwenpiv3_la_finetune/checkpoints/steps_60000_pytorch_model.pt"
RUN_DIR="/root/feihong/starVLA/Checkpoints/qwen25_la_var_productvq_g16_s1248_metaworld_mt50_20k_gbs512_save1k_fullcache"

if [[ ! -f "${TOKEN_CACHE}" ]]; then
  echo "Missing ${TOKEN_CACHE}; build it first with examples/simBenchmarks/MetaWorld/stage2_files/build_productvq_g16_s1248_metaworld_mt50_token_cache.sh" >&2
  exit 1
fi
if [[ ! -f "${LA_CKPT}" ]]; then
  echo "Missing ${LA_CKPT}; download MINT-SJTU/starvla_metaworld_qwenpiv3_la_finetune first." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
export PATH="$(pwd)/.venv/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_REQUIRE_LOGIN="${WANDB_REQUIRE_LOGIN:-1}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/root/feihong/.secrets/wandb_api_key}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_API_KEY_FILE}")"
fi

if [[ "${WANDB_MODE}" != "offline" && "${WANDB_MODE}" != "disabled" && "${WANDB_REQUIRE_LOGIN}" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${NETRC:-${HOME}/.netrc}" 2>/dev/null; then
    echo "W&B online mode is enabled, but no W&B login was found." >&2
    echo "On a fresh training machine, run one of:" >&2
    echo "  wandb login" >&2
    echo "  export WANDB_API_KEY=<your_wandb_api_key>" >&2
    echo "  install a key file at ${WANDB_API_KEY_FILE}" >&2
    echo "Set WANDB_MODE=offline or WANDB_REQUIRE_LOGIN=0 only if you intentionally do not want this guard." >&2
    exit 2
  fi
  "${PYTHON_BIN}" - <<'VERIFY_WANDB'
import sys
import wandb

try:
    if not wandb.login(verify=True, relogin=False):
        raise RuntimeError("wandb.login returned False")
except Exception as exc:
    print(f"W&B login verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    print("Run 'wandb login' or export WANDB_API_KEY before launching training.", file=sys.stderr)
    sys.exit(2)
VERIFY_WANDB
fi

mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/train.log"

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes "${NUM_PROCESSES:-8}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29554}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/MetaWorld/stage2_files/train_qwen25_la_var_productvq_g16_s1248_metaworld_mt50_60k_gbs512.yaml \
  2>&1 | tee -a "${LOG_FILE}"
