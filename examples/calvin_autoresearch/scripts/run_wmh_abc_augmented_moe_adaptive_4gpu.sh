#!/usr/bin/env bash
set -euo pipefail

# WMH one-click launcher for GTY's ABC augmented MoE adaptive training.
# It writes an explicit config file first, so launch-time overrides cannot be
# silently ignored by the training entrypoint.

BASE="${BASE:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
RUNTIME="${RUNTIME:-${BASE}/shared/runtime}"
CODE="${CODE:-${RUNTIME}/code/starVLA}"
GTY="${GTY:-${BASE}/members/GTY}"
MEMBER="${MEMBER:-WMH}"
MEMBER_ROOT="${MEMBER_ROOT:-${BASE}/members/${MEMBER}}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${MEMBER_ROOT}/runs}"

MODEL_DIR="${MODEL_DIR:-${BASE}/shared/models/base}"
BASE_VLM="${BASE_VLM:-${MODEL_DIR}/Qwen3-VL-4B-Instruct-Action}"
DATA_ROOT="${DATA_ROOT:-${BASE}/shared/datasets/calvin_lerobot}"

SRC_CONFIG="${SRC_CONFIG:-${GTY}/train_files/starvla_calvin_abc_augmented_moe_adaptive.yaml}"
ENTRY="${ENTRY:-${GTY}/train_files/run_train_moe_adaptive_entry.py}"
DS_CONFIG="${DS_CONFIG:-${CODE}/starVLA/config/deepseeds/deepspeed_zero2.yaml}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
NUM_PROCESSES="${NUM_PROCESSES:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-60000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
PORT="${PORT:-auto}"
PORT_START="${PORT_START:-29620}"
PORT_END="${PORT_END:-29720}"
WANDB_MODE="${WANDB_MODE:-disabled}"
TAIL_LOG="${TAIL_LOG:-1}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

TS="$(date +%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-abc_augmented_moe_adaptive_${MEMBER}_${TS}}"
RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/nohup.out}"
CONFIG_FILE="${CONFIG_FILE:-${RUN_DIR}/config.launch.yaml}"

log() {
  printf '[wmh-adaptive-train] %s\n' "$*"
}

require_path() {
  local path="$1"
  local desc="$2"
  if [[ ! -e "${path}" ]]; then
    printf '[wmh-adaptive-train] missing %s: %s\n' "${desc}" "${path}" >&2
    exit 2
  fi
}

choose_free_port() {
  local port
  for port in $(seq "${PORT_START}" "${PORT_END}"); do
    if ! (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1; then
      printf '%s\n' "${port}"
      return 0
    fi
  done
  printf '[wmh-adaptive-train] no free port found in %s-%s\n' "${PORT_START}" "${PORT_END}" >&2
  exit 2
}

if [[ -z "${NUM_PROCESSES}" ]]; then
  NUM_PROCESSES="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
fi

require_path "${RUNTIME}/starvla_env.sh" "runtime env"
require_path "${CODE}" "runtime StarVLA code"
require_path "${GTY}" "GTY member directory"
require_path "${SRC_CONFIG}" "source training config"
require_path "${ENTRY}" "training entrypoint"
require_path "${DS_CONFIG}" "DeepSpeed config"
require_path "${BASE_VLM}" "base VLM"
require_path "${DATA_ROOT}" "CALVIN data root"

mkdir -p "${RUN_DIR}"
: > "${LOG_FILE}"

source "${RUNTIME}/starvla_env.sh"

if [[ "${PORT}" == "0" || "${PORT}" == "auto" ]]; then
  PORT="$(choose_free_port)"
fi

log "writing launch config: ${CONFIG_FILE}"
"${STARVLA_PYTHON}" - "${CONFIG_FILE}" \
  "${RUN_ID}" \
  "${RUN_ROOT_DIR}" \
  "${BASE_VLM}" \
  "${DATA_ROOT}" \
  "${SRC_CONFIG}" \
  "${BATCH_SIZE}" \
  "${MAX_TRAIN_STEPS}" \
  "${SAVE_INTERVAL}" \
  "${LOGGING_FREQUENCY}" \
  "${GRADIENT_ACCUMULATION_STEPS}" <<'PY'
import sys
from omegaconf import OmegaConf

(
    out,
    run_id,
    run_root,
    base_vlm,
    data_root,
    src_config,
    batch_size,
    max_steps,
    save_interval,
    logging_frequency,
    grad_accum,
) = sys.argv[1:]

cfg = OmegaConf.load(src_config)
cfg.run_id = run_id
cfg.run_root_dir = run_root
cfg.framework.qwenvl.base_vlm = base_vlm
cfg.datasets.vla_data.data_root_dir = data_root
cfg.datasets.vla_data.data_mix = "calvin_abc_augmented"
cfg.datasets.vla_data.per_device_batch_size = int(batch_size)
cfg.trainer.max_train_steps = int(max_steps)
cfg.trainer.save_interval = int(save_interval)
cfg.trainer.logging_frequency = int(logging_frequency)
cfg.trainer.gradient_accumulation_steps = int(grad_accum)
cfg.trainer.freeze_modules = "qwen_vl_interface"

OmegaConf.save(cfg, out)
print(f"saved {out}")
PY

export PYTHONPATH="${GTY}/train_files:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export WANDB_MODE
export TRANSFORMERS_VERBOSITY

cd "${CODE}"

cmd=(
  "${STARVLA_PYTHON}" -m accelerate.commands.accelerate_cli launch
  --config_file "${DS_CONFIG}"
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${PORT}"
  "${ENTRY}"
  --config_yaml "${CONFIG_FILE}"
)

printf '%q ' "${cmd[@]}" > "${RUN_DIR}/launch_command.txt"
printf '\n' >> "${RUN_DIR}/launch_command.txt"

WRAPPER="${RUN_DIR}/train_wrapper.sh"
{
  printf '#!/usr/bin/env bash\n'
  printf 'set -o pipefail\n'
  printf 'echo "[wmh-adaptive-train] train command started at $(date)"\n'
  printf 'echo "[wmh-adaptive-train] host=$(hostname) pid=$$"\n'
  printf 'cd %q\n' "${CODE}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  printf 'status=$?\n'
  printf 'echo "[wmh-adaptive-train] train command exited at $(date) exit_code=${status}"\n'
  printf 'exit "${status}"\n'
} > "${WRAPPER}"
chmod +x "${WRAPPER}"

log "RUN_ID=${RUN_ID}"
log "RUN_DIR=${RUN_DIR}"
log "LOG_FILE=${LOG_FILE}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "NUM_PROCESSES=${NUM_PROCESSES}"
log "BATCH_SIZE(per device)=${BATCH_SIZE}"
log "PORT=${PORT}"
log "TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY}"

# Put the training launcher in a new session so Ctrl-C on the foreground
# tail/shell cannot send signals to the distributed training process group.
nohup setsid bash "${WRAPPER}" </dev/null > "${LOG_FILE}" 2>&1 &
pid="$!"
echo "${pid}" > "${RUN_DIR}/launch.pid"
log "started pid=${pid}"

sleep 2
if ! kill -0 "${pid}" 2>/dev/null; then
  log "training process exited immediately; recent log:"
  tail -n 120 "${LOG_FILE}" || true
  exit 1
fi

log "to monitor later: tail -f ${LOG_FILE}"
log "Ctrl-C here only stops log following; training keeps running."

if [[ "${TAIL_LOG}" == "1" ]]; then
  tail -n 80 -f "${LOG_FILE}"
fi
