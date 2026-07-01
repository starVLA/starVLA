#!/usr/bin/env bash

# Shared VLM LoRA CLI overrides for train_starvla.py launchers.
# Call append_vlm_lora_args after declaring EXTRA_ARGS=().

append_vlm_lora_args() {
  if ! declare -p EXTRA_ARGS >/dev/null 2>&1; then
    EXTRA_ARGS=()
  fi

  local lora_enabled_value="${VLM_LORA_ENABLED:-${LORA_ENABLED:-${USE_LORA:-}}}"
  if [[ -n "${lora_enabled_value}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.enabled "${lora_enabled_value}")
  fi

  if [[ -n "${VLM_LORA_R:-${LORA_R:-}}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.r "${VLM_LORA_R:-${LORA_R:-}}")
  fi
  if [[ -n "${VLM_LORA_ALPHA:-${LORA_ALPHA:-}}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.alpha "${VLM_LORA_ALPHA:-${LORA_ALPHA:-}}")
  fi
  if [[ -n "${VLM_LORA_DROPOUT:-${LORA_DROPOUT:-}}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.dropout "${VLM_LORA_DROPOUT:-${LORA_DROPOUT:-}}")
  fi
  if [[ -n "${VLM_LORA_BIAS:-${LORA_BIAS:-}}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.bias "${VLM_LORA_BIAS:-${LORA_BIAS:-}}")
  fi
  if [[ -n "${VLM_LORA_TARGET_MODULES:-${LORA_TARGET_MODULES:-}}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.target_modules "${VLM_LORA_TARGET_MODULES:-${LORA_TARGET_MODULES:-}}")
  fi
  if [[ -n "${VLM_LORA_ADAPTER_PATH:-${LORA_ADAPTER_PATH:-}}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.adapter_path "${VLM_LORA_ADAPTER_PATH:-${LORA_ADAPTER_PATH:-}}")
  fi
  if [[ -n "${VLM_LORA_ADAPTER_NAME:-${LORA_ADAPTER_NAME:-}}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.adapter_name "${VLM_LORA_ADAPTER_NAME:-${LORA_ADAPTER_NAME:-}}")
  fi
  if [[ -n "${VLM_LORA_IS_TRAINABLE:-${LORA_IS_TRAINABLE:-}}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.is_trainable "${VLM_LORA_IS_TRAINABLE:-${LORA_IS_TRAINABLE:-}}")
  fi
  if [[ -n "${VLM_LORA_SAVE_ADAPTER_ONLY:-${LORA_SAVE_ADAPTER_ONLY:-}}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.save_adapter_only "${VLM_LORA_SAVE_ADAPTER_ONLY:-${LORA_SAVE_ADAPTER_ONLY:-}}")
  fi
  if [[ -n "${VLM_LORA_MODULE_PATH:-}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.module_path "${VLM_LORA_MODULE_PATH}")
  fi
  if [[ -n "${VLM_LORA_ADAPTER_DIR_NAME:-}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.adapter_dir_name "${VLM_LORA_ADAPTER_DIR_NAME}")
  fi
  if [[ -n "${VLM_LORA_TASK_TYPE:-}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.task_type "${VLM_LORA_TASK_TYPE}")
  fi
  if [[ -n "${VLM_LORA_MODULES_TO_SAVE:-}" ]]; then
    EXTRA_ARGS+=(--framework.vlm.lora.modules_to_save "${VLM_LORA_MODULES_TO_SAVE}")
  fi

  local lora_lr="${VLM_LORA_LR:-${LORA_LR:-}}"
  if [[ -n "${lora_lr}" ]]; then
    local lora_lr_module="${VLM_LORA_LR_MODULE:-qwen_vl_interface}"
    EXTRA_ARGS+=("--trainer.learning_rate.${lora_lr_module}" "${lora_lr}")
  fi
}
