#!/usr/bin/env bash
# Serve one released 15-task checkpoint. Run from the StarVLA repository root in the model environment.
#   CKPT_ROOT=<hf download dir>  CASE=qwenoft-single|qwenoft-dual|wan-single|wan-dual|cosmos-single|cosmos-dual
#   QWEN_BASE=<local Qwen3-VL-4B-Instruct>  (QwenOFT only)   PORT=10101   GPU_ID=<gpu, default CUDA_VISIBLE_DEVICES or 0>
set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:?set CKPT_ROOT to the checkpoint download directory}"
CASE="${CASE:-qwenoft-single}"
PORT="${PORT:-10101}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export STARVLA_ROOT="${STARVLA_ROOT:-$(pwd)}"
export PYTHONPATH="${STARVLA_ROOT}:${STARVLA_ROOT}/examples/simBenchmarks/EvoDexBench/eval_files:${PYTHONPATH:-}"

model="${CASE%-*}"
embodiment="${CASE#*-}"
ckpt="${CKPT_ROOT}/${CASE}/final_model/pytorch_model.pt"
case "${model}" in
  qwenoft)
    # The client checks the canonical mixture name; the 15-task mixture shares its robot type and statistics.
    exec python -m starvla_policy.server --ckpt-path "${ckpt}" --port "${PORT}" \
      --config-override "framework.qwenvl.base_vlm=${QWEN_BASE:?set QWEN_BASE to the Qwen3-VL-4B-Instruct directory}" \
      --config-override "datasets.vla_data.data_mix=evodex_${embodiment}_h50_q99" ;;
  wan)
    exec python -m evodex_wm4a.serve --framework EvoWanOFT --ckpt-path "${ckpt}" --port "${PORT}" --seed 42 \
      --config-override "framework.evodex_wan.base_config_dir=${CKPT_ROOT}/base_configs/wan" \
      --config-override "datasets.vla_data.data_mix=evodex_${embodiment}_h8_q99" ;;
  cosmos)
    base="${CKPT_ROOT}/base_configs/Cosmos-Predict2-2B-Video2World"
    exec python -m evodex_wm4a.serve --framework EvoCosmosGR00T --ckpt-path "${ckpt}" --port "${PORT}" --seed 42 \
      --config-override "framework.evodex_cosmos.base_config_dir=${base}" \
      --config-override "framework.evodex_cosmos.base_config_alias=${base}" \
      --config-override "datasets.vla_data.data_mix=evodex_${embodiment}_h8_q99" ;;
  *) echo "Unknown CASE ${CASE}" >&2; exit 2 ;;
esac
