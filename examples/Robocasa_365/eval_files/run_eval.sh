#!/usr/bin/env bash
# RoboCasa365 walk-through evaluation — start the policy server in one terminal
# (starVLA env) and the simulation client in another (robocasa365 env).
set -euo pipefail

CKPT=${CKPT:-./playground/Checkpoints/robocasa365_qwenoft_OpenDrawer_100step/checkpoints/steps_100_pytorch_model.pt}
ENV_NAME=${ENV_NAME:-robocasa/OpenDrawer}
PORT=${PORT:-5678}
N_EPISODES=${N_EPISODES:-5}
N_ENVS=${N_ENVS:-1}
MAX_STEPS=${MAX_STEPS:-500}
N_ACT=${N_ACT:-8}

case "${1:-}" in
  server)
    # Run inside the `starVLA` env
    exec python deployment/model_server/server_policy.py \
      --ckpt_path "${CKPT}" \
      --port "${PORT}" \
      --use_bf16
    ;;
  client)
    # Run inside the `robocasa365` env
    exec python -m examples.Robocasa_365.eval_files.simulation_env \
      --args.pretrained-path "${CKPT}" \
      --args.env-name "${ENV_NAME}" \
      --args.port "${PORT}" \
      --args.n-episodes "${N_EPISODES}" \
      --args.n-envs "${N_ENVS}" \
      --args.max-episode-steps "${MAX_STEPS}" \
      --args.n-action-steps "${N_ACT}"
    ;;
  *)
    cat <<USAGE
Usage:
  # terminal 1 (conda env starVLA):
  bash examples/Robocasa_365/eval_files/run_eval.sh server
  # terminal 2 (conda env robocasa365):
  bash examples/Robocasa_365/eval_files/run_eval.sh client

Override defaults with env vars: CKPT, ENV_NAME, PORT, N_EPISODES, N_ENVS, MAX_STEPS, N_ACT.
USAGE
    ;;
esac
