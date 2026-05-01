"""Step-1 self test: load checkpoint + dummy obs → run policy → print actions.

This does NOT touch the network. It exercises:
  * checkpoint loading via ``baseframework.from_pretrained``
  * state min_max normalization
  * forward pass through the model
  * action un-normalization

Useful for validating a freshly trained checkpoint before any HTTP plumbing.

Example::

    python examples/RoboChallenge_table30v2/eval_files/local_self_test.py \\
        --checkpoint playground/Checkpoints/robochallenge_table30v2_qwenoft_shred_paper_100step/checkpoints/steps_100_pytorch_model.pt \\
        --robot_tag ur5 --prompt "shred the paper"
"""

from __future__ import annotations

import argparse
import io
import logging
import time

import numpy as np
from PIL import Image

from examples.RoboChallenge_table30v2.eval_files.model2robochallenge_interface import (
    ROBOT_SPECS,
    RoboChallengePolicy,
)


def _make_dummy_state(robot_tag: str, image_hw=(224, 224)) -> dict:
    """Build a fake ``/state.pkl`` payload matching the RC server schema."""
    spec = ROBOT_SPECS[robot_tag]
    images = {}
    for cam in spec.image_types:
        rgb = (np.random.rand(image_hw[0], image_hw[1], 3) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        images[cam] = buf.getvalue()
    state_vec = np.zeros(spec.state_dim, dtype=np.float32).tolist()  # neutral joints + closed gripper
    return {
        "images": images,
        "action": state_vec,
        "pending_actions": 0,
        "timestamp": time.time(),
        "state": "normal",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to steps_*_pytorch_model.pt")
    parser.add_argument("--robot_tag", default="ur5", choices=list(ROBOT_SPECS))
    parser.add_argument("--prompt", default="shred the paper")
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--n_warmup", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("self_test")

    policy = RoboChallengePolicy(
        checkpoint_path=args.checkpoint,
        robot_tag=args.robot_tag,
        n_action_steps=args.n_action_steps,
        device=args.device,
        use_bf16=not args.no_bf16,
    )

    spec = ROBOT_SPECS[args.robot_tag]
    logger.info("Loaded. spec=%s", spec)

    state = _make_dummy_state(args.robot_tag)

    # warmup (jit/compile, autotune)
    for _ in range(args.n_warmup):
        policy.run_policy(state, prompt=args.prompt)

    times = []
    for i in range(args.n_runs):
        t0 = time.time()
        actions = policy.run_policy(state, prompt=args.prompt)
        dt = time.time() - t0
        times.append(dt)
        actions_arr = np.asarray(actions)
        logger.info(
            "run %d: %.1f ms, actions shape=%s  first action=%s",
            i, dt * 1e3, actions_arr.shape, np.round(actions_arr[0], 4).tolist(),
        )
    logger.info("avg latency over %d runs: %.1f ms", len(times), 1e3 * float(np.mean(times)))
    logger.info("OK ✅  policy returns %d steps × %d-d", len(actions), len(actions[0]))


if __name__ == "__main__":
    main()
