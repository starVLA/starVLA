"""Step-2 self test: drive the upstream RC mock_server with our policy.

This is a near drop-in for ``RoboChallengeInference/test.py`` but with
``DummyPolicy`` swapped for :class:`RoboChallengePolicy`.

Prerequisites:

1. Clone upstream once and start the mock server, e.g.::

       cd playground/Code
       git clone -b cvpr https://github.com/RoboChallenge/RoboChallengeInference.git
       cd RoboChallengeInference
       # edit mock_server/mock_settings.py to point at your raw episode dir
       cd mock_server && python3 mock_robot_server.py   # listens on 0.0.0.0:9098

2. In another shell run *this* script with ``--rc_repo`` pointing at the cloned
   path (we add it to ``sys.path`` so we can import upstream's
   ``robot.interface_client.InterfaceClient`` without vendoring it)::

       python examples/RoboChallenge_table30v2/eval_files/test_with_mock_server.py \\
           --checkpoint .../steps_100_pytorch_model.pt \\
           --robot_tag ur5 --prompt "shred the paper" \\
           --rc_repo $HOME/playground/Code/RoboChallengeInference \\
           --max_wait 60

Notes:
  * We use *two* action_types per cycle: ``leftjoint`` to fetch the joint
    state, ``leftpos`` to post ee-pose actions.  The mock server treats each
    request independently, so this is supported.
  * ``duration`` is the inter-action interval the robot will use to interpolate.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from examples.RoboChallenge_table30v2.eval_files.model2robochallenge_interface import (
    ROBOT_SPECS,
    RoboChallengePolicy,
)

DEFAULT_USER_ID = "test_user"
DEFAULT_JOB_ID = "test_job"
DEFAULT_ROBOT_ID = "test_robot"


def _add_rc_repo_to_path(rc_repo: str) -> None:
    rc_repo = os.path.abspath(os.path.expanduser(rc_repo))
    if not os.path.isdir(rc_repo):
        raise FileNotFoundError(f"--rc_repo not found: {rc_repo}")
    if rc_repo not in sys.path:
        sys.path.insert(0, rc_repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--robot_tag", default="ur5", choices=list(ROBOT_SPECS))
    parser.add_argument("--prompt", default="shred the paper")
    parser.add_argument(
        "--rc_repo",
        default=os.environ.get(
            "ROBOCHALLENGE_INFERENCE_PATH",
            os.path.expanduser("~/playground/Code/RoboChallengeInference"),
        ),
        help="Path to the cloned RoboChallengeInference repo (cvpr branch).",
    )
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--max_wait", type=int, default=600)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    logger = logging.getLogger("rc_mock_test")

    _add_rc_repo_to_path(args.rc_repo)
    from robot.interface_client import InterfaceClient  # type: ignore  # noqa: E402

    policy = RoboChallengePolicy(
        checkpoint_path=args.checkpoint,
        robot_tag=args.robot_tag,
        n_action_steps=args.n_action_steps,
        device=args.device,
    )
    spec = policy.spec

    client = InterfaceClient(DEFAULT_USER_ID, mock=True)
    client.update_job_info(DEFAULT_JOB_ID, DEFAULT_ROBOT_ID)

    image_size = [224, 224]

    # The mock server starts streaming frames as soon as start_motion() is called.
    client.start_motion()
    logger.info("Started mock motion. spec=%s", spec)

    start = time.time()
    n_iters = 0
    try:
        while True:
            state = client.get_state(image_size, spec.image_types, spec.state_action_type)
            if not state:
                time.sleep(0.5)
                continue
            if state.get("state") == "size_none":
                client.post_size() if hasattr(client, "post_size") else None
                time.sleep(0.5)
                continue
            if state.get("state") != "normal" or state.get("pending_actions", 0) != 0:
                time.sleep(0.05)
                continue

            t0 = time.time()
            actions = policy.run_policy(state, prompt=args.prompt)
            infer_dt = time.time() - t0
            logger.info(
                "iter %d  infer=%.1fms  net=%.1fms  pending=%d  first_action=%s",
                n_iters,
                infer_dt * 1e3,
                (t0 - state.get("timestamp", t0)) * 1e3,
                state.get("pending_actions", 0),
                [round(v, 4) for v in actions[0]],
            )
            client.post_actions(actions, args.duration, spec.post_action_type)
            n_iters += 1

            if time.time() - start > args.max_wait:
                logger.warning("max_wait reached (%ds), stopping after %d iters", args.max_wait, n_iters)
                break
    finally:
        client.end_motion()
        logger.info("Done. iters=%d  elapsed=%.1fs", n_iters, time.time() - start)


if __name__ == "__main__":
    main()
