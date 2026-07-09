"""MetaWorld MT50 evaluation script for starVLA.

Evaluates a starVLA policy on all 50 MetaWorld tasks, reporting per-task and
per-difficulty-bucket success rates.  Follows the same tyro CLI pattern as
``eval_libero.py``.

Usage::

    python examples/simBenchmarks/MetaWorld/eval_files/eval_metaworld.py \
        --args.host 127.0.0.1 --args.port 10095 \
        --args.video-out-path experiments/metaworld/logs
"""

import dataclasses
import json
import logging
import os
import pathlib
import shutil
import tempfile

import gymnasium as gym
import imageio
import metaworld  # noqa: F401  ensures Meta-World envs are registered
import numpy as np
import tqdm
import tyro

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MUJOCO_GL", "egl")
gym.logger.min_level = gym.logger.ERROR

from examples.simBenchmarks.MetaWorld.eval_files.model2metaworld_interface import (
    ModelClient,
    preprocess_metaworld_image,
)


# ---------------------------------------------------------------------------
# MT50 task definitions (index, slug, prompt, difficulty bucket)
# ---------------------------------------------------------------------------
MT50_TASKS = [
    (0, "nut-assembly-v3", "Pick up a nut and place it onto a peg", "hard"),
    (1, "basketball-v3", "Dunk the basketball into the basket", "medium"),
    (2, "bin-picking-v3", "Grasp the puck from one bin and place it into another bin", "medium"),
    (3, "box-close-v3", "Grasp the cover and close the box with it", "medium"),
    (4, "button-press-topdown-v3", "Press a button from the top", "easy"),
    (5, "button-press-topdown-wall-v3", "Bypass a wall and press a button from the top", "easy"),
    (6, "button-press-v3", "Press a button", "easy"),
    (7, "button-press-wall-v3", "Bypass a wall and press a button", "easy"),
    (8, "coffee-button-v3", "Push a button on the coffee machine", "easy"),
    (9, "coffee-pull-v3", "Pull a mug from a coffee machine", "medium"),
    (10, "coffee-push-v3", "Push a mug under a coffee machine", "medium"),
    (11, "dial-turn-v3", "Rotate a dial 180 degrees", "easy"),
    (12, "nut-disassemble-v3", "Pick a nut out of a peg", "very_hard"),
    (13, "door-close-v3", "Close a door with a revolving joint", "easy"),
    (14, "door-lock-v3", "Lock the door by rotating the lock clockwise", "easy"),
    (15, "door-v3", "Open a door with a revolving joint", "easy"),
    (16, "door-unlock-v3", "Unlock the door by rotating the lock counter-clockwise", "easy"),
    (17, "hand-insert-v3", "Insert the gripper into a hole", "hard"),
    (18, "drawer-close-v3", "Push and close a drawer", "easy"),
    (19, "drawer-open-v3", "Open a drawer", "easy"),
    (20, "faucet-open-v3", "Rotate the faucet counter-clockwise", "easy"),
    (21, "faucet-close-v3", "Rotate the faucet clockwise", "easy"),
    (22, "hammer-v3", "Hammer a screw on the wall", "medium"),
    (23, "handle-press-side-v3", "Press a handle down sideways", "easy"),
    (24, "handle-press-v3", "Press a handle down", "easy"),
    (25, "handle-pull-side-v3", "Pull a handle up sideways", "easy"),
    (26, "handle-pull-v3", "Pull a handle up", "easy"),
    (27, "lever-pull-v3", "Pull a lever down 90 degrees", "easy"),
    (28, "pick-place-wall-v3", "Pick a puck, bypass a wall and place the puck", "very_hard"),
    (29, "pick-out-of-hole-v3", "Pick up a puck from a hole", "hard"),
    (30, "pick-place-v3", "Pick and place a puck to a goal", "hard"),
    (31, "plate-slide-v3", "Slide a plate into a cabinet", "easy"),
    (32, "plate-slide-side-v3", "Slide a plate into a cabinet sideways", "easy"),
    (33, "plate-slide-back-v3", "Get a plate from the cabinet", "easy"),
    (34, "plate-slide-back-side-v3", "Get a plate from the cabinet sideways", "easy"),
    (35, "peg-insertion-side-v3", "Insert a peg sideways", "medium"),
    (36, "peg-unplug-side-v3", "Unplug a peg sideways", "easy"),
    (37, "soccer-v3", "Kick a soccer into the goal", "medium"),
    (38, "stick-push-v3", "Grasp a stick and push a box using the stick", "very_hard"),
    (39, "stick-pull-v3", "Grasp a stick and pull a box with the stick", "very_hard"),
    (40, "push-v3", "Push the puck to a goal", "hard"),
    (41, "push-wall-v3", "Bypass a wall and push a puck to a goal", "medium"),
    (42, "push-back-v3", "Push the puck back to a goal", "hard"),
    (43, "reach-v3", "Reach a goal position", "easy"),
    (44, "reach-wall-v3", "Bypass a wall and reach a goal", "easy"),
    (45, "shelf-place-v3", "Pick and place a puck onto a shelf", "very_hard"),
    (46, "sweep-into-goal-v3", "Sweep a puck into a hole", "medium"),
    (47, "sweep-v3", "Sweep a puck off the table", "medium"),
    (48, "window-open-v3", "Push and open a window", "easy"),
    (49, "window-close-v3", "Push and close a window", "easy"),
]

DIFFICULTY_BUCKETS = ["easy", "medium", "hard", "very_hard"]
CAMERA_NAME = "corner2"


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10095

    # MetaWorld evaluation parameters
    seed: int = 4042
    episodes_per_task: int = 10  # Number of rollouts per task
    max_steps: int = 400  # Maximum steps per episode
    max_tasks: int = -1  # If > 0, limit tasks evaluated per bucket (smoke test). -1 = all.
    levels: str = "easy,medium,hard,very_hard"  # Comma-separated difficulty buckets

    # Output
    video_out_path: str = "experiments/metaworld/logs"

    # Dataset key for un-normalization. None = auto.
    unnorm_key: str | None = None


def eval_metaworld(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    np.random.seed(args.seed)

    # Parse difficulty levels.
    levels = [s.strip().lower() for s in args.levels.split(",") if s.strip()]
    for lv in levels:
        if lv not in DIFFICULTY_BUCKETS:
            raise ValueError(f"Unknown difficulty level: {lv}. Must be one of {DIFFICULTY_BUCKETS}")

    # Group tasks by difficulty.
    bucket_tasks = {b: [] for b in DIFFICULTY_BUCKETS}
    for idx, slug, prompt, difficulty in MT50_TASKS:
        bucket_tasks[difficulty].append((idx, slug, prompt))

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Build MT50 env.
    envs = gym.make_vec(
        "Meta-World/MT50",
        vector_strategy="sync",
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=CAMERA_NAME,
    )
    total_envs = len(envs.envs)

    # Connect to server.
    client = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
    )

    # Accumulators per bucket.
    bucket_successes = {b: 0 for b in DIFFICULTY_BUCKETS}
    bucket_trials = {b: 0 for b in DIFFICULTY_BUCKETS}
    total_episodes, total_successes = 0, 0

    for level in levels:
        tasks = bucket_tasks[level]
        if not tasks:
            logging.info(f"Skip bucket {level} (0 tasks)")
            continue

        if args.max_tasks > 0:
            tasks = tasks[: args.max_tasks]

        logging.info(f"\n==== Bucket: {level} ({len(tasks)} tasks) ====")

        for task_idx, slug, prompt in tqdm.tqdm(tasks, desc=level):
            if task_idx >= total_envs:
                logging.warning(f"Task index {task_idx} ({slug}) out of range, skipping")
                continue

            sub_env = envs.envs[task_idx]
            task_successes = 0

            for ep in tqdm.tqdm(range(args.episodes_per_task), desc=slug, leave=False):
                # Randomize goal position each episode.
                for obj in (sub_env, getattr(sub_env, "unwrapped", None)):
                    fn = getattr(obj, "iterate_goal_position", None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass
                        break

                obs, _ = sub_env.reset(seed=args.seed + ep)
                client.reset(task_description=prompt)

                # Initial no-op step for environment stabilization.
                try:
                    a0 = np.zeros(sub_env.action_space.shape, dtype=np.float32)
                    a0 = np.clip(a0, sub_env.action_space.low, sub_env.action_space.high)
                    obs, _, _, _, _ = sub_env.step(a0)
                except Exception:
                    pass

                replay_images = []
                step = 0
                done = False

                while step < args.max_steps and not done:
                    img_rgb = preprocess_metaworld_image(sub_env.render())
                    replay_images.append(img_rgb)

                    action = client.step(image=img_rgb, prompt=prompt, step=step)
                    action = np.clip(action, sub_env.action_space.low, sub_env.action_space.high)
                    obs, _, terminated, truncated, info = sub_env.step(action)
                    step += 1

                    if isinstance(info, dict) and info.get("success", 0) == 1:
                        task_successes += 1
                        done = True
                        break
                    if terminated or truncated:
                        done = True
                        break

                total_episodes += 1
                bucket_trials[level] += 1

                # Save replay video.
                suffix = "success" if done and info.get("success", 0) == 1 else "failure"
                _save_video(
                    replay_images,
                    pathlib.Path(args.video_out_path)
                    / f"task{task_idx:02d}_{slug}_ep{ep + 1:03d}_{suffix}.mp4",
                )

            bucket_successes[level] += task_successes
            total_successes += task_successes
            task_rate = task_successes / args.episodes_per_task
            logging.info(
                f"[Task {task_idx} {slug}] SR={task_rate:.3f} "
                f"({task_successes}/{args.episodes_per_task})"
            )

        bsr = bucket_successes[level] / max(1, bucket_trials[level])
        logging.info(f"==== Bucket {level}: SR={bsr:.3f} ({bucket_successes[level]}/{bucket_trials[level]}) ====")

    # Final summary.
    logging.info("\n==== Per-bucket success rates ====")
    evaluated_rates = []
    for lv in levels:
        if bucket_trials[lv] > 0:
            rate = bucket_successes[lv] / bucket_trials[lv]
            evaluated_rates.append(rate)
            logging.info(f"  {lv:10s}: {rate:.3f} ({bucket_successes[lv]}/{bucket_trials[lv]})")

    overall = (sum(evaluated_rates) / len(evaluated_rates)) if evaluated_rates else 0.0
    logging.info(f"\n==== Overall (mean of bucket SRs): {overall:.3f} ====")
    logging.info(f"Total episodes: {total_episodes}, total successes: {total_successes}")

    envs.close()

    # Save summary JSON.
    summary = {
        "overall": overall,
        "per_bucket": {lv: bucket_successes[lv] / max(1, bucket_trials[lv]) for lv in levels},
        "episodes_per_task": args.episodes_per_task,
        "seed": args.seed,
        "max_steps": args.max_steps,
    }
    summary_path = pathlib.Path(args.video_out_path) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved to {summary_path}")


def _save_video(frames: list, path: pathlib.Path) -> None:
    """Encode mp4 via a local temp file (avoids OSSFS seek-back issues)."""
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    try:
        imageio.mimsave(tmp_path, [np.asarray(f) for f in frames], fps=10)
        shutil.copyfile(tmp_path, str(path))
    except Exception as e:
        logging.warning(f"Failed to save video {path}: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    tyro.cli(eval_metaworld)
