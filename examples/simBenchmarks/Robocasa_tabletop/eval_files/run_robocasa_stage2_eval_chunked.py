#!/usr/bin/env python3
"""Run RoboCasa stage2 eval in resumable chunks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.Robocasa_tabletop.eval_files.robocasa_eval_tasks import load_tasks, task_slug


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_completed(chunk_dir: Path, expected_episodes: int) -> bool:
    complete_path = chunk_dir / "COMPLETE.json"
    if not complete_path.exists():
        return False
    try:
        data = read_json(complete_path)
    except Exception:
        return False
    episodes = data.get("episodes") or []
    return data.get("status") == "complete" and len(episodes) == expected_episodes


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def run_chunk(args: argparse.Namespace, *, task: str, task_index: int, trial_start: int, chunk_episodes: int) -> bool:
    slug = task_slug(task)
    chunk_dir = args.output_root / slug / f"r{trial_start:03d}_n{chunk_episodes:03d}"
    if chunk_completed(chunk_dir, chunk_episodes):
        print(f"[robocasa_eval] skip completed chunk task={task_index} trial_start={trial_start} dir={chunk_dir}")
        return True

    port = args.base_port + args.worker_index * 100 + task_index
    cmd = [
        args.starvla_python,
        "examples/simBenchmarks/Robocasa_tabletop/eval_files/run_robocasa_ckpt_eval.py",
        "--ckpt",
        args.ckpt,
        "--output-dir",
        str(chunk_dir),
        "--env-name",
        task,
        "--repo-root",
        str(args.repo_root),
        "--starvla-python",
        args.starvla_python,
        "--robocasa-python",
        args.robocasa_python,
        "--port",
        str(port),
        "--server-gpu",
        str(args.gpu),
        "--sim-gpu",
        str(args.gpu),
        "--n-episodes",
        str(chunk_episodes),
        "--n-envs",
        str(args.n_envs),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--n-action-steps",
        str(args.n_action_steps),
        "--server-ready-timeout",
        str(args.server_ready_timeout),
        "--server-idle-timeout",
        str(args.server_idle_timeout),
        "--sim-timeout",
        str(args.sim_timeout),
        "--attempts",
        "1",
        "--force",
    ]
    if args.use_bf16:
        cmd.append("--use-bf16")
    if args.no_video:
        cmd.append("--no-video")
    if args.action_stats_every > 0:
        cmd.extend(["--action-stats-every", str(args.action_stats_every)])
    if args.norm_action_stats_every > 0:
        cmd.extend(["--norm-action-stats-every", str(args.norm_action_stats_every)])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.repo_root}:{env.get('PYTHONPATH', '')}"

    for attempt in range(1, args.max_retries + 1):
        print(
            f"[robocasa_eval] task={task_index} trial_start={trial_start} "
            f"episodes={chunk_episodes} attempt={attempt} gpu={args.gpu} port={port}"
        )
        chunk_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            chunk_dir / "CHUNK_RUNNING.json",
            {
                "ckpt": args.ckpt,
                "task": task,
                "task_index": task_index,
                "trial_start": trial_start,
                "episodes": chunk_episodes,
                "attempt": attempt,
                "started_at": time.time(),
            },
        )
        result = subprocess.run(cmd, cwd=args.repo_root, env=env)
        if result.returncode == 0 and chunk_completed(chunk_dir, chunk_episodes):
            write_json(
                chunk_dir / "ROBOCASA_CHUNK_OK.json",
                {
                    "status": "complete",
                    "ckpt": args.ckpt,
                    "task": task,
                    "task_index": task_index,
                    "trial_start": trial_start,
                    "episodes": chunk_episodes,
                    "completed_at": time.time(),
                },
            )
            (chunk_dir / "CHUNK_RUNNING.json").unlink(missing_ok=True)
            return True
        if attempt < args.max_retries:
            time.sleep(args.retry_sleep)

    write_json(
        chunk_dir / "CHUNK_FAILED.json",
        {
            "status": "failed",
            "ckpt": args.ckpt,
            "task": task,
            "task_index": task_index,
            "trial_start": trial_start,
            "episodes": chunk_episodes,
            "failed_at": time.time(),
        },
    )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpt")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--starvla-python", default="/home/zhangfeihong/miniconda3/envs/starVLA/bin/python")
    parser.add_argument("--robocasa-python", default="/home/zhangfeihong/miniconda3/envs/robocasa/bin/python")
    parser.add_argument("--tasks-preset", default="gr1_5", choices=["gr1_5", "gr1_24"])
    parser.add_argument("--tasks-file", type=Path, default=None)
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--chunk-episodes", type=int, default=1)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=-1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=6700)
    parser.add_argument("--max-retries", type=int, default=100000)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--server-ready-timeout", type=float, default=900.0)
    parser.add_argument("--server-idle-timeout", type=int, default=1800)
    parser.add_argument("--sim-timeout", type=float, default=3600.0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=12)
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--action-stats-every", type=int, default=0)
    parser.add_argument("--norm-action-stats-every", type=int, default=0)
    args = parser.parse_args()

    tasks = load_tasks(preset=args.tasks_preset, tasks_file=args.tasks_file)
    task_end = len(tasks) if args.task_count < 0 else min(len(tasks), args.task_start + args.task_count)
    selected = [(idx, tasks[idx]) for idx in range(args.task_start, task_end)]
    selected = [(idx, task) for idx, task in selected if idx % args.worker_count == args.worker_index]
    if not selected:
        print(f"[robocasa_eval] no tasks for worker {args.worker_index}/{args.worker_count}")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    for task_index, task in selected:
        trial_start = 0
        while trial_start < args.trials_per_task:
            remaining = args.trials_per_task - trial_start
            chunk_episodes = min(args.chunk_episodes, remaining)
            if not run_chunk(args, task=task, task_index=task_index, trial_start=trial_start, chunk_episodes=chunk_episodes):
                return 1
            trial_start += chunk_episodes
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
