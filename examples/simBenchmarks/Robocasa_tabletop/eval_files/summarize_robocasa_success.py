#!/usr/bin/env python3
"""Summarize chunked RoboCasa eval results from COMPLETE.json files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.Robocasa_tabletop.eval_files.robocasa_eval_tasks import load_tasks, task_slug


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _chunk_complete(path: Path, expected_episodes: int | None) -> bool:
    complete_path = path / "COMPLETE.json"
    if not complete_path.exists():
        return False
    data = _read_json(complete_path)
    if data.get("status") != "complete":
        return False
    if expected_episodes is not None:
        episodes = data.get("episodes") or []
        if len(episodes) != expected_episodes:
            return False
    return True


def _chunk_stats(path: Path) -> tuple[int, int]:
    data = _read_json(path / "COMPLETE.json")
    episodes = data.get("episodes") or []
    successes = sum(1 for item in episodes if item.get("success") is True)
    return successes, len(episodes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_root", type=Path, help="RoboCasa eval root containing per-task chunk directories.")
    parser.add_argument("--expected-episodes-per-chunk", type=int, default=None)
    parser.add_argument("--tasks-preset", default=None, choices=["gr1_5", "gr1_24"])
    parser.add_argument("--tasks-file", type=Path, default=None)
    parser.add_argument("--trials-per-task", type=int, default=None)
    parser.add_argument("--chunk-episodes", type=int, default=None)
    parser.add_argument("--require-complete", action="store_true", help="Fail if any discovered chunk is not complete.")
    args = parser.parse_args()

    eval_root = args.eval_root
    if not eval_root.exists():
        raise SystemExit(f"Missing eval root: {eval_root}")

    total_successes = 0
    total_episodes = 0
    incomplete = []
    task_summaries = []

    if args.tasks_preset is not None or args.tasks_file is not None:
        if args.trials_per_task is None or args.chunk_episodes is None:
            raise SystemExit("--trials-per-task and --chunk-episodes are required with --tasks-preset/--tasks-file")
        tasks = load_tasks(preset=args.tasks_preset or "gr1_5", tasks_file=args.tasks_file)
        expected_task_dirs = [eval_root / task_slug(task) for task in tasks]
    else:
        expected_task_dirs = sorted(path for path in eval_root.iterdir() if path.is_dir() and path.name != "logs")

    for task_dir in expected_task_dirs:
        if args.trials_per_task is not None and args.chunk_episodes is not None:
            chunk_dirs = []
            trial_start = 0
            while trial_start < args.trials_per_task:
                remaining = args.trials_per_task - trial_start
                chunk = min(args.chunk_episodes, remaining)
                chunk_dirs.append(task_dir / f"r{trial_start:03d}_n{chunk:03d}")
                trial_start += chunk
        elif task_dir.exists():
            chunk_dirs = sorted(path for path in task_dir.iterdir() if path.is_dir() and path.name.startswith("r"))
        else:
            chunk_dirs = []

        if not chunk_dirs:
            complete_path = task_dir / "COMPLETE.json"
            if complete_path.exists():
                chunk_dirs = [task_dir]
            else:
                continue

        task_successes = 0
        task_episodes = 0
        task_complete_chunks = 0
        for chunk_dir in chunk_dirs:
            if not _chunk_complete(chunk_dir, args.expected_episodes_per_chunk):
                incomplete.append(str(chunk_dir))
                continue
            successes, episodes = _chunk_stats(chunk_dir)
            task_successes += successes
            task_episodes += episodes
            task_complete_chunks += 1

        total_successes += task_successes
        total_episodes += task_episodes
        rate = task_successes / task_episodes if task_episodes else 0.0
        task_summaries.append(
            {
                "task": task_dir.name,
                "complete_chunks": task_complete_chunks,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": rate,
            }
        )
        print(
            f"{task_dir.name}: chunks={task_complete_chunks}, "
            f"episodes={task_episodes}, success={rate * 100:.2f}%"
        )

    overall = total_successes / total_episodes if total_episodes else 0.0
    print(f"overall: episodes={total_episodes}, successes={total_successes}, success={overall * 100:.2f}%")
    if incomplete:
        print(f"incomplete_chunks: {len(incomplete)}")
        for item in incomplete[:20]:
            print(f"  {item}")
        if len(incomplete) > 20:
            print(f"  ... {len(incomplete) - 20} more")
        if args.require_complete:
            return 1

    summary = {
        "eval_root": str(eval_root),
        "tasks": task_summaries,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": overall,
        "incomplete_chunks": incomplete,
    }
    (eval_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
