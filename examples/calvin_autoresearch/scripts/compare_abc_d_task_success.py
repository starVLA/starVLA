#!/usr/bin/env python3
"""Compare CALVIN ABC and D eval metrics, with ABC LeRobot distribution fallback."""

import argparse
import json
from collections import Counter
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def first_epoch_entry(path):
    data = read_json(path)
    if "0" in data:
        return data["0"]
    return next(iter(data.values()))


def extract_task_success(metrics_path):
    entry = first_epoch_entry(metrics_path)
    source = "metrics.per_atomic_task" if "per_atomic_task" in entry else "results.task_info"
    raw = entry.get("per_atomic_task") or entry.get("task_info") or {}
    out = {}
    for task, stats in raw.items():
        attempts = int(stats.get("attempts", stats.get("total", 0)))
        successes = int(stats.get("successes", stats.get("success", 0)))
        out[task] = {
            "attempts": attempts,
            "successes": successes,
            "success_rate": successes / attempts if attempts else None,
        }
    return {"source": source, "num_tasks": len(out), "tasks": out}


def load_lerobot_task_distribution(dataset_root):
    dataset_root = Path(dataset_root)
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing LeRobot episodes metadata: {episodes_path}")
    counts = Counter()
    episode_count = 0
    for line in episodes_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        episode_count += 1
        episode = json.loads(line)
        for task in episode.get("tasks", []):
            counts[str(task)] += 1
    return {
        "source": str(episodes_path),
        "total_episodes": episode_count,
        "num_tasks": len(counts),
        "tasks": {
            task: {"episodes": int(count), "episode_fraction": count / episode_count if episode_count else None}
            for task, count in counts.most_common()
        },
    }


def compare_task_success(abc_success, d_success):
    abc_tasks = abc_success["tasks"]
    d_tasks = d_success["tasks"]
    common = sorted(set(abc_tasks).intersection(d_tasks))
    return {
        task: {
            "abc_success_rate": abc_tasks[task]["success_rate"],
            "d_success_rate": d_tasks[task]["success_rate"],
            "delta_d_minus_abc": (
                d_tasks[task]["success_rate"] - abc_tasks[task]["success_rate"]
                if d_tasks[task]["success_rate"] is not None and abc_tasks[task]["success_rate"] is not None
                else None
            ),
            "abc_attempts": abc_tasks[task]["attempts"],
            "d_attempts": d_tasks[task]["attempts"],
        }
        for task in common
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-metrics", required=True, help="D eval metrics.json or results.json")
    parser.add_argument("--abc-metrics", help="Optional ABC closed-loop metrics.json or results.json")
    parser.add_argument(
        "--abc-lerobot",
        default="/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D",
        help="ABC LeRobot dataset root. This gives train distribution, not closed-loop success.",
    )
    parser.add_argument("--out", help="Output JSON path")
    args = parser.parse_args()

    d_success = extract_task_success(args.d_metrics)
    report = {
        "d_closed_loop_task_success": d_success,
        "notes": [
            "Closed-loop ABC-vs-D success comparison requires an ABC closed-loop eval metrics file.",
            "The ABC LeRobot fallback reports training task distribution only, not success rate.",
        ],
    }

    if args.abc_metrics:
        abc_success = extract_task_success(args.abc_metrics)
        report["abc_closed_loop_task_success"] = abc_success
        report["abc_vs_d_common_task_success"] = compare_task_success(abc_success, d_success)

    if args.abc_lerobot:
        report["abc_lerobot_train_task_distribution"] = load_lerobot_task_distribution(args.abc_lerobot)

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output)
        print(args.out)
    else:
        print(output)


if __name__ == "__main__":
    main()
