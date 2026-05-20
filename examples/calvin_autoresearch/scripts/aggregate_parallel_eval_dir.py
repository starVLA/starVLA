#!/usr/bin/env python3
"""Aggregate sharded CALVIN eval outputs from worker_* directories."""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


ACTION_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def count_success(results):
    if not results:
        return [0.0] * 5
    count = Counter(results)
    return [sum(count[j] for j in reversed(range(i, 6))) / len(results) for i in range(1, 6)]


def basic_stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def vec(values=None):
    values = values or []
    out = [0.0] * len(ACTION_DIM_NAMES)
    for i, value in enumerate(values[: len(out)]):
        out[i] = float(value)
    return out


def add_vec(a, b):
    return [x + y for x, y in zip(a, b)]


def max_vec(a, b):
    return [max(x, y) for x, y in zip(a, b)]


def div_vec(values, denom):
    if denom <= 0:
        return [0.0] * len(values)
    return [value / denom for value in values]


def sqrt_vec(values):
    return [value**0.5 for value in values]


def merge_action_stats(stats_list):
    stats_list = [stats for stats in stats_list if stats and int(stats.get("count", 0)) > 0]
    dim = len(ACTION_DIM_NAMES)
    if not stats_list:
        return {
            "count": 0,
            "dim_names": ACTION_DIM_NAMES,
            "saturation_limits": [1.0] * dim,
            "mean_abs": [0.0] * dim,
            "rms": [0.0] * dim,
            "max_abs": [0.0] * dim,
            "saturation_rate": [0.0] * dim,
            "jitter": {
                "count": 0,
                "mean_abs": [0.0] * dim,
                "max_abs": [0.0] * dim,
                "mean_l2": 0.0,
                "max_l2": 0.0,
                "gripper_switches": 0,
                "gripper_switch_rate": 0.0,
            },
        }

    total_count = 0
    sum_abs = [0.0] * dim
    sum_sq = [0.0] * dim
    max_abs = [0.0] * dim
    sat_count = [0.0] * dim
    limits = stats_list[0].get("saturation_limits", [1.0] * dim)
    jitter_count = 0
    jitter_sum_abs = [0.0] * dim
    jitter_max_abs = [0.0] * dim
    jitter_sum_l2 = 0.0
    jitter_max_l2 = 0.0
    gripper_switches = 0

    for stats in stats_list:
        count = int(stats.get("count", 0))
        total_count += count
        sum_abs = add_vec(sum_abs, vec(stats.get("_sum_abs", [x * count for x in stats.get("mean_abs", [])])))
        sum_sq = add_vec(sum_sq, vec(stats.get("_sum_sq", [x * x * count for x in stats.get("rms", [])])))
        max_abs = max_vec(max_abs, vec(stats.get("max_abs")))
        sat_count = add_vec(
            sat_count,
            vec(stats.get("_saturation_count", [x * count for x in stats.get("saturation_rate", [])])),
        )
        jitter = stats.get("jitter", {})
        current_jitter_count = int(jitter.get("count", 0))
        jitter_count += current_jitter_count
        jitter_sum_abs = add_vec(
            jitter_sum_abs,
            vec(jitter.get("_sum_abs", [x * current_jitter_count for x in jitter.get("mean_abs", [])])),
        )
        jitter_max_abs = max_vec(jitter_max_abs, vec(jitter.get("max_abs")))
        jitter_sum_l2 += float(jitter.get("_sum_l2", float(jitter.get("mean_l2", 0.0)) * current_jitter_count))
        jitter_max_l2 = max(jitter_max_l2, float(jitter.get("max_l2", 0.0)))
        gripper_switches += int(jitter.get("gripper_switches", 0))

    return {
        "count": total_count,
        "dim_names": ACTION_DIM_NAMES,
        "saturation_limits": limits,
        "mean_abs": div_vec(sum_abs, total_count),
        "rms": sqrt_vec(div_vec(sum_sq, total_count)),
        "max_abs": max_abs,
        "saturation_rate": div_vec(sat_count, total_count),
        "jitter": {
            "count": jitter_count,
            "mean_abs": div_vec(jitter_sum_abs, jitter_count),
            "max_abs": jitter_max_abs,
            "mean_l2": jitter_sum_l2 / jitter_count if jitter_count else 0.0,
            "max_l2": jitter_max_l2,
            "gripper_switches": gripper_switches,
            "gripper_switch_rate": gripper_switches / jitter_count if jitter_count else 0.0,
        },
    }


def load_worker_results(root):
    all_results = []
    task_success = Counter()
    task_total = Counter()
    for result_path in sorted(root.glob("worker_*/results.json")):
        data = json.loads(result_path.read_text())
        entry = data.get("0") or next(iter(data.values()))
        all_results.extend(int(x) for x in entry.get("results", []))
        for task, info in entry.get("task_info", {}).items():
            task_success[task] += int(info.get("success", 0))
            task_total[task] += int(info.get("total", 0))
    return all_results, task_success, task_total


def write_results_summary(root):
    all_results, task_success, task_total = load_worker_results(root)
    if not all_results:
        raise SystemExit(f"No worker results found under {root}")
    chain_sr = {str(i + 1): float(sr) for i, sr in enumerate(count_success(all_results))}
    summary = {
        "0": {
            "avg_seq_len": float(sum(all_results) / len(all_results)),
            "chain_sr": chain_sr,
            "task_info": {
                task: {"success": int(task_success[task]), "total": int(task_total[task])}
                for task in sorted(task_total)
            },
            "results": all_results,
            "num_sequences": len(all_results),
        }
    }
    (root / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def attach_near_miss_to_results(root, result_summary, metrics_summary):
    entry = metrics_summary.get("0") if metrics_summary else None
    if not entry:
        return result_summary
    near_miss = entry.get("near_miss", {})
    result_entry = result_summary.setdefault("0", {})
    result_entry["near_miss"] = near_miss
    result_entry["near_miss_rate"] = near_miss.get("any_task_rate")
    result_entry["near_miss_related_rate"] = near_miss.get("related_task_rate")
    (root / "results.json").write_text(json.dumps(result_summary, indent=2, sort_keys=True))
    return result_summary


def load_sequence_records(root):
    records = []
    for records_path in sorted(root.glob("worker_*/metrics_sequences_epoch_*.jsonl")):
        for line in records_path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_records(records):
    results = [int(record.get("success_count", 0)) for record in records]
    success_len_hist = {str(i): int(Counter(results).get(i, 0)) for i in range(6)}
    conditional = {}
    for position in range(1, 6):
        attempts = sum(
            1
            for record in records
            if len(record.get("tasks", [])) >= position and int(record.get("success_count", 0)) >= position - 1
        )
        successes = sum(1 for record in records if int(record.get("success_count", 0)) >= position)
        conditional[str(position)] = {
            "attempts": attempts,
            "successes": successes,
            "success_rate": successes / attempts if attempts else None,
        }

    failure_position = Counter()
    failure_steps = []
    all_subtasks = []
    chain_acc = defaultdict(lambda: {"attempts": 0, "full_successes": 0, "success_len_sum": 0})
    per_task = defaultdict(
        lambda: {
            "attempts": 0,
            "successes": 0,
            "success_steps": [],
            "failure_steps": [],
            "near_miss_any_task": 0,
            "near_miss_related_task": 0,
            "achieved_other_tasks": Counter(),
        }
    )

    for record in records:
        tasks = [str(task) for task in record.get("tasks", [])]
        success_count = int(record.get("success_count", 0))
        chain_key = " -> ".join(tasks)
        chain_acc[chain_key]["attempts"] += 1
        chain_acc[chain_key]["full_successes"] += int(success_count >= len(tasks))
        chain_acc[chain_key]["success_len_sum"] += success_count
        failed_position = record.get("failed_subtask_position")
        failure_position[str(failed_position) if failed_position is not None else "complete"] += 1
        if record.get("failure_step") is not None:
            failure_steps.append(record["failure_step"])

        for subtask in record.get("subtasks", []):
            all_subtasks.append(subtask)
            task = str(subtask.get("task"))
            stats = per_task[task]
            stats["attempts"] += 1
            if subtask.get("success"):
                stats["successes"] += 1
                stats["success_steps"].append(subtask.get("success_step"))
            else:
                stats["failure_steps"].append(subtask.get("failure_step"))
                stats["near_miss_any_task"] += int(bool(subtask.get("near_miss_any_task", False)))
                stats["near_miss_related_task"] += int(bool(subtask.get("near_miss_related_task", False)))
            for achieved_task in subtask.get("achieved_other_tasks", []):
                stats["achieved_other_tasks"][achieved_task] += 1

    failed_subtasks = [subtask for subtask in all_subtasks if not subtask.get("success")]
    near_miss_any = sum(int(bool(subtask.get("near_miss_any_task", False))) for subtask in failed_subtasks)
    near_miss_related = sum(int(bool(subtask.get("near_miss_related_task", False))) for subtask in failed_subtasks)

    return {
        "num_sequences": len(records),
        "avg_seq_len": sum(results) / len(results) if results else 0.0,
        "chain_sr": {str(i + 1): float(sr) for i, sr in enumerate(count_success(results))},
        "success_len_histogram": success_len_hist,
        "conditional_success": conditional,
        "failure_position": dict(sorted(failure_position.items())),
        "failure_step": basic_stats(failure_steps),
        "near_miss": {
            "failed_subtasks": len(failed_subtasks),
            "any_task_count": near_miss_any,
            "any_task_rate": near_miss_any / len(failed_subtasks) if failed_subtasks else None,
            "related_task_count": near_miss_related,
            "related_task_rate": near_miss_related / len(failed_subtasks) if failed_subtasks else None,
        },
        "per_atomic_task": {
            task: {
                "attempts": stats["attempts"],
                "successes": stats["successes"],
                "failures": stats["attempts"] - stats["successes"],
                "success_rate": stats["successes"] / stats["attempts"] if stats["attempts"] else None,
                "success_step": basic_stats(stats["success_steps"]),
                "failure_step": basic_stats(stats["failure_steps"]),
                "near_miss_any_task_rate": (
                    stats["near_miss_any_task"] / (stats["attempts"] - stats["successes"])
                    if stats["attempts"] > stats["successes"]
                    else None
                ),
                "near_miss_related_task_rate": (
                    stats["near_miss_related_task"] / (stats["attempts"] - stats["successes"])
                    if stats["attempts"] > stats["successes"]
                    else None
                ),
                "achieved_other_tasks": dict(stats["achieved_other_tasks"].most_common()),
            }
            for task, stats in sorted(per_task.items())
        },
        "task_chain": {
            chain: {
                "attempts": stats["attempts"],
                "full_successes": stats["full_successes"],
                "full_success_rate": stats["full_successes"] / stats["attempts"] if stats["attempts"] else None,
                "avg_success_len": stats["success_len_sum"] / stats["attempts"] if stats["attempts"] else None,
            }
            for chain, stats in sorted(chain_acc.items())
        },
        "action_stats": {
            "raw_model_action": merge_action_stats([subtask.get("raw_action_stats") for subtask in all_subtasks]),
            "env_action_after_gripper_binarization": merge_action_stats(
                [subtask.get("env_action_stats") for subtask in all_subtasks]
            ),
        },
    }


def write_metrics_summary(root):
    records = load_sequence_records(root)
    if not records:
        print("No worker metrics_sequences_epoch_*.jsonl found; skipped metrics.json aggregation.")
        return None
    records_path = root / "metrics_sequences_epoch_0.jsonl"
    with records_path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")
    summary = summarize_records(records)
    summary["sequence_records_path"] = records_path.name
    metrics = {"0": summary}
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: aggregate_parallel_eval_dir.py /path/to/eval_dir")
    root = Path(sys.argv[1])
    result_summary = write_results_summary(root)
    metrics_summary = write_metrics_summary(root)
    result_summary = attach_near_miss_to_results(root, result_summary, metrics_summary)

    print("Aggregated results:", root / "results.json")
    print("Average successful sequence length:", result_summary["0"]["avg_seq_len"])
    for i, sr in result_summary["0"]["chain_sr"].items():
        print(f"{i}: {sr * 100:.1f}%")
    if metrics_summary is not None:
        near_miss = result_summary["0"].get("near_miss", {})
        if near_miss:
            any_rate = near_miss.get("any_task_rate")
            related_rate = near_miss.get("related_task_rate")
            print(f"near_miss_rate: {any_rate * 100:.1f}%" if any_rate is not None else "near_miss_rate: n/a")
            print(
                f"near_miss_related_rate: {related_rate * 100:.1f}%"
                if related_rate is not None
                else "near_miss_related_rate: n/a"
            )
        print("Aggregated detailed metrics:", root / "metrics.json")


if __name__ == "__main__":
    main()
