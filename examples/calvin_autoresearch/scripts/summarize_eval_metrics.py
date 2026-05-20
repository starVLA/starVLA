#!/usr/bin/env python3
"""Print a compact summary for CALVIN eval metrics.json."""

import json
import sys
from pathlib import Path


def pct(value):
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def first_epoch_entry(path):
    path = Path(path)
    if path.is_dir():
        path = path / "metrics.json"
    data = json.loads(path.read_text())
    return data.get("0") or next(iter(data.values()))


def print_action_stats(name, stats):
    dims = stats.get("dim_names", ["x", "y", "z", "roll", "pitch", "yaw", "gripper"])
    mean_abs = stats.get("mean_abs", [])
    max_abs = stats.get("max_abs", [])
    saturation = stats.get("saturation_rate", [])
    jitter = stats.get("jitter", {})
    print(f"\n{name}")
    for i, dim in enumerate(dims):
        ma = mean_abs[i] if i < len(mean_abs) else None
        mx = max_abs[i] if i < len(max_abs) else None
        sat = saturation[i] if i < len(saturation) else None
        print(f"  {dim}: mean_abs={ma:.5g} max_abs={mx:.5g} saturation={pct(sat)}")
    print(
        "  jitter:"
        f" mean_l2={float(jitter.get('mean_l2', 0.0)):.5g}"
        f" max_l2={float(jitter.get('max_l2', 0.0)):.5g}"
        f" gripper_switch_rate={pct(jitter.get('gripper_switch_rate'))}"
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_eval_metrics.py /path/to/metrics.json_or_eval_dir")
    entry = first_epoch_entry(sys.argv[1])
    print(f"num_sequences: {entry.get('num_sequences')}")
    print(f"avg_seq_len: {entry.get('avg_seq_len')}")

    print("\nchain_sr")
    for key, value in sorted(entry.get("chain_sr", {}).items(), key=lambda item: int(item[0])):
        print(f"  {key}/5: {pct(value)}")

    print("\nconditional_success")
    for key, item in sorted(entry.get("conditional_success", {}).items(), key=lambda row: int(row[0])):
        print(f"  position {key}: {item.get('successes')}/{item.get('attempts')} ({pct(item.get('success_rate'))})")

    print("\nfailure_position")
    for key, value in sorted(entry.get("failure_position", {}).items()):
        print(f"  {key}: {value}")

    near = entry.get("near_miss", {})
    print(
        "\nnear_miss"
        f"\n  near_miss_rate: {pct(near.get('any_task_rate'))}"
        f"\n  near_miss_related_rate: {pct(near.get('related_task_rate'))}"
        f"\n  any_task: {near.get('any_task_count')}/{near.get('failed_subtasks')} ({pct(near.get('any_task_rate'))})"
        f"\n  related_task: {near.get('related_task_count')}/{near.get('failed_subtasks')} ({pct(near.get('related_task_rate'))})"
    )

    print("\nworst_atomic_tasks")
    task_rows = []
    for task, item in entry.get("per_atomic_task", {}).items():
        attempts = int(item.get("attempts", 0))
        if attempts:
            task_rows.append((float(item.get("success_rate") or 0.0), attempts, task, item))
    for sr, attempts, task, item in sorted(task_rows)[:15]:
        print(
            f"  {task}: {item.get('successes')}/{attempts} ({pct(sr)})"
            f" failure_step_mean={item.get('failure_step', {}).get('mean')}"
            f" near_miss_related={pct(item.get('near_miss_related_task_rate'))}"
        )

    action_stats = entry.get("action_stats", {})
    if action_stats:
        print_action_stats("raw_model_action", action_stats.get("raw_model_action", {}))
        print_action_stats(
            "env_action_after_gripper_binarization",
            action_stats.get("env_action_after_gripper_binarization", {}),
        )


if __name__ == "__main__":
    main()
