#!/usr/bin/env python3
"""Reproduce the checkpoint- and task-level RoboCasa evaluation analysis."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


NEW_RUN = Path(
    "qwen_var_productvq_g16_s124816_robocasa_closebalanced_e256_bestworst_e47_"
    "100k_lr1e4_warmup5000_gbs512_fullcache"
)
OLD_RUN = Path(
    "qwen_var_productvq_g16_s124816_robocasa_best_recon_e128_"
    "100k_lr1e4_warmup5000_gbs512_fullcache"
)
NEW_STEPS = (78000, 80000, 86000, 90000, 100000)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_path(run: Path, step: int) -> Path:
    return (
        run
        / "robocasa_eval"
        / f"steps_{step}_pytorch_model_gr1_24_50eps_chunk50_robust"
        / "summary.json"
    )


def wilson_interval(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / episodes
    denom = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / denom
    half = z * math.sqrt(p * (1.0 - p) / episodes + z * z / (4.0 * episodes * episodes)) / denom
    return center - half, center + half


def difference_interval(success_a: int, total_a: int, success_b: int, total_b: int) -> tuple[float, float]:
    p_a = success_a / total_a
    p_b = success_b / total_b
    se = math.sqrt(p_a * (1.0 - p_a) / total_a + p_b * (1.0 - p_b) / total_b)
    return (p_a - p_b) - 1.959963984540054 * se, (p_a - p_b) + 1.959963984540054 * se


def short_task_name(name: str) -> str:
    text = name.removeprefix("gr1_unified_").removesuffix("_GR1ArmsAndWaistFourierHands")
    text = text.removesuffix("_Env")
    return text.replace("PosttrainPnPNovel", "Novel")


def task_map(summary: dict) -> dict[str, dict]:
    return {row["task"]: row for row in summary["tasks"]}


def pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("analysis_results.json"),
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    new_summaries = {
        step: read_json(summary_path(repo / NEW_RUN, step)) for step in NEW_STEPS
    }
    old_summary = read_json(summary_path(repo / OLD_RUN, 90000))

    checkpoint_rows = []
    for step, summary in new_summaries.items():
        successes = int(summary["total_successes"])
        episodes = int(summary["total_episodes"])
        low, high = wilson_interval(successes, episodes)
        checkpoint_rows.append(
            {
                "checkpoint": f"{step // 1000}k",
                "step": step,
                "successes": successes,
                "episodes": episodes,
                "success_rate": successes / episodes,
                "wilson95_low": low,
                "wilson95_high": high,
            }
        )

    old_successes = int(old_summary["total_successes"])
    old_episodes = int(old_summary["total_episodes"])
    old_low, old_high = wilson_interval(old_successes, old_episodes)
    old_row = {
        "checkpoint": "E128 90k",
        "step": 90000,
        "successes": old_successes,
        "episodes": old_episodes,
        "success_rate": old_successes / old_episodes,
        "wilson95_low": old_low,
        "wilson95_high": old_high,
    }

    best = max(checkpoint_rows, key=lambda row: row["success_rate"])
    diff_low, diff_high = difference_interval(
        best["successes"], best["episodes"], old_successes, old_episodes
    )

    maps = {step: task_map(summary) for step, summary in new_summaries.items()}
    old_tasks = task_map(old_summary)
    task_names = list(maps[86000])
    task_rows = []
    for task_index, task in enumerate(task_names):
        rates = [maps[step][task]["success_rate"] for step in NEW_STEPS]
        successes = [maps[step][task]["successes"] for step in NEW_STEPS]
        best_index = max(range(len(NEW_STEPS)), key=lambda idx: successes[idx])
        task_rows.append(
            {
                "task_index": task_index,
                "task": short_task_name(task),
                "full_task": task,
                "family": "close" if task_index < 6 else "novel",
                "success_86k": maps[86000][task]["success_rate"],
                "success_old_90k": old_tasks[task]["success_rate"],
                "delta_86k_vs_old_90k": maps[86000][task]["success_rate"]
                - old_tasks[task]["success_rate"],
                "mean_new": statistics.fmean(rates),
                "stdev_new": statistics.pstdev(rates),
                "min_new": min(rates),
                "max_new": max(rates),
                "range_new": max(rates) - min(rates),
                "best_observed_checkpoint": f"{NEW_STEPS[best_index] // 1000}k",
                "best_observed_success_rate": successes[best_index] / 50.0,
            }
        )

    family_rows = []
    for family in ("close", "novel"):
        family_tasks = [row["full_task"] for row in task_rows if row["family"] == family]
        for step in NEW_STEPS:
            successes = sum(maps[step][task]["successes"] for task in family_tasks)
            episodes = sum(maps[step][task]["episodes"] for task in family_tasks)
            family_rows.append(
                {
                    "family": family,
                    "checkpoint": f"{step // 1000}k",
                    "successes": successes,
                    "episodes": episodes,
                    "success_rate": successes / episodes,
                }
            )

    pairwise_checkpoint_correlations = []
    for left_index, left_step in enumerate(NEW_STEPS):
        for right_step in NEW_STEPS[left_index + 1 :]:
            left = [maps[left_step][task]["success_rate"] for task in task_names]
            right = [maps[right_step][task]["success_rate"] for task in task_names]
            pairwise_checkpoint_correlations.append(
                {
                    "left": f"{left_step // 1000}k",
                    "right": f"{right_step // 1000}k",
                    "pearson_task_rate": pearson(left, right),
                }
            )

    oracle_successes = sum(
        max(maps[step][task]["successes"] for step in NEW_STEPS) for task in task_names
    )
    change_counts = {
        "improved": sum(row["delta_86k_vs_old_90k"] > 0 for row in task_rows),
        "unchanged": sum(row["delta_86k_vs_old_90k"] == 0 for row in task_rows),
        "declined": sum(row["delta_86k_vs_old_90k"] < 0 for row in task_rows),
    }

    output = {
        "protocol": {
            "tasks": 24,
            "episodes_per_task": 50,
            "episodes_per_checkpoint": 1200,
            "chunk_episodes": 50,
            "precision": "fp32",
            "n_envs": 1,
            "max_episode_steps": 720,
            "n_action_steps": 12,
            "effective_seed_control": False,
        },
        "old_baseline": old_row,
        "new_checkpoints": checkpoint_rows,
        "best_observed": best,
        "best_vs_old": {
            "absolute_delta": best["success_rate"] - old_row["success_rate"],
            "relative_lift": best["success_rate"] / old_row["success_rate"] - 1.0,
            "wald95_delta_low": diff_low,
            "wald95_delta_high": diff_high,
        },
        "task_change_counts_86k_vs_old90k": change_counts,
        "task_rows": task_rows,
        "persistent_hard_tasks": sorted(task_rows, key=lambda row: row["mean_new"])[:8],
        "checkpoint_sensitive_tasks": sorted(
            task_rows, key=lambda row: row["range_new"], reverse=True
        )[:8],
        "family_rows": family_rows,
        "pairwise_checkpoint_correlations": pairwise_checkpoint_correlations,
        "taskwise_oracle_upper_bound": {
            "successes": oracle_successes,
            "episodes": 1200,
            "success_rate": oracle_successes / 1200.0,
            "warning": "Optimistically selected and measured on the same stochastic evaluations; not a valid held-out estimate.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
