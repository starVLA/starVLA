#!/usr/bin/env python
"""Diagnose CALVIN left/right mirror action and state transform candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.datasets import canonicalize_calvin_task


ACTION_DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
STATE_DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
PAIR_TASKS = [
    ("move_slider_left", "move_slider_right"),
    ("push_red_block_left", "push_red_block_right"),
    ("push_blue_block_left", "push_blue_block_right"),
    ("push_pink_block_left", "push_pink_block_right"),
]

ACTION_CANDIDATES = {
    "A0_x": {"action.x": "negate"},
    "A1_x_yaw": {"action.x": "negate", "action.yaw": "negate"},
    "A2_x_roll": {"action.x": "negate", "action.roll": "negate"},
    "A3_x_roll_yaw": {"action.x": "negate", "action.roll": "negate", "action.yaw": "negate"},
}


def _episode_path(dataset: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return dataset / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _load_episodes(dataset: Path):
    with (dataset / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            episode = json.loads(line)
            task_text = str(episode.get("tasks", [""])[0])
            yield {
                "episode_index": int(episode["episode_index"]),
                "length": int(episode["length"]),
                "task_text": task_text,
                "canonical_task": canonicalize_calvin_task(task_text),
            }


def _select_episodes(dataset: Path, max_per_task: int) -> dict[str, list[dict]]:
    wanted = {task for pair in PAIR_TASKS for task in pair}
    selected = {task: [] for task in wanted}
    for episode in _load_episodes(dataset):
        task = episode["canonical_task"]
        if task in wanted and len(selected[task]) < max_per_task:
            selected[task].append(episode)
        if all(len(items) >= max_per_task for items in selected.values()):
            break
    return selected


def _stack_episode_arrays(dataset: Path, episodes: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    actions = []
    states = []
    loaded = []
    for episode in episodes:
        path = _episode_path(dataset, int(episode["episode_index"]))
        if not path.exists():
            continue
        data = pd.read_parquet(path, columns=["state", "actions"])
        state = np.stack(data["state"].to_numpy()).astype(np.float64)
        action = np.stack(data["actions"].to_numpy()).astype(np.float64)
        if state.ndim != 2 or state.shape[1] != len(STATE_DIMS):
            continue
        if action.ndim != 2 or action.shape[1] != len(ACTION_DIMS):
            continue
        states.append(state)
        actions.append(action)
        loaded.append(int(episode["episode_index"]))
    if not actions:
        return np.zeros((0, len(ACTION_DIMS))), np.zeros((0, len(STATE_DIMS))), loaded
    return np.concatenate(actions, axis=0), np.concatenate(states, axis=0), loaded


def _summed_action_x_by_episode(dataset: Path, episodes: list[dict]) -> list[float]:
    values = []
    for episode in episodes:
        path = _episode_path(dataset, int(episode["episode_index"]))
        if not path.exists():
            continue
        data = pd.read_parquet(path, columns=["actions"])
        action = np.stack(data["actions"].to_numpy()).astype(np.float64)
        values.append(float(action[:, 0].sum()))
    return values


def _summary(array: np.ndarray, names: list[str]) -> dict:
    out = {}
    if array.size == 0:
        return out
    for idx, name in enumerate(names):
        values = array[:, idx]
        out[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "q01": float(np.quantile(values, 0.01)),
            "q50": float(np.quantile(values, 0.50)),
            "q99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
        }
    return out


def _quantile_distance(left: np.ndarray, right: np.ndarray, quantiles: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    left_q = np.quantile(left, quantiles)
    right_q = np.quantile(right, quantiles)
    return float(np.mean(np.abs(left_q - right_q)))


def _distance_by_dim(left: np.ndarray, right: np.ndarray, names: list[str]) -> dict:
    quantiles = np.linspace(0.01, 0.99, 99)
    return {name: _quantile_distance(left[:, idx], right[:, idx], quantiles) for idx, name in enumerate(names)}


def _apply_action_candidate(actions: np.ndarray, candidate: dict[str, str]) -> np.ndarray:
    out = actions.copy()
    for key, op in candidate.items():
        if op != "negate":
            continue
        _, dim_name = key.split(".", 1)
        out[:, ACTION_DIMS.index(dim_name)] *= -1.0
    return out


def _apply_state_candidate(states: np.ndarray, candidate: dict) -> np.ndarray:
    out = states.copy()
    if candidate["type"] == "none":
        return out
    if candidate["type"] == "mirror_x":
        out[:, STATE_DIMS.index("x")] = 2.0 * float(candidate["center"]) - out[:, STATE_DIMS.index("x")]
    return out


def _bounds_violation(values: np.ndarray, reference: np.ndarray, dims: list[str]) -> dict:
    out = {}
    for idx, name in enumerate(dims):
        lo = float(np.min(reference[:, idx]))
        hi = float(np.max(reference[:, idx]))
        col = values[:, idx]
        out[name] = float(np.mean((col < lo) | (col > hi)))
    return out


def _roundtrip_error_action(actions: np.ndarray, candidate: dict[str, str]) -> float:
    if actions.size == 0:
        return 0.0
    restored = _apply_action_candidate(_apply_action_candidate(actions, candidate), candidate)
    return float(np.max(np.abs(restored - actions)))


def _roundtrip_error_state(states: np.ndarray, candidate: dict) -> float:
    if states.size == 0:
        return 0.0
    restored = _apply_state_candidate(_apply_state_candidate(states, candidate), candidate)
    return float(np.max(np.abs(restored - states)))


def _mean_or_nan(values: list[float]) -> float:
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _build_state_candidates(all_state_x: np.ndarray, pair_state_x: dict[str, np.ndarray]) -> dict:
    global_mean = float(np.mean(all_state_x))
    global_midrange = float((np.min(all_state_x) + np.max(all_state_x)) / 2.0)
    candidates = {
        "S0_none": {"type": "none", "center": None},
        "S1_center_0": {"type": "mirror_x", "center": 0.0},
        "S2_global_mean": {"type": "mirror_x", "center": global_mean},
        "S3_global_midrange": {"type": "mirror_x", "center": global_midrange},
    }
    for pair_name, values in pair_state_x.items():
        candidates[f"S4_pair_center:{pair_name}"] = {
            "type": "mirror_x",
            "center": float(np.mean(values)),
            "pair_only": pair_name,
        }
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None, help="Recorded for provenance; not required by this diagnostic.")
    parser.add_argument("--max-episodes-per-task", type=int, default=200)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    selected = _select_episodes(args.dataset, args.max_episodes_per_task)
    task_data = {}
    all_states = []
    pair_state_x = {}
    for left_task, right_task in PAIR_TASKS:
        for task in (left_task, right_task):
            actions, states, loaded = _stack_episode_arrays(args.dataset, selected[task])
            task_data[task] = {
                "actions": actions,
                "states": states,
                "loaded_episodes": loaded,
                "action_x_sums": _summed_action_x_by_episode(args.dataset, selected[task]),
            }
            if len(states):
                all_states.append(states)
        left_states = task_data[left_task]["states"]
        right_states = task_data[right_task]["states"]
        if len(left_states) and len(right_states):
            pair_state_x[f"{left_task}<->{right_task}"] = np.concatenate(
                [left_states[:, STATE_DIMS.index("x")], right_states[:, STATE_DIMS.index("x")]]
            )

    if not all_states:
        raise SystemExit("No selected task state data loaded.")
    all_states_arr = np.concatenate(all_states, axis=0)
    all_state_x = all_states_arr[:, STATE_DIMS.index("x")]
    state_candidates = _build_state_candidates(all_state_x, pair_state_x)

    report = {
        "dataset": str(args.dataset),
        "config": str(args.config) if args.config else None,
        "max_episodes_per_task": args.max_episodes_per_task,
        "task_pairs": [],
        "task_summaries": {},
        "candidate_summary": {
            "action": defaultdict(lambda: defaultdict(list)),
            "state": defaultdict(list),
        },
    }

    for task, data in task_data.items():
        sums = data["action_x_sums"]
        report["task_summaries"][task] = {
            "loaded_episodes": data["loaded_episodes"],
            "num_frames": int(data["actions"].shape[0]),
            "action_summary": _summary(data["actions"], ACTION_DIMS),
            "state_summary": _summary(data["states"], STATE_DIMS),
            "episode_action_x_sum_mean": _mean_or_nan(sums),
            "episode_action_x_sum_std": float(np.std(sums)) if sums else float("nan"),
        }

    for left_task, right_task in PAIR_TASKS:
        left_actions = task_data[left_task]["actions"]
        right_actions = task_data[right_task]["actions"]
        left_states = task_data[left_task]["states"]
        right_states = task_data[right_task]["states"]
        pair_name = f"{left_task}<->{right_task}"

        base_action_dist = _distance_by_dim(left_actions, right_actions, ACTION_DIMS)
        base_state_dist = _distance_by_dim(left_states, right_states, STATE_DIMS)
        pair_report = {
            "pair": pair_name,
            "left_task": left_task,
            "right_task": right_task,
            "left_frames": int(left_actions.shape[0]),
            "right_frames": int(right_actions.shape[0]),
            "base_action_distance": base_action_dist,
            "base_state_distance": base_state_dist,
            "action_candidates": {},
            "state_candidates": {},
        }

        for name, candidate in ACTION_CANDIDATES.items():
            mirrored_left = _apply_action_candidate(left_actions, candidate)
            mirrored_right = _apply_action_candidate(right_actions, candidate)
            left_to_right = _distance_by_dim(mirrored_left, right_actions, ACTION_DIMS)
            right_to_left = _distance_by_dim(mirrored_right, left_actions, ACTION_DIMS)
            x_base = base_action_dist["x"]
            yaw_base = base_action_dist["yaw"]
            roll_base = base_action_dist["roll"]
            x_score = 0.5 * (left_to_right["x"] + right_to_left["x"])
            yaw_score = 0.5 * (left_to_right["yaw"] + right_to_left["yaw"])
            roll_score = 0.5 * (left_to_right["roll"] + right_to_left["roll"])
            result = {
                "transform": candidate,
                "left_to_right_distance": left_to_right,
                "right_to_left_distance": right_to_left,
                "mean_x_distance": x_score,
                "x_improvement_ratio": float((x_base - x_score) / x_base) if x_base > 0 else float("nan"),
                "mean_yaw_distance": yaw_score,
                "yaw_improvement_ratio": float((yaw_base - yaw_score) / yaw_base) if yaw_base > 0 else float("nan"),
                "mean_roll_distance": roll_score,
                "roll_improvement_ratio": float((roll_base - roll_score) / roll_base) if roll_base > 0 else float("nan"),
                "bounds_violation_left_to_right": _bounds_violation(mirrored_left, right_actions, ACTION_DIMS),
                "roundtrip_max_abs_error": _roundtrip_error_action(left_actions, candidate),
            }
            pair_report["action_candidates"][name] = result
            report["candidate_summary"]["action"][name]["x"].append(result["x_improvement_ratio"])
            report["candidate_summary"]["action"][name]["yaw"].append(result["yaw_improvement_ratio"])
            report["candidate_summary"]["action"][name]["roll"].append(result["roll_improvement_ratio"])

        for name, candidate in state_candidates.items():
            if candidate.get("pair_only") not in {None, pair_name}:
                continue
            mirrored_left = _apply_state_candidate(left_states, candidate)
            mirrored_right = _apply_state_candidate(right_states, candidate)
            left_to_right = _distance_by_dim(mirrored_left, right_states, STATE_DIMS)
            right_to_left = _distance_by_dim(mirrored_right, left_states, STATE_DIMS)
            x_base = base_state_dist["x"]
            x_score = 0.5 * (left_to_right["x"] + right_to_left["x"])
            result = {
                "transform": candidate,
                "left_to_right_distance": left_to_right,
                "right_to_left_distance": right_to_left,
                "mean_x_distance": x_score,
                "x_improvement_ratio": float((x_base - x_score) / x_base) if x_base > 0 else float("nan"),
                "bounds_violation_left_to_right": _bounds_violation(mirrored_left, right_states, STATE_DIMS),
                "roundtrip_max_abs_error": _roundtrip_error_state(left_states, candidate),
            }
            pair_report["state_candidates"][name] = result
            report["candidate_summary"]["state"][name].append(result["x_improvement_ratio"])

        report["task_pairs"].append(pair_report)

    report["candidate_summary"]["action"] = {
        name: {
            "mean_x_improvement_ratio": _mean_or_nan(values["x"]),
            "mean_yaw_improvement_ratio": _mean_or_nan(values["yaw"]),
            "mean_roll_improvement_ratio": _mean_or_nan(values["roll"]),
            "per_pair_x": values["x"],
            "per_pair_yaw": values["yaw"],
            "per_pair_roll": values["roll"],
        }
        for name, values in report["candidate_summary"]["action"].items()
    }
    report["candidate_summary"]["state"] = {
        name: {"mean_x_improvement_ratio": _mean_or_nan(values), "per_pair": values}
        for name, values in report["candidate_summary"]["state"].items()
    }

    args.output.mkdir(parents=True, exist_ok=True)
    summary_json = args.output / "summary.json"
    summary_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        "# Left/Right Mirror Candidate Report",
        "",
        f"dataset: `{args.dataset}`",
        f"config: `{args.config}`",
        f"max_episodes_per_task: `{args.max_episodes_per_task}`",
        "",
        "## Action Candidate Summary",
        "",
        "| candidate | mean x improvement | mean yaw improvement | mean roll improvement | per-pair x improvement |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, summary in sorted(report["candidate_summary"]["action"].items()):
        per_pair = ", ".join(f"{value:.3f}" for value in summary["per_pair_x"])
        lines.append(
            f"| `{name}` | {summary['mean_x_improvement_ratio']:.3f} | "
            f"{summary['mean_yaw_improvement_ratio']:.3f} | "
            f"{summary['mean_roll_improvement_ratio']:.3f} | {per_pair} |"
        )
    lines.extend(["", "## State Candidate Summary", "", "| candidate | mean x improvement | per-pair x improvement |", "| --- | ---: | --- |"])
    for name, summary in sorted(report["candidate_summary"]["state"].items()):
        per_pair = ", ".join(f"{value:.3f}" for value in summary["per_pair"])
        lines.append(f"| `{name}` | {summary['mean_x_improvement_ratio']:.3f} | {per_pair} |")
    lines.extend(["", "## Pair Details", ""])
    for pair in report["task_pairs"]:
        lines.append(f"### {pair['pair']}")
        lines.append("")
        lines.append(f"- frames: left={pair['left_frames']}, right={pair['right_frames']}")
        lines.append(f"- base action.x distance: {pair['base_action_distance']['x']:.6f}")
        lines.append(f"- base state.x distance: {pair['base_state_distance']['x']:.6f}")
        best_action = max(
            pair["action_candidates"].items(),
            key=lambda item: item[1]["x_improvement_ratio"],
        )
        best_state = max(
            pair["state_candidates"].items(),
            key=lambda item: item[1]["x_improvement_ratio"],
        )
        lines.append(
            f"- best action candidate by x: `{best_action[0]}` "
            f"({best_action[1]['x_improvement_ratio']:.3f})"
        )
        lines.append(
            f"- best state candidate by x: `{best_state[0]}` "
            f"({best_state[1]['x_improvement_ratio']:.3f})"
        )
        lines.append("")

    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"saved report to {summary_json}")
    print((args.output / "summary.md").read_text())


if __name__ == "__main__":
    main()
