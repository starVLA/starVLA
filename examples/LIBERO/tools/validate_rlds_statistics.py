#!/usr/bin/env python3
"""
Validate modified LIBERO RLDS statistics files for training compatibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REQUIRED_STATS_KEYS = ("mean", "std", "min", "max", "q01", "q99")
DATA_MIX_TO_SUITES = {
    "libero_goal": ["libero_goal_no_noops"],
    "libero_object": ["libero_object_no_noops"],
    "libero_spatial": ["libero_spatial_no_noops"],
    "libero_10": ["libero_10_no_noops"],
    "libero_all": [
        "libero_spatial_no_noops",
        "libero_object_no_noops",
        "libero_goal_no_noops",
        "libero_10_no_noops",
    ],
}


def resolve_suite_names(data_root_dir: Path, data_mix: str) -> List[str]:
    if data_mix in DATA_MIX_TO_SUITES:
        return DATA_MIX_TO_SUITES[data_mix]

    candidates = [item.strip() for item in str(data_mix).split(",") if item.strip()]
    resolved = []
    for name in candidates:
        if (data_root_dir / name).exists():
            resolved.append(name)
        elif (data_root_dir / f"{name}_no_noops").exists():
            resolved.append(f"{name}_no_noops")
        else:
            raise FileNotFoundError(f"Cannot resolve RLDS suite '{name}' under {data_root_dir}")
    if not resolved:
        raise ValueError(f"Empty RLDS data_mix: {data_mix}")
    return resolved


def _extract_nested_stats(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    if isinstance(payload.get("action"), dict):
        return payload
    for value in payload.values():
        if isinstance(value, dict) and isinstance(value.get("action"), dict):
            return value
    return None


def _extract_num_transitions(stats_payload: Dict[str, Any], split: str, suite_dir: Path) -> int:
    for key in (f"num_{split}_transitions", "num_transitions"):
        if key in stats_payload:
            try:
                return int(stats_payload[key])
            except Exception:
                pass

    info_path = suite_dir / "dataset_info.json"
    if not info_path.exists():
        return -1
    try:
        info = json.loads(info_path.read_text())
    except Exception:
        return -1

    splits = info.get("splits", {})
    if isinstance(splits, list):
        for split_info in splits:
            if split_info.get("name") == split and "shardLengths" in split_info:
                try:
                    return int(sum(int(x) for x in split_info["shardLengths"]))
                except Exception:
                    return -1
    if isinstance(splits, dict):
        split_info = splits.get(split) or splits.get("train")
        if isinstance(split_info, dict) and "numExamples" in split_info:
            try:
                return int(split_info["numExamples"])
            except Exception:
                return -1
    return -1


def load_suite_statistics(suite_dir: Path, split: str) -> Dict[str, Any]:
    stats_files = sorted(suite_dir.glob("dataset_statistics*.json"))
    if not stats_files:
        raise FileNotFoundError(f"Missing dataset_statistics*.json under {suite_dir}")

    for stats_file in stats_files:
        try:
            payload = json.loads(stats_file.read_text())
        except Exception:
            continue
        stats = _extract_nested_stats(payload)
        if stats is None:
            continue
        stats["num_transitions"] = _extract_num_transitions(payload, split, suite_dir)
        stats["__source_statistics_file__"] = str(stats_file)
        return stats

    raise ValueError(f"Cannot parse valid statistics file under {suite_dir}")


def _validate_modality_stats(modality: str, stats: Dict[str, Any], expected_dim: int | None) -> List[str]:
    errors: List[str] = []
    if not isinstance(stats, dict):
        return [f"{modality}: expected dict, got {type(stats)}"]

    for key in REQUIRED_STATS_KEYS:
        if key not in stats:
            errors.append(f"{modality}: missing key `{key}`")

    if errors:
        return errors

    lens = {key: len(np.asarray(stats[key]).reshape(-1)) for key in REQUIRED_STATS_KEYS}
    unique_lens = set(lens.values())
    if len(unique_lens) != 1:
        errors.append(f"{modality}: inconsistent lengths {lens}")
        return errors

    dim = next(iter(unique_lens))
    if expected_dim is not None and dim != expected_dim:
        errors.append(f"{modality}: dim mismatch, got {dim}, expected {expected_dim}")

    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    if np.any(q01 > q99):
        errors.append(f"{modality}: found q01 > q99")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root-dir", type=str, required=True)
    parser.add_argument("--data-mix", type=str, default="libero_all")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--expected-action-dim", type=int, default=7)
    parser.add_argument("--expected-proprio-dim", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.data_root_dir)
    suites = resolve_suite_names(root, args.data_mix)
    print(f"Checking suites: {suites}")

    all_errors: List[str] = []
    for suite in suites:
        version_dir = root / suite / "1.0.0"
        suite_dir = version_dir if version_dir.exists() else root / suite
        try:
            stats = load_suite_statistics(suite_dir, split=args.split)
        except Exception as exc:
            all_errors.append(f"{suite}: failed to load statistics: {exc}")
            continue

        source = stats.get("__source_statistics_file__", "unknown")
        action_stats = stats.get("action", {})
        proprio_stats = stats.get("proprio", {})
        num_transitions = stats.get("num_transitions", "unknown")
        print(f"[{suite}] source={source} num_transitions={num_transitions}")

        suite_errors = _validate_modality_stats("action", action_stats, args.expected_action_dim)
        if proprio_stats:
            suite_errors.extend(
                _validate_modality_stats("proprio", proprio_stats, args.expected_proprio_dim)
            )
        else:
            print(f"[{suite}] warning: missing `proprio` stats (state normalization will be unavailable)")

        if suite_errors:
            all_errors.extend([f"{suite}: {err}" for err in suite_errors])
        else:
            print(f"[{suite}] OK")

    if all_errors:
        print("\nValidation failed:")
        for err in all_errors:
            print(f"- {err}")
        sys.exit(1)

    print("\nAll suites passed statistics validation.")


if __name__ == "__main__":
    main()
