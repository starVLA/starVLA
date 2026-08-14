#!/usr/bin/env python3
"""Prepare StarVLA metadata files for the MetaWorld LeRobot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


STATS_FORMAT_VERSION = 2
STATS_CACHE_CONFIG = {"mode": "abs"}
DEFAULT_COLUMNS = ("observation.state", "action")


def _copy_modality(dataset_dir: Path, modality_template: Path) -> Path:
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    output = meta_dir / "modality.json"
    shutil.copyfile(modality_template, output)
    return output


def _stack_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.vstack([np.asarray(value, dtype=np.float32) for value in frame[column]])


def _compute_stats_from_episode_stats(
    dataset_dir: Path,
    columns: tuple[str, ...],
) -> dict[str, dict[str, list[float]]] | None:
    episodes_stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    if not episodes_stats_path.exists():
        return None

    accumulators: dict[str, dict[str, np.ndarray | float]] = {}
    for column in columns:
        accumulators[column] = {
            "count": 0.0,
            "sum": None,
            "sum_sq": None,
            "min": None,
            "max": None,
        }

    with episodes_stats_path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Aggregating MetaWorld episode stats"):
            payload = json.loads(line)
            episode_stats = payload.get("stats", {})
            for column in columns:
                column_stats = episode_stats.get(column)
                if column_stats is None:
                    continue

                count_value = column_stats.get("count", 0)
                count_array = np.asarray(count_value, dtype=np.float64).reshape(-1)
                count = float(count_array[0]) if count_array.size else 0.0
                if count <= 0:
                    continue

                mean = np.asarray(column_stats["mean"], dtype=np.float64)
                std = np.asarray(column_stats["std"], dtype=np.float64)
                min_value = np.asarray(column_stats["min"], dtype=np.float64)
                max_value = np.asarray(column_stats["max"], dtype=np.float64)

                acc = accumulators[column]
                acc["count"] = float(acc["count"]) + count
                next_sum = mean * count
                next_sum_sq = (std**2 + mean**2) * count
                if acc["sum"] is None:
                    acc["sum"] = next_sum
                    acc["sum_sq"] = next_sum_sq
                    acc["min"] = min_value
                    acc["max"] = max_value
                else:
                    acc["sum"] = np.asarray(acc["sum"]) + next_sum
                    acc["sum_sq"] = np.asarray(acc["sum_sq"]) + next_sum_sq
                    acc["min"] = np.minimum(np.asarray(acc["min"]), min_value)
                    acc["max"] = np.maximum(np.asarray(acc["max"]), max_value)

    stats: dict[str, dict[str, list[float]]] = {}
    for column, acc in accumulators.items():
        count = float(acc["count"])
        if count <= 0 or acc["sum"] is None or acc["sum_sq"] is None:
            raise RuntimeError(f"No episode statistics were found for column {column!r}")

        mean = np.asarray(acc["sum"]) / count
        variance = np.asarray(acc["sum_sq"]) / count - mean**2
        std = np.sqrt(np.maximum(variance, 0.0))
        min_value = np.asarray(acc["min"])
        max_value = np.asarray(acc["max"])

        stats[column] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": min_value.tolist(),
            "max": max_value.tolist(),
            # The MetaWorld configs use min_max normalization. q01/q99 are kept
            # for schema compatibility and can be regenerated with the parquet
            # slow path if a q99-normalized config is introduced later.
            "q01": min_value.tolist(),
            "q99": max_value.tolist(),
        }
    return stats


def _compute_stats(dataset_dir: Path, columns: tuple[str, ...]) -> dict[str, dict[str, list[float]]]:
    parquet_paths = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")

    arrays: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    for parquet_path in tqdm(parquet_paths, desc="Reading MetaWorld low-dim columns"):
        frame = pd.read_parquet(parquet_path, columns=list(columns))
        for column in columns:
            arrays[column].append(_stack_column(frame, column))

    stats: dict[str, dict[str, list[float]]] = {}
    for column, chunks in arrays.items():
        values = np.concatenate(chunks, axis=0)
        stats[column] = {
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    return stats


def _write_stats(dataset_dir: Path, stats: dict[str, dict[str, list[float]]]) -> Path:
    output = dataset_dir / "meta" / "stats_gr00t.json"
    payload = {
        "__format_version": STATS_FORMAT_VERSION,
        "__cache_config": STATS_CACHE_CONFIG,
        "statistics": stats,
    }
    tmp_output = output.with_suffix(".tmp")
    with tmp_output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    tmp_output.replace(output)
    return output


def _write_steps_cache(dataset_dir: Path, *, force: bool = False) -> Path:
    output = dataset_dir / "meta" / "steps_data_index.pkl"
    if output.exists() and not force:
        print(f"Keeping existing {output}")
        return output

    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing {episodes_path}")

    steps: list[tuple[int, int]] = []
    num_trajectories = 0
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            num_trajectories += 1
            steps.extend((episode_index, base_index) for base_index in range(length))

    config_dict = {
        "delete_pause_frame": False,
        "dataset_name": dataset_dir.name,
    }
    config_key = hashlib.md5(str(sorted(config_dict.items())).encode()).hexdigest()[:12]
    payload = {
        "config_key": config_key,
        "steps": steps,
        "num_trajectories": num_trajectories,
        "total_steps": len(steps),
        "computed_timestamp": "from_episodes_jsonl",
        "delete_pause_frame": False,
    }

    tmp_output = output.with_suffix(".tmp")
    with tmp_output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_output.replace(output)
    print(f"Wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Path to metaworld_mt50_lerobot")
    parser.add_argument(
        "--modality-template",
        type=Path,
        default=Path(__file__).with_name("modality.json"),
        help="StarVLA modality.json template to copy into dataset_dir/meta",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Only copy modality.json; do not build meta/stats_gr00t.json.",
    )
    parser.add_argument(
        "--force-stats",
        action="store_true",
        help="Recompute meta/stats_gr00t.json even if it already exists.",
    )
    parser.add_argument(
        "--force-parquet-stats",
        action="store_true",
        help="Scan parquet files instead of using meta/episodes_stats.jsonl.",
    )
    parser.add_argument(
        "--skip-steps",
        action="store_true",
        help="Do not build meta/steps_data_index.pkl from meta/episodes.jsonl.",
    )
    parser.add_argument(
        "--force-steps",
        action="store_true",
        help="Rebuild meta/steps_data_index.pkl even if it already exists.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    modality_path = _copy_modality(dataset_dir, args.modality_template.expanduser().resolve())
    print(f"Wrote {modality_path}")

    if not args.skip_steps:
        _write_steps_cache(dataset_dir, force=args.force_steps)

    stats_path = dataset_dir / "meta" / "stats_gr00t.json"
    if args.skip_stats:
        return
    if stats_path.exists() and not args.force_stats:
        print(f"Keeping existing {stats_path}")
        return

    stats = None if args.force_parquet_stats else _compute_stats_from_episode_stats(dataset_dir, DEFAULT_COLUMNS)
    if stats is None:
        stats = _compute_stats(dataset_dir, DEFAULT_COLUMNS)
    output = _write_stats(dataset_dir, stats)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
