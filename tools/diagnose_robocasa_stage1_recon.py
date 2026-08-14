#!/usr/bin/env python3
"""Diagnose RoboCasa-GR1 VAR stage1 reconstruction error by task and phase."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from starVLA.dataloader.var_stage1_action_dataset import VARStage1ActionDataset
from starVLA.model.modules.action_tokenizer import load_frozen_var_action_tokenizer
from starVLA.training.train_var_stage1 import collate_action_batch, load_starvla_base_config


DIM_GROUPS = {
    "left_arm": list(range(0, 7)),
    "right_arm": list(range(7, 14)),
    "left_hand": list(range(14, 20)),
    "right_hand": list(range(20, 26)),
    "waist": list(range(26, 29)),
    "arms": list(range(0, 14)),
    "hands": list(range(14, 26)),
}

TIME_GROUPS = {
    "t00_03": list(range(0, 4)),
    "t04_07": list(range(4, 8)),
    "t08_11": list(range(8, 12)),
    "t12_15": list(range(12, 16)),
}


@dataclass
class Moments:
    n: int = 0
    sum_sq: float = 0.0
    sum_abs: float = 0.0
    max_abs: float = 0.0

    def update(self, err: np.ndarray) -> None:
        if err.size == 0:
            return
        abs_err = np.abs(err)
        self.n += int(err.size)
        self.sum_sq += float(np.square(err, dtype=np.float64).sum())
        self.sum_abs += float(abs_err.sum(dtype=np.float64))
        local_max = float(abs_err.max())
        if local_max > self.max_abs:
            self.max_abs = local_max

    def to_dict(self) -> dict[str, float | int]:
        mse = self.sum_sq / self.n if self.n else float("nan")
        mae = self.sum_abs / self.n if self.n else float("nan")
        rmse = math.sqrt(mse) if self.n else float("nan")
        return {
            "n_values": self.n,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "max_abs": self.max_abs,
        }


@dataclass
class Bucket:
    sample_count: int = 0
    action: Moments = field(default_factory=Moments)
    velocity: Moments = field(default_factory=Moments)
    dim_groups: dict[str, Moments] = field(default_factory=lambda: defaultdict(Moments))
    time_groups: dict[str, Moments] = field(default_factory=lambda: defaultdict(Moments))
    phase: dict[str, Moments] = field(default_factory=lambda: defaultdict(Moments))
    token_hist: Counter[int] = field(default_factory=Counter)


def simplify_task_name(dataset_name: str) -> str:
    name = dataset_name.split(".", 1)[-1]
    suffixes = [
        "_GR1ArmsAndWaistFourierHands_1000",
        "_GR1ArmsAndWaistFourierHands",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.startswith("Posttrain"):
        name = name[len("Posttrain") :]
    return name


def task_type(task_name: str) -> str:
    return "close" if task_name.startswith("PnP") and task_name.endswith("Close") else "pnp"


def trajectory_phase(base_index: int, trajectory_length: int, horizon: int) -> str:
    max_start = max(int(trajectory_length) - int(horizon), 1)
    ratio = float(base_index) / float(max_start)
    if ratio < 0.25:
        return "traj_q1"
    if ratio < 0.50:
        return "traj_q2"
    if ratio < 0.75:
        return "traj_q3"
    return "traj_q4"


def find_task_ranges(full_windows: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    if not full_windows:
        return ranges
    start = 0
    current = int(full_windows[0][0])
    for idx, (dataset_index, _, _) in enumerate(full_windows):
        dataset_index = int(dataset_index)
        if dataset_index != current:
            ranges.append((current, start, idx))
            current = dataset_index
            start = idx
    ranges.append((current, start, len(full_windows)))
    return ranges


def select_indices(
    task_ranges: list[tuple[int, int, int]],
    *,
    samples_per_task: int,
    seed: int,
    strategy: str,
) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for _, start, end in task_ranges:
        count = end - start
        if samples_per_task <= 0 or samples_per_task >= count:
            task_indices = np.arange(start, end, dtype=np.int64)
        elif strategy == "random":
            task_indices = np.sort(rng.choice(np.arange(start, end, dtype=np.int64), size=samples_per_task, replace=False))
        else:
            task_indices = np.linspace(start, end - 1, num=samples_per_task, dtype=np.int64)
        selected.extend(int(i) for i in task_indices)
    return selected


def make_length_maps(dataset: VARStage1ActionDataset) -> dict[int, dict[int, int]]:
    maps: dict[int, dict[int, int]] = {}
    for dataset_index, source in enumerate(dataset.source_dataset.datasets):
        maps[dataset_index] = {
            int(traj_id): int(length)
            for traj_id, length in zip(source.trajectory_ids, source.trajectory_lengths, strict=False)
        }
    return maps


def update_bucket(
    bucket: Bucket,
    *,
    err: np.ndarray,
    vel_err: np.ndarray,
    phase_name: str,
    tokens: np.ndarray,
) -> None:
    bucket.sample_count += 1
    bucket.action.update(err)
    bucket.velocity.update(vel_err)
    bucket.phase[phase_name].update(err)
    for group_name, dims in DIM_GROUPS.items():
        bucket.dim_groups[group_name].update(err[:, dims])
    for group_name, times in TIME_GROUPS.items():
        bucket.time_groups[group_name].update(err[times, :])
    bucket.token_hist.update(int(x) for x in tokens.reshape(-1))


def metric_row(name: str, bucket: Bucket) -> dict[str, Any]:
    row = {
        "name": name,
        "sample_count": bucket.sample_count,
        "token_unique": len(bucket.token_hist),
    }
    row.update(bucket.action.to_dict())
    vel = bucket.velocity.to_dict()
    row.update({f"velocity_{key}": value for key, value in vel.items()})
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=Path("examples/simBenchmarks/Robocasa_tabletop/train_files/train_var_stage1_robocasa_gr1_e128_aeinit_productvq_g16_s1_2_4_8_16_current_from_latest_epoch032.yaml"),
    )
    parser.add_argument(
        "--stage1-artifact",
        type=Path,
        default=Path("/root/nas/feihong/starVLA/Checkpoints/var_stage1_robocasa_gr1_e128_productvq_resume_local_from_epoch016_mirror/best_recon.ckpt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/feihong/starVLA/stage1_diagnostics/robocasa_e128_best_recon"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--samples-per-task", type=int, default=2048)
    parser.add_argument("--sample-strategy", choices=["linspace", "random"], default="linspace")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config_yaml)
    base_cfg = load_starvla_base_config(cfg)
    dataset = VARStage1ActionDataset(
        base_cfg,
        mode="train",
        balance_dataset_weights=bool(cfg.data.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(cfg.data.get("balance_trajectory_weights", False)),
        seed=int(cfg.experiment.get("seed", args.seed)),
        return_raw_actions=True,
        window_mode=str(cfg.data.get("window_mode", "full")),
    )

    task_ranges = find_task_ranges(dataset._full_windows)
    selected_indices = select_indices(
        task_ranges,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        strategy=args.sample_strategy,
    )
    length_maps = make_length_maps(dataset)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    artifact = load_frozen_var_action_tokenizer(args.stage1_artifact, device=device)
    tokenizer = artifact.tokenizer
    tokenizer.eval()

    subset = Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_action_batch,
        persistent_workers=args.num_workers > 0,
    )

    task_buckets: dict[str, Bucket] = defaultdict(Bucket)
    type_buckets: dict[str, Bucket] = defaultdict(Bucket)
    overall = Bucket()
    dataset_index_to_task: dict[int, str] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="stage1 recon diagnostic"):
            actions = batch["actions"].to(device=device, dtype=torch.float32, non_blocking=True)
            output = tokenizer(actions)
            recon = output["recon"]
            flat_tokens = output["flat_token_ids"].detach().cpu().numpy()

            err = (recon - actions).detach().cpu().numpy()
            vel_err = (
                (recon[:, 1:, :] - recon[:, :-1, :]) - (actions[:, 1:, :] - actions[:, :-1, :])
            ).detach().cpu().numpy()

            for sample_idx, metadata in enumerate(batch["metadata"]):
                dataset_index = int(metadata.get("dataset_index", -1))
                task_name = simplify_task_name(str(metadata["dataset_name"]))
                dataset_index_to_task[dataset_index] = task_name
                phase_name = trajectory_phase(
                    int(metadata["base_index"]),
                    length_maps[dataset_index][int(metadata["trajectory_id"])],
                    int(dataset.action_spec.horizon),
                )
                sample_err = err[sample_idx]
                sample_vel_err = vel_err[sample_idx]
                sample_tokens = flat_tokens[sample_idx]
                update_bucket(
                    task_buckets[task_name],
                    err=sample_err,
                    vel_err=sample_vel_err,
                    phase_name=phase_name,
                    tokens=sample_tokens,
                )
                update_bucket(
                    type_buckets[task_type(task_name)],
                    err=sample_err,
                    vel_err=sample_vel_err,
                    phase_name=phase_name,
                    tokens=sample_tokens,
                )
                update_bucket(
                    overall,
                    err=sample_err,
                    vel_err=sample_vel_err,
                    phase_name=phase_name,
                    tokens=sample_tokens,
                )

    task_rows = [metric_row(task, bucket) for task, bucket in sorted(task_buckets.items())]
    type_rows = [metric_row(name, bucket) for name, bucket in sorted(type_buckets.items())]
    overall_row = metric_row("overall", overall)

    dim_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    for task, bucket in sorted(task_buckets.items()):
        for group_name, moments in sorted(bucket.dim_groups.items()):
            dim_rows.append({"task": task, "group": group_name, **moments.to_dict()})
        for phase_name, moments in sorted(bucket.phase.items()):
            phase_rows.append({"task": task, "phase": phase_name, **moments.to_dict()})
        for time_name, moments in sorted(bucket.time_groups.items()):
            time_rows.append({"task": task, "time_group": time_name, **moments.to_dict()})

    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "elapsed_seconds": time.time() - start_time,
        "dataset_len": len(dataset),
        "selected_len": len(selected_indices),
        "device": str(device),
        "model_config": artifact.checkpoint["model_config"],
        "action_spec": artifact.checkpoint["action_spec"],
        "task_ranges": [
            {
                "dataset_index": dataset_index,
                "task": dataset_index_to_task.get(dataset_index, str(dataset_index)),
                "start": start,
                "end": end,
                "count": end - start,
            }
            for dataset_index, start, end in task_ranges
        ],
        "overall": overall_row,
        "by_type": type_rows,
        "by_task": task_rows,
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(args.output_dir / "per_task_summary.csv", task_rows)
    write_csv(args.output_dir / "per_type_summary.csv", type_rows)
    write_csv(args.output_dir / "per_task_dim_group.csv", dim_rows)
    write_csv(args.output_dir / "per_task_traj_phase.csv", phase_rows)
    write_csv(args.output_dir / "per_task_time_group.csv", time_rows)

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "selected_len": len(selected_indices),
        "overall_mse": overall_row["mse"],
        "by_type": type_rows,
        "top_task_mse": sorted(task_rows, key=lambda row: float(row["mse"]), reverse=True)[:8],
    }, indent=2))


if __name__ == "__main__":
    main()
