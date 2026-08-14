"""Trend guardrails for per-task Stage-1 reconstruction MAE.

The monitored MAE is collected during the training pass. It is useful for
detecting plateaus and instability, but it is not a held-out validation metric.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any


def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if getter is not None:
        return getter(key, default)
    return cfg[key] if key in cfg else default


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float(0.5 * (ordered[midpoint - 1] + ordered[midpoint]))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def load_baseline_task_mae(path: str | Path, *, expected_tasks: int) -> dict[str, float]:
    baseline_path = Path(path)
    with baseline_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    tasks = payload.get("tasks", {})
    if len(tasks) != expected_tasks:
        raise ValueError(
            f"Baseline {baseline_path} contains {len(tasks)} tasks; expected {expected_tasks}."
        )
    values = {str(name): float(record["mae"]) for name, record in tasks.items()}
    invalid = [name for name, value in values.items() if not math.isfinite(value) or value < 0.0]
    if invalid:
        raise ValueError(f"Baseline {baseline_path} contains invalid task MAE values: {invalid}")
    return values


def _current_task_metrics(
    reconstruction_summary: dict[str, Any],
    *,
    baseline_task_mae: dict[str, float],
    expected_tasks: int,
    expected_close_tasks: int,
    expected_total_samples: int,
    tail_k: int,
) -> tuple[dict[str, float], dict[str, int], dict[str, Any]]:
    task_records = reconstruction_summary.get("tasks", {})
    if len(task_records) != expected_tasks:
        raise RuntimeError(f"Stage-1 MAE guardrail expected {expected_tasks} tasks, found {len(task_records)}.")

    current_names = set(task_records)
    baseline_names = set(baseline_task_mae)
    if current_names != baseline_names:
        missing = sorted(baseline_names - current_names)
        unexpected = sorted(current_names - baseline_names)
        raise RuntimeError(f"Stage-1 task set changed: missing={missing}, unexpected={unexpected}.")

    task_mae: dict[str, float] = {}
    task_counts: dict[str, int] = {}
    for name, record in task_records.items():
        mae = float(record.get("mae", float("nan")))
        count = int(record.get("count", 0))
        if not math.isfinite(mae) or mae < 0.0 or count <= 0:
            raise RuntimeError(f"Invalid Stage-1 task metric for {name}: mae={mae}, count={count}.")
        task_mae[str(name)] = mae
        task_counts[str(name)] = count

    total_samples = sum(task_counts.values())
    if expected_total_samples > 0 and total_samples != expected_total_samples:
        raise RuntimeError(
            f"Stage-1 MAE guardrail expected {expected_total_samples} samples, found {total_samples}."
        )

    close_values = [value for name, value in task_mae.items() if name.endswith("Close")]
    if len(close_values) != expected_close_tasks:
        raise RuntimeError(
            f"Stage-1 MAE guardrail expected {expected_close_tasks} Close tasks, found {len(close_values)}."
        )

    ordered = sorted(task_mae.items(), key=lambda item: item[1], reverse=True)
    tail = ordered[:tail_k]
    values = list(task_mae.values())
    summary = {
        "task_count": len(task_mae),
        "total_samples": total_samples,
        "task_mae_mean": float(sum(values) / len(values)),
        "task_mae_worst": float(ordered[0][1]),
        "task_mae_worst_name": ordered[0][0],
        "task_mae_tail_mean": float(sum(value for _, value in tail) / len(tail)),
        "task_mae_tail_names": [name for name, _ in tail],
        "close_task_mae_mean": float(sum(close_values) / len(close_values)),
    }
    return task_mae, task_counts, summary


def _smoothed_primary_records(
    history: list[dict[str, Any]],
    current_epoch_record: dict[str, Any],
    *,
    start_epoch: int,
    window: int,
) -> list[dict[str, Any]]:
    raw_records = [*history, current_epoch_record]
    usable: list[tuple[int, float, dict[str, Any] | None]] = []
    for record in raw_records:
        value = record.get("task_mae_mean")
        if value is None or not math.isfinite(float(value)):
            continue
        usable.append((int(record["epoch"]), float(value), record.get("task_mae_guardrail")))
    usable.sort(key=lambda item: item[0])

    smoothed: list[dict[str, Any]] = []
    for index, (epoch, raw_value, prior_guardrail) in enumerate(usable):
        if epoch < start_epoch or index + 1 < window:
            continue
        window_values = [value for _, value, _ in usable[index - window + 1 : index + 1]]
        smoothed.append(
            {
                "epoch": epoch,
                "raw": raw_value,
                "smoothed": _median(window_values),
                "promotion_guard_ok": True
                if prior_guardrail is None
                else bool(prior_guardrail.get("promotion_guard_ok", True)),
            }
        )
    return smoothed


def evaluate_stage1_task_mae_guardrail(
    *,
    epoch: int,
    history: list[dict[str, Any]],
    current_epoch_record: dict[str, Any],
    reconstruction_summary: dict[str, Any] | None,
    baseline_task_mae: dict[str, float],
    codebook_usage_ratio: float,
    cfg: Any,
) -> dict[str, Any]:
    if reconstruction_summary is None:
        raise RuntimeError("Stage-1 task MAE guardrail requires reconstruction_by_task statistics.")

    expected_tasks = int(_get(cfg, "expected_tasks", 24))
    expected_close_tasks = int(_get(cfg, "expected_close_tasks", 6))
    expected_total_samples = int(_get(cfg, "expected_total_samples", 0))
    tail_k = int(_get(cfg, "tail_k", 4))
    task_mae, task_counts, current = _current_task_metrics(
        reconstruction_summary,
        baseline_task_mae=baseline_task_mae,
        expected_tasks=expected_tasks,
        expected_close_tasks=expected_close_tasks,
        expected_total_samples=expected_total_samples,
        tail_k=tail_k,
    )

    baseline_ordered = sorted(baseline_task_mae.items(), key=lambda item: item[1], reverse=True)
    baseline_values = list(baseline_task_mae.values())
    baseline_close = [value for name, value in baseline_task_mae.items() if name.endswith("Close")]
    baseline = {
        "task_mae_mean": float(sum(baseline_values) / len(baseline_values)),
        "task_mae_worst": float(baseline_ordered[0][1]),
        "task_mae_tail_mean": float(sum(value for _, value in baseline_ordered[:tail_k]) / tail_k),
        "close_task_mae_mean": float(sum(baseline_close) / len(baseline_close)),
    }

    compare_tolerance = float(_get(cfg, "task_compare_tolerance_rel", 0.001))
    task_ratios = {name: task_mae[name] / baseline_task_mae[name] for name in task_mae}
    improved = sorted(name for name, ratio in task_ratios.items() if ratio < 1.0 - compare_tolerance)
    regressed = sorted(name for name, ratio in task_ratios.items() if ratio > 1.0 + compare_tolerance)
    regressed_gt_5pct = sorted(name for name, ratio in task_ratios.items() if ratio > 1.05)
    worst_relative_name, worst_relative_ratio = max(task_ratios.items(), key=lambda item: item[1])

    tail_regression_rel = float(_get(cfg, "tail_regression_rel", 0.02))
    close_regression_rel = float(_get(cfg, "close_regression_rel", 0.05))
    promotion_guard_ok = (
        current["task_mae_tail_mean"] <= baseline["task_mae_tail_mean"] * (1.0 + tail_regression_rel)
        and current["close_task_mae_mean"] <= baseline["close_task_mae_mean"] * (1.0 + close_regression_rel)
    )

    current_epoch_record["task_mae_mean"] = current["task_mae_mean"]
    current_epoch_record["task_mae_worst"] = current["task_mae_worst"]
    current_epoch_record["task_mae_worst_name"] = current["task_mae_worst_name"]
    start_epoch = int(_get(cfg, "start_epoch", 25))
    smoothing_window = int(_get(cfg, "smoothing_window", 3))
    smoothed_records = _smoothed_primary_records(
        history,
        current_epoch_record,
        start_epoch=start_epoch,
        window=smoothing_window,
    )
    if not smoothed_records:
        raise RuntimeError("Stage-1 task MAE guardrail has no valid smoothed primary records.")
    smoothed_records[-1]["promotion_guard_ok"] = promotion_guard_ok

    min_delta_rel = float(_get(cfg, "min_delta_rel", 0.005))
    baseline_epoch = int(_get(cfg, "baseline_epoch", 47))
    best_score = float(baseline["task_mae_mean"])
    best_epoch = baseline_epoch
    for record in smoothed_records:
        if int(record["epoch"]) <= baseline_epoch or not record["promotion_guard_ok"]:
            continue
        score = float(record["smoothed"])
        if score < best_score * (1.0 - min_delta_rel):
            best_score = score
            best_epoch = int(record["epoch"])

    current_smoothed = float(smoothed_records[-1]["smoothed"])
    epochs_without_improvement = int(epoch) - best_epoch
    patience = int(_get(cfg, "patience", 8))
    plateau_stop = epochs_without_improvement >= patience

    hard_regression_rel = float(_get(cfg, "hard_regression_rel", 0.10))
    hard_regression_patience = int(_get(cfg, "hard_regression_patience", 3))
    hard_threshold = best_score * (1.0 + hard_regression_rel)
    hard_regression_streak = 0
    for record in reversed(smoothed_records):
        if float(record["smoothed"]) <= hard_threshold:
            break
        hard_regression_streak += 1
    hard_regression_stop = hard_regression_streak >= hard_regression_patience

    codebook_min_usage = float(_get(cfg, "codebook_min_usage_ratio", 0.30))
    codebook_failure_patience = int(_get(cfg, "codebook_failure_patience", 2))
    usage_values = [
        float(record["codebook_usage_ratio"])
        for record in [*history, current_epoch_record]
        if record.get("codebook_usage_ratio") is not None
    ]
    codebook_failure_streak = 0
    for value in reversed(usage_values):
        if math.isfinite(value) and value >= codebook_min_usage:
            break
        codebook_failure_streak += 1
    codebook_stop = codebook_failure_streak >= codebook_failure_patience

    stop_reasons: list[str] = []
    if plateau_stop:
        stop_reasons.append(
            f"no >= {min_delta_rel:.2%} smoothed macro-MAE improvement for {epochs_without_improvement} epochs"
        )
    if hard_regression_stop:
        stop_reasons.append(
            f"smoothed macro-MAE stayed > {hard_regression_rel:.1%} above best for "
            f"{hard_regression_streak} evaluations"
        )
    if codebook_stop:
        stop_reasons.append(
            f"codebook usage stayed below {codebook_min_usage:.2%} for {codebook_failure_streak} epochs"
        )

    return {
        "epoch": int(epoch),
        "metric_scope": "online full training pass; not held-out validation",
        **current,
        "baseline_epoch": int(_get(cfg, "baseline_epoch", 47)),
        "baseline_task_mae_mean": baseline["task_mae_mean"],
        "baseline_task_mae_worst": baseline["task_mae_worst"],
        "baseline_task_mae_tail_mean": baseline["task_mae_tail_mean"],
        "baseline_close_task_mae_mean": baseline["close_task_mae_mean"],
        "task_mae_mean_smoothed": current_smoothed,
        "best_smoothed_task_mae_mean": best_score,
        "best_smoothed_epoch": best_epoch,
        "epochs_without_meaningful_improvement": epochs_without_improvement,
        "meaningful_new_best": best_epoch == int(epoch),
        "promotion_guard_ok": promotion_guard_ok,
        "tasks_improved_vs_baseline": len(improved),
        "tasks_regressed_vs_baseline": len(regressed),
        "tasks_regressed_gt_5pct": len(regressed_gt_5pct),
        "improved_task_names": improved,
        "regressed_task_names": regressed,
        "regressed_gt_5pct_task_names": regressed_gt_5pct,
        "worst_relative_regression_task": worst_relative_name,
        "worst_relative_regression_ratio": worst_relative_ratio,
        "codebook_usage_ratio": float(codebook_usage_ratio),
        "hard_regression_streak": hard_regression_streak,
        "codebook_failure_streak": codebook_failure_streak,
        "should_stop": bool(stop_reasons),
        "stop_reasons": stop_reasons,
        "thresholds": {
            "smoothing_window": smoothing_window,
            "min_delta_rel": min_delta_rel,
            "patience": patience,
            "tail_regression_rel": tail_regression_rel,
            "close_regression_rel": close_regression_rel,
            "hard_regression_rel": hard_regression_rel,
            "hard_regression_patience": hard_regression_patience,
            "codebook_min_usage_ratio": codebook_min_usage,
            "codebook_failure_patience": codebook_failure_patience,
        },
        "task_counts": task_counts,
    }


def write_stage1_task_mae_guardrail_outputs(
    *,
    output_dir: str | Path,
    epoch: int,
    history: list[dict[str, Any]],
    reconstruction_summary: dict[str, Any],
    baseline_task_mae: dict[str, float],
    guardrail_record: dict[str, Any],
) -> None:
    root = Path(output_dir)
    _atomic_json(root / "task_mae_guardrail_latest.json", guardrail_record)
    guardrail_history = [
        record["task_mae_guardrail"]
        for record in history
        if isinstance(record.get("task_mae_guardrail"), dict)
    ]
    _atomic_json(root / "task_mae_guardrail_history.json", guardrail_history)

    csv_path = root / f"task_mae_epoch_{epoch:03d}.csv"
    tmp_path = csv_path.with_name(f"{csv_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "task", "count", "mae", "baseline_epoch_047_mae", "delta", "delta_percent"])
        for task_name, record in sorted(reconstruction_summary["tasks"].items()):
            mae = float(record["mae"])
            baseline = float(baseline_task_mae[task_name])
            writer.writerow(
                [
                    int(epoch),
                    task_name,
                    int(record["count"]),
                    f"{mae:.12g}",
                    f"{baseline:.12g}",
                    f"{mae - baseline:.12g}",
                    f"{100.0 * (mae / baseline - 1.0):.8f}",
                ]
            )
    os.replace(tmp_path, csv_path)
