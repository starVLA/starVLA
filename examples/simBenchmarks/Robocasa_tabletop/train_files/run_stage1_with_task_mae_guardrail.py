#!/usr/bin/env python3
"""Run Stage-1 and stop it when the 24-task MAE guardrail says to stop."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Any

from omegaconf import OmegaConf

from starVLA.training.stage1_task_mae_guardrail import (
    evaluate_stage1_task_mae_guardrail,
    load_baseline_task_mae,
    write_stage1_task_mae_guardrail_outputs,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _terminate_process_group(child: subprocess.Popen[Any], *, reason: str) -> None:
    if child.poll() is not None:
        return
    print(f"[guardrail] stopping Stage-1 process group: {reason}", flush=True)
    for sig, timeout in ((signal.SIGINT, 60.0), (signal.SIGTERM, 30.0), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def _guardrail_records(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    return {int(record["epoch"]): record for record in payload}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--poll_seconds", type=float, default=20.0)
    parser.add_argument("--checkpoint_settle_seconds", type=float, default=5.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    config_path = Path(args.config_yaml)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    cfg = OmegaConf.load(config_path)
    output_dir = Path(str(cfg.experiment.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    guardrail_cfg = cfg.train.get("task_mae_guardrail", None)
    if not guardrail_cfg or not bool(guardrail_cfg.get("enabled", False)):
        raise ValueError("train.task_mae_guardrail.enabled must be true for guarded execution.")
    baseline_epoch = int(guardrail_cfg.get("baseline_epoch", 47))
    baseline_path = Path(str(guardrail_cfg.baseline_task_metrics_path))
    baseline_task_mae = load_baseline_task_mae(
        baseline_path,
        expected_tasks=int(guardrail_cfg.get("expected_tasks", 24)),
    )

    stopped_marker = output_dir / "STOPPED_EARLY.json"
    if stopped_marker.exists():
        print(f"[guardrail] refusing to restart a stopped run: {stopped_marker}", file=sys.stderr)
        return 2

    guardrail_history_path = output_dir / "task_mae_guardrail_history.json"
    decisions = _guardrail_records(guardrail_history_path)
    print(
        f"[guardrail] armed: baseline_epoch={baseline_epoch}, tasks={len(baseline_task_mae)}, "
        f"existing_guarded_epochs={sorted(decisions)}",
        flush=True,
    )

    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child = subprocess.Popen(
        [
            sys.executable,
            str(repo_root / "starVLA/training/train_var_stage1.py"),
            "--config_yaml",
            str(config_path),
        ],
        cwd=repo_root,
        env=child_env,
        start_new_session=True,
    )
    _atomic_json(
        output_dir / "process.json",
        {
            "monitor_pid": os.getpid(),
            "training_pid": child.pid,
            "config_yaml": str(config_path),
            "started_at_unix": time.time(),
        },
    )

    received_signal: int | None = None

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signum

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    def process_ready_epochs() -> bool:
        history_path = output_dir / "history.json"
        if not history_path.exists():
            return False
        history: list[dict[str, Any]] = _load_json(history_path)
        for current in sorted(history, key=lambda record: int(record["epoch"])):
            epoch = int(current["epoch"])
            if epoch <= baseline_epoch or epoch in decisions:
                continue

            checkpoint_path = output_dir / f"epoch_{epoch:03d}.ckpt"
            reconstruction_path = output_dir / f"reconstruction_by_task_epoch_{epoch:03d}.json"
            if not checkpoint_path.exists() or not reconstruction_path.exists():
                break
            if time.time() - checkpoint_path.stat().st_mtime < args.checkpoint_settle_seconds:
                break

            reconstruction_summary = _load_json(reconstruction_path)
            prior_history: list[dict[str, Any]] = []
            for record in sorted(history, key=lambda item: int(item["epoch"])):
                record_epoch = int(record["epoch"])
                if record_epoch >= epoch:
                    break
                copied = dict(record)
                if record_epoch in decisions:
                    copied["task_mae_guardrail"] = decisions[record_epoch]
                prior_history.append(copied)

            current_copy = dict(current)
            decision = evaluate_stage1_task_mae_guardrail(
                epoch=epoch,
                history=prior_history,
                current_epoch_record=current_copy,
                reconstruction_summary=reconstruction_summary,
                baseline_task_mae=baseline_task_mae,
                codebook_usage_ratio=float(current["codebook_usage_ratio"]),
                cfg=guardrail_cfg,
            )
            current_copy["task_mae_guardrail"] = decision
            decisions[epoch] = decision
            history_for_output = [
                *prior_history,
                current_copy,
            ]
            write_stage1_task_mae_guardrail_outputs(
                output_dir=output_dir,
                epoch=epoch,
                history=history_for_output,
                reconstruction_summary=reconstruction_summary,
                baseline_task_mae=baseline_task_mae,
                guardrail_record=decision,
            )

            print(
                f"[guardrail] epoch={epoch} macro={decision['task_mae_mean']:.9f} "
                f"median3={decision['task_mae_mean_smoothed']:.9f} "
                f"worst={decision['task_mae_worst']:.9f} "
                f"tail4={decision['task_mae_tail_mean']:.9f} "
                f"close={decision['close_task_mae_mean']:.9f} "
                f"improved_vs_e47={decision['tasks_improved_vs_baseline']}/24 "
                f"patience_used={decision['epochs_without_meaningful_improvement']}/"
                f"{decision['thresholds']['patience']}",
                flush=True,
            )
            for task_name, record in sorted(reconstruction_summary["tasks"].items()):
                current_mae = float(record["mae"])
                baseline_mae = float(baseline_task_mae[task_name])
                print(
                    f"[guardrail][epoch={epoch:03d}] {task_name} "
                    f"mae={current_mae:.9f} delta_vs_e47={100.0 * (current_mae / baseline_mae - 1.0):+.3f}%",
                    flush=True,
                )

            if bool(decision["should_stop"]):
                _atomic_json(
                    stopped_marker,
                    {
                        "stopped_early": True,
                        "epoch": epoch,
                        "reasons": decision["stop_reasons"],
                        "last_complete_checkpoint": str(checkpoint_path),
                        "guardrail": decision,
                    },
                )
                _terminate_process_group(child, reason="; ".join(decision["stop_reasons"]))
                return True
        return False

    stopped_by_guardrail = False
    try:
        while child.poll() is None:
            if received_signal is not None:
                _terminate_process_group(child, reason=f"monitor received signal {received_signal}")
                return 128 + received_signal
            try:
                if process_ready_epochs():
                    stopped_by_guardrail = True
                    break
            except Exception as exc:
                error_payload = {
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "time_unix": time.time(),
                }
                _atomic_json(output_dir / "MONITOR_ERROR.json", error_payload)
                _terminate_process_group(child, reason=f"guardrail monitor error: {exc!r}")
                raise
            time.sleep(args.poll_seconds)

        if not stopped_by_guardrail:
            process_ready_epochs()
        return_code = child.poll()
        if stopped_by_guardrail:
            return 0
        if return_code == 0:
            _atomic_json(
                output_dir / "COMPLETED.json",
                {
                    "completed": True,
                    "last_guarded_epoch": max(decisions) if decisions else baseline_epoch,
                    "completed_at_unix": time.time(),
                },
            )
        return int(return_code or 0)
    finally:
        if child.poll() is None:
            _terminate_process_group(child, reason="guardrail monitor exiting")


if __name__ == "__main__":
    raise SystemExit(main())
