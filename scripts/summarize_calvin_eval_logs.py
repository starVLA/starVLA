#!/usr/bin/env python3
"""Summarize CALVIN parallel eval metrics from worker logs.

This script is intentionally CPU-only and dependency-free.  It reads the logs
written by start_multi_calvin_eval_.sh and reports the current aggregate
metrics, including partially running workers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_BASE = Path("/inspire/qb-ilm2/project/26summer-camp-10/26220056")
DEFAULT_REPO = DEFAULT_BASE / "starVLA"
DEFAULT_RUN_DIR = DEFAULT_BASE / "runs" / "calvin_parallel"
PUBLIC_TEN_LOG = Path("/inspire/qb-ilm2/project/26summer-camp-10/public/ten/log")

WORKER_RE = re.compile(r"calvin_eval_worker(?P<worker>\d+)_gpu(?P<gpu>\d+)_port(?P<port>\d+)\.log$")
TQDM_RE = re.compile(
    r"1/5\s*:\s*(?P<sr1>[0-9.]+)%\s*\|\s*"
    r"2/5\s*:\s*(?P<sr2>[0-9.]+)%\s*\|\s*"
    r"3/5\s*:\s*(?P<sr3>[0-9.]+)%\s*\|\s*"
    r"4/5\s*:\s*(?P<sr4>[0-9.]+)%\s*\|\s*"
    r"5/5\s*:\s*(?P<sr5>[0-9.]+)%.*?"
    r"(?P<done>\d+)\s*/\s*(?P<total>\d+)"
)
FINAL_AVG_RE = re.compile(r"Average successful sequence length:\s*(?P<avg>[0-9.eE+-]+)")
FINAL_CHAIN_RE = re.compile(r"^(?P<idx>[1-5]):\s*(?P<sr>[0-9.]+)%\s*$")
ROLLOUT_RE = re.compile(
    r"\[rollout\]\s+seq=(?P<seq>\d+)\s+sub=(?P<sub>\d+)\s+task=(?P<task>[A-Za-z0-9_]+)\s+step=0\s+begin"
)
RESULTS_RE = re.compile(r"Results for Epoch")
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)|\bERROR\b|ConnectionClosed|TimeoutError")


@dataclass
class WorkerSummary:
    worker: int
    log: str
    gpu: int | None = None
    port: int | None = None
    planned: int | None = None
    completed: int = 0
    status: str = "running"
    chain_counts: list[float] = field(default_factory=lambda: [0.0] * 5)
    chain_sr_percent: list[float] = field(default_factory=lambda: [0.0] * 5)
    avg_len: float = 0.0
    transition_success: Counter[str] = field(default_factory=Counter)
    transition_total: Counter[str] = field(default_factory=Counter)
    pending_transition: str | None = None
    pending_count: int = 0
    has_final_summary: bool = False
    error_seen: bool = False


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(read_text(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def find_latest_log_dir(repo: Path, extra_bases: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for base in [repo / "log", *extra_bases]:
        if not base.exists():
            continue
        for timestamp_dir in base.iterdir():
            calvin_dir = timestamp_dir / "calvin"
            if timestamp_dir.is_dir() and timestamp_dir.name.isdigit() and calvin_dir.exists():
                candidates.append(calvin_dir)
    return max(candidates, key=lambda p: p.parent.name) if candidates else None


def discover_logs(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("calvin_eval_worker*_gpu*_port*.log"), key=lambda p: worker_id_from_path(p)[0])


def worker_id_from_path(path: Path) -> tuple[int, int, int]:
    match = WORKER_RE.search(path.name)
    if not match:
        return (10**9, -1, -1)
    return (int(match.group("worker")), int(match.group("gpu")), int(match.group("port")))


def load_planned_counts(split_dir: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    manifest = load_json(split_dir / "manifest.json")
    if isinstance(manifest, list):
        for item in manifest:
            try:
                counts[int(item["worker"])] = int(item["count"])
            except (KeyError, TypeError, ValueError):
                pass

    for count_file in split_dir.glob("eval_sequences_worker_*.count"):
        match = re.search(r"worker_(\d+)\.count$", count_file.name)
        if not match:
            continue
        try:
            counts[int(match.group(1))] = int(read_text(count_file).strip())
        except ValueError:
            pass
    return counts


def transition_name(prev_task: str | None, task: str) -> str:
    return f"{prev_task or 'START'}->{task}"


def rollout_events(text: str) -> list[tuple[int, int, str]]:
    events: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for match in ROLLOUT_RE.finditer(text):
        event = (int(match.group("seq")), int(match.group("sub")), match.group("task"))
        key = (event[0], event[1])
        if key in seen:
            continue
        seen.add(key)
        events.append(event)
    return events


def infer_transitions_from_rollouts(text: str) -> tuple[Counter[str], Counter[str], str | None, int]:
    """Infer A->B transition outcomes from subtask starts.

    CALVIN evaluates a sequence until the first failed subtask.  Therefore, when
    the log starts a later subtask in the same sequence, the previous subtask
    succeeded.  At a sequence boundary, the previous subtask failed unless that
    worker has already printed the final summary.
    """
    events = rollout_events(text)
    if not events:
        return Counter(), Counter(), None, 0

    success: Counter[str] = Counter()
    total: Counter[str] = Counter()
    tasks_by_seq_sub = {(seq, sub): task for seq, sub, task in events}
    has_final_summary = bool(RESULTS_RE.search(text))

    pending_transition = None
    pending_count = 0
    for idx, (seq, sub, task) in enumerate(events):
        prev_task = tasks_by_seq_sub.get((seq, sub - 1)) if sub > 0 else None
        trans = transition_name(prev_task, task)

        if idx + 1 < len(events):
            next_seq, next_sub, _ = events[idx + 1]
            if next_seq == seq and next_sub == sub + 1:
                known = True
                solved = True
            elif next_seq > seq:
                known = True
                solved = sub == 4
            else:
                known = False
                solved = False
        elif has_final_summary:
            known = True
            solved = sub == 4
        else:
            known = False
            solved = False

        if known:
            total[trans] += 1
            if solved:
                success[trans] += 1
        else:
            pending_transition = trans
            pending_count = 1

    return success, total, pending_transition, pending_count


def parse_worker_log(log_path: Path, planned: int | None) -> WorkerSummary:
    worker, gpu, port = worker_id_from_path(log_path)
    summary = WorkerSummary(worker=worker, gpu=gpu if gpu >= 0 else None, port=port if port >= 0 else None, log=str(log_path), planned=planned)
    text = read_text(log_path)
    summary.error_seen = bool(TRACEBACK_RE.search(text))

    tqdm_matches = list(TQDM_RE.finditer(text))
    if tqdm_matches:
        last = tqdm_matches[-1]
        summary.completed = int(last.group("done"))
        # Prefer the total embedded in the log.  The shared eval_splits
        # directory is overwritten by later runs, so it may not match old logs.
        summary.planned = int(last.group("total"))
        summary.chain_sr_percent = [float(last.group(f"sr{i}")) for i in range(1, 6)]
        summary.chain_counts = [summary.completed * sr / 100.0 for sr in summary.chain_sr_percent]
        summary.avg_len = sum(summary.chain_sr_percent) / 100.0

    final_avg_match = FINAL_AVG_RE.search(text)
    if final_avg_match:
        summary.avg_len = float(final_avg_match.group("avg"))
        summary.has_final_summary = True

    final_chain: dict[int, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        chain_match = FINAL_CHAIN_RE.match(line)
        if chain_match:
            final_chain[int(chain_match.group("idx"))] = float(chain_match.group("sr"))
            continue

    if len(final_chain) == 5:
        summary.chain_sr_percent = [final_chain[i] for i in range(1, 6)]
        if summary.completed == 0 and summary.planned is not None:
            summary.completed = summary.planned
        summary.chain_counts = [summary.completed * sr / 100.0 for sr in summary.chain_sr_percent]
        summary.has_final_summary = True

    inferred_success, inferred_total, pending_transition, pending_count = infer_transitions_from_rollouts(text)
    summary.transition_success = inferred_success
    summary.transition_total = inferred_total
    summary.pending_transition = pending_transition
    summary.pending_count = pending_count

    if summary.has_final_summary:
        summary.status = "finished"
    elif summary.error_seen:
        summary.status = "error_or_stopped"
    elif summary.completed > 0:
        summary.status = "running"
    else:
        summary.status = "starting"

    return summary


def summarize(log_dir: Path, split_dir: Path) -> dict[str, Any]:
    planned_counts = load_planned_counts(split_dir)
    workers = []
    total_completed = 0
    total_planned = 0
    chain_counts = [0.0] * 5
    transition_success: Counter[str] = Counter()
    transition_total: Counter[str] = Counter()
    pending_transitions: Counter[str] = Counter()

    for log_path in discover_logs(log_dir):
        worker_id, _, _ = worker_id_from_path(log_path)
        worker = parse_worker_log(log_path, planned_counts.get(worker_id))
        workers.append(worker)
        if worker.planned is not None:
            total_planned += worker.planned
        total_completed += worker.completed
        for idx, value in enumerate(worker.chain_counts):
            chain_counts[idx] += value
        transition_success.update(worker.transition_success)
        transition_total.update(worker.transition_total)
        if worker.pending_transition and worker.pending_count:
            pending_transitions[worker.pending_transition] += worker.pending_count

    chain_sr_percent = [(count / total_completed * 100.0) if total_completed else 0.0 for count in chain_counts]
    avg_len = sum(chain_counts) / total_completed if total_completed else 0.0
    transition_items = {
        transition: {
            "success": int(transition_success[transition]),
            "total": int(transition_total[transition]),
            "sr_percent": (transition_success[transition] / transition_total[transition] * 100.0) if transition_total[transition] else 0.0,
            "pending": int(pending_transitions.get(transition, 0)),
        }
        for transition in transition_total
    }
    for transition in set(pending_transitions) - set(transition_items):
        transition_items[transition] = {"success": 0, "total": 0, "sr_percent": 0.0, "pending": int(pending_transitions[transition])}
    transitions = dict(
        sorted(
            transition_items.items(),
            key=lambda item: (item[1]["sr_percent"], -item[1]["total"], item[0]),
        )
    )

    return {
        "log_dir": str(log_dir),
        "split_dir": str(split_dir),
        "total_completed": total_completed,
        "total_planned": total_planned or None,
        "avg_len": avg_len,
        "chain_sr_percent": {f"{idx}/5": chain_sr_percent[idx - 1] for idx in range(1, 6)},
        "transitions": transitions,
        "workers": [worker_to_dict(worker) for worker in workers],
    }


def worker_to_dict(worker: WorkerSummary) -> dict[str, Any]:
    return {
        "worker": worker.worker,
        "gpu": worker.gpu,
        "port": worker.port,
        "status": worker.status,
        "completed": worker.completed,
        "planned": worker.planned,
        "avg_len": worker.avg_len,
        "chain_sr_percent": {f"{idx}/5": worker.chain_sr_percent[idx - 1] for idx in range(1, 6)},
        "pending_transition": worker.pending_transition,
        "error_seen": worker.error_seen,
        "log": worker.log,
    }


def print_report(data: dict[str, Any], include_pending: bool) -> None:
    planned = data["total_planned"]
    progress = f"{data['total_completed']}/{planned}" if planned else str(data["total_completed"])
    print(f"Log dir: {data['log_dir']}")
    print(f"Progress: {progress} sequences")
    print(f"Avg len: {data['avg_len']:.4f}")
    print(
        "Chain SR: "
        + " | ".join(f"{key}: {value:.1f}%" for key, value in data["chain_sr_percent"].items())
    )
    print()
    print("Workers:")
    print("  worker  gpu  port   status            progress   avg_len   1/5    2/5    3/5    4/5    5/5")
    for worker in data["workers"]:
        worker_planned = worker["planned"]
        worker_progress = f"{worker['completed']}/{worker_planned}" if worker_planned is not None else str(worker["completed"])
        chain = worker["chain_sr_percent"]
        print(
            f"  {worker['worker']:>6}  {str(worker['gpu']):>3}  {str(worker['port']):>5}  "
            f"{worker['status']:<16}  {worker_progress:>8}  {worker['avg_len']:>8.4f}  "
            f"{chain['1/5']:>5.1f}  {chain['2/5']:>5.1f}  {chain['3/5']:>5.1f}  "
            f"{chain['4/5']:>5.1f}  {chain['5/5']:>5.1f}"
        )

    if data["transitions"]:
        print()
        print("Transitions:")
        pending_header = "  pending" if include_pending else ""
        print(f"  A->B                                              success/total      Rate{pending_header}")
        for transition, item in data["transitions"].items():
            pending = f"  {item['pending']:>7}" if include_pending else ""
            print(
                f"  {transition:<48}  {item['success']:>5}/{item['total']:<5}  "
                f"{item['sr_percent']:>6.1f}%{pending}"
            )


def clear_screen() -> None:
    if sys.stdout.isatty():
        os.system("clear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=None, help="CALVIN worker log dir, e.g. log/<timestamp>/calvin")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="starVLA repo path")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_RUN_DIR / "eval_splits", help="parallel eval split dir")
    parser.add_argument(
        "--extra-log-base",
        type=Path,
        action="append",
        default=[PUBLIC_TEN_LOG],
        help="additional base directory containing <timestamp>/calvin logs; can be repeated",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of a text table")
    parser.add_argument("--output", type=Path, default=None, help="write JSON summary to this path")
    parser.add_argument("--watch", type=float, default=0.0, help="refresh every N seconds")
    parser.add_argument("--include-pending", action="store_true", help="show currently running task attempts in task table")
    return parser.parse_args()


def resolve_log_dir(args: argparse.Namespace) -> Path:
    if args.log_dir is not None:
        return args.log_dir
    latest = find_latest_log_dir(args.repo, args.extra_log_base)
    if latest is None:
        raise SystemExit("ERROR: no log directory found. Pass --log-dir explicitly.")
    return latest


def run_once(args: argparse.Namespace, log_dir: Path) -> dict[str, Any]:
    data = summarize(log_dir, args.split_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print_report(data, args.include_pending)
    return data


def main() -> None:
    args = parse_args()
    log_dir = resolve_log_dir(args)
    if not log_dir.exists():
        raise SystemExit(f"ERROR: log dir not found: {log_dir}")

    if args.watch and args.json:
        raise SystemExit("ERROR: --watch and --json are not useful together; use table output for watch mode.")

    while True:
        clear_screen()
        run_once(args, log_dir)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
