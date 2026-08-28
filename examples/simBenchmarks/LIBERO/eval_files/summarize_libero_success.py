#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


TASK_RATE_RE = re.compile(r"Current task success rate:\s*([0-9.]+)")
TOTAL_RATE_RE = re.compile(r"Total success rate:\s*([0-9.]+)")
TASK_ID_RE = re.compile(r"Task id:\s*(\d+)")
EPISODE_RE = re.compile(r"Starting episode\s+(\d+)")
SUCCESS_RE = re.compile(r"Success:\s*(True|False)")
CHECKPOINT_STEP_RE = re.compile(r"steps_(\d+)_")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def parse_rates(log_path: Path) -> tuple[list[float], float | None]:
    text = log_path.read_text(errors="replace")
    task_rates = [float(match.group(1)) for match in TASK_RATE_RE.finditer(text)]
    total_matches = [float(match.group(1)) for match in TOTAL_RATE_RE.finditer(text)]
    return task_rates, total_matches[-1] if total_matches else None


def parse_episode_results(log_path: Path) -> list[tuple[int, int, bool]]:
    results: list[tuple[int, int, bool]] = []
    task_id: int | None = None
    episode: int | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        if match := TASK_ID_RE.search(line):
            task_id = int(match.group(1))
        if match := EPISODE_RE.search(line):
            episode = int(match.group(1)) - 1
        if match := SUCCESS_RE.search(line):
            if task_id is not None and episode is not None:
                results.append((task_id, episode, match.group(1) == "True"))
                episode = None
    return results


def collect_episode_rows(log_root: Path, suites: list[str], require_ok_marker: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for suite in suites:
        suite_dir = log_root / suite
        if not suite_dir.exists():
            continue
        for log_path in sorted(suite_dir.glob("*_chunked_t*_r*_n*.log")):
            text = log_path.read_text(errors="replace")
            if require_ok_marker and "EVAL_CHUNK_OK" not in text:
                continue
            if "Total success rate:" not in text:
                continue
            step_match = CHECKPOINT_STEP_RE.search(log_path.name)
            checkpoint_step = int(step_match.group(1)) if step_match else None
            for task_id, episode_idx, success in parse_episode_results(log_path):
                rows.append(
                    {
                        "suite": suite,
                        "checkpoint_step": checkpoint_step,
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "episode_number": episode_idx + 1,
                        "success": success,
                        "chunk_log": str(log_path.relative_to(log_root)),
                    }
                )
    return sorted(rows, key=lambda row: (SUITES.index(str(row["suite"])), row["task_id"], row["episode_idx"]))


def write_episode_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["suite", "checkpoint_step", "task_id", "episode_idx", "episode_number", "success", "chunk_log"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_episode_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LIBERO per-task success rates from eval logs.")
    parser.add_argument("log_root", type=Path, help="Directory containing libero_* subdirectories with .log files.")
    parser.add_argument("--log-name", default="steps_30000_pytorch_model.log", help="Single-run log filename.")
    parser.add_argument("--chunked", action="store_true", help="Parse chunked logs and aggregate unique task/trial results.")
    parser.add_argument(
        "--require-ok-marker",
        action="store_true",
        help="For chunked logs, ignore chunks that did not finish with EVAL_CHUNK_OK.",
    )
    parser.add_argument("--suite", action="append", choices=SUITES, help="Restrict summary to one suite. Repeatable.")
    parser.add_argument("--episode-csv", type=Path, help="Write per-task/per-episode success results as CSV.")
    parser.add_argument("--episode-jsonl", type=Path, help="Write per-task/per-episode success results as JSONL.")
    args = parser.parse_args()

    suites = args.suite or SUITES
    all_task_rates: list[float] = []

    for suite in suites:
        suite_dir = args.log_root / suite
        if args.chunked:
            by_task: dict[int, dict[int, bool]] = {}
            for log_path in sorted(suite_dir.glob("*_chunked_t*_r*_n*.log")):
                text = log_path.read_text(errors="replace")
                if args.require_ok_marker and "EVAL_CHUNK_OK" not in text:
                    continue
                if "Total success rate:" not in text:
                    continue
                for task_id, episode_idx, success in parse_episode_results(log_path):
                    by_task.setdefault(task_id, {})[episode_idx] = success
            task_rates = []
            for task_id in sorted(by_task):
                trials = by_task[task_id]
                rate = sum(trials.values()) / len(trials)
                task_rates.append(rate)
                print(f"{suite} task {task_id}: {len(trials)} trials, success={rate * 100:.2f}%")
            all_task_rates.extend(task_rates)
            if task_rates:
                task_mean = sum(task_rates) / len(task_rates)
                print(f"{suite}: {len(task_rates)} tasks, task-mean={task_mean * 100:.2f}%")
            else:
                print(f"{suite}: 0 completed tasks")
        else:
            log_path = suite_dir / args.log_name
            if not log_path.exists():
                print(f"{suite}: missing log {log_path}")
                continue

            task_rates, episode_total = parse_rates(log_path)
            all_task_rates.extend(task_rates)
            if task_rates:
                task_mean = sum(task_rates) / len(task_rates)
                total_part = "" if episode_total is None else f", episode-weighted={episode_total * 100:.2f}%"
                print(f"{suite}: {len(task_rates)} tasks, task-mean={task_mean * 100:.2f}%{total_part}")
            else:
                print(f"{suite}: 0 completed tasks")

    if all_task_rates:
        overall = sum(all_task_rates) / len(all_task_rates)
        print(f"overall_40_task_mean: {len(all_task_rates)} tasks, {overall * 100:.2f}%")
    else:
        print("overall_40_task_mean: no completed tasks")

    if args.chunked and (args.episode_csv or args.episode_jsonl):
        episode_rows = collect_episode_rows(args.log_root, suites, args.require_ok_marker)
        if args.episode_csv:
            write_episode_csv(args.episode_csv, episode_rows)
            print(f"episode_csv: {args.episode_csv} ({len(episode_rows)} rows)")
        if args.episode_jsonl:
            write_episode_jsonl(args.episode_jsonl, episode_rows)
            print(f"episode_jsonl: {args.episode_jsonl} ({len(episode_rows)} rows)")


if __name__ == "__main__":
    main()
