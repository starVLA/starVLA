#!/usr/bin/env python3
"""Summarize completed LIBERO evaluation logs as a markdown table.

Usage examples:
  python examples/simBenchmarks/LIBERO/eval_files/auto_eval_scripts/summarize_libero_eval.py \
    /root/nas/feihong/starVLA/Checkpoints/libero_phiwam_agra

  python examples/simBenchmarks/LIBERO/eval_files/auto_eval_scripts/summarize_libero_eval.py \
    /root/nas/feihong/starVLA/Checkpoints/libero_phiwam_agra/checkpoints/steps_40000_pytorch_model.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long / libero_10",
}

STEP_RE = re.compile(r"steps_(\d+)_pytorch_model")
TOTAL_SR_RE = re.compile(r"Total success rate:\s*([0-9.]+)")
TOTAL_EP_RE = re.compile(r"Total episodes:\s*(\d+)")


def resolve_model_roots(input_path: Path) -> list[Path]:
    """Return run/model roots that contain logs and checkpoints."""
    input_path = input_path.expanduser().resolve()

    if input_path.is_file():
        if "/checkpoints/" in str(input_path):
            return [input_path.parents[1]]
        return [input_path.parent]

    if (input_path / "logs").is_dir():
        return [input_path]

    if input_path.name == "checkpoints":
        return [input_path.parent]

    # A parent directory may contain multiple run roots.
    roots = sorted({path.parent for path in input_path.rglob("logs") if path.is_dir()})
    return roots


def checkpoint_filter(input_path: Path) -> set[str] | None:
    """If input is a checkpoint path, restrict output to that checkpoint basename."""
    if input_path.is_file() and STEP_RE.search(input_path.name):
        return {input_path.name}
    return None


def parse_log(log_path: Path) -> tuple[float, int | None] | None:
    """Parse the last total success rate and total episodes from one log file."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    rates = TOTAL_SR_RE.findall(text)
    if not rates:
        return None

    episodes = TOTAL_EP_RE.findall(text)
    return float(rates[-1]), int(episodes[-1]) if episodes else None


def step_sort_key(checkpoint_name: str) -> tuple[int, str]:
    match = STEP_RE.search(checkpoint_name)
    return (int(match.group(1)) if match else -1, checkpoint_name)


def collect_results(root: Path, only_checkpoints: set[str] | None = None) -> dict[str, dict[str, tuple[float, int | None]]]:
    """Collect {checkpoint_name: {suite: (success_rate, episodes)}} from logs."""
    results: dict[str, dict[str, tuple[float, int | None]]] = {}
    logs_root = root / "logs"
    if not logs_root.is_dir():
        return results

    for suite in SUITES:
        suite_dir = logs_root / suite
        if not suite_dir.is_dir():
            continue
        for log_path in sorted(suite_dir.glob("*.log")):
            if "_server_" in log_path.name:
                continue
            parsed = parse_log(log_path)
            if parsed is None:
                continue

            checkpoint_name = log_path.name
            # Logs are named like:
            #   run_checkpoints_steps_40000_pytorch_model.pt.log
            if checkpoint_name.endswith(".log"):
                checkpoint_name = checkpoint_name[:-4]
            checkpoint_name = checkpoint_name.split("_checkpoints_", 1)[-1]

            if only_checkpoints is not None and checkpoint_name not in only_checkpoints:
                continue

            results.setdefault(checkpoint_name, {})[suite] = parsed

    return results


def format_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_table(root: Path, results: dict[str, dict[str, tuple[float, int | None]]]) -> None:
    if not results:
        # print(f"No completed LIBERO evaluation logs found under: {root}")
        return

    print(f"\nResults under: {root}\n")
    headers = ["Checkpoint", *[SUITE_LABELS[s] for s in SUITES], "Avg"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---", *[":---:" for _ in SUITES], ":---:"]) + "|")

    for checkpoint_name in sorted(results, key=step_sort_key):
        suite_results = results[checkpoint_name]
        rates = [suite_results.get(suite, (None, None))[0] for suite in SUITES]
        completed_rates = [rate for rate in rates if rate is not None]
        avg = sum(completed_rates) / len(completed_rates) if completed_rates else None
        print(
            "| "
            + " | ".join([checkpoint_name, *[format_rate(rate) for rate in rates], format_rate(avg)])
            + " |"
        )

    # print("\nSuccess counts:")
    # print("| Checkpoint | Suite | Success / Episodes |")
    # print("|---|---|---:|")
    # for checkpoint_name in sorted(results, key=step_sort_key):
    #     for suite in SUITES:
    #         if suite not in results[checkpoint_name]:
    #             continue
    #         rate, episodes = results[checkpoint_name][suite]
    #         if episodes is None:
    #             count = "-"
    #         else:
    #             count = f"{round(rate * episodes)} / {episodes}"
    #         print(f"| {checkpoint_name} | {suite} | {count} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Checkpoint path, checkpoint directory, run root, or parent directory containing LIBERO eval logs.",
        # default="/root/nas/feihong/starVLA/Checkpoints",
    )
    args = parser.parse_args()

    roots = resolve_model_roots(args.path)
    only_checkpoints = checkpoint_filter(args.path.expanduser().resolve())
    if not roots:
        print(f"No candidate run roots found from: {args.path}")
        return

    for root in roots:
        print_table(root, collect_results(root, only_checkpoints=only_checkpoints))


if __name__ == "__main__":
    main()
