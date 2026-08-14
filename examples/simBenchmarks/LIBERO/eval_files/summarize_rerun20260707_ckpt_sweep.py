#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DEFAULT_RUN_IDS = [
    "qwen_var_productvq_g16_s1248_structemb_nextscale_100k_rerun20260707_A_baseline_lr2p5e5_warmup3k_ls002",
    "qwen_var_productvq_g16_s1248_structemb_nextscale_100k_rerun20260707_B_stable_lr2e5_warmup5k_ls002",
    "qwen_var_productvq_g16_s1248_structemb_nextscale_100k_rerun20260707_C_aggressive_lr3e5_warmup3k_ls002",
    "qwen_var_productvq_g16_s1248_structemb_nextscale_100k_rerun20260707_D_ls001_lr2p5e5_warmup3k",
]
DEFAULT_STEPS = list(range(26000, 40001, 1000))
SUITE_RE = re.compile(r"^(libero_[a-z0-9_]+):\s+(\d+) tasks, task-mean=([0-9.]+)%")
OVERALL_RE = re.compile(r"^overall_40_task_mean:\s+(\d+) tasks, ([0-9.]+)%")


def parse_summary(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"status": "missing"}
    if not path.exists():
        return result
    text = path.read_text(errors="replace")
    suite_rates: dict[str, float] = {}
    suite_task_counts: dict[str, int] = {}
    overall = None
    overall_tasks = 0
    for line in text.splitlines():
        if match := SUITE_RE.search(line):
            suite = match.group(1)
            suite_task_counts[suite] = int(match.group(2))
            suite_rates[suite] = float(match.group(3))
        elif match := OVERALL_RE.search(line):
            overall_tasks = int(match.group(1))
            overall = float(match.group(2))
    result.update(
        {
            "status": "complete" if overall_tasks == 40 and all(suite_task_counts.get(s, 0) == 10 for s in SUITES) else "partial",
            "overall_40_task_mean": overall,
            "overall_tasks": overall_tasks,
        }
    )
    for suite in SUITES:
        result[f"{suite}_task_mean"] = suite_rates.get(suite)
        result[f"{suite}_tasks"] = suite_task_counts.get(suite, 0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rerun20260707 LIBERO checkpoint sweep results.")
    parser.add_argument("--checkpoint-base", type=Path, default=Path("/root/feihong/starVLA/Checkpoints"))
    parser.add_argument("--eval-output-prefix", default="eval_sweep_26k_to_40k_40task_50ep_robust_seed7_20260709")
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--step", action="append", type=int, dest="steps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_ids = args.run_ids or DEFAULT_RUN_IDS
    steps = args.steps or DEFAULT_STEPS
    headers = [
        "run_id",
        "step",
        "status",
        "overall_40_task_mean",
        "overall_tasks",
        *[f"{suite}_task_mean" for suite in SUITES],
        *[f"{suite}_tasks" for suite in SUITES],
        "summary_path",
    ]
    rows = []
    for run_id in run_ids:
        for step in steps:
            summary_path = (
                args.checkpoint_base
                / run_id
                / f"{args.eval_output_prefix}_steps_{step}"
                / "logs"
                / "libero_40task_summary.txt"
            )
            parsed = parse_summary(summary_path)
            row = {
                "run_id": run_id,
                "step": step,
                "summary_path": str(summary_path),
                **parsed,
            }
            rows.append(row)

    lines = ["\t".join(headers)]
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.2f}"
            elif value is None:
                value = ""
            values.append(str(value))
        lines.append("\t".join(values))
    output_text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text)
        print(f"wrote {args.output}")
    print(output_text, end="")


if __name__ == "__main__":
    main()
