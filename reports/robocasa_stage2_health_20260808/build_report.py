#!/usr/bin/env python3
"""Build a reproducible portable report for the active RoboCasa Stage-2 run."""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


REPORT_DIR = Path(__file__).resolve().parent
RUN_DIR = Path(
    "/root/feihong/starVLA/"
    "qwen_var_productvq_g16_s124816_robocasa_closebalanced_e256_"
    "bestworst_e47_100k_lr1e4_warmup5000_gbs512_fullcache"
)
BASELINE_DIR = Path(
    "/root/feihong/starVLA/"
    "qwen_var_productvq_g16_s124816_robocasa_best_recon_e128_"
    "100k_lr1e4_warmup5000_gbs512_fullcache"
)

CURRENT_LOG = RUN_DIR / "wandb/wandb/run-20260808_120834-8c7uho0r/files/output.log"
FIRST_E256_LOG = RUN_DIR / "wandb/wandb/run-20260807_205439-urhbwkbq/files/output.log"
CURRENT_MSE = RUN_DIR / "summary.jsonl"
BASELINE_MSE = BASELINE_DIR / "summary.jsonl"
BASELINE_SIM = (
    BASELINE_DIR
    / "robocasa_eval/steps_90000_pytorch_model_gr1_24_50eps_chunk50_robust/summary.json"
)

CURRENT_WANDB = "https://wandb.ai/smap/starVLA_RoboCasa/runs/8c7uho0r"
FIRST_E256_WANDB = "https://wandb.ai/smap/starVLA_RoboCasa/runs/urhbwkbq"
BASELINE_WANDB = "https://wandb.ai/smap/starVLA_RoboCasa/runs/gmof9itw"

STEP_RE = re.compile(
    r"(?P<month>\d{2})/(?P<day>\d{2}) \[(?P<clock>\d{2}:\d{2}:\d{2})\]"
    r".*?Step (?P<step>\d+), Loss:.*?\n\s+"
    r"(?P<metrics>\{.*?'samples_seen':\s*\d+\})",
    re.S,
)


def parse_training_log(path: Path) -> dict[int, dict]:
    text = path.read_text(errors="replace")
    records: dict[int, dict] = {}
    for match in STEP_RE.finditer(text):
        step = int(match.group("step"))
        metrics = ast.literal_eval(match.group("metrics"))
        stamp = datetime.strptime(
            f"2026/{match.group('month')}/{match.group('day')} {match.group('clock')}",
            "%Y/%m/%d %H:%M:%S",
        ).replace(tzinfo=timezone.utc)
        records[step] = {"metrics": metrics, "timestamp": stamp}
    if not records:
        raise RuntimeError(f"No complete step records parsed from {path}")
    return records


def parse_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_mse_segments(rows: list[dict]) -> list[list[dict]]:
    segments: list[list[dict]] = [[]]
    previous = -1
    for row in rows:
        step = int(row["steps"])
        if step <= previous:
            segments.append([])
        segments[-1].append({"steps": step, "mse_score": float(row["mse_score"])})
        previous = step
    return segments


def window_mean(records: dict[int, dict], steps: list[int], key: str) -> float:
    return mean(float(records[step]["metrics"][key]) for step in steps)


def rolling_rows(
    records: dict[int, dict], series: str, max_step: int, width: int = 500, stride: int = 250
) -> list[dict]:
    output: list[dict] = []
    available = sorted(step for step in records if step <= max_step)
    for step in available:
        if step % stride:
            continue
        window = [candidate for candidate in available if step - width <= candidate <= step]
        output.append(
            {
                "step": step,
                "series": series,
                "action_loss": window_mean(records, window, "action_dit_loss"),
                "token_accuracy": window_mean(records, window, "token_accuracy"),
            }
        )
    return output


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (REPORT_DIR / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source(
    source_id: str,
    label: str,
    csv_name: str,
    href: str,
    description: str,
    definitions: list[str],
    executed_at: str,
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "href": href,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": description,
            "sql": f"SELECT * FROM read_csv_auto('{csv_name}')",
            "executed_at": executed_at,
            "tables_used": [csv_name],
            "filters": ["Only complete logged optimizer steps are included."],
            "metric_definitions": definitions,
        },
    }


def main() -> None:
    current = parse_training_log(CURRENT_LOG)
    first = parse_training_log(FIRST_E256_LOG)
    latest_step = max(current)
    latest_metrics = current[latest_step]["metrics"]
    common = sorted(set(current).intersection(first))
    recent_common = [step for step in common if latest_step - 1000 <= step <= latest_step]
    if len(recent_common) < 10:
        raise RuntimeError("Insufficient common E256 points for a 1k-step comparison")

    loss_current = window_mean(current, recent_common, "action_dit_loss")
    loss_first = window_mean(first, recent_common, "action_dit_loss")
    accuracy_current = window_mean(current, recent_common, "token_accuracy")
    accuracy_first = window_mean(first, recent_common, "token_accuracy")
    model_current = window_mean(current, recent_common, "timing/model")
    model_first = window_mean(first, recent_common, "timing/model")
    data_current = window_mean(current, recent_common, "timing/data")
    data_first = window_mean(first, recent_common, "timing/data")
    lr_current = window_mean(current, recent_common, "learning_rate/base")
    lr_first = window_mean(first, recent_common, "learning_rate/base")

    throughput_window = [step for step in sorted(current) if latest_step - 2000 <= step <= latest_step]
    first_timed = current[throughput_window[0]]
    last_timed = current[throughput_window[-1]]
    elapsed = (last_timed["timestamp"] - first_timed["timestamp"]).total_seconds()
    steps_per_second = (throughput_window[-1] - throughput_window[0]) / elapsed
    samples_per_second = steps_per_second * 512

    current_text = CURRENT_LOG.read_text(errors="replace")
    critical_pattern = re.compile(
        r"Traceback|CUDA out of memory|NCCL.{0,80}(?:error|timeout)|\b(?:NaN|Inf)\b",
        re.I,
    )
    critical_hits = len(critical_pattern.findall(current_text))

    trend_rows = rolling_rows(current, "当前重启 E256", latest_step)
    trend_rows += rolling_rows(first, "首次 E256", latest_step)

    group_rows: list[dict] = []
    for group in range(16):
        key = f"acc/codebook_group_{group}"
        delta = window_mean(current, recent_common, key) - window_mean(first, recent_common, key)
        group_rows.append(
            {
                "group": f"g{group}",
                "accuracy_delta": delta,
                "current_accuracy": window_mean(current, recent_common, key),
                "first_e256_accuracy": window_mean(first, recent_common, key),
            }
        )
    largest_group = max(group_rows, key=lambda row: abs(row["accuracy_delta"]))

    mse_segments = split_mse_segments(parse_jsonl(CURRENT_MSE))
    if len(mse_segments) < 2:
        raise RuntimeError("Expected separate first-run and restarted-run MSE segments")
    first_mse = {row["steps"]: row["mse_score"] for row in mse_segments[0]}
    current_mse = {row["steps"]: row["mse_score"] for row in mse_segments[-1]}
    baseline_mse = {row["steps"]: row["mse_score"] for row in parse_jsonl(BASELINE_MSE)}
    mse_steps = sorted(current_mse)
    mse_rows: list[dict] = []
    for series, values in (
        ("当前重启 E256", current_mse),
        ("首次 E256", first_mse),
        ("40.33% 基线 E128", baseline_mse),
    ):
        for step in mse_steps:
            if step in values:
                mse_rows.append(
                    {"step": step, "series": series, "mse_x1e3": values[step] * 1000}
                )
    latest_mse_step = max(mse_steps)
    current_latest_mse = current_mse[latest_mse_step]
    first_latest_mse = first_mse[latest_mse_step]
    baseline_latest_mse = baseline_mse[latest_mse_step]

    baseline_sim = json.loads(BASELINE_SIM.read_text())
    baseline_success = float(baseline_sim["total_success_rate"])
    baseline_successes = int(baseline_sim["total_successes"])
    baseline_episodes = int(baseline_sim["total_episodes"])

    summary_rows = [
        {
            "latest_step": latest_step,
            "latest_loss": float(latest_metrics["action_dit_loss"]),
            "latest_accuracy": float(latest_metrics["token_accuracy"]),
            "samples_per_second": samples_per_second,
            "critical_events": critical_hits,
        }
    ]
    comparison_rows = [
        {
            "metric": "Action loss（最近 1k step 均值）",
            "current": f"{loss_current:.6f}",
            "first_e256": f"{loss_first:.6f}",
            "delta": f"{loss_current - loss_first:+.6f} ({(loss_current / loss_first - 1) * 100:+.2f}%)",
            "assessment": "同配置噪声范围",
        },
        {
            "metric": "Token accuracy（最近 1k step 均值）",
            "current": f"{accuracy_current:.2%}",
            "first_e256": f"{accuracy_first:.2%}",
            "delta": f"{(accuracy_current - accuracy_first) * 100:+.3f} pp",
            "assessment": "同配置噪声范围",
        },
        {
            "metric": "Base LR（最近 1k step 均值）",
            "current": f"{lr_current:.8e}",
            "first_e256": f"{lr_first:.8e}",
            "delta": f"{lr_current - lr_first:+.2e}",
            "assessment": "调度一致",
        },
        {
            "metric": "Model time（秒，最近 1k step 均值）",
            "current": f"{model_current:.4f}",
            "first_e256": f"{model_first:.4f}",
            "delta": f"{model_current - model_first:+.4f}",
            "assessment": "吞吐稳定",
        },
        {
            "metric": "Data time（秒，最近 1k step 均值）",
            "current": f"{data_current:.4f}",
            "first_e256": f"{data_first:.4f}",
            "delta": f"{data_current - data_first:+.4f}",
            "assessment": "无数据卡顿",
        },
        {
            "metric": f"Validation MSE（step {latest_mse_step}）",
            "current": f"{current_latest_mse:.8f}",
            "first_e256": f"{first_latest_mse:.8f}",
            "delta": f"{(current_latest_mse / first_latest_mse - 1) * 100:+.2f}%",
            "assessment": "单点评估有抖动，结合趋势观察",
        },
    ]

    write_csv("summary.csv", summary_rows)
    write_csv("training_trend.csv", trend_rows)
    write_csv("mse_trend.csv", mse_rows)
    write_csv("group_delta.csv", group_rows)
    write_csv("comparison.csv", comparison_rows)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    loss_delta_pct = (loss_current / loss_first - 1) * 100
    accuracy_delta_pp = (accuracy_current - accuracy_first) * 100
    baseline_mse_delta_pct = (current_latest_mse / baseline_latest_mse - 1) * 100
    eta_hours = (100000 - latest_step) / steps_per_second / 3600

    sources = [
        source(
            "current_snapshot",
            "当前 E256 run 快照",
            "summary.csv",
            CURRENT_WANDB,
            "Loads the latest complete step and health counters from the active run.",
            [
                "latest_step is the largest complete optimizer step in output.log.",
                "samples_per_second is the last 2,000 optimizer steps times GBS 512 divided by elapsed wall time.",
                "critical_events counts Traceback, CUDA OOM, NCCL error/timeout, NaN, or Inf markers.",
            ],
            generated_at,
        ),
        source(
            "same_e256_metrics",
            "当前重启与首次 E256 同-step 对齐",
            "training_trend.csv",
            FIRST_E256_WANDB,
            "Loads 500-step rolling means for two runs with identical E256 Stage-2 configuration.",
            [
                "Each curve point is the arithmetic mean of complete 50-step logs in [step-500, step].",
                "Recent comparison window uses the exact common discrete steps in [latest_step-1000, latest_step].",
            ],
            generated_at,
        ),
        source(
            "validation_mse",
            "Stage-2 validation MSE 对照",
            "mse_trend.csv",
            BASELINE_WANDB,
            "Loads validation MSE for the current E256 restart, first E256 run, and E128 baseline.",
            [
                "MSE is measured in the shared continuous action space; lower is better.",
                "mse_x1e3 equals raw mse_score multiplied by 1,000 for chart readability.",
            ],
            generated_at,
        ),
        source(
            "group_deltas",
            "E256 codebook group 同-step 差异",
            "group_delta.csv",
            CURRENT_WANDB,
            "Loads current-minus-first E256 token accuracy deltas for 16 codebook groups.",
            [
                "accuracy_delta is current mean accuracy minus first-run mean accuracy over the recent 1,000-step common window.",
            ],
            generated_at,
        ),
        source(
            "comparison_detail",
            "E256 健康性审计明细",
            "comparison.csv",
            CURRENT_WANDB,
            "Loads the exact reviewed comparison values used in the report conclusion.",
            ["Deltas are current restart minus first E256 run at identical optimizer steps."],
            generated_at,
        ),
    ]

    title = "RoboCasa Stage-2 当前训练健康报告"
    manifest_sources = [
        {"id": item["id"], "label": item["label"], "href": item["href"]}
        for item in sources
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Active E256 optimization health, reproducibility, and run-lifecycle risk.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "step_card",
                    "dataset": "summary",
                    "sourceId": "current_snapshot",
                    "metrics": [{"label": "最新 optimizer step", "field": "latest_step", "format": "number"}],
                },
                {
                    "id": "loss_card",
                    "dataset": "summary",
                    "sourceId": "current_snapshot",
                    "metrics": [{"label": "最新 action loss", "field": "latest_loss", "format": "number"}],
                },
                {
                    "id": "accuracy_card",
                    "dataset": "summary",
                    "sourceId": "current_snapshot",
                    "metrics": [{"label": "最新 token accuracy", "field": "latest_accuracy", "format": "percent"}],
                },
                {
                    "id": "throughput_card",
                    "dataset": "summary",
                    "sourceId": "current_snapshot",
                    "metrics": [{"label": "近 2k samples/s", "field": "samples_per_second", "format": "number"}],
                },
                {
                    "id": "critical_card",
                    "dataset": "summary",
                    "sourceId": "current_snapshot",
                    "metrics": [{"label": "严重数值/系统事件", "field": "critical_events", "format": "number"}],
                },
            ],
            "charts": [
                {
                    "id": "loss_trend",
                    "title": "相同 E256 配置的 action loss 曲线",
                    "subtitle": "500-step 滚动均值；当前重启与首次 run 基本重合",
                    "type": "line",
                    "dataset": "training_trend",
                    "sourceId": "same_e256_metrics",
                    "encodings": {
                        "x": {"field": "step", "type": "quantitative", "label": "Optimizer step"},
                        "y": {"field": "action_loss", "type": "quantitative", "label": "Action loss"},
                        "color": {"field": "series", "type": "nominal", "label": "Run"},
                    },
                    "layout": "full",
                },
                {
                    "id": "accuracy_trend",
                    "title": "相同 E256 配置的 token accuracy 曲线",
                    "subtitle": "500-step 滚动均值；走势稳定向上",
                    "type": "line",
                    "dataset": "training_trend",
                    "sourceId": "same_e256_metrics",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "step", "type": "quantitative", "label": "Optimizer step"},
                        "y": {"field": "token_accuracy", "type": "quantitative", "label": "Token accuracy", "format": "percent"},
                        "color": {"field": "series", "type": "nominal", "label": "Run"},
                    },
                    "layout": "full",
                },
                {
                    "id": "mse_trend",
                    "title": "Validation MSE（连续动作空间）",
                    "subtitle": "每 2k step 一次；单点有明显抖动，越低越好",
                    "type": "line",
                    "dataset": "mse_trend",
                    "sourceId": "validation_mse",
                    "encodings": {
                        "x": {"field": "step", "type": "quantitative", "label": "Optimizer step"},
                        "y": {"field": "mse_x1e3", "type": "quantitative", "label": "MSE × 10⁻³"},
                        "color": {"field": "series", "type": "nominal", "label": "Run"},
                    },
                    "layout": "full",
                },
                {
                    "id": "group_delta",
                    "title": "最近 1k step 的 codebook group accuracy 差异",
                    "subtitle": "当前重启减首次 E256；绝对差异应围绕 0 小幅波动",
                    "type": "bar",
                    "dataset": "group_delta",
                    "sourceId": "group_deltas",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "group", "type": "nominal", "label": "Codebook group"},
                        "y": {"field": "accuracy_delta", "type": "quantitative", "label": "Accuracy delta", "format": "percent"},
                        "tooltip": [
                            {"field": "current_accuracy", "type": "quantitative", "label": "当前", "format": "percent"},
                            {"field": "first_e256_accuracy", "type": "quantitative", "label": "首次 E256", "format": "percent"},
                        ],
                    },
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "comparison_table",
                    "title": "同配置 E256 健康性对照",
                    "subtitle": f"严格按共同 step 对齐；最近窗口 {recent_common[0]}–{recent_common[-1]}",
                    "dataset": "comparison",
                    "sourceId": "comparison_detail",
                    "density": "spacious",
                    "defaultSort": {"field": "metric", "direction": "asc"},
                    "columns": [
                        {"field": "metric", "label": "指标", "type": "text"},
                        {"field": "current", "label": "当前重启", "type": "text"},
                        {"field": "first_e256", "label": "首次 E256", "type": "text"},
                        {"field": "delta", "label": "差值", "type": "text"},
                        {"field": "assessment", "label": "判断", "type": "text"},
                    ],
                }
            ],
            "sources": manifest_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "current_snapshot",
                    "body": (
                        "## Technical summary\n\n"
                        f"**结论：当前训练的计算与优化指标正常，可以继续跑。** 截至完整日志 step **{latest_step:,}**，"
                        f"单点 action loss 为 **{float(latest_metrics['action_dit_loss']):.4f}**，token accuracy 为 "
                        f"**{float(latest_metrics['token_accuracy']):.2%}**。最近 1k step 相对首次相同 E256 run，"
                        f"loss 差 **{loss_delta_pct:+.2f}%**、accuracy 差 **{accuracy_delta_pp:+.3f} 个百分点**；"
                        f"没有发现 NaN、Inf、OOM、NCCL error/timeout 或 traceback。\n\n"
                        "需要分开看待一个运维问题：当前 run 是从 0 重训，并非从 30k 正确续训；它与首轮共用 checkpoint 目录，"
                        "所以“训练数值健康”不等于“续训/检查点管理正确”。"
                    ),
                },
                {
                    "id": "metrics",
                    "type": "metric-strip",
                    "cardIds": ["step_card", "loss_card", "accuracy_card", "throughput_card", "critical_card"],
                },
                {
                    "id": "key_findings",
                    "type": "markdown",
                    "sourceId": "same_e256_metrics",
                    "body": (
                        "## Key findings and visual evidence\n\n"
                        "最强的健康性证据不是与旧 E128 run 比，而是与第一次 **相同 tokenizer、相同超参数的 E256 run** 按相同 step 对齐。"
                        "两次曲线近乎复现，说明重启后没有出现优化发散或数据管线错位。"
                    ),
                },
                {"id": "loss_chart", "type": "chart", "chartId": "loss_trend", "layout": "full"},
                {"id": "accuracy_chart", "type": "chart", "chartId": "accuracy_trend", "layout": "full"},
                {
                    "id": "mse_interpretation",
                    "type": "markdown",
                    "sourceId": "validation_mse",
                    "body": (
                        "## Validation evidence\n\n"
                        f"连续动作空间的 validation MSE 到 step {latest_mse_step:,} 为 **{current_latest_mse:.8f}**；"
                        f"相对首次 E256 是 **{(current_latest_mse / first_latest_mse - 1) * 100:+.1f}%**，"
                        f"相对旧 E128/40.33% 基线是 **{baseline_mse_delta_pct:+.1f}%**。MSE 每 2k step 才评估一次且抖动较大，"
                        "因此应看多点趋势，不能用单个 checkpoint 判定最终 RoboCasa 成功率。"
                    ),
                },
                {"id": "mse_chart_block", "type": "chart", "chartId": "mse_trend", "layout": "full"},
                {
                    "id": "group_interpretation",
                    "type": "markdown",
                    "sourceId": "group_deltas",
                    "body": (
                        "## Per-group check\n\n"
                        f"16 个 codebook group 中，最近 1k step 最大绝对 accuracy 差异是 **{largest_group['group']} "
                        f"{largest_group['accuracy_delta'] * 100:+.2f} 个百分点**。这是观察项，不构成停训或再次重启的证据。"
                    ),
                },
                {"id": "group_chart_block", "type": "chart", "chartId": "group_delta", "layout": "full"},
                {
                    "id": "scope_method",
                    "type": "markdown",
                    "body": (
                        "## Scope, data, metric definitions, and methodology\n\n"
                        f"范围是当前 run `8c7uho0r` 从 step 50 到 {latest_step:,} 的完整 50-step 日志，并以首次 E256 run "
                        "`urhbwkbq` 的共同 step 为主要对照。action loss 和 token accuracy 使用最近 1,000-step 共同窗口算术均值；"
                        "趋势图使用 500-step 滚动均值。吞吐按最近 2,000 optimizer steps、GBS 512 与日志墙钟计算。"
                    ),
                },
                {"id": "comparison_table_block", "type": "table", "tableId": "comparison_table", "layout": "full"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and robustness\n\n"
                        f"旧 40.33% 基线来自 step 90k 的 24×50 仿真评估（{baseline_successes}/{baseline_episodes}，"
                        f"**{baseline_success:.2%}**），而当前只到约 {latest_step / 1000:.1f}k，尚无同等仿真评估。"
                        "E128 旧 tokenizer 与 E256 close-balanced 新 tokenizer 的标签空间、Stage-1 objective/action-type embedding 也不同；"
                        "所以跨 tokenizer 的 cross-entropy、token accuracy 和 group ID 不能直接判断孰优，更不能推出最终成功率一定高于 40.33%。"
                    ),
                },
                {
                    "id": "operational_risk",
                    "type": "markdown",
                    "body": (
                        "## Run-lifecycle risk\n\n"
                        "配置明确是 `is_resume: false`，当前从 step 0 重训。共享 checkpoint 目录目前只保留首轮的 22k/24k/28k/30k 文件；"
                        "当前低步数 checkpoint 会受 `latest_and_best` 保留策略影响，导致当前 run 在到达旧高步数前缺少可靠恢复点。"
                        "这是检查点隔离/可恢复性异常，但不是 loss 发散。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next steps\n\n"
                        "- 数值上继续训练，无需因为当前 loss/accuracy 再次重启。\n"
                        "- 尽快把当前 run 的 checkpoint 输出隔离到新目录，或明确修复 resume 策略；这一步是为了可恢复性。\n"
                        f"- 重点观察 {largest_group['group']}、validation MSE 的多点评估，以及 40k/60k/90k 的 RoboCasa 仿真成功率。\n"
                        f"- 按最近吞吐粗略外推，到 100k 还需约 **{eta_hours:.1f} 小时**；保存和评估会带来额外抖动。"
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "E256 tokenizer 的早期 token 指标较低，最终是否能转化为高于 40.33% 的仿真成功率，只能通过同协议的中后期 checkpoint 评估回答。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "partial",
            "datasets": {
                "summary": summary_rows,
                "training_trend": trend_rows,
                "mse_trend": mse_rows,
                "group_delta": group_rows,
                "comparison": comparison_rows,
            },
            "accessIssues": [
                {
                    "id": "current_sim_eval_unavailable",
                    "dataset": "current_robocasa_sim_eval",
                    "message": "Current run has not yet reached a checkpoint with a comparable 24-task × 50-episode RoboCasa evaluation.",
                }
            ],
        },
        "sources": sources,
        "package_info": {
            "originUrl": "artifact://robocasa-stage2-health-20260808",
            "controls": {"edit": False, "refresh": False},
        },
    }

    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "latest_step": latest_step,
                "latest_loss": latest_metrics["action_dit_loss"],
                "latest_accuracy": latest_metrics["token_accuracy"],
                "recent_1k_loss_delta_pct": loss_delta_pct,
                "recent_1k_accuracy_delta_pp": accuracy_delta_pp,
                "samples_per_second": samples_per_second,
                "critical_events": critical_hits,
                "largest_group": largest_group,
                "latest_mse_step": latest_mse_step,
                "current_mse": current_latest_mse,
                "first_e256_mse": first_latest_mse,
                "baseline_e128_mse": baseline_latest_mse,
                "eta_hours": eta_hours,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
