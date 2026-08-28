#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


REPO = Path("/root/feihong/starVLA")
REPORT_DIR = REPO / "reports/libero_stage2_early_eval_20260812"
CHECKPOINT_ROOT = REPO / "Checkpoints"
RUNS = {
    "A: LR 2.5e-5 / warmup 3k": "qwen_var_productvq_g16_s1248_structemb_nextscale_100k_rerun20260707_A_baseline_lr2p5e5_warmup3k_ls002",
    "B: LR 2.0e-5 / warmup 5k": "qwen_var_productvq_g16_s1248_structemb_nextscale_100k_rerun20260707_B_stable_lr2e5_warmup5k_ls002",
}
PREFIX = "eval_sweep_26k_to_40k_40task_50ep_robust_seed7_20260709"
SUITE_RE = re.compile(r"^(libero_[a-z0-9_]+):\s+10 tasks, task-mean=([0-9.]+)%$")
OVERALL_RE = re.compile(r"^overall_40_task_mean:\s+40 tasks, ([0-9.]+)%$")


def parse_summary(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    suites: dict[str, float] = {}
    overall = None
    for line in path.read_text(errors="replace").splitlines():
        if match := SUITE_RE.match(line):
            suites[match.group(1)] = float(match.group(2))
        elif match := OVERALL_RE.match(line):
            overall = float(match.group(1))
    if overall is None or len(suites) != 4:
        return None
    return {"overall": overall, **suites}


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, run_id in RUNS.items():
        for step in range(26000, 40001, 1000):
            summary = (
                CHECKPOINT_ROOT
                / run_id
                / f"{PREFIX}_steps_{step}"
                / "logs/libero_40task_summary.txt"
            )
            parsed = parse_summary(summary)
            if parsed is None:
                continue
            rows.append(
                {
                    "run": label,
                    "step": step,
                    "step_k": step / 1000,
                    "overall_success": parsed["overall"] / 100,
                    "libero_10_success": parsed["libero_10"] / 100,
                    "spatial_success": parsed["libero_spatial"] / 100,
                    "object_success": parsed["libero_object"] / 100,
                    "goal_success": parsed["libero_goal"] / 100,
                    "tasks": 40,
                    "episodes": 2000,
                    "per_device_batch": 16,
                    "gpu_count": 8,
                    "gradient_accumulation": 1,
                    "global_batch": 128,
                }
            )
    return rows


def build_artifact(rows: list[dict[str, object]]) -> dict[str, object]:
    anchors = [
        row
        for row in rows
        if (row["run"].startswith("A:") and row["step"] in {26000, 30000, 32000})
        or (row["run"].startswith("B:") and row["step"] in {28000, 30000, 35000})
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "LIBERO Stage2 早期 Eval 与 Batch Size 核对",
            "description": "26k–40k checkpoint 的完整 40-task eval 与全局 batch 口径核对。",
            "generatedAt": "2026-08-12T12:30:00Z",
            "charts": [
                {
                    "id": "early_eval_curve",
                    "title": "Stage2 早期 checkpoint 成功率",
                    "subtitle": "每点为 40 tasks × 50 episodes，seed 7；A/B 是独立训练 run",
                    "type": "line",
                    "dataset": "early_eval",
                    "sourceId": "robust_eval_summaries",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "step_k", "type": "quantitative", "label": "Optimizer step (k)"},
                        "y": {"field": "overall_success", "type": "quantitative", "label": "40-task mean success", "format": "percent"},
                        "color": {"field": "run", "type": "nominal", "label": "Training run"},
                        "tooltip": [
                            {"field": "libero_10_success", "type": "quantitative", "label": "LIBERO-10", "format": "percent"},
                            {"field": "episodes", "type": "quantitative", "label": "Episodes"},
                        ],
                    },
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "anchor_table",
                    "title": "关键 checkpoint 精确值",
                    "subtitle": "完整评测锚点；成功率按 2,000 episodes 汇总",
                    "dataset": "anchors",
                    "sourceId": "robust_eval_summaries",
                    "density": "spacious",
                    "defaultSort": {"field": "step", "direction": "asc"},
                    "columns": [
                        {"field": "run", "label": "Run", "type": "text"},
                        {"field": "step", "label": "Step", "type": "number"},
                        {"field": "overall_success", "label": "40-task mean", "type": "number", "format": "percent"},
                        {"field": "libero_10_success", "label": "LIBERO-10", "type": "number", "format": "percent"},
                        {"field": "global_batch", "label": "Global batch", "type": "number"},
                    ],
                }
            ],
            "sources": [
                {
                    "id": "robust_eval_summaries",
                    "label": "LIBERO robust checkpoint sweep summaries",
                    "query": {
                        "engine": "python",
                        "language": "python",
                        "description": "Parses only summaries containing all four suites, 40 tasks, and 2,000 episode rows.",
                        "tables_used": ["libero_40task_summary.txt", "libero_40task_episodes.csv"],
                        "filters": ["seed=7", "50 episodes per task", "complete 40-task summaries only", "steps 26k–40k"],
                        "metric_definitions": [
                            "overall_success is total successful episodes divided by 2,000 episodes across 40 tasks.",
                            "global_batch equals 8 GPUs × 16 samples per GPU × gradient accumulation 1.",
                        ],
                    },
                }
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# LIBERO Stage2 早期 Eval 与 Batch Size 核对"},
                {
                    "id": "executive_summary",
                    "type": "markdown",
                    "sourceId": "robust_eval_summaries",
                    "body": "## Executive Summary\n\n- **你的记忆是对的。** A 基线在 26k 和 30k 都达到 **97.65%**；B 稳定版在 28k 达到 **97.85%**。\n- **这些结果对应 global batch 128。** 启动口径是 8 GPUs × per-device batch 16 × accumulation 1；当前新极核配置已经完全一致。\n- **不建议现在把 per-device batch 提到 32。** 那会把 global batch 改成 256，并同时改变每 step 看到的样本数与优化噪声，削弱新旧 Stage1 的公平归因。",
                },
                {
                    "id": "finding_curve",
                    "type": "markdown",
                    "sourceId": "robust_eval_summaries",
                    "body": "## 97%+ 确实在 30k 左右已经出现\n\nA run 从 26k 到 32k 的完整评测基本稳定在 **97.15%–97.65%**；30k 是 **97.65%**，其中 LIBERO-10 为 **95.00%**。B run 的峰值更早，28k 为 **97.85%**，但 30k 回落到 **96.30%**，说明单个 checkpoint 有明显波动，不能把‘30k’理解成固定最优点。",
                },
                {"id": "curve", "type": "chart", "chartId": "early_eval_curve", "layout": "full"},
                {
                    "id": "batch_implication",
                    "type": "markdown",
                    "sourceId": "robust_eval_summaries",
                    "body": "## Batch Size 应保持 128，优先调整评测与保存窗口\n\n现有证据支持保持每卡 16、8 卡、累积 1。新实验唯一希望验证的核心变量是加权后的 Stage1；若同步把 global batch 翻倍，就无法区分涨跌来自 tokenizer 还是 Stage2 优化尺度。更高价值的调整是从 **24k/26k 开始每 1k 保存**，优先评测 26k、28k、30k、32k、35k，并把 40k 设为第一阶段决策点。",
                },
                {"id": "anchors", "type": "table", "tableId": "anchor_table", "layout": "full"},
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## Recommended next steps\n\n- 新 Stage2 保持 global batch 128。\n- 第一阶段训练到 40k，保存 24k–40k 的 1k 间隔 checkpoint。\n- 用同一 seed、同一精度、40 tasks × 50 episodes 对 26k/28k/30k/32k/35k 做 sweep。\n- 若早期没有超过旧链路，再恢复到 100k，并额外保留 99k。",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": "## Further questions\n\n新 Stage1 是否让 LIBERO-10 在相同步数超过旧值，需要重点看 task 03/06/08 的逐任务成功率，而不只是 40-task mean。",
                },
                {
                    "id": "caveats",
                    "type": "markdown",
                    "body": "## Caveats and assumptions\n\nA 与 B 是独立训练 run，不能拼成一条连续曲线。旧 60k/save-all run 在 40k–50k 的下降也属于另一条训练链路，不用于推断 A/B 的后期走势。所有主结论仅使用带完整 40-task 汇总和 2,000-row episode CSV 的评测。",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-08-12T12:30:00Z",
            "status": "ready",
            "datasets": {"early_eval": rows, "anchors": anchors},
            "accessIssues": [],
        },
        "sources": [
            {
                "id": "robust_eval_summaries",
                "label": "LIBERO robust checkpoint sweep summaries",
                "query": {
                    "engine": "python",
                    "language": "python",
                    "description": "Parses complete local checkpoint-sweep summaries with episode-level coverage checks.",
                    "tables_used": ["libero_40task_summary.txt", "libero_40task_episodes.csv"],
                    "filters": ["seed=7", "50 episodes per task", "40 tasks", "steps 26k–40k"],
                    "metric_definitions": [
                        "Success rate is successful rollout episodes divided by evaluated episodes.",
                        "Global batch is GPU count times per-device batch times gradient accumulation.",
                    ],
                },
            }
        ],
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    if not rows:
        raise RuntimeError("No complete eval summaries found")
    with (REPORT_DIR / "early_eval_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(build_artifact(rows), ensure_ascii=False, indent=2) + "\n"
    )
    print(f"complete rows={len(rows)}")


if __name__ == "__main__":
    main()
