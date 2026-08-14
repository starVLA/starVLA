#!/usr/bin/env python3
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = REPORT_DIR / "artifact.json"


SCALE_ROWS = [
    {"scale": "s1", "e256_accuracy": 0.606742, "e128_accuracy": 0.837084, "raw_gap": -0.230342, "gap_contribution": -0.00743, "gap_share": 0.055},
    {"scale": "s2", "e256_accuracy": 0.364894, "e128_accuracy": 0.630043, "raw_gap": -0.265149, "gap_contribution": -0.01711, "gap_share": 0.126},
    {"scale": "s4", "e256_accuracy": 0.397264, "e128_accuracy": 0.639984, "raw_gap": -0.242720, "gap_contribution": -0.03132, "gap_share": 0.231},
    {"scale": "s8", "e256_accuracy": 0.491840, "e128_accuracy": 0.663794, "raw_gap": -0.171954, "gap_contribution": -0.04438, "gap_share": 0.328},
    {"scale": "s16", "e256_accuracy": 0.562896, "e128_accuracy": 0.630862, "raw_gap": -0.067966, "gap_contribution": -0.03508, "gap_share": 0.259},
]


CONFIG_ROWS = [
    {
        "item": "Stage-2 数据与优化超参数",
        "e128": "相同 RoboCasa 数据、GBS 512、100k steps",
        "e256": "相同 RoboCasa 数据、GBS 512、100k steps",
        "implication": "排除数据量与 Stage-2 训练配方差异",
    },
    {
        "item": "token cache 形状",
        "e128": "5,660,058 × 496",
        "e256": "5,660,058 × 496",
        "implication": "样本数和每样本 token 位置数相同",
    },
    {
        "item": "VQ 拓扑",
        "e128": "ProductVQ；512 codes；16 groups；scales 1/2/4/8/16",
        "e256": "ProductVQ；512 codes；16 groups；scales 1/2/4/8/16",
        "implication": "拓扑相同，但 code ID 的含义并不固定",
    },
    {
        "item": "latent 维度",
        "e128": "128（每 group 8 维）",
        "e256": "256（每 group 16 维）",
        "implication": "每组表征空间翻倍，分类难度与分区语义改变",
    },
    {
        "item": "Stage-1 表征目标",
        "e128": "标准重建；best_recon checkpoint",
        "e256": "close/late/task-balanced + adaptive worst-task；best_worst_task_mae",
        "implication": "tokenizer 被优化成不同的离散标签空间",
    },
    {
        "item": "action-type embedding",
        "e128": "关闭",
        "e256": "开启",
        "implication": "token 条件与分配方式进一步不同",
    },
    {
        "item": "Stage-1 code 使用",
        "e128": "511 / 512",
        "e256": "512 / 512",
        "implication": "两者都无明显 codebook collapse",
    },
]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replace_by_id(items, item):
    items[:] = [existing for existing in items if existing.get("id") != item["id"]]
    items.append(item)


def main():
    write_csv(REPORT_DIR / "token_accuracy_gap_by_scale.csv", SCALE_ROWS)
    write_csv(REPORT_DIR / "tokenizer_comparison.csv", CONFIG_ROWS)

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    manifest = artifact["manifest"]
    datasets = artifact["snapshot"]["datasets"]
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    datasets["token_accuracy_gap_by_scale"] = SCALE_ROWS
    datasets["tokenizer_comparison"] = CONFIG_ROWS

    chart = {
        "id": "token_accuracy_gap_by_scale",
        "title": "E256 相对 40.33% E128 的 accuracy 差距贡献",
        "subtitle": "共同末 10k；E256−E128。五个尺度合计 −13.53 个百分点",
        "type": "bar",
        "dataset": "token_accuracy_gap_by_scale",
        "sourceId": "token_gap_diagnosis",
        "valueFormat": "percent",
        "encodings": {
            "x": {"field": "scale", "type": "nominal", "label": "Token scale"},
            "y": {"field": "gap_contribution", "type": "quantitative", "label": "对总体 accuracy 差距的贡献", "format": "percent"},
            "tooltip": [
                {"field": "e256_accuracy", "type": "quantitative", "label": "当前 E256 accuracy", "format": "percent"},
                {"field": "e128_accuracy", "type": "quantitative", "label": "40.33% E128 accuracy", "format": "percent"},
                {"field": "raw_gap", "type": "quantitative", "label": "该尺度原始差距", "format": "percent"},
                {"field": "gap_share", "type": "quantitative", "label": "占总差距比例", "format": "percent"},
            ],
        },
        "layout": "full",
    }
    replace_by_id(manifest["charts"], chart)

    table = {
        "id": "tokenizer_comparison_table",
        "title": "两次训练使用的离散标签空间并不相同",
        "subtitle": "Stage-2 数据规模相同；差异来自 Stage-1 tokenizer 表征与 checkpoint 选择",
        "dataset": "tokenizer_comparison",
        "sourceId": "tokenizer_config_evidence",
        "density": "spacious",
        "columns": [
            {"field": "item", "label": "核对项", "type": "text"},
            {"field": "e128", "label": "40.33% E128", "type": "text"},
            {"field": "e256", "label": "当前 E256", "type": "text"},
            {"field": "implication", "label": "含义", "type": "text"},
        ],
    }
    replace_by_id(manifest["tables"], table)

    source_manifest = {
        "id": "token_gap_diagnosis",
        "label": "E256 与 40.33% E128 的同-step token 指标分解",
        "href": "https://wandb.ai/smap/starVLA_RoboCasa/runs/gmof9itw",
    }
    replace_by_id(manifest["sources"], source_manifest)
    config_source_manifest = {
        "id": "tokenizer_config_evidence",
        "label": "Stage-1 tokenizer 配置与 cache 审计",
        "href": "https://wandb.ai/smap/starVLA_RoboCasa/runs/8c7uho0r",
    }
    replace_by_id(manifest["sources"], config_source_manifest)

    artifact_sources = artifact["sources"]
    replace_by_id(artifact_sources, {
        **source_manifest,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": "Loads exact common-step late-window token accuracy decomposed by multiscale token position weights.",
            "sql": "SELECT * FROM read_csv_auto('token_accuracy_gap_by_scale.csv')",
            "executed_at": generated_at,
            "tables_used": ["token_accuracy_gap_by_scale.csv"],
            "filters": ["Exact common logged steps 85,750–95,750 (201 points)", "E256 run 8c7uho0r versus E128 run gmof9itw"],
            "metric_definitions": [
                "raw_gap is E256 accuracy minus E128 accuracy within each scale.",
                "gap_contribution is raw_gap multiplied by the scale token-position weight 1/31, 2/31, 4/31, 8/31, or 16/31.",
                "The five contributions sum to the overall −13.531 percentage-point gap.",
            ],
        },
    })
    replace_by_id(artifact_sources, {
        **config_source_manifest,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": "Loads audited Stage-1 tokenizer and Stage-2 cache configuration differences.",
            "sql": "SELECT * FROM read_csv_auto('tokenizer_comparison.csv')",
            "executed_at": generated_at,
            "tables_used": ["tokenizer_comparison.csv"],
            "filters": ["Only fields that materially affect token-label comparability are included."],
            "metric_definitions": [
                "Cache shape is samples by token positions; 496 equals 16 groups times (1+2+4+8+16) scale positions.",
                "Dimensions per group equal tokenizer embed_dim divided by 16 product-quantization groups.",
            ],
        },
    })

    diagnosis_blocks = [
        {
            "id": "token_accuracy_diagnosis",
            "type": "markdown",
            "sourceId": "token_gap_diagnosis",
            "body": (
                "## 为什么当前 token accuracy 比 40.33% 那次低\n\n"
                "**主因是 tokenizer 变了，所以两次 accuracy 不是同一把尺子；这不是当前训练末段退化。** "
                "在严格共同的末 10k step 中，当前 E256 为 **51.18%**，旧 E128 为 **64.71%**，差 **−13.53 个百分点**。"
                "但差距从早期就存在：2k 为 −9.65 个百分点，8k 为 −12.43 个百分点，90k 为 −13.41 个百分点；"
                "与此同时当前 run 自己的末 10k accuracy 斜率仍为正。\n\n"
                "代码中的 accuracy 是 496 个位置上 `argmax token ID == target token ID` 的精确命中率。"
                "E128 与 E256 虽然样本数、token 数、512-code ProductVQ、16 groups 和五个 scales 相同，"
                "但 latent 从 128 增至 256（每组 8→16 维），Stage-1 目标、action-type embedding 和 checkpoint 选择也改变；"
                "因此同一个 group 编号和 token ID 不再代表相同的动作区域，分类熵与难度基线也不可直接对齐。"
            ),
        },
        {"id": "token_gap_chart_block", "type": "chart", "chartId": "token_accuracy_gap_by_scale", "layout": "full"},
        {
            "id": "token_accuracy_action_counterexample",
            "type": "markdown",
            "sourceId": "validation_mse",
            "body": (
                "### Exact-token 命中更低，不等于解码后的动作更差\n\n"
                "最直接的反例在 step 90k：当前 E256 token accuracy **50.52%**，旧 E128 **63.92%**；"
                "但共享连续动作空间的 validation MSE 是 **0.00010718 vs 0.00016909**，E256 反而低 **36.6%**。"
                "错误 token 可能对应邻近或功能相似的 codeword，多尺度、多 group 的误差经 decoder 合成后，对连续动作的影响也不同。"
                "分组结果同样呈交叉：有些 E256 group 低近 60 个百分点，有些却高 37 个百分点，"
                "这更像离散分区语义重排，而不是统一的学习能力下降。\n\n"
                "因此应使用三层判断：同 tokenizer 内看 loss/accuracy 趋势；跨 tokenizer 看连续动作 MSE；"
                "最终是否超过 **40.33%** 仍以相同协议的 24×50 RoboCasa 闭环仿真为准。"
            ),
        },
        {"id": "tokenizer_comparison_table_block", "type": "table", "tableId": "tokenizer_comparison_table", "layout": "full"},
    ]

    blocks = manifest["blocks"]
    diagnosis_ids = {block["id"] for block in diagnosis_blocks}
    blocks[:] = [block for block in blocks if block.get("id") not in diagnosis_ids]
    insert_at = next((idx + 1 for idx, block in enumerate(blocks) if block.get("id") == "accuracy_chart"), len(blocks))
    blocks[insert_at:insert_at] = diagnosis_blocks

    for block in blocks:
        if block.get("id") == "technical_summary":
            block["body"] += (
                "\n\n与 40.33% 的 E128 基线相比，当前 exact-token accuracy 低约 **13.53 个百分点**，"
                "但该比较跨越不同 tokenizer，不能解释为策略变差；step 90k 时当前连续动作 MSE 反而低 **36.6%**。"
            )
        elif block.get("id") == "limitations":
            block["body"] += (
                " Exact-token accuracy 要求离散 ID 完全相等；它不会反映错误 codeword 与目标 codeword 的距离，"
                "也不能控制两套 tokenizer 的标签熵与分区语义差异。"
            )

    manifest["generatedAt"] = generated_at
    artifact["snapshot"]["generatedAt"] = generated_at
    artifact["package_info"]["generatedAt"] = generated_at
    ARTIFACT_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
