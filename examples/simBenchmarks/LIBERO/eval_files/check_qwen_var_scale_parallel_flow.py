"""Sanity-check QwenVARScaleParallel next-scale training/inference flow.

This test avoids loading Qwen-VL. It builds a tiny fake instance that reuses the
real QwenVARScaleParallel methods and verifies:

- [1,2,4,8] scale layout maps to [16,32,64,128] slots.
- teacher-forced training conditions each scale only on previous scales.
- per-factor classifiers work with variable-length scale slot subsets.
- inference feeds predicted previous-scale codes into later scales.
"""

from __future__ import annotations

import json
import types
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

from starVLA.model.framework.VLM4A.QwenVARParallel import ActionCodeQueryBlock
from starVLA.model.framework.VLM4A.QwenVARScaleParallel import QwenVARScaleParallel


class _FakeStage1Tokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scales = [1, 2, 4, 8]
        self.product_codebook_groups = 16
        self.quantization_mode = "product_vq"
        self.codebook_size = 512
        self.token_dim = sum(self.scales) * self.product_codebook_groups
        self.product_codebooks = nn.ModuleList([nn.Embedding(self.codebook_size, 2) for _ in range(16)])

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(token_ids.shape[0], 8, 7, device=token_ids.device)


def _build_tiny_model() -> QwenVARScaleParallel:
    model = QwenVARScaleParallel.__new__(QwenVARScaleParallel)
    nn.Module.__init__(model)

    hidden_size = 32
    model.config = SimpleNamespace(framework={"parallel_head": {"label_smoothing": 0.0}})
    model.qwen_vl_interface = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=hidden_size), device=torch.device("cpu"))
    )
    model.stage1_tokenizer = _FakeStage1Tokenizer()
    model.token_dim = int(model.stage1_tokenizer.token_dim)
    model.num_factor_slots = int(model.stage1_tokenizer.product_codebook_groups)
    model.parallel_classifier_type = "per_factor"

    scale_indices = []
    time_indices = []
    factor_indices = []
    for scale_idx, scale in enumerate(model.stage1_tokenizer.scales):
        for time_idx in range(scale):
            for factor_idx in range(model.num_factor_slots):
                scale_indices.append(scale_idx)
                time_indices.append(time_idx)
                factor_indices.append(factor_idx)
    model.slot_scale_indices = torch.tensor(scale_indices, dtype=torch.long)
    model.slot_time_indices = torch.tensor(time_indices, dtype=torch.long)
    model.slot_factor_indices = torch.tensor(factor_indices, dtype=torch.long)

    model.action_token_queries = nn.Parameter(torch.randn(model.token_dim, hidden_size) * 0.01)
    model.action_query_cross_attn = nn.ModuleList(
        [ActionCodeQueryBlock(hidden_size, num_heads=4, mlp_ratio=2.0, dropout=0.0)]
    )
    model.action_token_norm = nn.LayerNorm(hidden_size)
    model.action_token_dropout = nn.Dropout(0.0)
    model.action_token_classifier = nn.Linear(hidden_size, model.stage1_tokenizer.codebook_size)
    model.action_factor_classifiers = nn.ModuleList(
        [nn.Linear(hidden_size, model.stage1_tokenizer.codebook_size) for _ in range(model.num_factor_slots)]
    )
    model.code_condition_projectors = nn.ModuleList([nn.Linear(2, hidden_size) for _ in range(16)])
    model.code_condition_norm = nn.LayerNorm(hidden_size)
    model.code_condition_dropout = nn.Dropout(0.0)

    records: list[dict[str, int | None]] = []
    original_predict_scale = QwenVARScaleParallel._predict_scale_logits

    def _recording_predict_scale(
        self,
        context_states,
        pooled_context,
        key_padding_mask,
        code_memory,
        slot_indices,
    ):
        records.append(
            {
                "slot_count": int(slot_indices.numel()),
                "memory_count": None if code_memory is None else int(code_memory.shape[1]),
            }
        )
        return original_predict_scale(
            self,
            context_states,
            pooled_context,
            key_padding_mask,
            code_memory,
            slot_indices,
        )

    def _encode_context(self, examples):
        batch_size = len(examples)
        context_states = torch.randn(batch_size, 5, hidden_size)
        pooled_context = context_states.mean(dim=1)
        return context_states, pooled_context, None

    model._flow_records = records
    model._predict_scale_logits = types.MethodType(_recording_predict_scale, model)
    model._encode_context = types.MethodType(_encode_context, model)
    return model


def main() -> None:
    torch.manual_seed(0)
    model = _build_tiny_model()
    scale_slot_counts = [
        int((model.slot_scale_indices == scale_idx).sum().item())
        for scale_idx in range(len(model.stage1_tokenizer.scales))
    ]
    assert scale_slot_counts == [16, 32, 64, 128], scale_slot_counts
    assert model.token_dim == 240, model.token_dim

    batch_size = 2
    target_tokens = torch.randint(0, model.stage1_tokenizer.codebook_size, (batch_size, model.token_dim))
    dummy_image = Image.new("RGB", (1, 1), color=(0, 0, 0))
    examples = [{"action_tokens": target_tokens[row], "image": dummy_image, "lang": "dummy"} for row in range(batch_size)]

    metrics = model.forward(examples)
    assert metrics["action_loss"].ndim == 0
    assert torch.isfinite(metrics["action_loss"])
    assert 0.0 <= float(metrics["token_accuracy"]) <= 1.0
    assert [item["slot_count"] for item in model._flow_records] == [16, 32, 64, 128]
    assert [item["memory_count"] for item in model._flow_records] == [None, 16, 48, 112]

    model._flow_records.clear()
    out = model.predict_action(examples)
    assert out["action_tokens"].shape == (batch_size, model.token_dim)
    assert out["normalized_actions"].shape == (batch_size, 8, 7)
    assert [item["slot_count"] for item in model._flow_records] == [16, 32, 64, 128]
    assert [item["memory_count"] for item in model._flow_records] == [None, 16, 48, 112]

    print(
        json.dumps(
            {
                "token_dim": model.token_dim,
                "scale_slot_counts": scale_slot_counts,
                "teacher_forcing_memory_counts": [None, 16, 48, 112],
                "status": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
