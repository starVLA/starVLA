# Copyright 2026 starVLA community. All rights reserved.
"""QwenVARParallel: query-based parallel Stage 1 action-code policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenVAR import QwenVAR
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenVARParallelDefaultConfig:
    """Default framework config for parallel Stage 1 code-slot prediction."""

    name: str = "QwenVARParallel"
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-VARAction",
            "attn_implementation": "flash_attention_2",
        }
    )
    stage1_tokenizer: dict = field(
        default_factory=lambda: {
            "artifact": "playground/Checkpoints/var_stage1_pi05_libero/best_recon.ckpt",
            "stage1_config": "examples/LIBERO/train_files/train_var_stage1_pi05_libero.yaml",
            "freeze": True,
            "token_cache": None,
        }
    )
    action_token_text: dict = field(
        default_factory=lambda: {
            "prefix": "<var_action_",
            "suffix": ">",
        }
    )
    parallel_head: dict = field(
        default_factory=lambda: {
            "num_cross_attention_layers": 2,
            "num_attention_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "label_smoothing": 0.0,
            "classifier": "shared",
        }
    )


class ActionCodeQueryBlock(nn.Module):
    """Cross-attention block for action-code queries over VLM context tokens."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        ffn_hidden = max(hidden_size, int(round(hidden_size * float(mlp_ratio))))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_queries = self.query_norm(queries)
        norm_context = self.context_norm(context)
        attn_out, _ = self.cross_attn(
            norm_queries,
            norm_context,
            norm_context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + self.dropout(attn_out)
        queries = queries + self.dropout(self.ffn(self.ffn_norm(queries)))
        return queries


@FRAMEWORK_REGISTRY.register("QwenVARParallel")
class QwenVARParallel(QwenVAR):
    """Qwen-VL policy that predicts all frozen Stage 1 code slots in parallel.

    Unlike ``QwenVAR``, this class does not train Qwen as a left-to-right
    language model over ``<var_action_i>`` text. It encodes image/language once,
    lets one learnable query per Stage 1 code slot cross-attend to the full VLM
    context, and classifies every codebook index in one forward pass.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(QwenVARParallelDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        stage1_path = self.config.framework.stage1_tokenizer.get("artifact", None) or self.config.framework.stage1_tokenizer.get("checkpoint", None)
        if stage1_path is None:
            raise ValueError("QwenVARParallel requires framework.stage1_tokenizer.artifact or .checkpoint.")

        from starVLA.model.modules.action_tokenizer import VARTokenTextCodec, load_frozen_var_action_tokenizer

        artifact = load_frozen_var_action_tokenizer(stage1_path, device="cpu")
        self.stage1_artifact_id = artifact.artifact_id
        self.action_spec = artifact.action_spec
        self.stage1_tokenizer = artifact.tokenizer
        self.stage1_tokenizer.eval()
        for param in self.stage1_tokenizer.parameters():
            param.requires_grad_(False)

        token_cfg = self.config.framework.action_token_text
        self.var_token_codec = VARTokenTextCodec(
            codebook_size=self.stage1_tokenizer.codebook_size,
            prefix=str(token_cfg.get("prefix", "<var_action_")),
            suffix=str(token_cfg.get("suffix", ">")),
        )
        self.token_dim = int(self.stage1_tokenizer.token_dim)
        self._action_token_id_set: set[int] | None = None

        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        codebook_size = int(self.stage1_tokenizer.codebook_size)
        head_cfg = self.config.framework.get("parallel_head", {})
        num_heads = int(head_cfg.get("num_attention_heads", 8))
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_attention_heads={num_heads}.")
        dropout = float(head_cfg.get("dropout", 0.0))

        slot_scale_indices, slot_time_indices, slot_factor_indices = self._build_slot_layout()
        self.register_buffer("slot_scale_indices", slot_scale_indices, persistent=False)
        self.register_buffer("slot_time_indices", slot_time_indices, persistent=False)
        self.register_buffer("slot_factor_indices", slot_factor_indices, persistent=False)
        self.num_factor_slots = int(slot_factor_indices.max().item()) + 1 if slot_factor_indices.numel() else 1

        self.action_token_queries = nn.Parameter(torch.empty(self.token_dim, hidden_size))
        self.action_query_cross_attn = nn.ModuleList(
            [
                ActionCodeQueryBlock(
                    hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=float(head_cfg.get("mlp_ratio", 4.0)),
                    dropout=dropout,
                )
                for _ in range(int(head_cfg.get("num_cross_attention_layers", 2)))
            ]
        )
        self.action_token_norm = nn.LayerNorm(hidden_size)
        self.action_token_dropout = nn.Dropout(dropout)
        self.action_token_classifier = nn.Linear(hidden_size, codebook_size)
        classifier = str(head_cfg.get("classifier", "shared"))
        if classifier not in {"shared", "per_factor"}:
            raise ValueError(f"Unsupported QwenVARParallel classifier: {classifier!r}")
        self.parallel_classifier_type = classifier
        self.action_factor_classifiers = nn.ModuleList(
            [nn.Linear(hidden_size, codebook_size) for _ in range(self.num_factor_slots)]
        )
        nn.init.normal_(self.action_token_queries, mean=0.0, std=hidden_size**-0.5)

    def _build_slot_layout(self) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        scale_indices: list[int] = []
        time_indices: list[int] = []
        factor_indices: list[int] = []
        groups = int(
            self.stage1_tokenizer.product_codebook_groups
            if self.stage1_tokenizer.quantization_mode == "product_vq"
            else 1
        )
        for scale_idx, scale in enumerate(self.stage1_tokenizer.scales):
            for time_idx in range(int(scale)):
                for factor_idx in range(groups):
                    scale_indices.append(scale_idx)
                    time_indices.append(time_idx)
                    factor_indices.append(factor_idx)
        if len(scale_indices) != self.token_dim:
            raise ValueError(f"Stage 1 token layout produced {len(scale_indices)} slots, expected {self.token_dim}.")
        return (
            torch.as_tensor(scale_indices, dtype=torch.long),
            torch.as_tensor(time_indices, dtype=torch.long),
            torch.as_tensor(factor_indices, dtype=torch.long),
        )

    def _pool_condition(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1, :]
        last_indices = attention_mask.long().cumsum(dim=1).argmax(dim=1)
        # The cumsum trick returns 0 for all-left-padding edge cases; use the
        # actual last non-pad index when masks are valid.
        non_pad = attention_mask.bool()
        for row in range(non_pad.shape[0]):
            row_indices = torch.nonzero(non_pad[row], as_tuple=False).flatten()
            if row_indices.numel() > 0:
                last_indices[row] = row_indices[-1]
        return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), last_indices]

    def _key_padding_mask(self, attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        return ~attention_mask.bool()

    def _classify_queries(self, query_states: torch.Tensor, slot_indices: torch.Tensor | None = None) -> torch.Tensor:
        query_states = self.action_token_dropout(self.action_token_norm(query_states.float()))
        if self.parallel_classifier_type == "shared":
            return self.action_token_classifier(query_states)

        logits = query_states.new_empty(
            query_states.shape[0],
            query_states.shape[1],
            int(self.stage1_tokenizer.codebook_size),
        )
        factor_indices = self.slot_factor_indices.to(query_states.device)
        if slot_indices is not None:
            factor_indices = factor_indices.index_select(0, slot_indices.to(query_states.device))
        if factor_indices.numel() != query_states.shape[1]:
            raise ValueError(
                f"factor index count {factor_indices.numel()} does not match query count {query_states.shape[1]}"
            )
        for factor_idx, classifier in enumerate(self.action_factor_classifiers):
            mask = factor_indices == factor_idx
            if mask.any():
                factor_logits = classifier(query_states[:, mask, :])
                if logits.dtype != factor_logits.dtype:
                    logits = logits.to(dtype=factor_logits.dtype)
                logits[:, mask, :] = factor_logits
        return logits

    def _postprocess_decoded_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        reorder = self.config.framework.stage1_tokenizer.get("decoded_action_reorder", None)
        if reorder is None:
            return normalized_actions
        indices = torch.as_tensor([int(item) for item in reorder], device=normalized_actions.device, dtype=torch.long)
        if indices.numel() != normalized_actions.shape[-1]:
            raise ValueError(
                "decoded_action_reorder length must match action_dim: "
                f"got {indices.numel()}, expected {normalized_actions.shape[-1]}"
            )
        return normalized_actions.index_select(-1, indices)

    def _predict_token_logits(self, examples: List[dict]) -> torch.Tensor:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        qwen_inputs = self._build_qwenvl_inputs(images=batch_images, instructions=instructions)

        use_cuda_autocast = self.qwen_vl_interface.model.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda_autocast):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        context_states = outputs.hidden_states[-1].float()
        pooled_context = self._pool_condition(context_states, qwen_inputs.get("attention_mask", None))
        query_states = self.action_token_queries[None, :, :].expand(context_states.shape[0], -1, -1).float()
        query_states = query_states + pooled_context[:, None, :]
        key_padding_mask = self._key_padding_mask(qwen_inputs.get("attention_mask", None))
        for block in self.action_query_cross_attn:
            query_states = block(query_states, context_states, key_padding_mask=key_padding_mask)
        return self._classify_queries(query_states)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict[str, torch.Tensor]:
        logits = self._predict_token_logits(examples)
        target_tokens = torch.stack([example["action_tokens"].long() for example in examples], dim=0).to(logits.device)
        label_smoothing = float(self.config.framework.get("parallel_head", {}).get("label_smoothing", 0.0))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_tokens.reshape(-1),
            label_smoothing=label_smoothing,
        )
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(target_tokens)
            token_accuracy = correct.float().mean()
            token_losses = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                target_tokens.reshape(-1),
                reduction="none",
            ).view_as(target_tokens)

        metrics: dict[str, torch.Tensor] = {"action_loss": loss, "token_accuracy": token_accuracy}
        scale_indices = self.slot_scale_indices.to(logits.device)
        factor_indices = self.slot_factor_indices.to(logits.device)
        with torch.no_grad():
            for scale_idx, scale in enumerate(self.stage1_tokenizer.scales):
                mask = scale_indices == scale_idx
                if mask.any():
                    metrics[f"loss/scale_{int(scale)}"] = token_losses[:, mask].mean()
                    metrics[f"acc/scale_{int(scale)}"] = correct[:, mask].float().mean()
            for factor_idx in range(self.num_factor_slots):
                mask = factor_indices == factor_idx
                if mask.any():
                    metrics[f"loss/codebook_group_{factor_idx}"] = token_losses[:, mask].mean()
                    metrics[f"acc/codebook_group_{factor_idx}"] = correct[:, mask].float().mean()
        return metrics

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict[str, np.ndarray]:
        if not isinstance(examples, list):
            examples = [examples]
        converted = []
        for example in examples:
            item = dict(example)
            item["image"] = to_pil_preserve(example["image"])
            converted.append(item)

        logits = self._predict_token_logits(converted)
        token_ids = logits.argmax(dim=-1).long()
        token_ids = token_ids.to(next(self.stage1_tokenizer.parameters()).device)
        normalized_actions = self.stage1_tokenizer.decode(token_ids)
        normalized_actions = self._postprocess_decoded_actions(normalized_actions)
        normalized_actions = normalized_actions.detach().cpu().float().numpy()
        return {
            "normalized_actions": normalized_actions,
            "action_tokens": token_ids.detach().cpu().numpy(),
            "generation_diagnostics": [
                {
                    "valid_token_count": self.token_dim,
                    "padded": False,
                    "parallel": True,
                    "classifier": self.parallel_classifier_type,
                }
                for _ in converted
            ],
        }
