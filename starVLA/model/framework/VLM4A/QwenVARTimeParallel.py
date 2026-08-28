# Copyright 2026 starVLA community. All rights reserved.
"""Time-autoregressive, factor-parallel Stage 1 action-code policy."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenVARParallel import QwenVARParallel
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenVARTimeParallel")
class QwenVARTimeParallel(QwenVARParallel):
    """Predict one action timestep at a time, with factor groups in parallel.

    For the g16_s8 product-VQ tokenizer the Stage 1 code map is [T=8, G=16].
    This model predicts the 16 codebook-group tokens for each timestep jointly,
    while conditioning each timestep on the code embeddings from previous
    timesteps. It reduces decoding steps from T * G to T without collapsing the
    entire action chunk into a single non-autoregressive prediction.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        if self.stage1_tokenizer.quantization_mode != "product_vq":
            raise ValueError("QwenVARTimeParallel currently expects product_vq Stage 1 tokens.")

        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        head_cfg = self.config.framework.get("parallel_head", {})
        dropout = float(head_cfg.get("dropout", 0.0))
        num_heads = int(head_cfg.get("num_attention_heads", 8))
        factor_layers = int(head_cfg.get("num_factor_self_attention_layers", 1))
        ffn_hidden = max(hidden_size, int(round(hidden_size * float(head_cfg.get("mlp_ratio", 4.0)))))

        self.code_condition_projectors = nn.ModuleList(
            [nn.Linear(codebook.embedding_dim, hidden_size) for codebook in self.stage1_tokenizer.product_codebooks]
        )
        if len(self.code_condition_projectors) != self.num_factor_slots:
            raise ValueError(
                f"Expected {self.num_factor_slots} factor projectors, got {len(self.code_condition_projectors)}."
            )
        self.code_condition_norm = nn.LayerNorm(hidden_size)
        self.code_condition_dropout = nn.Dropout(
            float(head_cfg.get("code_condition_dropout", dropout))
        )
        self.factor_self_attn = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=ffn_hidden,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(factor_layers)
            ]
        )

        unique_times = torch.unique(self.slot_time_indices, sorted=True)
        self.num_time_steps = int(unique_times.numel())
        if self.num_time_steps <= 0:
            raise ValueError("Stage 1 token layout has no time slots.")

    def _encode_context(self, examples: List[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        key_padding_mask = self._key_padding_mask(qwen_inputs.get("attention_mask", None))
        return context_states, pooled_context, key_padding_mask

    def _slot_indices_for_time(self, time_idx: int, device: torch.device) -> torch.Tensor:
        mask = self.slot_time_indices.to(device) == int(time_idx)
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _classify_slot_queries(self, query_states: torch.Tensor, slot_indices: torch.Tensor) -> torch.Tensor:
        query_states = self.action_token_dropout(self.action_token_norm(query_states.float()))
        if self.parallel_classifier_type == "shared":
            return self.action_token_classifier(query_states)

        logits = query_states.new_empty(
            query_states.shape[0],
            query_states.shape[1],
            int(self.stage1_tokenizer.codebook_size),
        )
        factor_indices = self.slot_factor_indices.to(query_states.device).index_select(0, slot_indices)
        for factor_idx, classifier in enumerate(self.action_factor_classifiers):
            mask = factor_indices == factor_idx
            if mask.any():
                factor_logits = classifier(query_states[:, mask, :])
                if logits.dtype != factor_logits.dtype:
                    logits = logits.to(dtype=factor_logits.dtype)
                logits[:, mask, :] = factor_logits
        return logits

    def _code_condition_embeddings(self, tokens: torch.Tensor, slot_indices: torch.Tensor) -> torch.Tensor:
        factor_indices = self.slot_factor_indices.to(tokens.device).index_select(0, slot_indices)
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        pieces = torch.empty(
            tokens.shape[0],
            slot_indices.numel(),
            hidden_size,
            device=tokens.device,
            dtype=torch.float32,
        )

        for factor_idx, (codebook, projector) in enumerate(
            zip(self.stage1_tokenizer.product_codebooks, self.code_condition_projectors, strict=True)
        ):
            mask = factor_indices == factor_idx
            if mask.any():
                code_vectors = codebook(tokens[:, mask].long()).float()
                pieces[:, mask, :] = projector(code_vectors).float()

        query_pos = self.action_token_queries.index_select(0, slot_indices).float()
        pieces = pieces + query_pos.unsqueeze(0)
        return self.code_condition_dropout(self.code_condition_norm(pieces))

    def _predict_time_logits(
        self,
        context_states: torch.Tensor,
        pooled_context: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        code_memory: torch.Tensor | None,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor:
        query_states = self.action_token_queries.index_select(0, slot_indices)[None, :, :]
        query_states = query_states.expand(context_states.shape[0], -1, -1).float()
        query_states = query_states + pooled_context[:, None, :]

        if code_memory is None:
            attention_context = context_states
            attention_mask = key_padding_mask
        else:
            attention_context = torch.cat([context_states, code_memory.to(context_states.dtype)], dim=1)
            if key_padding_mask is None:
                attention_mask = None
            else:
                extra_mask = torch.zeros(
                    code_memory.shape[:2],
                    device=key_padding_mask.device,
                    dtype=key_padding_mask.dtype,
                )
                attention_mask = torch.cat([key_padding_mask, extra_mask], dim=1)

        for block in self.action_query_cross_attn:
            query_states = block(query_states, attention_context, key_padding_mask=attention_mask)
        for block in self.factor_self_attn:
            query_states = block(query_states)
        return self._classify_slot_queries(query_states, slot_indices)

    def _predict_token_logits_teacher_forced(self, examples: List[dict], target_tokens: torch.Tensor) -> torch.Tensor:
        context_states, pooled_context, key_padding_mask = self._encode_context(examples)
        logits = context_states.new_empty(
            target_tokens.shape[0],
            self.token_dim,
            int(self.stage1_tokenizer.codebook_size),
        )
        code_memory = None
        for time_idx in range(self.num_time_steps):
            slot_indices = self._slot_indices_for_time(time_idx, target_tokens.device)
            time_logits = self._predict_time_logits(
                context_states,
                pooled_context,
                key_padding_mask,
                code_memory,
                slot_indices,
            )
            logits.index_copy_(1, slot_indices, time_logits.float())
            time_tokens = target_tokens.index_select(1, slot_indices)
            time_memory = self._code_condition_embeddings(time_tokens, slot_indices)
            code_memory = time_memory if code_memory is None else torch.cat([code_memory, time_memory], dim=1)
        return logits

    def forward(self, examples: List[dict] = None, **kwargs) -> dict[str, torch.Tensor]:
        target_tokens = torch.stack([example["action_tokens"].long() for example in examples], dim=0)
        target_tokens = target_tokens.to(self.qwen_vl_interface.model.device)
        logits = self._predict_token_logits_teacher_forced(examples, target_tokens)
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
        time_indices = self.slot_time_indices.to(logits.device)
        factor_indices = self.slot_factor_indices.to(logits.device)
        with torch.no_grad():
            for time_idx in range(self.num_time_steps):
                mask = time_indices == time_idx
                if mask.any():
                    metrics[f"loss/time_{time_idx}"] = token_losses[:, mask].mean()
                    metrics[f"acc/time_{time_idx}"] = correct[:, mask].float().mean()
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

        context_states, pooled_context, key_padding_mask = self._encode_context(converted)
        token_ids = torch.empty(context_states.shape[0], self.token_dim, device=context_states.device, dtype=torch.long)
        code_memory = None
        for time_idx in range(self.num_time_steps):
            slot_indices = self._slot_indices_for_time(time_idx, context_states.device)
            time_logits = self._predict_time_logits(
                context_states,
                pooled_context,
                key_padding_mask,
                code_memory,
                slot_indices,
            )
            time_tokens = time_logits.argmax(dim=-1).long()
            token_ids.index_copy_(1, slot_indices, time_tokens)
            time_memory = self._code_condition_embeddings(time_tokens, slot_indices)
            code_memory = time_memory if code_memory is None else torch.cat([code_memory, time_memory], dim=1)

        token_ids = token_ids.to(next(self.stage1_tokenizer.parameters()).device)
        normalized_actions = self.stage1_tokenizer.decode(token_ids).detach().cpu().float().numpy()
        return {
            "normalized_actions": normalized_actions,
            "action_tokens": token_ids.detach().cpu().numpy(),
            "generation_diagnostics": [
                {
                    "valid_token_count": self.token_dim,
                    "padded": False,
                    "parallel": True,
                    "time_autoregressive": True,
                    "factor_parallel": True,
                    "decoding_steps": self.num_time_steps,
                    "classifier": self.parallel_classifier_type,
                }
                for _ in converted
            ],
        }
