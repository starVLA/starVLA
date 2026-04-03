# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
Qwen-VL + Flow-matching head to directly predict continuous actions

LangForceV5:
(1) Assert language span consistency between prior/post branches (token-level exact match)
(2) Hard-token LLR + Shortcut gate
(3) Optional detach of prior condition to avoid pushing backbone to vision-only shortcut
"""

import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# ===== Qwen special tokens (you confirmed) =====
VISION_START_TOKEN_INDEX = 151652  # <|vision_start|>
VISION_END_TOKEN_INDEX = 151654  # <|vision_end|>
IMAGE_TOKEN_INDEX = 151655  # <|image_pad|>
VIDEO_TOKEN_INDEX = 151656  # <|video_pad|>
IM_START_TOKEN_INDEX = 151644  # <|im_start|>
IM_END_TOKEN_INDEX = 151645  # <|im_end|>

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for LangForce
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class LangForceDefaultConfig:
    """LangForce framework default parameters.

    Dual-branch VLA with Bayesian decomposition:
      - Prior branch (V + A + L) and posterior branch (V + L + A)
      - LLR regularizer with optional hard-token/gate mechanisms
    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.
    """

    # --- Registry identifier ---
    name: str = "LangForce"

    # === VLM backbone (Qwen3-VL with latent action query tokens) ===
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        "attn_implementation": "flash_attention_2",
        # Number of latent action query tokens injected
        "num_latent_action_query": 32,
    })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "DiT-B",
        "action_hidden_dim": 1024,
        "hidden_size": 1024,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "action_dim": 7,
        "state_dim": 7,
        "future_action_window_size": 7,
        "action_horizon": 8,
        "past_action_window_size": 0,
        "repeated_diffusion_steps": 4,
        "num_inference_timesteps": 4,
        "diffusion_model_cfg": {
            "cross_attention_dim": 2048,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": 16,
            "output_dim": 1024,
            "positional_embeddings": None,
        },
    })

    # === Observation image size (optional resize before encoding) ===
    obs_image_size: Optional[list] = None

    # === LangForce-specific loss / regularizer weights ===
    # KL weight: maximize LLR via -kl_weight * kl_loss
    kl_weight: float = 0.1
    # Weight for prior branch flow-matching loss
    prior_loss_weight: float = 0.3
    # Whether to assert language span token-level match between prior/post
    assert_lang_span_match: bool = True
    # Whether to detach prior condition (prevent vision-only drift)
    detach_prior_cond: bool = True
    # Hard-token LLR: use top-k hardest tokens under posterior
    use_hard_token_llr: bool = False
    hard_token_k: int = 16
    # Shortcut gate: down-weight LLR when log p(L|V) is already low
    use_kl_gate: bool = False
    kl_gate_momentum: float = 0.99
    kl_gate_temp: float = 0.5
    kl_gate_tau_scale: float = 0.7
    kl_gate_min: float = 0.0
    kl_gate_max: float = 1.0


@FRAMEWORK_REGISTRY.register("LangForce")
class LangForce(baseframework):
    """
    LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries (arxiv 2601.15197)

    Dual-branch VLA with:
      - Prior branch: (V + A + L) => proposal-like p(a|v) head
      - Posterior branch: (V + L + A) => pi(a|v,l)
      - LLR regularizer: maximize log p(L|V,A_prior) - sg(log p(L|V))
        with:
          * Hard-token LLR (top-k hardest tokens under post)
          * Shortcut gate (down-weight LLR when log p(L|V) is already very low)
      - Optional detach prior cond (protect backbone from vision-only drift)

    Additionally:
      - Training-time assertion: extracted language spans in prior/post must match exactly (token-level).
        If mismatch => raise AssertionError with decoded spans.
      - LangForce utilizes Qwen3-VL and extends the vocabulary with specialized tokens that serve as Latent Action Queries.
        Run the provided example script add_token.py in https://github.com/ZGC-EmbodyAI/LangForce to update the tokenizer with these additional tokens.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(LangForceDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # align cross_attention_dim to VLM hidden_size at runtime
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.num_latent_action_query = self.config.framework.qwenvl.get("num_latent_action_query", 32)
        self.latent_action_query = "".join([f"<|action_{i}|>" for i in range(self.num_latent_action_query)])
        self.action_token_ids = None  # cached {'first','last'}

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # ===== Loss weights =====
        self.kl_weight = float(self.config.framework.get("kl_weight", 0.1))  # maximize LLR via -kl_weight * kl_loss
        self.prior_loss_weight = float(self.config.framework.get("prior_loss_weight", 0.3))

        # ===== (0) training assert switch =====
        self.assert_lang_span_match = bool(self.config.framework.get("assert_lang_span_match", True))

        # ===== (1) detach prior cond switch =====
        self.detach_prior_cond = bool(self.config.framework.get("detach_prior_cond", True))

        # ===== (2) Hard-token LLR =====
        self.use_hard_token_llr = bool(self.config.framework.get("use_hard_token_llr", False))
        self.hard_token_k = int(self.config.framework.get("hard_token_k", 16))
        assert self.hard_token_k > 0

        # ===== (3) Shortcut gate =====
        # gate computed from posterior language-span NLL: high NLL => log p(L|V) low => gate small
        self.use_kl_gate = bool(self.config.framework.get("use_kl_gate", False))
        self.kl_gate_momentum = float(self.config.framework.get("kl_gate_momentum", 0.99))
        self.kl_gate_temp = float(self.config.framework.get("kl_gate_temp", 0.5))
        self.kl_gate_tau_scale = float(self.config.framework.get("kl_gate_tau_scale", 0.7))  # scale EMA threshold
        self.kl_gate_min = float(self.config.framework.get("kl_gate_min", 0.0))
        self.kl_gate_max = float(self.config.framework.get("kl_gate_max", 1.0))

        # cache some special token ids from tokenizer lazily
        self._im_end_id = None

        # EMA buffer for posterior language-span NLL
        self.register_buffer("post_nll_ema", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("post_nll_ema_inited", torch.tensor(0, dtype=torch.uint8))

    # ---------------------------------------------------------------------
    # Token id helpers
    # ---------------------------------------------------------------------
    def _ensure_action_token_ids(self, tokenizer):
        if self.action_token_ids is None:
            self.action_token_ids = {
                "first": tokenizer.convert_tokens_to_ids("<|action_0|>"),
                "last": tokenizer.convert_tokens_to_ids(f"<|action_{self.num_latent_action_query-1}|>"),
            }

    def _ensure_im_end_id(self, tokenizer):
        if self._im_end_id is None:
            self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _find_last_pos(self, seq_1d: torch.Tensor, token_id: int) -> int:
        idx = (seq_1d == int(token_id)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return -1
        return int(idx[-1].item())

    def _find_first_pos_after(self, seq_1d: torch.Tensor, token_id: int, start: int) -> int:
        if start < 0:
            start = 0
        sub = seq_1d[start:]
        idx = (sub == int(token_id)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return -1
        return int(start + idx[0].item())

    # ---------------------------------------------------------------------
    # Action block helpers
    # ---------------------------------------------------------------------
    def _get_action_block_start(self, input_ids_1d: torch.Tensor, tokenizer) -> int:
        self._ensure_action_token_ids(tokenizer)
        first_id = self.action_token_ids["first"]
        last_id = self.action_token_ids["last"]

        pos = (input_ids_1d == int(first_id)).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            return -1

        start = int(pos[0].item())
        end = start + self.num_latent_action_query
        if end > input_ids_1d.shape[0]:
            return -1
        if int(input_ids_1d[end - 1].item()) != int(last_id):
            return -1
        return start

    def _extract_action_query_hidden_states(
        self,
        hidden_states: torch.Tensor,  # [B, S, H]
        input_ids: torch.Tensor,  # [B, S]
        tokenizer,
        return_starts: bool = False,
    ):
        self._ensure_action_token_ids(tokenizer)

        B = hidden_states.shape[0]
        out = []
        starts = []
        for b in range(B):
            start = self._get_action_block_start(input_ids[b], tokenizer)
            assert start != -1, "No valid contiguous action token block found in the sequence."
            end = start + self.num_latent_action_query
            out.append(hidden_states[b, start:end, :])
            starts.append(start)

        out = torch.stack(out, dim=0)  # [B, K, H]
        if return_starts:
            return out, torch.tensor(starts, device=input_ids.device, dtype=torch.long)
        return out

    # ---------------------------------------------------------------------
    # SHIFT-correct token-level NLL span
    # ---------------------------------------------------------------------
    def _token_nll_span(
        self,
        logits_1d: torch.Tensor,  # [S, V]
        input_ids_1d: torch.Tensor,  # [S]
        start: int,
        end: int,
        ignore_ids: Optional[Set[int]] = None,
    ):
        """
        Return (nll_vec, target_ids_vec) for tokens in [start,end),
        using next-token alignment:
          token at position j is scored by logits[j-1] (requires j>0).
        """
        if end <= start:
            return None, None
        S = int(input_ids_1d.shape[0])
        start = max(0, int(start))
        end = min(S, int(end))
        if end <= start:
            return None, None

        j = torch.arange(start, end, device=input_ids_1d.device, dtype=torch.long)
        j = j[j > 0]
        if j.numel() == 0:
            return None, None

        targets = input_ids_1d[j].long()

        if ignore_ids is not None and len(ignore_ids) > 0:
            keep = torch.ones_like(targets, dtype=torch.bool)
            for tid in ignore_ids:
                keep &= targets != int(tid)
            j = j[keep]
            if j.numel() == 0:
                return None, None
            targets = input_ids_1d[j].long()

        pred_pos = j - 1
        pred_logits = logits_1d[pred_pos].float()  # [T, V]
        nll = F.cross_entropy(pred_logits, targets, reduction="none")  # [T]
        return nll, targets

    # ---------------------------------------------------------------------
    # Compute LLR with:
    #   - strict span equality assertion (training)
    #   - hard-token LLR (top-k)
    #   - shortcut gate based on posterior NLL
    # ---------------------------------------------------------------------
    def _compute_language_llr_from_boundaries(
        self,
        priori_logits: torch.Tensor,  # [B, S, V]
        posteriori_logits: torch.Tensor,  # [B, S, V] (detached)
        priori_input_ids: torch.Tensor,  # [B, S]
        posteriori_input_ids: torch.Tensor,  # [B, S]
        priori_action_starts: torch.Tensor,  # [B]
        posteriori_action_starts: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        self._ensure_im_end_id(tokenizer)

        pad_id = tokenizer.pad_token_id
        ignore_ids: Set[int] = set()
        if pad_id is not None:
            ignore_ids.add(int(pad_id))
        ignore_ids.add(int(IMAGE_TOKEN_INDEX))
        ignore_ids.add(int(VIDEO_TOKEN_INDEX))
        ignore_ids.add(int(VISION_START_TOKEN_INDEX))
        ignore_ids.add(int(VISION_END_TOKEN_INDEX))
        ignore_ids.add(int(IM_START_TOKEN_INDEX))
        ignore_ids.add(int(IM_END_TOKEN_INDEX))

        B = int(priori_input_ids.shape[0])
        K = self.num_latent_action_query

        llr_vals = []
        post_nll_means = []

        for b in range(B):
            ids_prior = priori_input_ids[b]
            ids_post = posteriori_input_ids[b]

            a_start_prior = int(priori_action_starts[b].item())
            a_start_post = int(posteriori_action_starts[b].item())

            # ===== prior language span: [action_end : im_end) =====
            lang_start_prior = a_start_prior + K
            if lang_start_prior >= ids_prior.shape[0]:
                continue
            im_end = self._find_first_pos_after(ids_prior, self._im_end_id, lang_start_prior)
            lang_end_prior = im_end if im_end != -1 else int(ids_prior.shape[0])
            if lang_end_prior <= lang_start_prior:
                continue

            # ===== post language span: [last(vision_end)+1 : action_start) =====
            v_end_post = self._find_last_pos(ids_post, VISION_END_TOKEN_INDEX)
            if v_end_post == -1:
                continue
            lang_start_post = v_end_post + 1
            lang_end_post = a_start_post
            if lang_end_post <= lang_start_post:
                continue

            # ===== (1) strict assertion: token-level equality =====
            if self.training and self.assert_lang_span_match:
                prior_span_ids = ids_prior[lang_start_prior:lang_end_prior]
                post_span_ids = ids_post[lang_start_post:lang_end_post]

                if (prior_span_ids.numel() != post_span_ids.numel()) or (not torch.equal(prior_span_ids, post_span_ids)):
                    # decode for human-readable debugging
                    prior_text = tokenizer.decode(prior_span_ids.tolist())
                    post_text = tokenizer.decode(post_span_ids.tolist())

                    raise AssertionError(
                        "\n[LangForceV5] Language span mismatch detected!\n"
                        f"Sample b={b}\n"
                        f"PRIOR span idx: [{lang_start_prior}:{lang_end_prior}]  (len={prior_span_ids.numel()})\n"
                        f"POST  span idx: [{lang_start_post}:{lang_end_post}]  (len={post_span_ids.numel()})\n"
                        f"PRIOR span: {prior_text!r}\n"
                        f"POST  span: {post_text!r}\n"
                        f"PRIOR token ids (first 50): {prior_span_ids[:50].tolist()}\n"
                        f"POST  token ids (first 50): {post_span_ids[:50].tolist()}\n"
                        "This indicates your boundary-based language extraction is inconsistent (likely prompt/template issue)."
                    )

            # ===== (2) hard-token LLR needs token-level aligned targets =====
            nll_prior, tok_prior = self._token_nll_span(
                logits_1d=priori_logits[b],
                input_ids_1d=ids_prior,
                start=lang_start_prior,
                end=lang_end_prior,
                ignore_ids=ignore_ids,
            )
            nll_post, tok_post = self._token_nll_span(
                logits_1d=posteriori_logits[b],
                input_ids_1d=ids_post,
                start=lang_start_post,
                end=lang_end_post,
                ignore_ids=ignore_ids,
            )
            if nll_prior is None or nll_post is None:
                continue

            # record post nll mean for gate
            post_nll_mean = nll_post.mean().detach()
            post_nll_means.append(post_nll_mean)

            # logp_prior - logp_post = (-nll_prior) - (-nll_post) = nll_post - nll_prior
            if self.use_hard_token_llr:
                # require same target token sequence
                if (
                    tok_prior is None
                    or tok_post is None
                    or tok_prior.shape != tok_post.shape
                    or (not torch.equal(tok_prior, tok_post))
                ):
                    # This should not happen if your spans match, but keep safe fallback.
                    llr = nll_post.mean() - nll_prior.mean()
                else:
                    k = min(self.hard_token_k, int(nll_post.numel()))
                    if k <= 0:
                        continue
                    idx = torch.topk(nll_post.detach(), k=k, largest=True).indices
                    llr = (nll_post[idx] - nll_prior[idx]).mean()
            else:
                llr = nll_post.mean() - nll_prior.mean()

            llr_vals.append(llr)

        if len(llr_vals) == 0:
            return torch.tensor(0.0, device=priori_logits.device, dtype=torch.float32)

        llr_vals_t = torch.stack(llr_vals).float()  # [M]
        post_nll_means_t = torch.stack(post_nll_means).float()  # [M]

        # ===== (2) shortcut gate: update EMA threshold =====
        if self.use_kl_gate and self.training:
            batch_mean = post_nll_means_t.mean().detach()
            with torch.no_grad():
                if int(self.post_nll_ema_inited.item()) == 0:
                    self.post_nll_ema.copy_(batch_mean)
                    self.post_nll_ema_inited.fill_(1)
                else:
                    m = self.kl_gate_momentum
                    self.post_nll_ema.copy_(m * self.post_nll_ema + (1.0 - m) * batch_mean)

        # ===== gate computation =====
        if self.use_kl_gate:
            tau = self.post_nll_ema.detach() * float(self.kl_gate_tau_scale)
            temp = max(float(self.kl_gate_temp), 1e-6)
            # high nll => log p(L|V) low => gate small
            g = torch.sigmoid((tau - post_nll_means_t) / temp)
            # optional clamp/scale
            if self.kl_gate_min != 0.0 or self.kl_gate_max != 1.0:
                g = float(self.kl_gate_min) + (float(self.kl_gate_max) - float(self.kl_gate_min)) * g
        else:
            g = torch.ones_like(post_nll_means_t)

        # weighted LLR
        return (g * llr_vals_t).mean()

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        batch_images = [example["image"] for example in examples]  # [B, [PIL...]]
        instructions_priori = [self.latent_action_query + example["lang"] for example in examples]  # A + L
        instructions_posteriori = [example["lang"] + self.latent_action_query for example in examples]  # L + A

        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # ===== Step 1: Priori Branch (V + A + L) =====
        qwen_inputs_priori = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions_priori
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs_priori = self.qwen_vl_interface(
                **qwen_inputs_priori,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            priori_last_hidden = qwenvl_outputs_priori.hidden_states[-1]  # [B, S, H]
            priori_action_hidden, priori_action_starts = self._extract_action_query_hidden_states(
                priori_last_hidden,
                qwen_inputs_priori["input_ids"],
                self.qwen_vl_interface.processor.tokenizer,
                return_starts=True,
            )  # [B, K, H], [B]
            priori_logits = qwenvl_outputs_priori.logits  # [B, S, V]

        # ===== Step 2: Posteriori Branch (V + L + A) =====
        qwen_inputs_posteriori = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions_posteriori
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs_posteriori = self.qwen_vl_interface(
                **qwen_inputs_posteriori,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            posteriori_last_hidden = qwenvl_outputs_posteriori.hidden_states[-1]  # [B, S, H]
            posteriori_action_hidden, posteriori_action_starts = self._extract_action_query_hidden_states(
                posteriori_last_hidden,
                qwen_inputs_posteriori["input_ids"],
                self.qwen_vl_interface.processor.tokenizer,
                return_starts=True,
            )  # [B, K, H], [B]

            # detach baseline logits: do not allow worsening log p(L|V) to inflate LLR
            posteriori_logits = qwenvl_outputs_posteriori.logits.detach()  # [B, S, V]

        # ===== Step 3: LLR loss (Hard-token + Gate + Assert) =====
        kl_loss = self._compute_language_llr_from_boundaries(
            priori_logits=priori_logits,
            posteriori_logits=posteriori_logits,
            priori_input_ids=qwen_inputs_priori["input_ids"],
            posteriori_input_ids=qwen_inputs_posteriori["input_ids"],
            priori_action_starts=priori_action_starts,
            posteriori_action_starts=posteriori_action_starts,
        )

        # ===== Step 4: Action head losses =====
        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = torch.tensor(
                np.array(actions), device=priori_action_hidden.device, dtype=priori_action_hidden.dtype
            )
            actions_target = actions_t[:, -(self.future_action_window_size + 1) :, :]  # [B, chunk_len, action_dim]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )

            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=priori_action_hidden.device, dtype=priori_action_hidden.dtype
                )

            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)

            # (3) detach prior condition switch
            if self.detach_prior_cond:
                priori_cond_base = priori_action_hidden.detach()
            else:
                priori_cond_base = priori_action_hidden

            priori_cond = priori_cond_base.repeat(repeated_diffusion_steps, 1, 1).float()
            posteriori_cond = posteriori_action_hidden.repeat(repeated_diffusion_steps, 1, 1).float()
            state_repeated = state_tensor.repeat(repeated_diffusion_steps, 1, 1) if state_tensor is not None else None

            prior_loss = self.action_model(priori_cond, actions_target_repeated, state_repeated)
            main_loss = self.action_model(posteriori_cond, actions_target_repeated, state_repeated)

        # ===== Step 5: Total loss (keep your preferred convex mixture) =====
        total_loss = (
            (1.0 - self.prior_loss_weight) * main_loss + self.prior_loss_weight * prior_loss - self.kl_weight * kl_loss
        )

        return {
            "action_loss": total_loss,
            # optional logs:
            "main_loss": main_loss.detach(),
            "prior_loss": prior_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> dict:
        """
        Inference uses Posteriori branch: (V + L + action_query)
        """
        if type(examples) is not list:
            examples = [examples]

        # robustly preserve PIL for each view
        batch_images = []
        for ex in examples:
            imgs = ex["image"]
            if isinstance(imgs, list):
                batch_images.append([to_pil_preserve(im) for im in imgs])
            else:
                batch_images.append([to_pil_preserve(imgs)])

        instructions_posteriori = [ex["lang"] + self.latent_action_query for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions_posteriori
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

            last_hidden = qwenvl_outputs.hidden_states[-1]
            action_hidden = self._extract_action_query_hidden_states(
                last_hidden, qwen_inputs["input_ids"], self.qwen_vl_interface.processor.tokenizer, return_starts=False
            )  # [B, K, H]

        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(action_hidden.device, dtype=action_hidden.dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(action_hidden, state_tensor)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml", type=str, default="./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml"
    )
    args, clipargs = parser.parse_known_args()

    try:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()
    except (ImportError, RuntimeError):
        pass

    # args.config_yaml = "examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml"
    cfg = OmegaConf.load(args.config_yaml)

    model: LangForce = LangForce(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "Put all the toys in the child's room ... inside the toy box.",
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "Put all the toys in the child's room ... inside the toy box.",
    }

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    out = model(batch)
    print("Action Loss:", out["action_loss"].item(), "KL Loss:", out["kl_loss"].item())

    pred = model.predict_action([sample])
    print("Pred shape:", pred["normalized_actions"].shape)

    # # optional dataloader test (requires data)
    # vla_dataset_cfg = cfg.datasets.vla_data
    # from torch.utils.data import DataLoader
    # from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
    # cfg.datasets.vla_data.include_state = "False"
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    # train_dataloader = DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=collate_fn)
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     model(batch)
    #     break
    print("Finished")
