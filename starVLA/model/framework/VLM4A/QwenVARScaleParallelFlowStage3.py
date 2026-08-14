# Copyright 2026 starVLA community. All rights reserved.
"""Stage3 for scale-parallel VAR pretraining plus a flow action expert."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenVARScaleParallel import QwenVARScaleParallel
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    get_action_model as get_layerwise_action_model,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("QwenVARScaleParallelPiFlowStage3")
class QwenVARScaleParallelPiFlowStage3(QwenVARScaleParallel):
    """Stage2 scale-parallel VAR CE plus a strict QwenPI_v3-style Flow head.

    This variant follows the π-style implementation used by ``QwenPI_v3``:
    proprioceptive state is discretized into text tokens and appended to the
    language instruction, QwenVL emits layer-wise hidden states, each selected
    layer is projected into the Action DiT hidden space, and a layer-wise
    cross-DiT flow head learns continuous action chunks. The Stage2 VAR branch
    is left intact and still contributes its original CE loss.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None)
        if action_dit_hidden_dim is None:
            action_dit_hidden_dim = llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_layerwise_action_model(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)
        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if llm_hidden_size == self.action_dit_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                    )
                )
                for _ in range(self.num_action_dit_layers)
            ]
        )
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _project_vl_hidden_for_action(self, vl_embs_list: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(vl_embs_list) != len(self.project_layers):
            raise ValueError(
                f"Layer number mismatch: got {len(vl_embs_list)} VL layers, "
                f"but project_layers has {len(self.project_layers)} layers."
            )
        return [proj(vl_h) for proj, vl_h in zip(self.project_layers, vl_embs_list)]

    def _encode_flow_hidden_states(self, *, images: list, instructions: list[str]) -> list[torch.Tensor]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        use_cuda_autocast = self.qwen_vl_interface.model.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda_autocast):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            vl_embs_list = list(outputs.hidden_states[-self.num_action_dit_layers :])
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)
        return vl_embs_list

    def forward(self, examples: List[dict] = None, **kwargs) -> dict[str, torch.Tensor]:
        stage2_metrics = QwenVARScaleParallel.forward(self, examples=examples, **kwargs)
        var_ce_loss = stage2_metrics["action_loss"]
        if var_ce_loss is None or not torch.isfinite(var_ce_loss.detach()).all().item():
            trainable_param = next(param for param in self.qwen_vl_interface.parameters() if param.requires_grad)
            var_ce_loss = trainable_param.float().sum() * 0.0

        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        states = [example["state"] for example in examples] if "state" in examples[0] else None
        if states is not None:
            instructions = add_discretized_state_to_instruction(instructions, states)

        vl_embs_list = self._encode_flow_hidden_states(images=batch_images, instructions=instructions)
        base_hidden = vl_embs_list[-1]
        actions = torch.tensor(
            np.array([example["action"] for example in examples]),
            device=base_hidden.device,
            dtype=base_hidden.dtype,
        )
        actions_target = actions[:, -self.action_horizon :, :]

        repeated_diffusion_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 2))
        actions_target = actions_target.repeat(repeated_diffusion_steps, 1, 1)
        vl_embs_list = [hidden.repeat(repeated_diffusion_steps, 1, 1) for hidden in vl_embs_list]

        with torch.autocast("cuda", dtype=torch.float32, enabled=base_hidden.device.type == "cuda"):
            flow_loss = self.action_model(vl_embs_list, actions_target, None)

        stage3_cfg = self.config.framework.get("stage3", {})
        ce_weight = float(stage3_cfg.get("var_ce_weight", 1.0))
        flow_weight = float(stage3_cfg.get("flow_loss_weight", 1.0))
        total_loss = ce_weight * var_ce_loss + flow_weight * flow_loss

        metrics = {key: value for key, value in stage2_metrics.items() if key != "action_loss"}
        metrics.update(
            {
                "action_loss": total_loss,
                "var_ce_loss": var_ce_loss.detach(),
                "flow_loss": flow_loss.detach(),
                "stage3/var_ce_weight": torch.tensor(ce_weight, device=total_loss.device),
                "stage3/flow_loss_weight": torch.tensor(flow_weight, device=total_loss.device),
            }
        )
        return metrics

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict[str, np.ndarray]:
        decoder = str(kwargs.get("action_decoder", os.getenv("STAGE3_ACTION_DECODER", "flow"))).lower()
        if decoder in {"token", "tokens", "var", "stage2"}:
            out = QwenVARScaleParallel.predict_action(self, examples=examples, **kwargs)
            out.update(
                {
                    "decoder": "token",
                    "stage1_artifact_id": self.stage1_artifact_id,
                    "stage2_head": "scale_parallel",
                    "flow_head": "bypassed",
                }
            )
            return out
        if decoder != "flow":
            raise ValueError(f"Unsupported Stage3 action decoder: {decoder!r}. Expected 'flow' or 'token'.")

        if not isinstance(examples, list):
            examples = [examples]
        converted = []
        for example in examples:
            item = dict(example)
            item["image"] = to_pil_preserve(example["image"])
            converted.append(item)

        batch_images = [example["image"] for example in converted]
        instructions = [example["lang"] for example in converted]
        states = [example["state"] for example in converted] if "state" in converted[0] else None
        if states is not None:
            instructions = add_discretized_state_to_instruction(instructions, states)

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vl_embs_list = self._encode_flow_hidden_states(images=batch_images, instructions=instructions)
        action_dtype = self.action_model.dtype
        vl_embs_list = [hidden.to(dtype=action_dtype) for hidden in vl_embs_list]
        with torch.autocast("cuda", dtype=torch.float32, enabled=vl_embs_list[-1].device.type == "cuda"):
            actions = self.action_model.predict_action(vl_embs_list, None)
        return {
            "normalized_actions": actions.detach().float().cpu().numpy(),
            "stage1_artifact_id": self.stage1_artifact_id,
            "decoder": "flow",
            "stage2_head": "scale_parallel",
            "flow_head": "QwenPI_v3",
        }


@FRAMEWORK_REGISTRY.register("QwenVARScaleParallelFlowStage3")
class QwenVARScaleParallelFlowStage3(QwenVARScaleParallelPiFlowStage3):
    """Backward-compatible registry alias for the π-style Stage3 implementation."""
