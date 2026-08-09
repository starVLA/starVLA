# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Ported from RoboTTT (arxiv 2607.15275, "RoboTTT: Context Scaling for Robot Policies").
"""
RoboTTT Framework
A faithful port of NVIDIA's RoboTTT: GR00T N1.7 (Qwen3-VL backbone + flow-matching DiT
action head) augmented with **Test-Time-Training (TTT) layers** so the policy can
condition on a long visuomotor trajectory (context) without growing inference latency.

Architecture (RoboTTT §3.1):
  - Qwen3-VL backbone produces per-timestep VL tokens.
  - The DiT action head has a TTT layer after each of its attention blocks: attention
    processes single-step tokens (register + proprio + noised-action), TTT compresses
    information across the time dimension into fast weights (updated by gradient descent
    at both train and inference time).
  - A learned gate (init ≈ 0.001) scales the TTT contribution (preserving pretrained
    capabilities at the start of training).

Training: ``forward(examples)`` treats ``examples`` as a **trajectory** — a list of
per-timestep single-sample dicts (B=1, T=len(examples)), matching the paper's per-device
batch size 1 for long context. The head runs sequence action forcing (independent
flow-matching noise per timestep) + TBPTT (self-contained in the action head; no trainer
change). An optional per-timestep ``loss_mask`` (from ``example["loss_mask"]``) makes
selected timesteps context-only (for in-context-video imitation / DAgger Distillation).

Inference: ``predict_action(examples)`` rolls the fast weights over the context trajectory
and denoises the final (current) timestep's action chunk, reusing the carried fast weights.
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.RoboTTT_ActionHeader import get_action_model_robottt
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


@dataclass
class RoboTTTDefaultConfig:
    """RoboTTT framework defaults = N1.7 defaults + TTT fields.

    YAML values override these; extra YAML keys are preserved.
    """

    name: str = "RoboTTT"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            # === N1.7 flow-matching head defaults (see Gr00tN1d7DefaultConfig) ===
            "input_embedding_dim": 1536,
            "hidden_size": 1024,
            "max_action_dim": 132,
            "max_state_dim": 132,
            "state_history_length": 1,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 16,
            "use_vlln": True,
            "vl_self_attention_cfg": {"num_layers": 0},
            "use_alternate_vl_dit": True,
            "attend_text_every_n_blocks": 2,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "state_dropout_prob": 0.0,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "max_num_embodiments": 32,
            "tune_projector": True,
            "tune_diffusion_model": True,
            "tune_vlln": True,
            "repeated_diffusion_steps": 1,
            "diffusion_model_cfg": {
                "positional_embeddings": None,
                "num_layers": 16,
                "num_attention_heads": 32,
                "attention_head_dim": 48,
                "norm_type": "ada_norm",
                "dropout": 0.2,
                "final_dropout": True,
                "output_dim": 1024,
                "interleave_self_attention": True,
            },
            # === RoboTTT TTT fields ===
            "num_registers": 4,  # per-timestep register tokens carrying VL info across time
            "tbptt_segment_length": 64,  # truncate the TTT graph every N timesteps (memory = segment, not sequence)
            "ttt_cfg": {
                "num_heads": 8,
                "mlp_inner_dim": 768,
                "base_lr": 0.1,  # constant base inner lr (Appendix A.1)
                "gate_init": 0.001,  # near-zero gate preserves pretrained capabilities (Eq. 3)
                "rope_theta": 10000.0,
                "chunk_size": 8,  # TTT mini-batch size for parallelism
            },
        }
    )


@FRAMEWORK_REGISTRY.register("RoboTTT")
class RoboTTT(baseframework):
    """RoboTTT: GR00T N1.7 + TTT layers across the trajectory time dimension.

    ``forward(examples)`` consumes a trajectory (list of per-timestep dicts, B=1).
    ``predict_action(examples)`` consumes a context trajectory and returns the current
    step's action chunk.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(RoboTTTDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        vl_hidden_size = self.qwen_vl_interface.model.config.hidden_size
        am = self.config.framework.action_model
        am.backbone_embedding_dim = vl_hidden_size
        am.diffusion_model_cfg["cross_attention_dim"] = vl_hidden_size

        self.action_model = get_action_model_robottt(config=self.config)

        self.action_horizon = int(am.action_horizon)
        self.max_action_dim = int(am.max_action_dim)
        self.max_state_dim = int(am.max_state_dim)
        self.real_action_dim = int(am.action_dim)
        self.real_state_dim = int(am.state_dim)
        self.state_history_length = int(am.state_history_length)
        self.use_alternate_vl_dit = bool(am.use_alternate_vl_dit)
        self.image_token_id = int(getattr(self.qwen_vl_interface.model.config, "image_token_id", 151655))

    # ── helpers ─────────────────────────────────────────────────────────
    def _encode_backbone(self, batch_images, instructions):
        """Run the VLM on a list of (images, instruction) per timestep.

        Returns ``(last_hidden [1, T, S, H], backbone_attn_mask [1,T,S] bool,
        image_mask [1,T,S] bool)`` — the trajectory axis is the batch axis of the VLM
        call, reshaped to ``[1, T, ...]`` (B=1 trajectory).
        """
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        input_ids = qwen_inputs.get("input_ids", None)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
            last_hidden = outputs.hidden_states[-1]  # [T, S, H] (T = #timesteps as batch)

        # [T, S, H] -> [1, T, S, H]
        last_hidden = last_hidden.unsqueeze(0)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.unsqueeze(0)
        image_mask = None
        if self.use_alternate_vl_dit and input_ids is not None:
            image_mask = (input_ids == self.image_token_id).to(device=last_hidden.device, dtype=torch.bool)
            image_mask = image_mask.unsqueeze(0)
        return last_hidden, backbone_attention_mask, image_mask

    def _trajectory_actions(self, examples, device, dtype):
        """Build padded actions [1, T, H, max_D] + action_mask from per-timestep dicts."""
        per_ts = [np.array(ex["action"]) for ex in examples]  # each [T_full, real_D]
        arr = np.stack(per_ts, axis=0)  # [T, T_full, real_D]
        arr = arr[:, -self.action_horizon :, :]  # [T, H, real_D]
        T, H, D = arr.shape
        padded = np.zeros((T, H, self.max_action_dim), dtype=arr.dtype)
        padded[..., :D] = arr
        mask = np.zeros((T, H, self.max_action_dim), dtype=arr.dtype)
        mask[..., :D] = 1.0
        actions = torch.from_numpy(padded[None]).to(device=device, dtype=dtype)  # [1,T,H,max_D]
        action_mask = torch.from_numpy(mask[None]).to(device=device, dtype=dtype)
        return actions, action_mask

    def _trajectory_state(self, examples, device, dtype):
        """Build state [1, T, state_history_length, max_S] from per-timestep dicts (or None)."""
        if "state" not in examples[0]:
            return None
        per_ts = [np.atleast_1d(np.array(ex["state"], dtype=np.float32)) for ex in examples]
        arr = np.stack(per_ts, axis=0)  # [T, real_S]
        T, D = arr.shape
        # Tile to state_history_length.
        arr = np.repeat(arr[:, None, :], self.state_history_length, axis=1)  # [T, Hs, real_S]
        if D < self.max_state_dim:
            padded = np.zeros((T, self.state_history_length, self.max_state_dim), dtype=arr.dtype)
            padded[..., :D] = arr
            arr = padded
        return torch.from_numpy(arr[None]).to(device=device, dtype=dtype)  # [1,T,Hs,max_S]

    # ── forward (sequence training) ────────────────────────────────────
    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        """``examples`` = a trajectory: list of per-timestep dicts (B=1, T=len(examples))."""
        batch_images = [ex["image"] for ex in examples]  # [T, [PIL]]
        instructions = [ex["lang"] for ex in examples]  # [T, str]
        last_hidden, backbone_attention_mask, image_mask = self._encode_backbone(batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.float32):
            device = last_hidden.device
            dtype = last_hidden.dtype
            actions, action_mask = self._trajectory_actions(examples, device, dtype)
            state = self._trajectory_state(examples, device, dtype)
            embodiment_id = torch.tensor(
                [int(ex.get("embodiment_id", 0)) for ex in examples[:1]],
                device=device,
                dtype=torch.long,
            )  # [1]
            loss_mask = None
            if "loss_mask" in examples[0]:
                loss_mask = torch.tensor(
                    np.array([float(ex["loss_mask"]) for ex in examples]),
                    device=device,
                    dtype=dtype,
                ).unsqueeze(
                    0
                )  # [1, T]

            out = self.action_model.forward_sequence(
                last_hidden,
                actions,
                state,
                embodiment_id,
                action_mask,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                loss_mask=loss_mask,
            )
        return {"action_loss": out["action_loss"]}

    # ── predict_action (inference) ──────────────────────────────────────
    # NOTE: ``no_grad`` (not ``inference_mode``): the TTT layer updates fast weights by
    # computing an *inner gradient* at inference (``ttt_layer.TTTLayer.forward`` calls
    # ``torch.autograd.grad`` under a local ``torch.enable_grad()``). ``inference_mode``
    # produces tensors with no ``grad_fn`` that even ``enable_grad`` cannot rescue, so
    # the TTT inner loop raises "element 0 of tensors does not require grad".
    @torch.no_grad()
    def predict_action(self, examples: List[dict], **kwargs: Any) -> dict:
        """``examples`` = context trajectory (list of per-timestep dicts; last = current).

        Returns ``{"normalized_actions": np.ndarray [1, action_horizon, real_action_dim]}``.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(ex["image"]) for ex in examples]
        instructions = [ex["lang"] for ex in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        last_hidden, backbone_attention_mask, image_mask = self._encode_backbone(batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.float32):
            device = last_hidden.device
            dtype = last_hidden.dtype
            state = self._trajectory_state(examples, device, dtype)
            embodiment_id = torch.tensor(
                [int(ex.get("embodiment_id", 0)) for ex in examples[:1]],
                device=device,
                dtype=torch.long,
            )
            pred = self.action_model.predict_action(
                last_hidden,
                state,
                embodiment_id,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )  # [1, H, max_D]

        pred = pred[:, :, : self.real_action_dim]
        normalized_actions = pred.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/LIBERO/train_files/starvla_robottt_libero.yaml",
        help="Path to YAML config",
    )
    args, _ = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    model: RoboTTT = RoboTTT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Fake 4-timestep trajectory (B=1).
    traj = [
        {
            "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
            "image": [image],
            "lang": "Assemble the car roof.",
        }
        for _ in range(4)
    ]
    traj[1]["lang"] = "Screw the bolt."
    traj[2]["lang"] = "Drill the hole."
    traj[3]["lang"] = "Hand off the part."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(traj)
    print(f"Action Loss: {forward_output['action_loss'].item()}")

    predict_output = model.predict_action(examples=traj)
    normalized_actions = predict_output["normalized_actions"]
    print(f"Predicted normalized action shape: {normalized_actions.shape}")
    print(f"Predicted action (first step): {normalized_actions[0, 0]}")
    print("Finished")
