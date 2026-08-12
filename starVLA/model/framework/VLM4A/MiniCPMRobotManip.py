# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
MiniCPM-RobotManip Framework
============================

Fine-tune the released **MiniCPM-RobotManip** generalist VLA
(https://huggingface.co/openbmb/MiniCPM-RobotManip) inside starVLA.

Unlike the ``MiniCPMGR00T`` example (which trains a *fresh* GR00T head on top of
the plain ``openbmb/MiniCPM-V-4.6`` backbone), this framework loads the released
1.5B checkpoint **as-is** through its shipped ``trust_remote_code`` model class
(``MiniCPMV_VLA`` = MiniCPM-V-4.6 backbone + a pretrained 80-D flow-matching
GR00T action head) and fine-tunes it.

The released repo ships an inference-only ``predict_action``; the flow-matching
*training* loss is added here using the LIBERO branch of the mixed post-training
recipe: clean-action target, xyz masked-MSE x500, rotation6D masked-MSE x10,
and gripper masked-L1.

Action space: unified 80-D layout. LIBERO is single-arm, so the 10-D absolute
EE6D target (``observation.xvla_abs_ee6d`` = xyz(3) + rot6d(6) + gripper(1)) is
placed in the left-arm end-effector slot ``[7:17]``; all other channels are
masked out of the loss. ``embodiment_id = 0``.
"""

from typing import List, Optional

import numpy as np
import torch
from torch.distributions import Beta

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# 80-D unified layout: left_arm[0:17] = joint(7) + xyz(3) + rot6d(6) + gripper(1).
# The 10-D EE6D target maps contiguously onto the left-arm eef slot [7:17].
EE6D_SLOT_START = 7
EE6D_SLOT_END = 17
EE6D_XYZ_END = EE6D_SLOT_START + 3
EE6D_ROT6D_END = EE6D_XYZ_END + 6

LIBERO_PROMPT_TEMPLATE = (
    "The robot is LIBERO Franka, a simulated single-arm Franka manipulator. "
    "Its action control method is absolute single-arm end-effector pose in the unified 80D layout "
    "with gripper closed command, and its action FPS is 20 Hz. Task: {instruction}"
)


@FRAMEWORK_REGISTRY.register("MiniCPMRobotManip")
class MiniCPM_RobotManip(baseframework):
    """Released MiniCPM-RobotManip (MiniCPM-V 4.6 + 80-D GR00T head) fine-tuning."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.config = config
        fw = config.framework
        model_id = fw.get("base_vlm", "openbmb/MiniCPM-RobotManip")
        # The released MiniCPMV_VLA composite only accepts "eager" / "flash_attention_2"
        # (it rejects "sdpa"); default to "eager" for portability.
        attn_impl = fw.get("attn_implementation", "eager")
        if attn_impl == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                logger.warning("flash_attn not installed; falling back to eager")
                attn_impl = "eager"

        # Released self-contained checkpoint: MiniCPMV_VLA = vlm + action_head.
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

        # Optional VLM gradient checkpointing (trades a little compute for a large
        # activation-memory saving; recommended for full-size fine-tuning batches).
        grad_ckpt = bool(getattr(getattr(config, "trainer", object()), "gradient_checkpointing", False)) or bool(
            fw.get("gradient_checkpointing", False)
        )
        if grad_ckpt:
            try:
                self.model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                if hasattr(self.model.vlm, "enable_input_require_grads"):
                    self.model.vlm.enable_input_require_grads()
                logger.info("[MiniCPMRobotManip] VLM gradient_checkpointing ENABLED")
            except Exception as exc:
                logger.warning(f"[MiniCPMRobotManip] failed to enable gradient_checkpointing: {exc}")

        head = self.model.action_head
        self.action_horizon = int(head.action_horizon)
        self.action_dim = int(head.action_dim)
        self.state_dim = int(head.state_dim)
        self.num_timestep_buckets = int(head.num_timestep_buckets)
        self.action_head_dtype = self.model.action_head_dtype

        # Flow-matching noise schedule (matches the released training recipe).
        self.repeated_diffusion_steps = int(fw.get("repeated_diffusion_steps", 8))
        self._beta_dist = Beta(
            float(fw.get("noise_beta_alpha", 1.5)),
            float(fw.get("noise_beta_beta", 1.0)),
        )
        self._noise_s = float(fw.get("noise_s", 0.999))
        self.xyz_loss_scale = float(fw.get("xyz_loss_scale", 500.0))
        self.rot6d_loss_scale = float(fw.get("rot6d_loss_scale", 10.0))
        self.libero_prompt_template = str(fw.get("libero_prompt_template", LIBERO_PROMPT_TEMPLATE))

    # ------------------------------------------------------------------ utils
    def _format_instruction(self, instruction: str) -> str:
        # Evaluation already supplies the full robot prompt. Avoid adding it
        # twice during direct predict_action() validation.
        if instruction.startswith("The robot is LIBERO Franka,"):
            return instruction
        return self.libero_prompt_template.format(instruction=instruction)

    def _build_vlm_inputs(self, images: List[list], instructions: List[str]):
        """Tokenize (multi-view images + instruction) with the shipped processor."""
        messages = []
        for imgs, instruction in zip(images, instructions, strict=True):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": self._format_instruction(instruction)})
            messages.append([{"role": "user", "content": content}])
        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return batch.to(self.model.device)

    def _to_80d(self, arr, device):
        """(B, T, 10) EE6D -> (B, T, 80) unified layout + valid mask."""
        t = torch.as_tensor(np.asarray(arr), dtype=torch.float32, device=device)
        b, horizon, dim = t.shape
        assert dim == (EE6D_SLOT_END - EE6D_SLOT_START), f"expected 10-D EE6D, got {dim}-D"
        out = torch.zeros(b, horizon, self.action_dim, dtype=torch.float32, device=device)
        mask = torch.zeros(b, horizon, self.action_dim, dtype=torch.float32, device=device)
        out[:, :, EE6D_SLOT_START:EE6D_SLOT_END] = t
        mask[:, :, EE6D_SLOT_START:EE6D_SLOT_END] = 1.0
        return out, mask

    def _sample_time(self, batch_size, device, dtype):
        sample = self._beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self._noise_s)
        return (self._noise_s - sample) / self._noise_s

    @staticmethod
    def _reduce_qwenvla(loss_per_elem, valid_mask):
        """Apply the released recipe's per-channel masked reduction."""
        masked = loss_per_elem * valid_mask
        valid_counts = valid_mask.sum(dim=1)  # (B, D)
        channel_valid = valid_counts > 0
        per_channel = masked.sum(dim=1) / valid_counts.clamp_min(1e-8)
        per_channel = per_channel * channel_valid.to(per_channel.dtype)
        n_valid_channels = channel_valid.sum(dim=1).clamp_min(1)
        return (per_channel.sum(dim=1) / n_valid_channels).mean()

    def _group_weighted_action_loss(self, pred, target, valid_mask):
        """Released-recipe loss: xyz MSE x500 + rot6d MSE x10 + gripper L1."""
        xyz_slice = slice(EE6D_SLOT_START, EE6D_XYZ_END)
        rot_slice = slice(EE6D_XYZ_END, EE6D_ROT6D_END)
        gripper_slice = slice(EE6D_ROT6D_END, EE6D_SLOT_END)

        xyz_loss = self._reduce_qwenvla(
            (pred[:, :, xyz_slice] - target[:, :, xyz_slice]).square(),
            valid_mask[:, :, xyz_slice],
        )
        rot6d_loss = self._reduce_qwenvla(
            (pred[:, :, rot_slice] - target[:, :, rot_slice]).square(),
            valid_mask[:, :, rot_slice],
        )
        gripper_loss = self._reduce_qwenvla(
            (pred[:, :, gripper_slice] - target[:, :, gripper_slice]).abs(),
            valid_mask[:, :, gripper_slice],
        )
        action_loss = self.xyz_loss_scale * xyz_loss + self.rot6d_loss_scale * rot6d_loss + gripper_loss
        return {
            "action_loss": action_loss,
            "xyz_loss": xyz_loss,
            "rot6d_loss": rot6d_loss,
            "gripper_loss": gripper_loss,
        }

    # ---------------------------------------------------------------- forward
    def forward(self, examples: Optional[List[dict]] = None, **kwargs) -> dict:
        if examples is None:
            raise ValueError("examples must be provided")
        images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions10 = [ex["action"] for ex in examples]  # each (H, 10)
        states10 = [ex["state"] for ex in examples]  # each (1, 10)

        vlm_inputs = self._build_vlm_inputs(images, instructions)
        vl_out = self.model._vlm_forward(vlm_inputs)
        vl_embs = vl_out.hidden_states[-1].to(self.action_head_dtype)

        device = vl_embs.device
        action80, valid_mask = self._to_80d(np.stack(actions10), device)
        state80, _ = self._to_80d(np.stack(states10), device)
        embodiment_id = torch.zeros(vl_embs.shape[0], dtype=torch.long, device=device)

        # The released checkpoint intentionally keeps the action head in FP32.
        # Disable the trainer's outer BF16 autocast for this block while leaving
        # the VLM forward above in BF16.
        device_type = device.type
        with torch.autocast(device_type, enabled=False):
            r = self.repeated_diffusion_steps
            vl = vl_embs.repeat(r, 1, 1)
            act = action80.repeat(r, 1, 1)
            msk = valid_mask.repeat(r, 1, 1)
            st = state80.repeat(r, 1, 1)
            eid = embodiment_id.repeat(r)

            head = self.model.action_head
            noise = torch.randn_like(act)
            t = self._sample_time(act.shape[0], device, act.dtype)
            tb = t[:, None, None]
            # clean-action flow matching: t=0 -> clean, t=1 -> pure noise.
            noisy = tb * noise + (1.0 - tb) * act
            t_disc = (t * self.num_timestep_buckets).long()
            pred = head._predict(noisy, vl, st, t_disc, eid)  # (B, H, 80)
            losses = self._group_weighted_action_loss(pred, act, msk)

        return losses

    # ---------------------------------------------------------- predict_action
    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        states10 = [ex["state"] for ex in examples]

        vlm_inputs = self._build_vlm_inputs(images, instructions)
        device = self.model.device
        state80, _ = self._to_80d(np.stack(states10), device)
        embodiment_id = torch.zeros(len(examples), dtype=torch.long, device=device)

        actions80 = self.model.predict_action(
            state=state80,
            embodiment_id=embodiment_id,
            **vlm_inputs,
        )  # (B, H, 80)
        # Recover the single-arm 10-D EE6D action from the left-arm eef slot.
        actions10 = actions80[:, :, EE6D_SLOT_START:EE6D_SLOT_END]
        return {"normalized_actions": actions10.float().cpu().numpy()}
