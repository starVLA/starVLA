# ruff: noqa: B905, E402, RUF005
# ``_dp_vendor/`` is a trimmed, vendored copy of real-stanford/diffusion_policy (MIT).
# It is vendored for version pinning because only the non-hybrid image-policy subset is needed.

import copy
import sys
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from starVLA.model.framework.base_framework import FRAMEWORK_REGISTRY, baseframework
from starVLA.model.framework.share_tools import merge_framework_config

_DP_VENDOR_PATH = Path(__file__).resolve().parent / "_dp_vendor"
if str(_DP_VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(_DP_VENDOR_PATH))

from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@contextmanager
def _model_dtype_no_autocast(policy: torch.nn.Module):
    model_param = next(policy.parameters())
    previous_default_dtype = torch.get_default_dtype()
    should_set_default_dtype = model_param.dtype in _FLOAT_DTYPES
    if should_set_default_dtype:
        torch.set_default_dtype(model_param.dtype)
    try:
        if model_param.device.type in ("cuda", "cpu"):
            with torch.autocast(model_param.device.type, enabled=False):
                yield
        else:
            yield
    finally:
        if should_set_default_dtype:
            torch.set_default_dtype(previous_default_dtype)


@dataclass
class DiffusionPolicyDefaultConfig:
    name: str = "DiffusionPolicy"
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    action_encoding: str = "joints"
    action_dim: int = 8
    state_dim: int = 8
    image_keys: tuple[str, ...] = ("cam0_rgb", "cam1_rgb")
    image_size: tuple[int, int] = (224, 224)
    pretrained_backbone: bool = True
    num_inference_steps: int | None = None
    obs_as_global_cond: bool = True
    diffusion_step_embed_dim: int = 256
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    num_train_timesteps: int = 100


def _get_image_size(config) -> tuple[int, int]:
    image_size = tuple(
        config.get("image_size", DiffusionPolicyDefaultConfig.image_size)
    )
    return int(image_size[0]), int(image_size[1])


def _build_shape_meta(config) -> dict:
    image_height, image_width = _get_image_size(config)
    obs_meta = {
        image_key: {"shape": [3, image_height, image_width], "type": "rgb"}
        for image_key in tuple(
            config.get("image_keys", DiffusionPolicyDefaultConfig.image_keys)
        )
    }
    obs_meta["state"] = {"shape": [int(config.state_dim)], "type": "low_dim"}
    return {
        "action": {"shape": [int(config.action_dim)]},
        "obs": obs_meta,
    }


@FRAMEWORK_REGISTRY.register("DiffusionPolicy")
class DiffusionPolicy(baseframework):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = merge_framework_config(DiffusionPolicyDefaultConfig, config)
        framework_config = self.config.framework
        shape_meta = _build_shape_meta(framework_config)
        obs_encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=torchvision.models.resnet18(
                weights=(
                    "IMAGENET1K_V1"
                    if framework_config.get("pretrained_backbone", True)
                    else None
                )
            ),
            resize_shape=None,
            crop_shape=None,
            random_crop=False,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=True,
        )
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(framework_config.num_train_timesteps),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.action_model = DiffusionUnetImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=int(framework_config.horizon),
            n_action_steps=int(framework_config.n_action_steps),
            n_obs_steps=int(framework_config.n_obs_steps),
            num_inference_steps=framework_config.num_inference_steps,
            obs_as_global_cond=bool(framework_config.obs_as_global_cond),
            diffusion_step_embed_dim=int(framework_config.diffusion_step_embed_dim),
            down_dims=tuple(framework_config.down_dims),
            kernel_size=int(framework_config.kernel_size),
            n_groups=int(framework_config.n_groups),
            cond_predict_scale=bool(framework_config.cond_predict_scale),
        )
        self._set_identity_normalizers()
        ema_model = copy.deepcopy(self.action_model)
        self._ema = EMAModel(model=ema_model)

    # Checkpoint key prefix for EMA weights. Saving and loading must share this value so old
    # checkpoint detection continues to work.
    _EMA_KEY_PREFIX = "ema_averaged."

    def state_dict(self, *args, **kwargs):
        """Add the true training EMA weights to the standard state dict.

        The weights use the ``ema_averaged.*`` prefix. This override is necessary because
        ``self._ema`` is an upstream ``EMAModel``, a plain class rather than an ``nn.Module``.
        Its ``averaged_model`` is therefore omitted from the default ``state_dict()``. Training
        continuously updates the true EMA through ``self._ema.step`` in ``forward``, but without
        this override the entire EMA is dropped when saving, so version 1 and earlier checkpoints
        contain no EMA keys.

        Both save paths, intermediate ``train_starvla`` checkpoints and ``final_model``, call
        ``accelerator.get_state_dict``, which delegates to this method. They therefore include
        the true EMA automatically. The override applies only to ``DiffusionPolicy`` and does not
        change how other frameworks are saved.

        Intentional limitations: EMA entries are always detached tensors and do not honor
        ``keep_vars=True`` (the production save path has no such caller). This override is paired
        with :meth:`load_state_dict` only when Diffusion Policy is the top-level object being
        loaded; ``build_framework`` returns this class at the top level in production.
        """
        destination = super().state_dict(*args, **kwargs)
        # Support both the legacy positional signature (destination, prefix, keep_vars) and
        # the keyword form of prefix.
        prefix = kwargs.get("prefix", args[1] if len(args) >= 2 else "")
        for key, value in self._ema.averaged_model.state_dict().items():
            destination[prefix + self._EMA_KEY_PREFIX + key] = value.detach()
        return destination

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load the true EMA from new checkpoints and safely handle old checkpoints.

        The root cause addressed by fcad746 is that ``self._ema`` is an ``EMAModel``, a plain
        class rather than an ``nn.Module``, so its ``averaged_model`` parameters are omitted from
        ``DiffusionPolicy.state_dict()``. During ``__init__``, the EMA is created by deep-copying
        the **randomly initialized** ``action_model``. An old checkpoint has no EMA keys, so
        ``from_pretrained`` with ``strict=True`` updates only ``action_model`` and otherwise leaves
        the EMA randomly initialized. Since ``predict_action`` performs inference with
        ``self._ema.averaged_model``, DDPM ``clip_sample`` clips the random output and inverse
        normalization pins it to the boundaries in ``dataset_statistics``.

        Loading has two paths, selected by whether the checkpoint contains
        ``ema_averaged.*``:

        1. For a new checkpoint saved by the :meth:`state_dict` override, first reseed from the
           loaded ``action_model`` as a safe fallback, then strip the prefix and overwrite
           ``_ema.averaged_model`` with the true EMA. Strictness follows the caller. Under a
           ``strict=False`` warm start, incomplete EMA keys are allowed and missing entries keep
           the reseeded fallback values instead of remaining randomly initialized. Inference
           therefore uses the true training EMA wherever it was saved.
        2. For a version 1 or other old checkpoint, preserve the fcad746 behavior: after the base
           load, reseed the EMA from ``action_model``. The unsaved historical EMA cannot be
           recovered, but the converged raw weights provide usable inference weights with no
           additional drift.

        The ``ema_averaged.*`` entries must be removed before calling ``super`` because they are
        not part of the ``nn.Module`` hierarchy and would be reported as unexpected keys under
        ``strict=True``.

        Intentional limitations (deviations should be treated as bugs):

        - The vendored ``EMAModel.optimization_step`` is not persisted because this repository has
          no ``accelerator.save_state`` resume path. During warm-start fine-tuning, EMA decay warms
          up again (the first step has ``ema=raw``), matching the vendored from-scratch semantics.
        - ``assign=True`` applies only to ``action_model``, not the EMA; no production entry point
          uses it.
        - The returned missing and unexpected keys describe only base keys, not
          ``ema_averaged.*``.
        """
        # Clone to break storage aliasing with this instance's EMA parameters. During an in-place
        # fw.load_state_dict(fw.state_dict()) round trip, the EMA tensors returned by state_dict
        # share storage with averaged_model parameters. The fallback reseed below would first
        # rewrite them with raw weights; without cloning, the true EMA would be silently replaced
        # and the validation would incorrectly pass.
        ema_sd = {
            k[len(self._EMA_KEY_PREFIX) :]: v.detach().clone()
            for k, v in state_dict.items()
            if k.startswith(self._EMA_KEY_PREFIX)
        }
        base_sd = OrderedDict(
            (k, v)
            for k, v in state_dict.items()
            if not k.startswith(self._EMA_KEY_PREFIX)
        )
        # Preserve PyTorch state-dict metadata used for submodule version compatibility while
        # stripping only the EMA keys.
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            base_sd._metadata = metadata
        result = super().load_state_dict(base_sd, *args, **kwargs)
        # Match the caller's strictness. In the nn.Module signature, strict is the first optional
        # positional argument. Production uses both from_pretrained(strict=True) and the
        # trainer_tools warm-start path (strict=False).
        strict = kwargs.get("strict", args[0] if args else True)
        if ema_sd:
            # New checkpoint: reseed first so keys omitted by a strict=False partial load do not
            # remain randomly initialized, then overwrite with true checkpoint EMA entries.
            # Missing and extra keys follow the same caller-selected strictness as the base model.
            self._ema.averaged_model.load_state_dict(self.action_model.state_dict())
            self._ema.averaged_model.load_state_dict(ema_sd, strict=strict)
            # Verify every loaded key against the checkpoint. copy_ permits cross-dtype loading,
            # so align dtypes before comparison and raise on any discrepancy.
            ema_now = self._ema.averaged_model.state_dict()
            for k, v in ema_sd.items():
                if (
                    k in ema_now
                    and torch.is_floating_point(ema_now[k])
                    and not torch.allclose(
                        ema_now[k],
                        v.to(dtype=ema_now[k].dtype, device=ema_now[k].device),
                    )
                ):
                    raise RuntimeError(
                        f"DiffusionPolicy.load_state_dict: loaded true EMA differs from checkpoint ({k})"
                    )
        else:
            # Old checkpoint: reseed the EMA averaged_model, which has the same structure because
            # it is a deepcopy, from action_model after its checkpoint weights have been loaded.
            self._ema.averaged_model.load_state_dict(self.action_model.state_dict())
            # Verify every parameter matches action_model as hard evidence that reseeding worked;
            # raise instead of silently accepting any discrepancy.
            for (n_a, p_a), (n_e, p_e) in zip(
                self.action_model.named_parameters(),
                self._ema.averaged_model.named_parameters(),
            ):
                if p_a.shape != p_e.shape or not torch.allclose(p_a, p_e):
                    raise RuntimeError(
                        "DiffusionPolicy.load_state_dict: reseeded EMA parameter differs from action_model "
                        f"(action={n_a}{tuple(p_a.shape)} vs ema={n_e}{tuple(p_e.shape)})"
                    )
        return result

    def _set_identity_normalizers(self) -> None:
        # starVLA StateActionTransform/PolicyNormProcessor own normalization.
        # DP internal normalizers must be identity to avoid double normalization.
        framework_config = self.config.framework
        keys = list(
            framework_config.get("image_keys", DiffusionPolicyDefaultConfig.image_keys)
        ) + ["state", "action"]
        normalizer = LinearNormalizer()
        for key in keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        self.action_model.set_normalizer(normalizer)

    def _sync_ema_to_action_model(self) -> None:
        action_param = next(self.action_model.parameters())
        ema_param = next(self._ema.averaged_model.parameters())
        if (
            ema_param.device != action_param.device
            or ema_param.dtype != action_param.dtype
        ):
            self._ema.averaged_model.to(
                device=action_param.device, dtype=action_param.dtype
            )

    def _examples_to_dp_batch(
        self, examples: List[dict], require_action: bool = True
    ) -> dict:
        if not examples:
            raise ValueError("DP batch requires at least one example")

        framework_config = self.config.framework
        image_keys = list(
            framework_config.get("image_keys", DiffusionPolicyDefaultConfig.image_keys)
        )
        n_obs_steps = int(
            framework_config.get(
                "n_obs_steps", DiffusionPolicyDefaultConfig.n_obs_steps
            )
        )
        horizon = int(
            framework_config.get("horizon", DiffusionPolicyDefaultConfig.horizon)
        )
        action_dim = int(
            framework_config.get("action_dim", DiffusionPolicyDefaultConfig.action_dim)
        )
        state_dim = int(
            framework_config.get("state_dim", DiffusionPolicyDefaultConfig.state_dim)
        )
        image_height, image_width = _get_image_size(framework_config)
        target_image_size = (image_width, image_height)
        model_param = next(self.action_model.parameters())
        device = model_param.device
        model_dtype = model_param.dtype

        obs_dict = {}

        # Convention: example["image"][i] is paired with framework.image_keys[i].
        for view_idx, image_key in enumerate(image_keys):
            image_sequences = []
            for example in examples:
                try:
                    image_source = example["image"][view_idx]
                except (IndexError, TypeError) as exc:
                    raise ValueError(
                        f"DP image key {image_key} expects example['image'][{view_idx}]"
                    ) from exc

                if isinstance(image_source, (list, tuple)):
                    frames = list(image_source)
                else:
                    frames = [image_source]

                if len(frames) == 1 and n_obs_steps > 1:
                    frames = frames * n_obs_steps
                elif len(frames) < n_obs_steps:
                    raise ValueError(
                        f"DP image key {image_key} needs {n_obs_steps} frames, got {len(frames)}"
                    )
                else:
                    frames = frames[-n_obs_steps:]

                if isinstance(frames[0], np.ndarray) and all(
                    isinstance(f, np.ndarray)
                    and f.shape[:2] == (image_height, image_width)
                    for f in frames
                ):
                    # Fast path: batch uint8 HWC worker output for one H2D transfer and GPU
                    # normalization — only when every frame already matches the configured
                    # input size. Any other resolution falls through to the PIL path below,
                    # which resizes instead of tripping the encoder's shape assertion.
                    arr = np.stack(frames, axis=0)
                    seq = torch.from_numpy(arr).to(device, non_blocking=True)
                    seq = (
                        seq.permute(0, 3, 1, 2)
                        .to(torch.float32)
                        .div_(255.0)
                        .to(model_dtype)
                    )
                    image_sequences.append(seq)
                else:
                    frame_tensors = []
                    for image in frames:
                        if isinstance(image, np.ndarray):
                            image = Image.fromarray(image)
                        image = image.convert("RGB")
                        if image.size != target_image_size:
                            image = image.resize(target_image_size)
                        image_array = np.asarray(image, dtype=np.float32) / 255.0
                        image_tensor = (
                            torch.from_numpy(image_array)
                            .permute(2, 0, 1)
                            .to(device=device, dtype=model_dtype)
                        )
                        frame_tensors.append(image_tensor)
                    image_sequences.append(torch.stack(frame_tensors, dim=0))
            obs_dict[image_key] = torch.stack(image_sequences, dim=0)

        states = []
        for example in examples:
            state = example.get(
                "state", np.zeros((n_obs_steps, state_dim), dtype=np.float32)
            )
            state_tensor = torch.as_tensor(state, dtype=model_dtype, device=device)
            if state_tensor.ndim == 1:
                if state_tensor.shape[0] != state_dim:
                    raise ValueError(
                        f"DP state must have {state_dim} values, got {state_tensor.shape[0]}"
                    )
                state_tensor = state_tensor.unsqueeze(0).repeat(n_obs_steps, 1)
            elif state_tensor.ndim == 2:
                if state_tensor.shape[-1] != state_dim:
                    raise ValueError(
                        f"DP state must have shape [T, {state_dim}], got {tuple(state_tensor.shape)}"
                    )
                if state_tensor.shape[0] == 1:
                    state_tensor = state_tensor.repeat(n_obs_steps, 1)
                elif state_tensor.shape[0] < n_obs_steps:
                    raise ValueError(
                        f"DP state history must be at least {n_obs_steps}, got {state_tensor.shape[0]}"
                    )
                else:
                    state_tensor = state_tensor[-n_obs_steps:]
            else:
                raise ValueError(
                    f"DP state must have shape [{state_dim}] or [T, {state_dim}], got {tuple(state_tensor.shape)}"
                )
            states.append(state_tensor)
        obs_dict["state"] = torch.stack(states, dim=0)

        batch = {"obs": obs_dict}

        if not require_action:
            return batch

        actions = []
        for example in examples:
            action_tensor = torch.as_tensor(
                example["action"], dtype=model_dtype, device=device
            )
            if action_tensor.ndim != 2 or action_tensor.shape[-1] != action_dim:
                raise ValueError(
                    f"DP action must have shape [T, {action_dim}], got {tuple(action_tensor.shape)}"
                )
            if action_tensor.shape[0] < horizon:
                raise ValueError(
                    f"DP action horizon must be at least {horizon}, got {action_tensor.shape[0]}"
                )
            # Take the immediate prefix (t .. t+horizon-1). Dataloaders whose action window
            # starts at the current step then always train DP on the actions that follow the
            # sampled observation, even if they provide a longer window (e.g. one shared with
            # a longer-chunk framework such as ACT). Identical to the previous behavior when
            # the window length equals the horizon.
            actions.append(action_tensor[:horizon])
        batch["action"] = torch.stack(actions, dim=0)

        return batch

    def forward(self, examples: List[dict], **kwargs) -> dict:
        batch = self._examples_to_dp_batch(examples, require_action=True)
        with _model_dtype_no_autocast(self.action_model):
            loss = self.action_model.compute_loss(batch)
        if self.action_model.training:
            # Known timing caveat: StarVLA's trainer offers no post-optimizer hook to a
            # framework, so the EMA is stepped here (once per training forward, tracking the
            # weights as of the previous optimizer step). Compared to upstream Diffusion
            # Policy (EMA after optimizer.step) the EMA lags by exactly one update, which is
            # negligible over full training; with gradient accumulation the decay schedule
            # advances per micro-batch (towards unchanged weights). Wiring the update to the
            # optimizer would require trainer changes and is intentionally out of scope for
            # this additive integration.
            self._sync_ema_to_action_model()
            self._ema.step(self.action_model)
        return {"action_loss": loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        self._sync_ema_to_action_model()
        ema_policy = self._ema.averaged_model
        obs_dict = self._examples_to_dp_batch(examples, require_action=False)["obs"]
        was_training = ema_policy.training
        try:
            with _model_dtype_no_autocast(ema_policy):
                result = ema_policy.predict_action(obs_dict)
        finally:
            ema_policy.train(was_training)
        action = result["action"]
        normalized_actions = action.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}
