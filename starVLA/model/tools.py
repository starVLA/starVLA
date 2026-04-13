def auto_get_module_keys(module, max_depth=0, prefix_list=None, current_depth=0, current_prefix=""):
    """
    get all submodule keys of a module, support setting recursion depth and prefix list.

    :param module: the module to traverse.
    :param max_depth: the maximum recursion depth, default is 1.
    :param prefix_list: only include modules with specified prefix, default is None means no restriction.
    :param current_depth: the current recursion depth, internal use.
    :param current_prefix: the current prefix, internal use.
    :return: the list of module keys.
    """
    if current_depth > max_depth:
        return []

    module_keys = []
    for name, sub_module in module.named_children():
        full_name = f"{current_prefix}.{name}" if current_prefix else name
        if prefix_list is None or any(full_name.startswith(prefix) for prefix in prefix_list):
            module_keys.append(full_name)
        module_keys.extend(auto_get_module_keys(sub_module, max_depth, prefix_list, current_depth + 1, full_name))
    return module_keys


def is_module_trainable(module):
    """
    check if a module is trainable: if the module itself has parameters, then all its parameters require_grad must be True;
    if the module itself has no parameters, then its trainability depends on its submodules.
    """
    params = list(module.parameters(recurse=False))
    if params:
        return all(p.requires_grad for p in params)
    else:
        # for container modules with no direct parameters, consider them trainable (the final result depends on their submodules)
        return True


def auto_get_trainable_modules(module, prefix="", max_depth=None):
    """
    recursively traverse the module, return the list of all trainable module names.
    if all submodules of a module are trainable, then only return the name of the parent module, no longer recursively output the names of its submodules.

    parameters:
      - module: the module to traverse.
      - prefix: the name prefix of the current module (internal use).
      - max_depth: the maximum recursion depth, None means infinite recursion.

    return:
      a list of module names.
    """
    # get all direct submodules of the current module
    children = list(module.named_children())

    # if the maximum depth is reached or there are no submodules, return the current module (if trainable and prefix is not empty)
    if (max_depth is not None and max_depth <= 0) or not children:
        return [prefix] if prefix and is_module_trainable(module) else []

    child_keys = []
    all_children_trainable = True
    for name, child in children:
        full_name = f"{prefix}.{name}" if prefix else name
        # recursively get the trainable keys of the submodules
        keys = auto_get_trainable_modules(child, full_name, None if max_depth is None else max_depth - 1)
        if not keys:
            # if the submodule does not return any further submodules, check the submodule itself
            if is_module_trainable(child):
                keys = [full_name]
            else:
                all_children_trainable = False
        else:
            # if the submodule returns multiple names, it means that it cannot be merged
            if len(keys) > 1:
                all_children_trainable = False
        child_keys.extend(keys)

    # if the current module is trainable and all submodules are trainable, return the name of the current module
    if is_module_trainable(module) and all_children_trainable and child_keys:
        return [prefix] if prefix else child_keys
    else:
        return child_keys


def print_freeze_status(self):
    """
    for each top-level submodule, if all its parameters are in the same state (all frozen or all trainable), only print the top-level module.
    if some top-level submodule has mixed parameter states (some frozen, some trainable), list the state of each parameter under the submodule.
    """
    from collections import defaultdict

    # collect the state of parameters under each top-level module
    status_dict = defaultdict(lambda: {"Frozen": 0, "Trainable": 0, "params": []})
    for full_name, param in self.named_parameters():
        # full_name is like "qwen_vl_interface.model.layer.weight"
        top_module = full_name.split(".", 1)[0]  # get the top-level module name
        state = "Frozen" if not param.requires_grad else "Trainable"
        status_dict[top_module]["params"].append((full_name, state))
        status_dict[top_module][state] += 1

    print("=== module parameter freezing status ===")
    for top_module, info in status_dict.items():
        frozen_count = info["Frozen"]
        trainable_count = info["Trainable"]

        if frozen_count > 0 and trainable_count == 0:
            # all frozen
            print(f"{top_module:40s}  |  all Frozen ({frozen_count} parameters)")
        elif trainable_count > 0 and frozen_count == 0:
            # all trainable
            print(f"{top_module:40s}  |  all Trainable ({trainable_count} parameters)")
        else:
            # mixed state, first print the module name summary, then list the state of each parameter
            print(f"{top_module:40s}  |  mixed state → Frozen: {frozen_count}, Trainable: {trainable_count}")
            for pname, pstate in info["params"]:
                print(f"    {pname:60s}  |  {pstate}")
    print("=========================\n")


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry = {}

    def register(self, key: str):
        """Decorator: register a builder function or class"""

        def decorator(framework_class):
            if key in self._registry:
                # print(ImportWarning(f"{key} already registered to {self.name}"))
                pass
            self._registry[key] = framework_class
            return framework_class

        return decorator

    def __getitem__(self, key):
        return self._registry[key]

    def list(self):
        """
        List currently registered keys; if with_values=True (not used here) return mapping {key: value_obj}.
        Using class name as value is also intuitive, e.g., framework.__name__.
        """
        return {k: v for k, v in self._registry.items()}


FRAMEWORK_REGISTRY = Registry("frameworks")


import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms as TF

from starVLA.training.trainer_utils import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def read_mode_config(pretrained_checkpoint):
    """Re-export from share_tools — canonical version lives in starVLA.model.framework.share_tools."""
    from starVLA.model.framework.share_tools import read_mode_config as _read_mode_config

    return _read_mode_config(pretrained_checkpoint)


class FrameworkTools:
    """Model-agnostic utility helpers for action (un)normalization and trainable-module discovery.

    These are pure functions / static methods that do NOT depend on any model state.
    ``baseframework`` holds a class-level reference so that legacy call-sites like
    ``model.unnormalize_actions(...)`` keep working, but new code should prefer::

        from starVLA.model.tools import FrameworkTools
        actions = FrameworkTools.unnormalize_actions(norm_actions, stats)
    """

    @staticmethod
    def check_unnorm_key(norm_stats: dict, unnorm_key: str | None) -> str:
        """Infer or validate the dataset stats key used for un-normalization.

        Args:
            norm_stats: ``{dataset_key: stats_block}`` mapping.
            unnorm_key: Explicit key, or ``None`` to auto-resolve (only when single dataset).

        Returns:
            Resolved dataset key.

        Raises:
            AssertionError: If ambiguous (multiple datasets without key) or key not found.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    @staticmethod
    def get_action_stats(norm_stats: dict, unnorm_key: str | None = None) -> dict:
        """Retrieve raw action normalization statistics for a dataset.

        Args:
            norm_stats: Full norm-stats dict (loaded from ``dataset_statistics.json``).
            unnorm_key: Optional dataset key; auto-resolved when single dataset.

        Returns:
            Stats sub-dict (e.g. ``{"q01": ..., "q99": ..., "mask": ...}``).
        """
        unnorm_key = FrameworkTools.check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: dict,
        gripper_channel_idx: int = 6,
    ) -> np.ndarray:
        """Map normalized actions back to original value range.

        Steps:
            1. Clamp to [-1, 1]
            2. Threshold gripper channel to binary {0, 1}
            3. Linear rescale masked dims: ``original = 0.5*(norm+1)*(q99-q01) + q01``

        Args:
            normalized_actions: ``[T, action_dim]`` array.
            action_norm_stats: Dict with ``q01``, ``q99``, optional ``mask``.
            gripper_channel_idx: Which channel is the binary gripper (default 6 for 7-DoF).

        Returns:
            Unnormalized actions (same shape).
        """
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        if 0 <= gripper_channel_idx < normalized_actions.shape[-1]:
            normalized_actions[:, gripper_channel_idx] = np.where(
                normalized_actions[:, gripper_channel_idx] < 0.5, 0, 1
            )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    @staticmethod
    def get_trainable_module_keys(model, max_depth: int = 1) -> list:
        """Enumerate trainable sub-module names up to *max_depth*.

        Args:
            model: ``nn.Module`` instance.
            max_depth: How deep to traverse the module tree.

        Returns:
            List of module-path strings considered trainable.
        """
        return auto_get_trainable_modules(model, max_depth=max_depth)


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, nhead: int = 8, dropout: float = 0.0, kv_dim: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden if d_hidden is not None else d_model
        self.nhead = nhead
        self.head_dim = self.d_hidden // nhead
        assert self.d_hidden % nhead == 0, "d_hidden must be divisible by nhead"

        # Projections
        self.q_proj = nn.Linear(d_model, self.d_hidden)
        self.k_proj = nn.Linear(kv_dim, self.d_hidden)
        self.v_proj = nn.Linear(kv_dim, self.d_hidden)
        self.out_proj = nn.Linear(self.d_hidden, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image_feature: torch.Tensor, spatial_feature: torch.Tensor):
        """
        Args:
            image_feature: (B, N_img, d_model) — Query
            vggt_feature:   (B, N_vggt, kv_dim) — Key and Value

        Returns:
            fused_image_feature: (B, N_img, d_model)
        """
        B, N_img, _ = image_feature.shape
        _, N_spatial, _ = spatial_feature.shape

        # Project to d_hidden
        q = self.q_proj(image_feature)  # (B, N_img, d_hidden)
        k = self.k_proj(spatial_feature)  # (B, N_vggt, d_hidden)
        v = self.v_proj(spatial_feature)  # (B, N_vggt, d_hidden)

        # Reshape for multi-head: (B, N, d_hidden) -> (B, N, nhead, head_dim) -> (B, nhead, N, head_dim)
        q = q.view(B, N_img, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_img, head_dim)
        k = k.view(B, N_spatial, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_vggt, head_dim)
        v = v.view(B, N_spatial, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_vggt, head_dim)

        # Scaled Dot-Product Attention
        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nhead, N_img, N_vggt)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        # Weighted sum over values
        attn_output = torch.matmul(attn_weights, v)  # (B, nhead, N_img, head_dim)

        # Concatenate heads and project back
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, N_img, nhead, head_dim)
        attn_output = attn_output.view(B, N_img, self.d_hidden)  # (B, N_img, d_hidden)

        # Final projection to d_model
        output = self.out_proj(attn_output)  # (B, N_img, d_model)
        output = self.dropout_out(output)

        # Residual connection + LayerNorm
        output = self.norm(image_feature + output)

        return output


def preprocess_images(image_list, target_size, mode="crop"):  #  [B，[PLT]]
    batch_images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    # target_size = 518

    # First process all images and collect their shapes
    for imgs in image_list:
        epi_images = []
        for img in imgs:
            width, height = img.size

            if mode == "pad":
                # Make the largest dimension 518px while maintaining aspect ratio
                if width >= height:
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
                else:
                    new_height = target_size
                    new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
            else:  # mode == "crop"
                # Original behavior: set width to 518px
                new_width = target_size
                # Calculate height maintaining aspect ratio, divisible by 14
                new_height = round(height * (new_width / width) / 14) * 14

            # Resize with new dimensions (width, height)
            # img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = to_tensor(img)  # Convert to tensor (0, 1)

            # Center crop height if it's larger than 518 (only in crop mode)
            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                img = img[:, start_y : start_y + target_size, :]

            # For pad mode, pad to make a square of target_size x target_size
            if mode == "pad":
                h_padding = target_size - img.shape[1]
                w_padding = target_size - img.shape[2]

                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left

                    # Pad with white (value=1.0)
                    img = torch.nn.functional.pad(
                        img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )

            shapes.add((img.shape[1], img.shape[2]))
            epi_images.append(img)
        batch_images.append(torch.stack(epi_images))

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in batch_images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        batch_images = padded_images

    batch_images = torch.stack(batch_images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_list) == 1:
        # Verify shape is (1, C, H, W)
        if batch_images.dim() == 3:
            batch_images = batch_images.unsqueeze(0)
    return batch_images
