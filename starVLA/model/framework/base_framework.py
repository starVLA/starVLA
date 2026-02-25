"""
Base framework abstraction providing:
- Pretrained loading (config + normalization stats + weights)
- Action space utilities (dimension, stats, (un)normalization)
- Trainable module discovery helper
Note: No device placement or optimizer concerns handled here (delegated to trainer).
"""

import torch.nn as nn
from typing import List

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from typing import List

from pathlib import Path
from typing import Dict, List
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
import numpy as np
from starVLA.model.tools import auto_get_trainable_modules

from starVLA.model.framework.share_tools import read_mode_config
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.framework.share_tools import dict_to_namespace
from starVLA.model.framework.__init__ import build_framework

logger = initialize_overwatch(__name__)


# PreTrainedModel, AutoModel, PretrainedConfig,  are so good, find sometime to study them
# TODO @JinhuiYE find sometime to merge yaml config with transformer config

class baseframework(PreTrainedModel):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to:
      - Accept a structured config
      - Register components in __init__
      - Use provided helpers for action normalization handling
    """

    def __init__(
        self,
        hf_config = PretrainedConfig()
    ) -> None:
        """
        Initialize base nn.Module. Subclasses add components.
        """
        
        super().__init__(hf_config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        **kwargs,
    ) -> None:
        """
        Restore a model instance from a saved checkpoint.

        Workflow:
            1. Resolve checkpoint path
            2. Load config + dataset normalization statistics
            3. Build model with loaded config
            4. Load state_dict strictly (reports missing/unexpected keys)
            5. Attach normalization stats for later un-normalization

        Args:
            pretrained_checkpoint: Path to .pt file inside run/checkpoints directory.
            **kwargs: Extra constructor overrides passed to subclass.

        Returns:
            baseframework: Instantiated model (left on CPU; caller decides device).

        Raises:
            RuntimeError: If state_dict key mismatch occurs under strict=True.
            FileNotFoundError: If underlying files are missing (surfaced earlier).
        """
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(pretrained_checkpoint)  # read config and norm_stats

        config = dict_to_namespace(model_config)
        model_config = config
        model_config.trainer.pretrained_checkpoint = None
        # FrameworkModel = cls(config=model_config, **kwargs) # TODO find cls by config
        FrameworkModel = build_framework(cfg=model_config)
        # set for action un-norm
        FrameworkModel.norm_stats = norm_stats
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        if pretrained_checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            model_state_dict = load_file(str(pretrained_checkpoint))
        else:
            model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")
        # logger.info(f"Loading model weights from `{pretrained_checkpoint}`")
        model_keys = set(FrameworkModel.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())
        try:
            FrameworkModel.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            # must keep all keys matched
            common_keys = model_keys.intersection(checkpoint_keys)
            missing_keys = model_keys - common_keys
            unexpected_keys = checkpoint_keys - common_keys
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

            raise e

        # **ensure model is on GPU**
        FrameworkModel = FrameworkModel
        return FrameworkModel

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Infer or validate the dataset stats key used for un-normalization.

        Args:
            norm_stats: Dict[str, dict] mapping dataset key -> stats block.
            unnorm_key: Optional explicit dataset key.

        Returns:
            str: Resolved key.

        Raises:
            AssertionError: If multiple datasets present and key not provided,
                            or provided key not found.
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

    @classmethod
    def get_action_stats(self, unnorm_key=None):
        """
        Retrieve raw action normalization statistics.

        Args:
            unnorm_key: Optional dataset stats key.

        Returns:
            dict: Stats structure (e.g. q01, q99, mask).
        """
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    @property
    def trainable_module_keys(self, max_depth=1) -> List[str]:
        """
        Enumerate trainable submodule names up to a depth.

        Args:
            max_depth: Descent depth when traversing module tree.

        Returns:
            List[str]: Module path names considered trainable.
        """
        keys = auto_get_trainable_modules(self, max_depth=max_depth)  # auto check which modules are trainable
        return keys

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Map normalized actions (≈[-1, 1]) back to original value range.

        Steps:
            - Clamp values to [-1, 1]
            - Threshold channel index 6 to {0,1} (binary semantic)
            - Apply linear scaling for masked dimensions using:
                original = 0.5 * (norm + 1) * (q99 - q01) + q01

        Args:
            normalized_actions: Array shape [T, D] (or chunk length × action_dim).
            action_norm_stats: Dict containing:
                q01 (array-like): Lower percentile (per-dimension).
                q99 (array-like): Upper percentile (per-dimension).
                mask (optional bool array): True => apply de-normalization; False => keep original normalized value.

        Returns:
            np.ndarray: Unnormalized actions (same shape as input).
        """
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
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

    @classmethod
    def get_action_stats(self, unnorm_key=None, norm_stats=None):
        """
        Duplicate stats accessor (retained for backward compatibility).
        # in future, it will own to policy interface and pack as 
        """
        if norm_stats ==None:
            norm_stats = self.norm_stats
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]
