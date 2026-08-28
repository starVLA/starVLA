# Copyright 2026 starVLA community. All rights reserved.
"""Alpha-style controlled variant of scale-parallel VAR Stage 2."""

from __future__ import annotations

from typing import Optional

from starVLA.model.framework.VLM4A.QwenVARScaleParallel import QwenVARScaleParallel
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenVARAlphaScaleParallel")
class QwenVARAlphaScaleParallel(QwenVARScaleParallel):
    """Controlled VAR Stage 2 entry point with unchanged training logic.

    This class intentionally inherits the full implementation from
    ``QwenVARScaleParallel``. It exists to make the alpha-style experiment a
    separate framework/config/run while preserving the original VAR Stage 2
    objective, loss, prediction path, and Stage 1 tokenizer interface.

    The alpha constraints are enforced by the experiment config:
    raw LIBERO RGB views + language only, frozen Stage 1 action tokenizer,
    no state/history inputs, and no extra robot-specific pretraining beyond the
    Qwen3-VL backbone initialization.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.alpha_variant = True
        self.alpha_variant_name = "var_stage2_alpha_scale_parallel"
