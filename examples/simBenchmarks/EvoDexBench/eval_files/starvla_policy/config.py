"""Versioned StarVLA inference settings for Evo-DexBench's 30 Hz contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from dex_benchmark.policy_api.policy_features import (
    camera_roles_for_embodiment,
    roles_for_embodiment,
)

Framework = Literal["qwenoft", "wan", "cosmos"]
Embodiment = Literal["single", "dual"]
CONTROL_HZ = 30
HORIZONS: dict[Framework, int] = {"qwenoft": 50, "wan": 8, "cosmos": 8}
DEFAULT_EXECUTION: dict[Framework, int] = {"qwenoft": 10, "wan": 8, "cosmos": 8}
EMBODIMENT_IDS: dict[Embodiment, str] = {
    "single": "panda_shadow_single_right",
    "dual": "panda_shadow_bimanual",
}


@dataclass(frozen=True)
class StarVLAPolicyConfig:
    """One server/checkpoint interface; no temporal resampling is permitted."""

    host: str = "127.0.0.1"
    port: int = 10093
    framework: Framework = "qwenoft"
    embodiment: Embodiment = "single"
    execute_steps: int | None = None
    unnorm_key: str | None = None
    expected_checkpoint_path: str | None = None
    max_hand_target_step_rad: float | None = None

    def __post_init__(self) -> None:
        if not self.host.strip() or not 1 <= self.port <= 65535:
            raise ValueError("StarVLA host/port are invalid.")
        if self.framework not in HORIZONS:
            raise ValueError("framework must be qwenoft, wan, or cosmos.")
        if self.embodiment not in EMBODIMENT_IDS:
            raise ValueError("embodiment must be single or dual.")
        if self.execute_steps is not None and not 1 <= self.execute_steps <= self.chunk_size:
            raise ValueError(f"execute_steps must be in [1, {self.chunk_size}].")
        if self.unnorm_key is not None and not self.unnorm_key.strip():
            raise ValueError("unnorm_key must be non-empty when set.")
        if self.expected_checkpoint_path is not None and not self.expected_checkpoint_path.strip():
            raise ValueError("expected_checkpoint_path must be non-empty when set.")
        guard = self.max_hand_target_step_rad
        if guard is not None and (not math.isfinite(guard) or guard <= 0):
            raise ValueError("max_hand_target_step_rad must be positive and finite.")

    @property
    def chunk_size(self) -> int:
        return HORIZONS[self.framework]

    @property
    def steps_per_request(self) -> int:
        return DEFAULT_EXECUTION[self.framework] if self.execute_steps is None else self.execute_steps

    @property
    def roles(self) -> tuple[str, ...]:
        return roles_for_embodiment(self.embodiment)

    @property
    def camera_roles(self) -> tuple[str, ...]:
        return camera_roles_for_embodiment(self.embodiment)

    @property
    def state_dim(self) -> int:
        return 31 * len(self.roles)

    @property
    def action_dim(self) -> int:
        return 30 * len(self.roles)
