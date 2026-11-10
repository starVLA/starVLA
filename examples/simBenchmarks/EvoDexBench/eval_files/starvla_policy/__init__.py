"""StarVLA policy factories for the public Evo-DexBench policy API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import StarVLAPolicyConfig
    from .session import StarVLAPolicySession


def __getattr__(name: str) -> Any:
    if name == "StarVLAPolicyConfig":
        from .config import StarVLAPolicyConfig

        return StarVLAPolicyConfig
    if name == "StarVLAPolicySession":
        from .session import StarVLAPolicySession

        return StarVLAPolicySession
    raise AttributeError(name)


def create_starvla_policy(**kwargs: Any) -> StarVLAPolicySession:
    from .config import StarVLAPolicyConfig
    from .session import StarVLAPolicySession

    return StarVLAPolicySession(StarVLAPolicyConfig(**kwargs))


def create_qwenoft_policy(**kwargs: Any) -> StarVLAPolicySession:
    return create_starvla_policy(framework="qwenoft", **kwargs)


def create_wan_policy(**kwargs: Any) -> StarVLAPolicySession:
    return create_starvla_policy(framework="wan", **kwargs)


def create_cosmos_policy(**kwargs: Any) -> StarVLAPolicySession:
    return create_starvla_policy(framework="cosmos", **kwargs)


__all__ = [
    "StarVLAPolicyConfig",
    "StarVLAPolicySession",
    "create_cosmos_policy",
    "create_qwenoft_policy",
    "create_starvla_policy",
    "create_wan_policy",
]
