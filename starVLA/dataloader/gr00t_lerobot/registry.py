"""
Centralized registry that auto-discovers benchmark-specific data configs from
``examples/*/train_files/data_registry/`` and merges them with the base
registries defined in this package.

Three registries are maintained:

* ``DATASET_NAMED_MIXTURES``       – mixture_name → [(dataset, weight, robot_type)]
* ``ROBOT_TYPE_CONFIG_MAP``        – robot_type → DataConfig instance
* ``ROBOT_TYPE_TO_EMBODIMENT_TAG`` – robot_type → EmbodimentTag

``ROBOT_TYPE_TO_EMBODIMENT_TAG`` is **derived** from ``ROBOT_TYPE_CONFIG_MAP``
by reading each DataConfig class's ``embodiment_tag`` classvar (Proposal A).
Classes without the classvar fall back to ``EmbodimentTag.NEW_EMBODIMENT``.
Legacy bench files exposing their own ``ROBOT_TYPE_TO_EMBODIMENT_TAG`` dict are
still honored as overrides for backward compatibility.


Usage::

    from starVLA.dataloader.gr00t_lerobot.registry import (
        ROBOT_TYPE_CONFIG_MAP,
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
        DATASET_NAMED_MIXTURES,
    )
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

# Base registries (kept as fallback / seed values)
from starVLA.dataloader.gr00t_lerobot.data_config import (
    ROBOT_TYPE_CONFIG_MAP as _BASE_CONFIG_MAP,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import (
    EmbodimentTag,  # re-export for convenience
)
from starVLA.dataloader.gr00t_lerobot.mixtures import (
    DATASET_NAMED_MIXTURES as _BASE_MIXTURES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mutable copies – will be extended by discovered modules
# ---------------------------------------------------------------------------
ROBOT_TYPE_CONFIG_MAP: dict = dict(_BASE_CONFIG_MAP)
DATASET_NAMED_MIXTURES: dict = dict(_BASE_MIXTURES)

# Legacy explicit overrides (rarely needed; prefer the classvar on DataConfig).
_LEGACY_TAG_OVERRIDES: dict = {}


def _derive_tag_map() -> dict:
    """Build robot_type -> EmbodimentTag from ROBOT_TYPE_CONFIG_MAP.

    Reads each DataConfig instance's ``embodiment_tag`` classvar. Falls back to
    ``EmbodimentTag.NEW_EMBODIMENT`` if the classvar is missing. Legacy
    overrides from bench files take precedence.
    """
    out: dict = {}
    for rt, cfg in ROBOT_TYPE_CONFIG_MAP.items():
        tag = getattr(cfg, "embodiment_tag", None)
        if tag is None:
            logger.warning(
                "[registry] DataConfig for robot_type=%r has no `embodiment_tag` "
                "classvar; defaulting to EmbodimentTag.NEW_EMBODIMENT.", rt
            )
            tag = EmbodimentTag.NEW_EMBODIMENT
        out[rt] = tag
    out.update(_LEGACY_TAG_OVERRIDES)
    return out


ROBOT_TYPE_TO_EMBODIMENT_TAG: dict = {}  # populated by discover_and_merge()

# ---------------------------------------------------------------------------
# Discovery logic
# ---------------------------------------------------------------------------
_REGISTRY_DIR_NAME = "data_registry"
_DISCOVERED = False


def _find_registry_dirs() -> list[Path]:
    """Return all ``examples/*/train_files/data_registry/`` directories."""
    # Walk up from this file to the repo root
    # registry.py is at starVLA/dataloader/gr00t_lerobot/registry.py
    #   parents: [0]=gr00t_lerobot, [1]=dataloader, [2]=starVLA(pkg), [3]=repo root
    repo_root = Path(__file__).resolve().parents[3]
    examples_dir = repo_root / "examples"
    if not examples_dir.is_dir():
        return []
    dirs: list[Path] = []
    for bench_dir in sorted(examples_dir.iterdir()):
        registry_dir = bench_dir / "train_files" / _REGISTRY_DIR_NAME
        if registry_dir.is_dir():
            dirs.append(registry_dir)
    return dirs


def _load_module_from_path(module_name: str, file_path: Path):
    """Import a Python file as a module with the given name."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def discover_and_merge() -> None:
    """Scan ``examples/*/train_files/data_registry/`` and merge into global registries."""
    global _DISCOVERED, ROBOT_TYPE_TO_EMBODIMENT_TAG
    if _DISCOVERED:
        return
    _DISCOVERED = True

    for registry_dir in _find_registry_dirs():
        bench_name = registry_dir.parents[1].name  # examples/<BenchName>/train_files/data_registry
        prefix = f"_data_registry_{bench_name}"

        # --- data_config.py (may contain all three registries) ---
        cfg_file = registry_dir / "data_config.py"
        if cfg_file.is_file():
            mod = _load_module_from_path(f"{prefix}.data_config", cfg_file)
            if mod:
                if hasattr(mod, "ROBOT_TYPE_CONFIG_MAP"):
                    ROBOT_TYPE_CONFIG_MAP.update(mod.ROBOT_TYPE_CONFIG_MAP)
                    logger.debug(f"[registry] Loaded data_config from {bench_name}: {list(mod.ROBOT_TYPE_CONFIG_MAP.keys())}")
                # Legacy: bench file may still expose an explicit dict override.
                if hasattr(mod, "ROBOT_TYPE_TO_EMBODIMENT_TAG"):
                    _LEGACY_TAG_OVERRIDES.update(mod.ROBOT_TYPE_TO_EMBODIMENT_TAG)
                    logger.debug(f"[registry] Legacy embodiment_tag overrides from {bench_name}: {list(mod.ROBOT_TYPE_TO_EMBODIMENT_TAG.keys())}")
                if hasattr(mod, "DATASET_NAMED_MIXTURES"):
                    DATASET_NAMED_MIXTURES.update(mod.DATASET_NAMED_MIXTURES)
                    logger.debug(f"[registry] Loaded mixtures from {bench_name} (data_config): {list(mod.DATASET_NAMED_MIXTURES.keys())}")

    # Re-derive after all benches are loaded so classvars + legacy overrides combine.
    ROBOT_TYPE_TO_EMBODIMENT_TAG = _derive_tag_map()


# Run discovery on first import
discover_and_merge()
