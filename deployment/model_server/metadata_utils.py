"""Lightweight helpers for policy-server metadata."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional


def as_string_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


def training_video_keys_from_model_config(model_cfg: Dict[str, Any]) -> Optional[List[str]]:
    """Return the training video-key order when one data config defines it."""
    vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {})
    data_mix = vla_data_cfg.get("data_mix", None)
    if not data_mix:
        return None

    try:
        from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
        from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
    except Exception as exc:  # pragma: no cover - metadata should not block serving
        logging.warning("Could not import data config metadata: %s", exc)
        return None

    mix_entries = DATASET_NAMED_MIXTURES.get(data_mix, None)
    if not mix_entries:
        return None

    video_key_orders: List[List[str]] = []
    for _, _, robot_type in mix_entries:
        robot_cfg = ROBOT_TYPE_CONFIG_MAP.get(robot_type, None)
        keys = as_string_list(getattr(robot_cfg, "video_keys", None)) if robot_cfg is not None else None
        if keys:
            video_key_orders.append(keys)

    if not video_key_orders:
        return None

    first = video_key_orders[0]
    if all(keys == first for keys in video_key_orders):
        return first
    logging.warning(
        "Data mix %s uses multiple video-key orders; omitting vla_video_keys metadata: %s",
        data_mix,
        video_key_orders,
    )
    return None


def build_image_metadata_from_model_config(model_cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build image-contract metadata advertised by the policy server."""
    vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {})
    vla_data_obs = as_string_list(vla_data_cfg.get("obs", None))
    vla_video_keys = training_video_keys_from_model_config(model_cfg)

    metadata: Dict[str, List[str]] = {}
    if vla_video_keys is not None:
        metadata["vla_video_keys"] = vla_video_keys
    if vla_data_obs is not None:
        metadata["vla_data_obs"] = vla_data_obs

    if vla_video_keys is not None and vla_data_obs is not None and len(vla_video_keys) != len(vla_data_obs):
        logging.warning(
            "Using derived vla_video_keys over raw vla_data_obs for eval image ordering because they "
            "describe different image counts: vla_video_keys=%s, vla_data_obs=%s",
            vla_video_keys,
            vla_data_obs,
        )

    return metadata
