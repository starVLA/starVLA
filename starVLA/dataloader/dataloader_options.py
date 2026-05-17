"""Helpers for dataset-level PyTorch DataLoader options."""

from __future__ import annotations

from typing import Any

DATALOADER_DEFAULTS = {
    "num_workers": 4,
    "pin_memory": False,
    "persistent_workers": False,
    "prefetch_factor": 2,
    "drop_last": False,
    "timeout": 0,
}


def _get_config_value(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        value = config.get(key, default)
    else:
        value = getattr(config, key, default)
    return default if value is None else value


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"DataLoader option `{key}` must be a boolean, got {value!r}.")


def _as_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"DataLoader option `{key}` must be an integer, got {value!r}.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"DataLoader option `{key}` must be an integer, got {value!r}.") from exc


def build_dataloader_kwargs(data_cfg: Any) -> dict[str, Any]:
    """Build DataLoader kwargs from a dataset config, preserving legacy defaults."""
    num_workers = _as_int(_get_config_value(data_cfg, "num_workers", DATALOADER_DEFAULTS["num_workers"]), "num_workers")
    timeout = _as_int(_get_config_value(data_cfg, "timeout", DATALOADER_DEFAULTS["timeout"]), "timeout")
    persistent_workers = _as_bool(
        _get_config_value(data_cfg, "persistent_workers", DATALOADER_DEFAULTS["persistent_workers"]),
        "persistent_workers",
    )

    if num_workers < 0:
        raise ValueError("DataLoader option `num_workers` must be non-negative.")
    if timeout < 0:
        raise ValueError("DataLoader option `timeout` must be non-negative.")
    if num_workers == 0 and persistent_workers:
        raise ValueError("DataLoader option `persistent_workers=True` requires `num_workers > 0`.")

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": _as_bool(
            _get_config_value(data_cfg, "pin_memory", DATALOADER_DEFAULTS["pin_memory"]), "pin_memory"
        ),
        "persistent_workers": persistent_workers,
        "drop_last": _as_bool(_get_config_value(data_cfg, "drop_last", DATALOADER_DEFAULTS["drop_last"]), "drop_last"),
        "timeout": timeout,
    }
    if num_workers > 0:
        prefetch_factor = _as_int(
            _get_config_value(data_cfg, "prefetch_factor", DATALOADER_DEFAULTS["prefetch_factor"]),
            "prefetch_factor",
        )
        if prefetch_factor <= 0:
            raise ValueError("DataLoader option `prefetch_factor` must be positive when `num_workers > 0`.")
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs
