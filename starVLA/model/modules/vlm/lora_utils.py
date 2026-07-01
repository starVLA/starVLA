"""Utilities for optional VLM backbone LoRA fine-tuning.

Keep PEFT imports inside enabled LoRA paths so the repository remains importable
when PEFT is not installed and LoRA is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional


DEFAULT_QWEN_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Backward-compatible name used by the earlier Qwen-only LoRA implementation.
DEFAULT_TARGET_MODULES = DEFAULT_QWEN_TARGET_MODULES

DEFAULT_VLM_MODULE_PATH = "qwen_vl_interface.model"
DEFAULT_ADAPTER_DIR_NAME = "vlm_lora_adapter"
LEGACY_ADAPTER_DIR_NAME = "qwen_lora_adapter"
_MISSING = object()


@dataclass(frozen=True)
class LoraSettings:
    enabled: bool = False
    adapter_path: Optional[str] = None
    adapter_name: str = "default"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] | str | None = None
    is_trainable: bool = True
    save_adapter_only: bool = False
    module_path: str = DEFAULT_VLM_MODULE_PATH
    adapter_dir_name: str = DEFAULT_ADAPTER_DIR_NAME
    task_type: Optional[str] = "CAUSAL_LM"
    modules_to_save: list[str] | None = None

    def __post_init__(self) -> None:
        if self.target_modules is None:
            object.__setattr__(self, "target_modules", list(DEFAULT_QWEN_TARGET_MODULES))
        if not self.module_path:
            object.__setattr__(self, "module_path", DEFAULT_VLM_MODULE_PATH)
        if not self.adapter_dir_name:
            object.__setattr__(self, "adapter_dir_name", DEFAULT_ADAPTER_DIR_NAME)


def _get_value(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if hasattr(container, "get"):
        try:
            return container.get(key, default)
        except Exception:
            pass
    if isinstance(container, dict):
        return container.get(key, default)
    try:
        return getattr(container, key)
    except AttributeError:
        return default


def _set_value(container: Any, key: str, value: Any) -> None:
    if isinstance(container, dict):
        container[key] = value
        return
    try:
        container[key] = value
        return
    except Exception:
        pass
    setattr(container, key, value)


def _get_nested(container: Any, path: Iterable[str], default: Any = None) -> Any:
    current = container
    for key in path:
        current = _get_value(current, key, default)
        if current is default:
            return default
    return current


def _ensure_child(container: Any, key: str) -> Any:
    child = _get_value(container, key, _MISSING)
    if child is not _MISSING and child is not None:
        return child

    if isinstance(container, SimpleNamespace):
        new_child: Any = SimpleNamespace()
    else:
        new_child = {}
    _set_value(container, key, new_child)
    return _get_value(container, key, new_child)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_target_modules(value: Any) -> list[str] | str:
    if value is None:
        return list(DEFAULT_QWEN_TARGET_MODULES)
    if isinstance(value, str):
        value = value.strip()
        if value == "all-linear":
            return value
        return [module.strip() for module in value.split(",") if module.strip()]
    return [str(module).strip() for module in value if str(module).strip()]


def _as_optional_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"none", "null"}:
            return None
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _lora_config_path(cfg: Any) -> tuple[str, str, str]:
    canonical = _get_nested(cfg, ("framework", "vlm", "lora"), default=_MISSING)
    if canonical is not _MISSING:
        return ("framework", "vlm", "lora")

    legacy = _get_nested(cfg, ("framework", "qwenvl", "lora"), default=_MISSING)
    if legacy is not _MISSING:
        return ("framework", "qwenvl", "lora")

    return ("framework", "vlm", "lora")


def _get_lora_config(cfg: Any) -> Any:
    return _get_nested(cfg, _lora_config_path(cfg), default=None)


def get_lora_settings(cfg: Any) -> LoraSettings:
    """Return normalized LoRA settings from VLM LoRA config.

    ``framework.vlm.lora`` is the canonical path. ``framework.qwenvl.lora`` is
    accepted as a compatibility fallback for existing configs and checkpoints.
    """

    lora_cfg = _get_lora_config(cfg)
    return LoraSettings(
        enabled=_as_bool(_get_value(lora_cfg, "enabled", False)),
        adapter_path=_get_value(lora_cfg, "adapter_path", None),
        adapter_name=str(_get_value(lora_cfg, "adapter_name", "default")),
        r=int(_get_value(lora_cfg, "r", 16)),
        alpha=int(_get_value(lora_cfg, "alpha", 32)),
        dropout=float(_get_value(lora_cfg, "dropout", 0.05)),
        bias=str(_get_value(lora_cfg, "bias", "none")),
        target_modules=_as_target_modules(_get_value(lora_cfg, "target_modules", None)),
        is_trainable=_as_bool(_get_value(lora_cfg, "is_trainable", True)),
        save_adapter_only=_as_bool(_get_value(lora_cfg, "save_adapter_only", False)),
        module_path=str(_get_value(lora_cfg, "module_path", DEFAULT_VLM_MODULE_PATH)),
        adapter_dir_name=str(_get_value(lora_cfg, "adapter_dir_name", DEFAULT_ADAPTER_DIR_NAME)),
        task_type=_get_value(lora_cfg, "task_type", "CAUSAL_LM"),
        modules_to_save=_as_optional_list(_get_value(lora_cfg, "modules_to_save", None)),
    )


def is_lora_enabled(cfg: Any) -> bool:
    return get_lora_settings(cfg).enabled


def inject_lora_adapter_path(cfg: Any, adapter_path: str | Path, *, is_trainable: bool) -> None:
    """Inject an adapter path into the config path that LoRA settings use."""

    framework = _ensure_child(cfg, "framework")
    _, framework_key, lora_key = _lora_config_path(cfg)
    framework_section = _ensure_child(framework, framework_key)
    lora_section = _ensure_child(framework_section, lora_key)
    _set_value(lora_section, "enabled", True)
    _set_value(lora_section, "adapter_path", str(adapter_path))
    _set_value(lora_section, "is_trainable", is_trainable)


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def resolve_vlm_lora_adapter_dir(
    pretrained_checkpoint: str | Path,
    adapter_dir_name: str = DEFAULT_ADAPTER_DIR_NAME,
) -> Path | None:
    """Locate the PEFT adapter folder saved alongside supported checkpoints.

    Generic ``vlm_lora_adapter`` directories are preferred. Legacy
    ``qwen_lora_adapter`` directories remain load-compatible.
    """

    ckpt = Path(pretrained_checkpoint)
    candidates = []

    if ckpt.is_dir() and (
        ckpt.name.endswith(f"_{adapter_dir_name}")
        or ckpt.name.endswith(f"_{DEFAULT_ADAPTER_DIR_NAME}")
        or ckpt.name.endswith(f"_{LEGACY_ADAPTER_DIR_NAME}")
        or ckpt.name in {adapter_dir_name, DEFAULT_ADAPTER_DIR_NAME, LEGACY_ADAPTER_DIR_NAME}
        or (ckpt / "adapter_config.json").is_file()
    ):
        return ckpt

    if ckpt.is_dir():
        candidates.extend(
            [
                ckpt / adapter_dir_name,
                ckpt / DEFAULT_ADAPTER_DIR_NAME,
                ckpt / LEGACY_ADAPTER_DIR_NAME,
            ]
        )

    stem = ckpt.stem
    for suffix in ("_pytorch_model", "_model"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            candidates.extend(
                [
                    ckpt.with_name(f"{base}_{adapter_dir_name}"),
                    ckpt.with_name(f"{base}_{DEFAULT_ADAPTER_DIR_NAME}"),
                    ckpt.with_name(f"{base}_{LEGACY_ADAPTER_DIR_NAME}"),
                ]
            )

    if stem in {"pytorch_model", "model"}:
        candidates.extend(
            [
                ckpt.with_name(adapter_dir_name),
                ckpt.with_name(DEFAULT_ADAPTER_DIR_NAME),
                ckpt.with_name(LEGACY_ADAPTER_DIR_NAME),
            ]
        )

    candidates.extend(
        [
            ckpt.with_name(f"{stem}_{adapter_dir_name}"),
            ckpt.with_name(f"{stem}_{DEFAULT_ADAPTER_DIR_NAME}"),
            ckpt.with_name(f"{stem}_{LEGACY_ADAPTER_DIR_NAME}"),
        ]
    )

    for candidate in _dedupe_paths(candidates):
        if candidate.is_dir():
            return candidate
    return None


def resolve_qwen_lora_adapter_dir(pretrained_checkpoint: str | Path) -> Path | None:
    """Backward-compatible alias for the generic VLM adapter resolver."""

    return resolve_vlm_lora_adapter_dir(pretrained_checkpoint)


def _get_child(obj: Any, key: str) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        return obj[key]
    except Exception as exc:
        raise AttributeError(f"LoRA enabled, but `{type(obj).__name__}` has no child `{key}`") from exc


def _set_child(obj: Any, key: str, value: Any) -> None:
    try:
        setattr(obj, key, value)
        return
    except Exception:
        pass
    try:
        obj[key] = value
        return
    except Exception as exc:
        raise AttributeError(f"LoRA enabled, but cannot set child `{key}` on `{type(obj).__name__}`") from exc


def _resolve_module_path(root: Any, module_path: str) -> tuple[Any, str, Any]:
    parts = [part for part in module_path.split(".") if part]
    if not parts:
        raise ValueError("framework.vlm.lora.module_path must not be empty")

    parent = root
    for part in parts[:-1]:
        parent = _get_child(parent, part)
    attr = parts[-1]
    module = _get_child(parent, attr)
    if module is None:
        raise AttributeError(f"LoRA enabled, but `{module_path}` resolved to None")
    return parent, attr, module


def _task_type_value(task_type: Optional[str], TaskType: Any) -> Any:
    if task_type is None:
        return None
    task_type = str(task_type).strip()
    if not task_type or task_type.lower() in {"none", "null"}:
        return None
    return getattr(TaskType, task_type, task_type)


def apply_lora_to_vlm_backbone(vla_model: Any, cfg: Any, logger: Any = None) -> tuple[Any, LoraSettings]:
    """Wrap a configured VLM backbone module with a PEFT LoRA adapter."""

    settings = get_lora_settings(cfg)
    if not settings.enabled:
        return vla_model, settings

    parent, module_attr, base_vlm_model = _resolve_module_path(vla_model, settings.module_path)

    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "framework.vlm.lora.enabled=true requires the optional 'peft' package. "
            "Install it with the project requirements or disable LoRA."
        ) from exc

    if settings.adapter_path:
        wrapped_vlm_model = PeftModel.from_pretrained(
            base_vlm_model,
            settings.adapter_path,
            adapter_name=settings.adapter_name,
            is_trainable=settings.is_trainable,
        )
        message = f"Loaded VLM LoRA adapter from {settings.adapter_path}"
    else:
        lora_kwargs = {
            "r": settings.r,
            "lora_alpha": settings.alpha,
            "lora_dropout": settings.dropout,
            "bias": settings.bias,
            "target_modules": settings.target_modules,
        }
        task_type = _task_type_value(settings.task_type, TaskType)
        if task_type is not None:
            lora_kwargs["task_type"] = task_type
        if settings.modules_to_save is not None:
            lora_kwargs["modules_to_save"] = settings.modules_to_save

        peft_config = LoraConfig(**lora_kwargs)
        wrapped_vlm_model = get_peft_model(
            base_vlm_model,
            peft_config,
            adapter_name=settings.adapter_name,
        )
        if not settings.is_trainable:
            for param in wrapped_vlm_model.parameters():
                param.requires_grad = False
        message = (
            "Initialized VLM LoRA adapter "
            f"path={settings.module_path}, name={settings.adapter_name}, r={settings.r}, alpha={settings.alpha}"
        )

    _set_child(parent, module_attr, wrapped_vlm_model)
    if logger is not None:
        logger.info(message)
    return vla_model, settings


def apply_lora_to_qwen_vl_interface(vla_model: Any, cfg: Any, logger: Any = None) -> tuple[Any, LoraSettings]:
    """Backward-compatible alias for VLM backbone LoRA wrapping."""

    return apply_lora_to_vlm_backbone(vla_model, cfg, logger=logger)


def save_vlm_lora_adapter(
    vla_model: Any,
    output_dir: str,
    adapter_name: Optional[str] = None,
    module_path: str = DEFAULT_VLM_MODULE_PATH,
) -> None:
    """Save the PEFT adapter attached to the configured VLM backbone module."""

    _, _, vlm_model = _resolve_module_path(vla_model, module_path)
    if adapter_name:
        try:
            vlm_model.save_pretrained(output_dir, selected_adapters=[adapter_name])
            return
        except TypeError:
            pass
    vlm_model.save_pretrained(output_dir)


def save_qwen_lora_adapter(vla_model: Any, output_dir: str, adapter_name: Optional[str] = None) -> None:
    """Backward-compatible alias for saving VLM backbone LoRA adapters."""

    save_vlm_lora_adapter(vla_model, output_dir, adapter_name=adapter_name)
