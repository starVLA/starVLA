"""Device selection shared by the inference entry points (XPU, then CUDA, then CPU).

An explicit device always wins over detection and is never swapped for another backend.
"""

from typing import Union

import torch

DeviceLike = Union[str, torch.device, None]
_ACCELERATORS = ("xpu", "cuda")


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu``, then ``cuda``, then ``cpu``."""
    for device_type in _ACCELERATORS:
        if is_available(device_type):
            return device_type
    return "cpu"


def default_device(index: int = 0) -> torch.device:
    """Device of the best available backend (``cuda:0`` on a CUDA host, as before)."""
    device_type = default_device_type()
    return torch.device("cpu") if device_type == "cpu" else torch.device(device_type, index)


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Resolve ``None`` to the detected default, anything else verbatim (a bare ``xpu``/``cuda`` means index 0)."""
    if device is None or device == "":
        return default_device()
    device = torch.device(device)
    if device.type in _ACCELERATORS and device.index is None:
        return torch.device(device.type, 0)
    return device


def device_type(device: DeviceLike) -> str:
    """Backend name (``xpu``, ``cuda``, ``cpu``) of a device, a device string or a tensor or module device."""
    return torch.device(device).type


def autocast(device: DeviceLike, dtype: torch.dtype):
    """``torch.autocast`` on the backend of ``device``; disabled on CPU, as the hard-coded ``"cuda"`` was.

    XPU autocast does not accept float32, so a float32 region is left to the caller's dtypes there.
    """
    backend = device_type(device)
    enabled = backend == "cuda" or (backend == "xpu" and dtype != torch.float32)
    return torch.autocast(backend if backend != "cpu" else "cuda", dtype=dtype, enabled=enabled)


def manual_seed_all(seed: int) -> None:
    """Seed every accelerator backend that is present (``torch.cuda`` seeding is a no-op on an XPU build)."""
    if is_available("cuda"):
        torch.cuda.manual_seed_all(seed)
    if is_available("xpu"):
        torch.xpu.manual_seed_all(seed)


def cast_like(module: torch.nn.Module, value, device: DeviceLike):
    """Cast a tensor (or a list of tensors) to the dtype of ``module`` when ``device`` is not CUDA.

    Only CUDA autocast casts the inputs of a float32 region, so on other backends the cast is done by hand.
    ``None`` and CUDA devices are passed through unchanged.
    """
    if value is None or device_type(device) == "cuda":
        return value
    dtype = next(module.parameters()).dtype
    if isinstance(value, (list, tuple)):
        return type(value)(v.to(dtype) for v in value)
    return value.to(dtype)
