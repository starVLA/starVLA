"""Lightweight training throughput and memory metric helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

_BYTES_PER_GIB = 1024**3


def _tensor_length(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    return int(shape[0])


def count_batch_samples(batch: Any) -> int:
    """Infer the number of examples represented by a dataloader batch."""
    if batch is None:
        return 0
    if isinstance(batch, Mapping):
        attention_mask = batch.get("attention_mask")
        input_ids = batch.get("input_ids")
        if attention_mask is not None:
            attention_length = _tensor_length(attention_mask)
            input_length = _tensor_length(input_ids)
            if attention_length is not None and input_length == 1 and getattr(attention_mask, "ndim", 0) == 1:
                return max(attention_length - 1, 1)
            if attention_length is not None:
                return attention_length
        for value in batch.values():
            value_length = _tensor_length(value)
            if value_length is not None:
                return value_length
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return len(value)
        return 1
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        return len(batch)
    tensor_length = _tensor_length(batch)
    return tensor_length if tensor_length is not None else 1


def count_batches_samples(*batches: Any) -> int:
    """Infer total examples across one or more dataloader batches."""
    return sum(count_batch_samples(batch) for batch in batches)


def build_step_performance_metrics(
    data_time: float,
    model_time: float,
    sample_count: int,
    cuda: Any = torch.cuda,
) -> dict[str, float]:
    """Build grouped timing, throughput, and GPU-memory metrics for one loop iteration."""
    step_time = data_time + model_time
    metrics = {
        "timing/step": step_time,
        "throughput/samples_per_sec": sample_count / step_time if step_time > 0 else 0.0,
        "throughput/model_samples_per_sec": sample_count / model_time if model_time > 0 else 0.0,
        "throughput/data_wait_ratio": data_time / step_time if step_time > 0 else 0.0,
        "memory/gpu_allocated_gb": 0.0,
        "memory/gpu_reserved_gb": 0.0,
    }

    if cuda is not None and cuda.is_available():
        metrics["memory/gpu_allocated_gb"] = cuda.memory_allocated() / _BYTES_PER_GIB
        metrics["memory/gpu_reserved_gb"] = cuda.memory_reserved() / _BYTES_PER_GIB
    return metrics
