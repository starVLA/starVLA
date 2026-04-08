from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Any, Dict

import numpy as np
from PIL import Image
try:
    import torch
except ImportError:  # pragma: no cover - runtime environments still install torch
    torch = None


def _update_hasher(hasher: "hashlib._Hash", value: Any) -> None:
    if value is None:
        hasher.update(b"n")
        return

    if isinstance(value, bool):
        hasher.update(b"b")
        hasher.update(b"1" if value else b"0")
        return

    if isinstance(value, (int, float, str)):
        hasher.update(type(value).__name__.encode("utf-8"))
        hasher.update(str(value).encode("utf-8"))
        return

    if isinstance(value, bytes):
        hasher.update(b"bytes")
        hasher.update(value)
        return

    if isinstance(value, Image.Image):
        hasher.update(b"pil")
        hasher.update(value.mode.encode("utf-8"))
        hasher.update(str(value.size).encode("utf-8"))
        hasher.update(value.tobytes())
        return

    if torch is not None and torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"torch")
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
        return

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"numpy")
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(array.tobytes())
        return

    if isinstance(value, dict):
        hasher.update(b"dict")
        for key in sorted(value.keys(), key=lambda item: str(item)):
            _update_hasher(hasher, key)
            _update_hasher(hasher, value[key])
        return

    if isinstance(value, (list, tuple)):
        hasher.update(b"seq")
        hasher.update(str(len(value)).encode("utf-8"))
        for item in value:
            _update_hasher(hasher, item)
        return

    hasher.update(type(value).__name__.encode("utf-8"))
    hasher.update(repr(value).encode("utf-8"))


def estimate_cache_value_bytes(value: Any) -> int:
    if value is None:
        return 0

    if isinstance(value, bytes):
        return len(value)

    if isinstance(value, str):
        return len(value.encode("utf-8"))

    if isinstance(value, Image.Image):
        return len(value.tobytes())

    if torch is not None and torch.is_tensor(value):
        return int(value.nelement() * value.element_size())

    if isinstance(value, np.ndarray):
        return int(value.nbytes)

    if isinstance(value, dict):
        return sum(estimate_cache_value_bytes(item) for item in value.values())

    if isinstance(value, (list, tuple)):
        return sum(estimate_cache_value_bytes(item) for item in value)

    return 0


def build_multimodal_cache_key(
    *,
    images: Any,
    instructions: Any,
    override_key: str | None = None,
    extra: Any = None,
) -> str:
    if override_key is not None:
        return str(override_key)

    hasher = hashlib.blake2b(digest_size=20)
    _update_hasher(hasher, instructions)
    _update_hasher(hasher, images)
    if extra is not None:
        _update_hasher(hasher, extra)
    return hasher.hexdigest()


@dataclass
class CacheEntry:
    key: str
    value: Any
    size_bytes: int


class SessionInferenceCache:
    def __init__(self) -> None:
        self._entries: Dict[str, CacheEntry] = {}
        self._stats = defaultdict(lambda: {"hits": 0, "misses": 0})

    @staticmethod
    def _normalize_session_id(session_id: str | None) -> str:
        return str(session_id or "default")

    def get(self, session_id: str | None, cache_key: str) -> tuple[Any, bool]:
        normalized_session_id = self._normalize_session_id(session_id)
        entry = self._entries.get(normalized_session_id)
        if entry is not None and entry.key == cache_key:
            self._stats[normalized_session_id]["hits"] += 1
            return entry.value, True

        self._stats[normalized_session_id]["misses"] += 1
        return None, False

    def put(self, session_id: str | None, cache_key: str, value: Any) -> None:
        normalized_session_id = self._normalize_session_id(session_id)
        self._entries[normalized_session_id] = CacheEntry(
            key=cache_key,
            value=value,
            size_bytes=estimate_cache_value_bytes(value),
        )

    def clear(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._entries.clear()
            self._stats.clear()
            return

        normalized_session_id = self._normalize_session_id(session_id)
        self._entries.pop(normalized_session_id, None)
        self._stats.pop(normalized_session_id, None)

    def stats(self, session_id: str | None = None) -> Dict[str, int]:
        if session_id is not None:
            normalized_session_id = self._normalize_session_id(session_id)
            stats = self._stats[normalized_session_id]
            entry = self._entries.get(normalized_session_id)
            return {
                "hits": int(stats["hits"]),
                "misses": int(stats["misses"]),
                "cache_entries": 1 if entry is not None else 0,
                "cache_bytes": int(entry.size_bytes if entry is not None else 0),
            }

        hits = sum(stats["hits"] for stats in self._stats.values())
        misses = sum(stats["misses"] for stats in self._stats.values())
        cache_bytes = sum(entry.size_bytes for entry in self._entries.values())
        return {
            "hits": int(hits),
            "misses": int(misses),
            "sessions": len(self._entries),
            "cache_entries": len(self._entries),
            "cache_bytes": int(cache_bytes),
        }
