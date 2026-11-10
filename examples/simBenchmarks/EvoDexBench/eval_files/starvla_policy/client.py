"""Lightweight client for the pinned StarVLA WebSocket wire protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _pack_array(value: Any) -> Any:
    """Match StarVLA's msgpack ndarray representation without object pickling."""

    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in "VOc":
        raise TypeError(f"Unsupported NumPy dtype: {value.dtype}.")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"Cannot encode {type(value).__name__} for StarVLA.")


def _unpack_array(value: dict[Any, Any]) -> Any:
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in "VOc":
            raise TypeError(f"Unsupported NumPy dtype: {dtype}.")
        return np.ndarray(buffer=value[b"data"], dtype=dtype, shape=tuple(value[b"shape"]))
    if b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in "VOc":
            raise TypeError(f"Unsupported NumPy dtype: {dtype}.")
        return dtype.type(value[b"data"])
    return value


class StarVLAWebsocketClient:
    """Read server metadata once, then exchange physical-action requests."""

    def __init__(self, host: str, port: int) -> None:
        import msgpack
        from websockets.sync.client import connect

        self._msgpack = msgpack
        self._connection = connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            open_timeout=30,
            ping_interval=None,
        )
        try:
            metadata = self._decode(self._connection.recv(timeout=30))
            if not isinstance(metadata, Mapping):
                raise TypeError("StarVLA server metadata must be a mapping.")
        except BaseException:
            self._connection.close()
            raise
        self._metadata = dict(metadata)

    def _decode(self, response: bytes | str) -> Any:
        if isinstance(response, str):
            raise RuntimeError(f"StarVLA server error: {response}")
        return self._msgpack.unpackb(response, object_hook=_unpack_array)

    def get_server_metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def predict_action(self, query: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = self._msgpack.Packer(default=_pack_array).pack(dict(query))
        self._connection.send(payload)
        response = self._decode(self._connection.recv())
        if not isinstance(response, Mapping):
            raise TypeError("StarVLA prediction response must be a mapping.")
        return response

    def close(self) -> None:
        self._connection.close()
