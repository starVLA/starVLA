# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""ZMQ REP policy server speaking the Isaac-GR00T PolicyServer wire protocol.

Protocol (byte-compatible with ``gr00t/policy/server_client.py`` in
Isaac-GR00T N1.6, i.e. what GR00T-WBC-Bridge's ``GR00TN1Client`` speaks with
its default ``--server_codec custom``):

- transport: ZMQ REQ/REP, one msgpack blob per message.
- ndarrays:  encoded as ``{"__ndarray_class__": True, "as_npy": <np.save bytes>}``.
- request:   ``{"endpoint": <name>, "data": {...}}`` (endpoint defaults to
  ``get_action``).
- response:  the handler's return value; errors as ``{"error": str}``.

Endpoints mirror the GR00T server: ``ping``, ``kill``, ``get_action``,
``reset``, ``get_modality_config``.
"""

from __future__ import annotations

import io
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict

import msgpack
import numpy as np
import zmq

logger = logging.getLogger(__name__)


# -- msgpack codec (matches Isaac-GR00T MsgSerializer for ndarrays) -----------

def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _decode(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def pack(data: Any) -> bytes:
    return msgpack.packb(data, default=_encode, strict_types=False)


def unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode, raw=False)


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class ZmqGr00tPolicyServer:
    """Serve a GR00T-style policy object over the GR00T ZMQ protocol.

    ``policy`` must expose ``get_action(observation, options)``,
    ``reset(options)`` and (optionally) ``get_modality_config()`` — e.g.
    :class:`deployment.model_server.gr00t_obs_adapter.Gr00tCompatPolicy`.
    """

    def __init__(self, policy, host: str = "0.0.0.0", port: int = 5555) -> None:
        self._policy = policy
        self.running = False
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(f"tcp://{host}:{port}")

        self._endpoints: Dict[str, EndpointHandler] = {
            "ping": EndpointHandler(self._ping, requires_input=False),
            "kill": EndpointHandler(self._kill, requires_input=False),
            "get_action": EndpointHandler(policy.get_action),
            "reset": EndpointHandler(policy.reset),
            "get_modality_config": EndpointHandler(
                getattr(policy, "get_modality_config", lambda: {}),
                requires_input=False,
            ),
        }

    def _ping(self) -> dict:
        return {"status": "ok", "message": "Server is running"}

    def _kill(self) -> dict:
        self.running = False
        return {"status": "ok", "message": "Server shutting down"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def run(self) -> None:
        addr = self._sock.getsockopt_string(zmq.LAST_ENDPOINT)
        logger.info("GR00T-compat ZMQ server listening on %s", addr)
        self.running = True
        while self.running:
            # Poll so `kill` (or an external stop()) exits promptly instead of
            # blocking forever in recv().
            if not self._sock.poll(timeout=200):
                continue
            try:
                request = unpack(self._sock.recv())
                endpoint = request.get("endpoint", "get_action")
                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")
                spec = self._endpoints[endpoint]
                result = (
                    spec.handler(**request.get("data", {}))
                    if spec.requires_input
                    else spec.handler()
                )
                self._sock.send(pack(result))
            except Exception as e:  # noqa: BLE001 — REQ/REP must always reply
                logger.error("Error handling request: %s\n%s", e, traceback.format_exc())
                try:
                    self._sock.send(pack({"error": str(e)}))
                except zmq.ZMQError:
                    pass
        self.close()

    def stop(self) -> None:
        self.running = False

    def close(self) -> None:
        self._sock.close(linger=0)
        self._ctx.term()
