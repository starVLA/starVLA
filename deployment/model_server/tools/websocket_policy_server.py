# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import logging
import time

import websockets
import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # 新增参数，单位秒，-1表示永不关闭
        metadata: dict | None = None,
        
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    @staticmethod
    def _ok_response(req_id: str, response_type: str, data: dict | None = None) -> dict:
        response = {"status": "ok", "ok": True, "type": response_type, "request_id": req_id}
        if data is not None:
            response["data"] = data
        return response

    @staticmethod
    def _error_response(req_id: str, response_type: str, message: str, **extra) -> dict:
        error_payload = {"message": message}
        error_payload.update(extra)
        return {
            "status": "error",
            "ok": False,
            "type": response_type,
            "request_id": req_id,
            "error": error_payload,
        }

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server):
        """监控空闲时间，超时则关闭服务器"""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()  # 每次收到消息刷新活跃时间
                ret = self._route_message(msg)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logging.exception("Unexpected websocket server error")
                await websocket.send(
                    packer.pack(
                        self._error_response(
                            req_id="unknown",
                            response_type="internal_error",
                            message="Internal server error in websocket handler.",
                        )
                    )
                )
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        if not isinstance(msg, dict):
            return self._error_response(
                req_id="default",
                response_type="unknown",
                message="Top-level websocket message must be a dict.",
                payload_type=str(type(msg)),
            )

        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")  # default = infer
        payload = msg.get("payload", msg)  # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return self._ok_response(req_id=req_id, response_type="ping")

        # infer --> framework.predict_action
        elif mtype in {"infer", "predict_action"}:
            # Basic payload sanity
            if not isinstance(payload, dict):
                return self._error_response(
                    req_id=req_id,
                    response_type="inference_result",
                    message="Payload must be a dict.",
                    payload_type=str(type(payload)),
                )
            try:
                output_dict = self._policy.predict_action(**payload)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                return self._error_response(
                    req_id=req_id,
                    response_type="inference_result",
                    message=str(e),
                )
            return self._ok_response(req_id=req_id, response_type="inference_result", data=output_dict)

        # unknow request type
        else:
            return self._error_response(
                req_id=req_id,
                response_type="unknown",
                message=f"Unsupported message type '{mtype}'",
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
