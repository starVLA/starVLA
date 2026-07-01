# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import logging
import os
import socket

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def main(args) -> None:
    """Build the policy wrapper and start the websocket server.

    The wrapper now owns un-normalization + chunk_size discovery so that all
    eval clients (LIBERO / SimplerEnv / etc.) just need to forward `examples`
    and consume already-unnormalized actions from the response.
    """
    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # =========================================================================
    # !!! TRAIN / TEST CONSISTENCY — READ BEFORE SERVING !!!
    # -------------------------------------------------------------------------
    # This server replays the *training-time* observation contract. The eval
    # client MUST feed observations exactly as the model saw them at TRAIN time,
    # otherwise the success rate silently drops (no error is raised):
    #   - state   : is proprioceptive state used? (use_state) and its dim/order
    #   - img size: resize / crop resolution (e.g. 224x224)
    #   - img num : how many camera views are fed
    #   - img order: the ordering of those camera views
    #   - action normalization: unnorm_key / dataset stats must match training
    # `wrapper.metadata` (logged below and sent at handshake) exposes
    # action_chunk_size / state_keys / action_keys — cross-check these against
    # the training config used to produce `args.ckpt_path`.
    # =========================================================================
    logging.warning(
        "[TRAIN/TEST CONSISTENCY CHECK] serving ckpt=%s — verify eval observations "
        "(state / image size / image count / image order / action normalization) match "
        "the training config. metadata=%s",
        args.ckpt_path,
        wrapper.metadata,
    )

    # start websocket server; wrapper.metadata is sent at handshake.
    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    )
    logging.info("server running ... metadata=%s", wrapper.metadata)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
