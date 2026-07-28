# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""GR00T-protocol policy server (ZMQ) for starVLA checkpoints.

Drop-in replacement for Isaac-GR00T's ``gr00t/eval/run_gr00t_server.py``:
binds a ZMQ REP socket speaking the GR00T N1.6 msgpack protocol, so existing
GR00T clients — in particular the GR00T-WBC-Bridge deploy script
(``gr00t_n1_wbc_bridge_deploy.py`` with its default ``--server_codec custom``)
— work against a starVLA checkpoint with zero client changes.

The observation/action contract is derived from the checkpoint's training
DataConfig (state key order + per-key dims, action key split), so serving a
new embodiment only requires registering its DataConfig and training with it.

Example::

    python deployment/model_server/server_policy_gr00t_zmq.py \
        --ckpt_path /path/to/steps_N_pytorch_model.pt \
        --port 5555 --use_bf16
"""

import argparse
import logging

from deployment.model_server.gr00t_obs_adapter import Gr00tCompatPolicy
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.zmq_policy_server import ZmqGr00tPolicyServer


def main(args) -> None:
    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
        unnorm_key=args.unnorm_key,
    )
    policy = Gr00tCompatPolicy(
        wrapper,
        unnorm_key=args.unnorm_key,
        send_state=not args.no_state,
        fallback_instruction=args.fallback_instruction,
    )

    contract = policy.get_modality_config()
    logging.warning(
        "[TRAIN/TEST CONSISTENCY CHECK] serving ckpt=%s over the GR00T ZMQ protocol. "
        "Clients must send state keys %s (flattened in this order, dims %s) and will "
        "receive action keys %s (dims %s), chunk_size=%s. Cross-check against the "
        "client's bridge profile.",
        args.ckpt_path,
        contract["state_keys"],
        contract["state_key_dims"],
        contract["action_keys"],
        contract["action_key_dims"],
        wrapper.metadata.get("action_chunk_size"),
    )

    server = ZmqGr00tPolicyServer(policy, host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
        server.stop()


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument(
        "--unnorm_key", type=str, default=None,
        help="Dataset statistics key; required for multi-dataset checkpoints.",
    )
    parser.add_argument(
        "--no_state", action="store_true",
        help="Do not forward proprioceptive state (for state-less checkpoints).",
    )
    parser.add_argument(
        "--fallback_instruction", type=str, default="",
        help="Language instruction used when the observation carries none.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
