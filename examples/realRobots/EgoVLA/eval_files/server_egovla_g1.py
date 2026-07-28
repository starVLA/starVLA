"""Serve the starVLA EgoVLA model over the GR00T ZMQ protocol on :5555.

The GR00T-WBC-Bridge (its ``--server_port 5555 --groot_version n1.7``) connects
UNCHANGED. EgoVLA's 48-dim camera-frame output is decoded to the bridge's
joint-target action contract inside :class:`EgoVLAG1Policy` (EE pose -> pinocchio
IK; MANO hand -> Inspire; see egovla_g1_policy.py).

Requires the license-gated EgoVLA assets locally (see this folder's README.md):
the EgoVLA checkpoint (``ego_vla_checkpoint``), MANO models, and the hand-retarget
nets. Point ``--egovla_release`` (or env ``EGOVLA_RELEASE``) at the checkout.

Example::

    python examples/realRobots/EgoVLA/eval_files/server_egovla_g1.py --port 5555 \
        --egovla_release /path/to/EgoVLA_Release
"""
import argparse
import logging
import os

try:  # standalone script: bootstrap the repo root onto sys.path (repo isn't pip-installed)
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:  # imported as a package module: root already on sys.path
    pass


def main(args) -> None:
    if args.egovla_release:
        os.environ["EGOVLA_RELEASE"] = args.egovla_release
    from examples.realRobots.EgoVLA.eval_files.egovla_g1_policy import EgoVLAG1Policy
    from deployment.model_server.tools.zmq_policy_server import ZmqGr00tPolicyServer

    policy = EgoVLAG1Policy(
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        fallback_instruction=args.fallback_instruction,
    )
    contract = policy.get_modality_config()
    logging.warning(
        "[EgoVLA G1 server] serving on :%d. state_keys=%s action_keys=%s horizon=%s. "
        "Decodes EgoVLA 48-dim (camera-frame EE + MANO) -> G1 joint targets (arm IK, "
        "Inspire hand). Bridge connects unchanged.",
        args.port, contract["state_keys"], contract["action_keys"], contract["action_horizon"],
    )

    server = ZmqGr00tPolicyServer(policy, host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        logging.info("Shutting down EgoVLA G1 server...")
        server.stop()


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--egovla_release", type=str, default=None,
                   help="EgoVLA_Release checkout (has checkpoint + MANO + hand nets).")
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="VILA-style EgoVLA checkpoint dir; default <egovla_release>/checkpoints/...")
    p.add_argument("--fallback_instruction", type=str, default="place the red ball in the box")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
