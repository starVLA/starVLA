"""Benchmark state-normalization boundary around StarVLA's official policy server.

Run from the StarVLA repository root; the Evo-DexBench data configs are registered automatically from
``examples/simBenchmarks/EvoDexBench/train_files/data_registry``.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np


def apply_saved_state_transform(processor: Any, state: np.ndarray) -> np.ndarray:
    """Apply the saved q99 transform to physical state [T, 31 or 62]."""
    state = np.asarray(state, dtype=np.float32)
    widths = [processor.state_key_dims[key] for key in processor.state_keys]
    if state.ndim != 2 or state.shape[1] != sum(widths) or not np.isfinite(state).all():
        raise ValueError("Physical state shape or finite-value check failed")
    offset = 0
    fields = {}
    for key, width in zip(processor.state_keys, widths, strict=True):
        fields[key] = state[:, offset : offset + width].copy()
        offset += width
    transformed = processor.transform.apply(fields)
    return np.concatenate([np.asarray(transformed[key].detach().cpu()) for key in processor.state_keys], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-override", action="append", default=[])
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")

    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from deployment.model_server.seed_utils import set_seed_everywhere
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    class EvoPolicyWrapper(PolicyServerWrapper):
        def predict_action(
            self,
            examples: list[dict[str, Any]],
            unnorm_key: str | None = None,
            normalize_state: bool = False,
            **kwargs: Any,
        ) -> dict[str, np.ndarray]:
            if normalize_state:
                processor = self.get_norm_processor(unnorm_key)
                examples = [
                    {**example, "state": apply_saved_state_transform(processor, example["state"])}
                    for example in examples
                ]
            return super().predict_action(examples, unnorm_key=unnorm_key, **kwargs)

    set_seed_everywhere(args.seed)
    policy = EvoPolicyWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        config_overrides=args.config_override,
    )
    WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={**policy.metadata, "seed": args.seed},
    ).serve_forever()


if __name__ == "__main__":
    main()
