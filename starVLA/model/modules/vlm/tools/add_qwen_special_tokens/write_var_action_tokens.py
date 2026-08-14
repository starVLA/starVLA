"""Write VAR action special tokens for Qwen tokenizer augmentation."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write <var_action_i> tokens, one per line.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--prefix", type=str, default="<var_action_")
    parser.add_argument("--suffix", type=str, default=">")
    args = parser.parse_args()

    if args.codebook_size <= 0:
        raise ValueError(f"codebook_size must be positive, got {args.codebook_size}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for token_id in range(args.codebook_size):
            handle.write(f"{args.prefix}{token_id}{args.suffix}\n")
    print(f"Wrote {args.codebook_size} VAR action tokens to {args.output}")


if __name__ == "__main__":
    main()
