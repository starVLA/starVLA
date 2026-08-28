"""Text codec for VAR action tokens used by autoregressive VLM policies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class VARTokenTextCodec:
    """Map integer VAR codebook ids to VLM special-token strings."""

    codebook_size: int
    prefix: str = "<var_action_"
    suffix: str = ">"

    def __post_init__(self) -> None:
        if self.codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {self.codebook_size}.")
        pattern = re.escape(self.prefix) + r"(\d+)" + re.escape(self.suffix)
        object.__setattr__(self, "_pattern", re.compile(pattern))

    def token_string(self, token_id: int) -> str:
        token_id = int(token_id)
        if token_id < 0 or token_id >= self.codebook_size:
            raise ValueError(f"VAR token id {token_id} outside [0, {self.codebook_size}).")
        return f"{self.prefix}{token_id}{self.suffix}"

    def all_token_strings(self) -> list[str]:
        return [self.token_string(token_id) for token_id in range(self.codebook_size)]

    def ids_to_text(self, token_ids: Sequence[int] | Iterable[int]) -> str:
        return "".join(self.token_string(int(token_id)) for token_id in token_ids)

    def ids_from_text(self, text: str, *, expected_len: int | None = None, strict: bool = False) -> list[int]:
        ids = [int(match.group(1)) for match in self._pattern.finditer(text)]
        invalid = [token_id for token_id in ids if token_id < 0 or token_id >= self.codebook_size]
        if invalid:
            raise ValueError(f"Found VAR token ids outside [0, {self.codebook_size}): {invalid[:10]}")
        if expected_len is not None and len(ids) != expected_len and strict:
            raise ValueError(f"Expected {expected_len} VAR tokens, found {len(ids)}.")
        if expected_len is not None:
            ids = ids[:expected_len]
        return ids

    def tokenizer_id_range(self, tokenizer) -> tuple[int, int]:
        """Return the contiguous tokenizer-id range for the VAR special tokens."""

        vocab_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in self.all_token_strings()]
        missing = [token for token, token_id in zip(self.all_token_strings(), vocab_ids, strict=True) if token_id < 0]
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if unk_id is not None:
            missing.extend(
                token
                for token, token_id in zip(self.all_token_strings(), vocab_ids, strict=True)
                if token_id == unk_id
            )
        if missing:
            raise ValueError(f"Tokenizer is missing VAR action special tokens, first missing tokens: {missing[:5]}")
        min_id, max_id = min(vocab_ids), max(vocab_ids)
        if sorted(vocab_ids) != list(range(min_id, max_id + 1)):
            raise ValueError("VAR action special tokens are not contiguous in the tokenizer vocabulary.")
        return min_id, max_id

    def ids_from_tokenizer_ids(
        self,
        generated_ids: Sequence[int],
        tokenizer,
        *,
        expected_len: int | None = None,
    ) -> list[int]:
        tokens = tokenizer.convert_ids_to_tokens([int(token_id) for token_id in generated_ids])
        text = "".join(tokens)
        return self.ids_from_text(text, expected_len=expected_len)
