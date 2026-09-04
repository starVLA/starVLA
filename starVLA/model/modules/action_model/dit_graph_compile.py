"""Bucketed CUDA-graph execution for DiT-style action heads.

The action-head DiT is the shape-stable region of a VLA training step: its
query side (state/future/action tokens) is a per-run constant, and only the
encoder side (VLM hidden states, padded to the batch's longest sequence)
varies. That variation is small and discrete, so we pad the encoder length up
to a bucket and run the DiT under ``torch.compile(mode="reduce-overhead")``,
which replays one CUDA graph per observed shape. Padded encoder positions are
masked out of cross-attention, so bucketing is numerically a no-op (masked
keys receive exactly zero attention weight).

This mirrors the pattern production inference engines use for dynamic shapes
(vLLM ``cudagraph_capture_sizes`` / multimodal-encoder token budgets,
TensorRT-LLM ``cuda_graph_config.enable_padding``): discretize the varying
dimension, pad up to the nearest captured size, and fall back to eager
execution for anything outside the captured set.

Environment switches:
  STARVLA_DIT_GRAPH_STRICT=1  raise on any eager fallback (CI / debugging).
  STARVLA_CHECK_DIT_GRAPH=1   on the first hit of each bucket, compare the
                              compiled eval-mode output against eager and log
                              the max deviation.
"""

import math
import os

import torch
import torch.nn.functional as F


def _rank0() -> bool:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


class BucketedCompiledDiT:
    """Callable wrapper around a DiT module.

    Pads the encoder-length dimension of ``encoder_hidden_states`` (a list of
    per-layer tensors) and ``encoder_attention_mask`` up to a bucket, then
    dispatches to a ``torch.compile``d copy of the module. Any input the
    bucketing cannot serve (sequence longer than ``max_capture_len``, compile
    backend failure, gradient checkpointing enabled, pre-existing param grads
    — see below) falls back to the eager module.

    Gradient-accumulation restriction: reduce-overhead CUDA graphs write
    parameter gradients into static buffers, so letting autograd ACCUMULATE
    into ``param.grad`` across microbatches silently yields N x (the last
    microbatch's grad). Trainers must clear grads every microbatch backward
    (DeepSpeed ZeRO does; ``zero_grad(set_to_none=True)`` does). The wrapper
    detects a pre-existing ``param.grad`` before a compiled training call and
    serves that call eagerly, which keeps accumulation correct at eager speed.

    Output-lifetime contract: the returned tensor lives in a static buffer
    that the NEXT compiled call overwrites — consume (or clone) it before the
    next step, which the LayerwiseFM training forward does.

    Not an nn.Module on purpose: it owns no parameters and must stay invisible
    to ``state_dict`` / optimizer setup.
    """

    def __init__(
        self,
        dit: torch.nn.Module,
        bucket_multiple: int = 64,
        max_capture_len: int = 1024,
        capture_lens=None,
        log_every: int = 500,
        freeze_after: int = 4,
    ):
        if bucket_multiple <= 0:
            raise ValueError(f"bucket_multiple must be positive, got {bucket_multiple}")
        self.dit = dit
        self.bucket_multiple = int(bucket_multiple)
        self.max_capture_len = int(max_capture_len)
        self.capture_lens = sorted(int(c) for c in capture_lens) if capture_lens else None
        self.log_every = int(log_every)
        self.freeze_after = int(freeze_after)
        self._frozen = False
        self._params = list(dit.parameters())
        if self.max_capture_len <= 0:
            raise ValueError(f"max_capture_len must be positive, got {self.max_capture_len}")
        if self.capture_lens is None and self.bucket_multiple > self.max_capture_len:
            raise ValueError(
                f"bucket_multiple={self.bucket_multiple} exceeds max_capture_len="
                f"{self.max_capture_len}: no input could ever be served by a graph"
            )

        self._compiled = None
        self._disabled_reason = None
        self._warned = set()
        self._checked_buckets = set()
        self._calls = 0
        self._hits = 0
        self._strict = _env_flag("STARVLA_DIT_GRAPH_STRICT")
        self._check = _env_flag("STARVLA_CHECK_DIT_GRAPH")

    # ------------------------------------------------------------------
    def _bucket(self, seq_len: int):
        """Smallest capture length >= seq_len, or None if out of range."""
        if self.capture_lens is not None:
            for cap in self.capture_lens:
                if cap >= seq_len:
                    return cap
            return None
        bucket = int(math.ceil(seq_len / self.bucket_multiple) * self.bucket_multiple)
        return bucket if bucket <= self.max_capture_len else None

    def _fallback(self, reason: str, detail: str = "", **kwargs):
        # `reason` is the warn-once dedup key; volatile specifics go in `detail`.
        if self._strict:
            raise RuntimeError(
                f"STARVLA_DIT_GRAPH_STRICT=1 and DiT graph fell back to eager: {reason}{detail}"
            )
        if reason not in self._warned and _rank0():
            self._warned.add(reason)
            print(f"[dit-graph] falling back to eager (reported once): {reason}{detail}", flush=True)
        return self.dit(**kwargs)

    def _get_compiled(self):
        if self._compiled is None:
            import torch._dynamo

            # A handful of buckets (and train/eval variants) each compile their
            # own specialization; the dynamo default cache limit (8) would
            # silently stop compiling past that, so give the full capture set room.
            limit = 8 + (
                len(self.capture_lens) if self.capture_lens else self.max_capture_len // self.bucket_multiple
            )
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, limit)
            self._compiled = torch.compile(self.dit, mode="reduce-overhead")
        return self._compiled

    @staticmethod
    def _pad_inputs(encoder_hidden_states, encoder_attention_mask, bucket: int):
        seq_len = encoder_hidden_states[0].shape[1]
        pad = bucket - seq_len
        if pad == 0 and encoder_attention_mask is not None:
            return encoder_hidden_states, encoder_attention_mask
        padded_states = [F.pad(h, (0, 0, 0, pad)) for h in encoder_hidden_states]
        if encoder_attention_mask is None:
            # The eager path treats a missing mask as "attend to everything";
            # once we add padded keys a real mask is required to keep the
            # computation equivalent.
            base = encoder_hidden_states[0]
            encoder_attention_mask = torch.ones(
                base.shape[0], seq_len, dtype=torch.bool, device=base.device
            )
        padded_mask = F.pad(encoder_attention_mask.to(torch.bool), (0, pad), value=False)
        return padded_states, padded_mask

    def _check_bucket_once(self, bucket, call_kwargs, eager_kwargs):
        """First hit of a bucket: compare compiled vs eager in eval mode."""
        self._checked_buckets.add(bucket)
        was_training = self.dit.training
        self.dit.eval()
        try:
            with torch.no_grad():
                ref = self.dit(**eager_kwargs)
                # Run the compiled fn through warmup -> record -> replay so the
                # comparison covers an actual CUDA-graph replay, not only the
                # first (eagerly executed) warmup call.
                for _ in range(3):
                    out = self._get_compiled()(**call_kwargs)
            diff = (out - ref).abs().max().item()
            scale = ref.abs().max().item()
            rel = diff / scale if scale > 0 else diff
            if _rank0():
                level = "WARNING" if rel > 5e-2 else "check ok"
                print(
                    f"[dit-graph] {level}: bucket={bucket} compiled-vs-eager "
                    f"max_abs_diff={diff:.3e} max_rel={rel:.3e}",
                    flush=True,
                )
        finally:
            self.dit.train(was_training)

    # ------------------------------------------------------------------
    def __call__(self, hidden_states, encoder_hidden_states, timestep, encoder_attention_mask=None, **kwargs):
        eager_kwargs = dict(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )
        self._calls += 1
        if self._disabled_reason is not None:
            return self.dit(**eager_kwargs)
        if not isinstance(encoder_hidden_states, (list, tuple)):
            return self._fallback("encoder_hidden_states is not a per-layer list", **eager_kwargs)
        if getattr(self.dit, "gradient_checkpointing", False):
            return self._fallback("gradient checkpointing is enabled on the DiT", **eager_kwargs)
        if self.dit.training and torch.is_grad_enabled() and any(p.grad is not None for p in self._params):
            # Autograd would ACCUMULATE into param.grad, but graph-produced
            # grads live in static buffers that the next replay overwrites —
            # accumulation across microbatches would silently keep only the
            # last one. Serve this call eagerly; correct at eager speed.
            return self._fallback(
                "pre-existing param.grad detected (gradient accumulation without "
                "per-microbatch grad clearing); running eagerly to keep grads correct",
                **eager_kwargs,
            )

        seq_len = encoder_hidden_states[0].shape[1]
        bucket = self._bucket(seq_len)
        if bucket is None:
            return self._fallback(
                "encoder length exceeds max_capture_len",
                detail=f" ({seq_len} > {self.max_capture_len})",
                **eager_kwargs,
            )

        padded_states, padded_mask = self._pad_inputs(encoder_hidden_states, encoder_attention_mask, bucket)
        call_kwargs = dict(
            hidden_states=hidden_states,
            encoder_hidden_states=list(padded_states),
            timestep=timestep,
            encoder_attention_mask=padded_mask,
            **kwargs,
        )
        if self._check and bucket not in self._checked_buckets:
            try:
                self._check_bucket_once(bucket, call_kwargs, eager_kwargs)
            except Exception as exc:
                if self._strict:
                    raise
                # A failing debug check must not disable graphs for the run.
                print(
                    f"[dit-graph] rank {os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))}: "
                    f"STARVLA_CHECK_DIT_GRAPH probe failed for bucket {bucket}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                self._checked_buckets.add(bucket)
        try:
            out = self._get_compiled()(**call_kwargs)
        except Exception as exc:  # compile/capture backend failure -> permanent eager
            if self._strict:
                raise
            self._disabled_reason = f"{type(exc).__name__}: {exc}"
            # Ungated on purpose: the failure may be rank-local, and a silent
            # permanent eager rank would only show up as a throughput mystery.
            print(
                f"[dit-graph] rank {os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))}: "
                f"compile failed, disabling graphs for this run: {self._disabled_reason}",
                flush=True,
            )
            return self.dit(**eager_kwargs)

        self._hits += 1
        if not self._frozen and self.freeze_after > 0 and self._hits >= self.freeze_after:
            # Expected specializations are compiled within the first few calls.
            # Anything asking to recompile later (a ~20s stall each time) is
            # pathological: run those calls eagerly instead. Process-global by
            # necessity (torch.compiler stance API), which is fine here — this
            # is the only compiled region in a stock training run.
            self._frozen = True
            try:
                torch.compiler.set_stance("eager_on_recompile")
                if _rank0():
                    print(
                        f"[dit-graph] warmup done after {self._hits} compiled calls; "
                        "future recompiles will run eagerly (eager_on_recompile)",
                        flush=True,
                    )
            except Exception:
                pass  # older torch without set_stance: keep default behavior
        if self.log_every > 0 and self._calls % self.log_every == 0 and _rank0():
            print(
                f"[dit-graph] hit rate {self._hits}/{self._calls} "
                f"({100.0 * self._hits / self._calls:.1f}%)",
                flush=True,
            )
        return out


def maybe_wrap_dit(dit: torch.nn.Module, action_config):
    """Build a BucketedCompiledDiT from ``framework.action_model`` config keys.

    Returns None unless ``compile: true`` is set (callers keep using the eager
    module directly; returning the module itself would let nn.Module attribute
    assignment register it twice and duplicate state_dict keys).
    """
    if not bool(action_config.get("compile", False)):
        return None
    return BucketedCompiledDiT(
        dit,
        bucket_multiple=int(action_config.get("compile_bucket_multiple", 64)),
        max_capture_len=int(action_config.get("compile_max_capture_len", 1024)),
        capture_lens=action_config.get("compile_capture_lens", None),
        freeze_after=int(action_config.get("compile_freeze_after", 4)),
    )
