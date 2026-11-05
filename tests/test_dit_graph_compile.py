"""Tests for the bucketed CUDA-graph DiT wrapper (dit_graph_compile).

Layers of validation:
  1. Bucket selection / fallback logic (pure CPU, no compile).
  2. Encoder padding is a numerical no-op under the attention mask (eager).
  3. Compiled (reduce-overhead) output matches eager in eval mode (GPU only).
"""

import pytest
import torch

from starVLA.model.modules.action_model.dit_graph_compile import (
    BucketedCompiledDiT,
    maybe_wrap_dit,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

HEADS, HEAD_DIM, LAYERS = 2, 8, 4
INNER = HEADS * HEAD_DIM
SEQ_Q, SEQ_ENC, BATCH = 6, 181, 3


def make_dit(dropout=0.0):
    torch.manual_seed(0)
    return DiT(
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        output_dim=7,
        num_layers=LAYERS,
        dropout=dropout,
        final_dropout=False,
        interleave_self_attention=True,
        norm_type="ada_norm",
        positional_embeddings=None,
        cross_attention_dim=INNER,
    ).eval()


def make_inputs(device="cpu", seq_enc=SEQ_ENC, mask=True):
    torch.manual_seed(1)
    hidden = torch.randn(BATCH, SEQ_Q, INNER, device=device)
    encoders = [torch.randn(BATCH, seq_enc, INNER, device=device) for _ in range(LAYERS)]
    timestep = torch.randint(0, 1000, (BATCH,), device=device)
    enc_mask = torch.ones(BATCH, seq_enc, dtype=torch.bool, device=device) if mask else None
    # Make some positions padding-like so masking is actually exercised.
    if enc_mask is not None:
        enc_mask[:, -5:] = False
    return dict(
        hidden_states=hidden,
        encoder_hidden_states=encoders,
        timestep=timestep,
        encoder_attention_mask=enc_mask,
        return_pre_output=True,
    )


# ---------------------------------------------------------------- bucket logic
def test_bucket_multiple():
    w = BucketedCompiledDiT(make_dit(), bucket_multiple=64, max_capture_len=1024)
    assert w._bucket(1) == 64
    assert w._bucket(64) == 64
    assert w._bucket(181) == 192
    assert w._bucket(1024) == 1024
    assert w._bucket(1025) is None


def test_bucket_capture_lens_override():
    w = BucketedCompiledDiT(make_dit(), capture_lens=[192, 384])
    assert w._bucket(150) == 192
    assert w._bucket(200) == 384
    assert w._bucket(400) is None


def test_maybe_wrap_disabled_returns_none():
    assert maybe_wrap_dit(make_dit(), {"compile": False}) is None
    assert isinstance(maybe_wrap_dit(make_dit(), {"compile": True}), BucketedCompiledDiT)


# ---------------------------------------------------------------- fallback
def test_eager_fallback_when_too_long():
    dit = make_dit()
    w = BucketedCompiledDiT(dit, bucket_multiple=64, max_capture_len=128)
    inputs = make_inputs(seq_enc=200)
    with torch.no_grad():
        out = w(**inputs)
        ref = dit(**inputs)
    assert torch.equal(out, ref)
    assert w._hits == 0


def test_strict_mode_raises(monkeypatch):
    monkeypatch.setenv("STARVLA_DIT_GRAPH_STRICT", "1")
    w = BucketedCompiledDiT(make_dit(), bucket_multiple=64, max_capture_len=128)
    with pytest.raises(RuntimeError, match="fell back to eager"):
        w(**make_inputs(seq_enc=200))


# ---------------------------------------------------------------- pad no-op
def test_pad_synthesizes_mask_when_missing():
    inputs = make_inputs(mask=False)
    states, mask = BucketedCompiledDiT._pad_inputs(
        inputs["encoder_hidden_states"], None, 192
    )
    assert states[0].shape[1] == 192 and mask.shape == (BATCH, 192)
    assert mask[:, :SEQ_ENC].all() and not mask[:, SEQ_ENC:].any()


def test_pad_is_noop_under_mask():
    """Padded keys are masked to exactly zero attention weight, so padding the
    encoder must not change the DiT output (up to reduction-order noise in the
    attention kernel, which is zero for the math backend used on CPU)."""
    dit = make_dit()
    inputs = make_inputs()
    padded_states, padded_mask = BucketedCompiledDiT._pad_inputs(
        inputs["encoder_hidden_states"], inputs["encoder_attention_mask"], 192
    )
    with torch.no_grad():
        ref = dit(**inputs)
        out = dit(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=padded_states,
            timestep=inputs["timestep"],
            encoder_attention_mask=padded_mask,
            return_pre_output=True,
        )
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------- compiled vs eager
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for reduce-overhead")
def test_compiled_matches_eager_eval():
    dit = make_dit().cuda()
    w = BucketedCompiledDiT(dit, bucket_multiple=64, max_capture_len=1024)
    inputs = make_inputs(device="cuda")
    with torch.no_grad():
        ref = dit(**inputs)
        # CUDA-graph outputs live in static buffers that the next replay
        # overwrites; clone before invoking again (training consumes the
        # output before the next step, so this only matters in tests).
        out = w(**inputs).clone()
        out2 = w(**inputs).clone()  # second call exercises graph replay
    assert w._disabled_reason is None
    assert w._hits == 2  # both calls took the compiled path, not a fallback
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(out2, ref, rtol=1e-3, atol=1e-4)


def test_grad_accum_triggers_eager_fallback():
    """A pre-existing param.grad (vanilla gradient accumulation) must route the
    call to eager: graph-produced grads live in static buffers, so accumulating
    across microbatches would silently keep only the last microbatch."""
    dit = make_dit().train()
    w = BucketedCompiledDiT(dit, bucket_multiple=64, max_capture_len=1024)
    inputs = make_inputs()
    # microbatch 1 populates param.grad via the eager module
    dit(**inputs).float().pow(2).mean().backward()
    assert any(p.grad is not None for p in dit.parameters())
    out = w(**inputs)  # microbatch 2 must fall back, not compile
    assert w._hits == 0 and out is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for reduce-overhead")
def test_compiled_grads_match_eager_train():
    """Train-mode fwd+bwd through the wrapper (the wired-in path) produces the
    same parameter grads as eager. dropout=0 for determinism; second backward
    exercises graph replay."""
    dit = make_dit().cuda().train()
    inputs = make_inputs(device="cuda")

    def grads(fn):
        dit.zero_grad(set_to_none=True)
        fn(**inputs).float().pow(2).mean().backward()
        out = {n: p.grad.detach().clone() for n, p in dit.named_parameters() if p.grad is not None}
        dit.zero_grad(set_to_none=True)
        return out

    ref = grads(dit)
    w = BucketedCompiledDiT(dit, bucket_multiple=64, max_capture_len=1024)
    grads(w)          # warmup/record
    got = grads(w)    # replay
    assert w._hits == 2 and w._disabled_reason is None
    assert set(got) == set(ref)
    for name in ref:
        torch.testing.assert_close(got[name], ref[name], rtol=1e-3, atol=1e-4, msg=name)
