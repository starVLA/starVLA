import math
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.VLA_AdapterHeader import (
    MLPResNetBlock_Pro,
    RotaryPositionEmbedding,
    apply_rope,
    get_action_model,
)


def pair_rotation(tensor, inv_freq, offset=0):
    """Independent 2D rotation of adjacent coordinates, without apply_rope."""
    positions = torch.arange(offset, offset + tensor.shape[-2], device=tensor.device, dtype=inv_freq.dtype)
    angles = positions[:, None] * inv_freq
    cos, sin = angles.cos().to(tensor.dtype), angles.sin().to(tensor.dtype)
    even, odd = tensor[..., ::2], tensor[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


def block_reference(block, x, h_a=None, h_t=None, p=None):
    """Reconstruct joint attention using the independent pair-rotation formula."""
    batch, length, channels = x.shape

    def heads(tensor):
        return tensor.reshape(batch, -1, block.num_heads, block.head_dim).transpose(1, 2)

    def rotate(tensor):
        return pair_rotation(tensor, block.rope.inv_freq)

    query = rotate(heads(block.q_proj(x)))
    scores = [query @ rotate(heads(block.k_self(x))).transpose(-2, -1)]
    values = [heads(block.v_self(x))]
    conditions = [tensor[:, None] if tensor.ndim == 2 else tensor for tensor in (h_a, p) if tensor is not None]
    if conditions:
        adapter = torch.cat(conditions, 1)
        scores.append(query @ rotate(heads(block.k_adapter(adapter))).transpose(-2, -1))
        values.append(heads(block.v_adapter(adapter)))
    if h_t is not None:
        task = h_t[:, None] if h_t.ndim == 2 else h_t
        scores.append((query @ rotate(heads(block.k_task(task))).transpose(-2, -1)) * block.gating_factor.tanh())
        values.append(heads(block.v_task(task)))
    probabilities = (torch.cat(scores, -1) / math.sqrt(block.head_dim)).softmax(-1)
    attention = probabilities @ torch.cat(values, 2)
    attention = attention.transpose(1, 2).reshape(batch, length, channels)
    return block.ffn(block.o_proj(attention) + x)


def tiny_action_head(pro=True):
    config = OmegaConf.create(
        {
            "framework": {
                "qwenvl": {"vl_hidden_dim": 64},
                "action_model": {
                    "hidden_dim": 64,
                    "action_dim": 2,
                    "action_query_num": 2,
                    "num_actions_chunk": 3,
                    "use_pro_version": pro,
                },
            }
        }
    )
    return get_action_model(config)


class AdapterRoPETest(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(17)

    def test_tables_repeat_each_frequency_for_an_adjacent_pair(self):
        for width in (2, 4, 8, 64, 256):
            for length in (0, 5):
                for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                    with self.subTest(width=width, length=length, dtype=dtype):
                        rope = RotaryPositionEmbedding(width)
                        cos, sin = rope(length, torch.device("cpu"), dtype)
                        angles = torch.arange(length, dtype=rope.inv_freq.dtype)[:, None] * rope.inv_freq
                        torch.testing.assert_close(cos, angles.cos().repeat_interleave(2, -1).to(dtype))
                        torch.testing.assert_close(sin, angles.sin().repeat_interleave(2, -1).to(dtype))
                        self.assertEqual(cos.shape, (length, width))
                        self.assertEqual(cos.dtype, dtype)
                        self.assertEqual(sin.dtype, dtype)
                        self.assertEqual(cos.device, torch.device("cpu"))

    def test_rotation_matches_pairwise_reference_and_its_gradients(self):
        rope = RotaryPositionEmbedding(8).double()
        query = torch.randn(2, 3, 5, 8, dtype=torch.float64, requires_grad=True)
        key = torch.randn_like(query, requires_grad=True)
        before_query, before_key = query.detach().clone(), key.detach().clone()
        cos, sin = rope(5, query.device, query.dtype)
        actual_query, actual_key = apply_rope(query, key, cos, sin)
        expected_query = pair_rotation(query, rope.inv_freq)
        expected_key = pair_rotation(key, rope.inv_freq)
        torch.testing.assert_close(actual_query, expected_query)
        torch.testing.assert_close(actual_key, expected_key)
        actual_loss = actual_query.sin().sum() + actual_key.cos().sum()
        expected_loss = expected_query.sin().sum() + expected_key.cos().sum()
        actual_gradients = torch.autograd.grad(actual_loss, (query, key))
        expected_gradients = torch.autograd.grad(expected_loss, (query, key))
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(query.detach(), before_query)
        torch.testing.assert_close(key.detach(), before_key)

    def test_norms_and_same_position_dot_products_are_preserved(self):
        for width in (2, 4, 8, 64, 256):
            with self.subTest(width=width):
                rope = RotaryPositionEmbedding(width).double()
                query = torch.randn(2, 3, 16, width, dtype=torch.float64)
                key = torch.randn_like(query)
                cos, sin = rope(16, query.device, query.dtype)
                rotated_query, rotated_key = apply_rope(query, key, cos, sin)
                torch.testing.assert_close(rotated_query.square().sum(-1), query.square().sum(-1))
                torch.testing.assert_close(rotated_key.square().sum(-1), key.square().sum(-1))
                torch.testing.assert_close((rotated_query * rotated_key).sum(-1), (query * key).sum(-1))

    def test_joint_attention_scores_are_invariant_to_a_common_position_offset(self):
        rope = RotaryPositionEmbedding(8).double()
        query = torch.randn(2, 3, 4, 8, dtype=torch.float64)
        key = torch.randn(2, 3, 7, 8, dtype=torch.float64)
        cos, sin = rope(14, query.device, query.dtype)
        query_at_zero, _ = apply_rope(query, query, cos[:4], sin[:4])
        key_at_zero, _ = apply_rope(key, key, cos[:7], sin[:7])
        query_shifted, _ = apply_rope(query, query, cos[7:11], sin[7:11])
        key_shifted, _ = apply_rope(key, key, cos[7:14], sin[7:14])
        torch.testing.assert_close(
            query_at_zero @ key_at_zero.transpose(-2, -1), query_shifted @ key_shifted.transpose(-2, -1)
        )

    def test_position_zero_is_identity_and_gradcheck_passes(self):
        rope = RotaryPositionEmbedding(4).double()
        query = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
        key = torch.randn_like(query, requires_grad=True)
        cos, sin = rope(3, query.device, query.dtype)
        rotated_query, rotated_key = apply_rope(query, key, cos, sin)
        torch.testing.assert_close(rotated_query[..., 0, :], query[..., 0, :])
        torch.testing.assert_close(rotated_key[..., 0, :], key[..., 0, :])
        self.assertTrue(torch.autograd.gradcheck(lambda q, k: apply_rope(q, k, cos, sin), (query, key)))

    def test_real_pro_block_and_gradients_match_reference_for_all_branches(self):
        for branch in ("self", "action", "action_state", "vision", "all"):
            with self.subTest(branch=branch):
                block = MLPResNetBlock_Pro(64).double()
                with torch.no_grad():
                    block.gating_factor.fill_(0.35)
                x = torch.randn(2, 4, 64, dtype=torch.float64, requires_grad=True)
                h_a = (
                    torch.randn(2, 3, 64, dtype=torch.float64, requires_grad=True)
                    if branch in ("action", "action_state", "all")
                    else None
                )
                h_t = (
                    torch.randn(2, 5, 64, dtype=torch.float64, requires_grad=True)
                    if branch in ("vision", "all")
                    else None
                )
                p = (
                    torch.randn(2, 64, dtype=torch.float64, requires_grad=True)
                    if branch in ("action_state", "all")
                    else None
                )
                actual = block(x, h_a=h_a, h_t=h_t, p=p)
                expected = block_reference(block, x, h_a=h_a, h_t=h_t, p=p)
                torch.testing.assert_close(actual, expected)
                inputs = [tensor for tensor in (x, h_a, h_t, p) if tensor is not None]
                parameters = [block.q_proj.weight, block.k_self.weight]
                if h_a is not None:
                    parameters.append(block.k_adapter.weight)
                if h_t is not None:
                    parameters.extend((block.k_task.weight, block.gating_factor))
                gradients = inputs + parameters
                actual_gradients = torch.autograd.grad(actual.square().mean(), gradients)
                expected_gradients = torch.autograd.grad(expected.square().mean(), gradients)
                for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
                    torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_full_action_head_trains_and_predicts(self):
        head = tiny_action_head()
        hidden = torch.randn(2, 26, 6, 64, requires_grad=True)
        state = torch.randn(2, 64, requires_grad=True)
        target = torch.randn(2, 3, 2)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            prediction = head.predict_action(hidden, vision_hidden_len=4, state_projected=state, phase="Training")
            self.assertEqual(prediction.shape, target.shape)
            loss = (prediction - target).abs().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for parameter in (
                head.action_chunk_embeddings,
                head.model.mlp_resnet_blocks[0].q_proj.weight,
                head.model.fc2.weight,
            ):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
            optimizer.step()
        head.eval()
        with torch.no_grad():
            prediction = head.predict_action(hidden, vision_hidden_len=4, state_projected=state)
        self.assertEqual(prediction.shape, target.shape)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertFalse(prediction.requires_grad)
        checkpoint = head.state_dict()
        self.assertFalse(any("inv_freq" in name for name in checkpoint))
        result = head.load_state_dict(checkpoint, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_non_pro_action_head_does_not_use_rope(self):
        head = tiny_action_head(pro=False).eval()
        with (
            patch(
                "starVLA.model.modules.action_model.VLA_AdapterHeader.apply_rope",
                side_effect=AssertionError("Non-Pro path must not use RoPE"),
            ),
            torch.no_grad(),
        ):
            prediction = head.predict_action(torch.randn(2, 26, 6, 64), vision_hidden_len=4)
        self.assertEqual(prediction.shape, (2, 3, 2))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
