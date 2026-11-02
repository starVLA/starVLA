import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI


class TinyVLM(nn.Module):
    """Checkpoint-free backbone fixture; framework and action head are real."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(3, 64)
        self.model.config = SimpleNamespace(num_hidden_layers=2, hidden_size=64)
        self.last_context = None
        self.last_hidden_states = None

    def build_qwenvl_inputs(self, images, instructions):
        pixels = self.model.weight.new_ones(len(images), 5, 3)
        return {"pixels": pixels}

    def forward(self, pixels, **kwargs):
        self.last_context = (torch.is_inference_mode_enabled(), torch.is_grad_enabled())
        hidden = self.model(pixels)
        self.last_hidden_states = (hidden, hidden.tanh())
        return SimpleNamespace(hidden_states=self.last_hidden_states)


def tiny_qwenpi(state_dim=2):
    config = OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "action_dim": 2,
                    "state_dim": state_dim,
                    "action_horizon": 4,
                    "num_inference_timesteps": 4,
                    "num_target_vision_tokens": 1,
                    "diffusion_model_cfg": {"dropout": 0.0, "final_dropout": False},
                }
            },
            "datasets": {"vla_data": {}},
        }
    )
    with patch("starVLA.model.framework.VLM4A.QwenPI.get_vlm_model", return_value=TinyVLM()):
        return Qwen_PI(config).eval()


def observations(with_state=True):
    examples = [{"image": np.zeros((8, 8, 3), dtype=np.uint8), "lang": "move"} for _ in range(2)]
    if with_state:
        for example in examples:
            example["state"] = [[0.1, -0.2]]
    return examples


class IdentityActionEncoder(nn.Module):
    def forward(self, actions, timesteps):
        return actions


class AffineVelocity(nn.Module):
    """Nonidentity, nonsymmetric Jacobian makes the VJP direction observable."""

    def __init__(self):
        super().__init__()
        self.register_buffer("matrix", torch.tensor([[0.15, -0.10], [0.25, 0.20]]))
        self.register_buffer("bias", torch.tensor([0.1, -0.2]))

    def forward(self, hidden_states, **kwargs):
        return hidden_states @ self.matrix.T + self.bias


class QwenPIRealtimeTest(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(17)
        warning_context = warnings.catch_warnings()
        warning_context.__enter__()
        self.addCleanup(warning_context.__exit__, None, None, None)
        warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
        self.previous = np.linspace(-0.5, 0.5, 16, dtype=np.float32).reshape(2, 4, 2)

    def test_public_pigdm_supports_grad_no_grad_and_inference_contexts(self):
        for state_dim in (0, 2):
            for context in (torch.enable_grad, torch.no_grad, torch.inference_mode):
                with self.subTest(state_dim=state_dim, context=context.__name__):
                    model = tiny_qwenpi(state_dim)
                    parameters = list(model.parameters())
                    requires_grad = [parameter.requires_grad for parameter in parameters]
                    # Inference must neither accumulate nor clear existing parameter gradients.
                    model.qwen_vl_interface.model.weight.grad = torch.full_like(
                        model.qwen_vl_interface.model.weight, 0.25
                    )
                    model.action_model.action_decoder.layer1.weight.grad = torch.full_like(
                        model.action_model.action_decoder.layer1.weight, 0.5
                    )
                    gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
                    previous_copy = self.previous.copy()
                    with context():
                        outer_context = (torch.is_inference_mode_enabled(), torch.is_grad_enabled())
                        with patch("torch.autograd.grad", wraps=torch.autograd.grad) as vjp:
                            result = model.predict_action_realtime(
                                observations(bool(state_dim)), prev_action_chunk_normalized=self.previous
                            )
                        self.assertEqual((torch.is_inference_mode_enabled(), torch.is_grad_enabled()), outer_context)
                    self.assertEqual(vjp.call_count, 4)
                    for call in vjp.call_args_list:
                        self.assertTrue(call.args[0].requires_grad)
                        self.assertTrue(call.args[1].requires_grad)
                        self.assertFalse(call.args[1].is_inference())
                    self.assertEqual(model.qwen_vl_interface.last_context, (False, False))
                    for hidden in model.qwen_vl_interface.last_hidden_states:
                        self.assertFalse(hidden.requires_grad)
                        self.assertFalse(hidden.is_inference())
                    actions = result["normalized_actions"]
                    self.assertIsInstance(actions, np.ndarray)
                    self.assertEqual(actions.shape, (2, 4, 2))
                    self.assertEqual(actions.dtype, np.float32)
                    self.assertTrue(np.isfinite(actions).all())
                    np.testing.assert_array_equal(self.previous, previous_copy)
                    self.assertEqual([p.requires_grad for p in parameters], requires_grad)
                    for parameter, gradient in zip(parameters, gradients, strict=True):
                        if gradient is None:
                            self.assertIsNone(parameter.grad)
                        else:
                            torch.testing.assert_close(parameter.grad, gradient)

    def test_zero_guidance_matches_ordinary_sampling(self):
        model = tiny_qwenpi()
        examples = observations()
        torch.manual_seed(23)
        expected = model.predict_action(examples)["normalized_actions"]
        torch.manual_seed(23)
        result = model.predict_action_realtime(examples, self.previous, max_guidance_weight=0.0)["normalized_actions"]
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_outer_context_is_restored_when_sampling_raises(self):
        model = tiny_qwenpi()
        for context in (torch.enable_grad, torch.no_grad, torch.inference_mode):
            with self.subTest(context=context.__name__), context():
                outer_context = (torch.is_inference_mode_enabled(), torch.is_grad_enabled())
                with patch.object(
                    model.action_model, "predict_action_realtime", side_effect=ValueError("sampler failed")
                ):
                    with self.assertRaisesRegex(ValueError, "sampler failed"):
                        model.predict_action_realtime(observations(), self.previous)
                self.assertEqual((torch.is_inference_mode_enabled(), torch.is_grad_enabled()), outer_context)

    def test_fallback_matches_ordinary_sampling(self):
        model = tiny_qwenpi()
        for previous, delay in ((None, 1), (self.previous, 0), (self.previous, -1)):
            with self.subTest(previous_missing=previous is None, delay=delay):
                torch.manual_seed(23)
                expected = model.predict_action(observations())["normalized_actions"]
                torch.manual_seed(23)
                result = model.predict_action_realtime(observations(), previous, inference_delay=delay)[
                    "normalized_actions"
                ]
                np.testing.assert_array_equal(result, expected)

    def test_simulated_delay_preserves_the_prefix(self):
        model = tiny_qwenpi()
        with torch.inference_mode():
            result = model.predict_action_realtime(
                observations(), self.previous, inference_delay=2, mode="simulated_delay"
            )["normalized_actions"]
        np.testing.assert_array_equal(result[:, :2], self.previous[:, :2])
        self.assertTrue(np.isfinite(result).all())

    def test_training_gradients_still_work_after_realtime_inference(self):
        model = tiny_qwenpi()
        examples = observations()
        model.predict_action_realtime(examples, self.previous)
        model.train()
        for example in examples:
            example["action"] = np.zeros((4, 2), dtype=np.float32)
        loss = model(examples)["action_loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (
            model.qwen_vl_interface.model.weight,
            model.action_model.action_encoder.layer1.weight,
            model.action_model.state_encoder.layer1.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_pigdm_matches_an_analytical_affine_velocity_reference(self):
        head = tiny_qwenpi(state_dim=0).action_model
        head.config.add_pos_embed = False
        head.action_encoder = IdentityActionEncoder()
        head.action_decoder = nn.Identity()
        head.future_tokens = nn.Embedding(0, 2)
        head.model = AffineVelocity()
        matrix = head.model.matrix
        bias = head.model.bias
        previous = torch.from_numpy(self.previous)
        torch.manual_seed(31)
        expected = torch.randn(2, 4, 2)
        unguided = expected.clone()
        identity = torch.eye(2)
        weights = torch.tensor([1.0, 0.0, 0.0, 0.0])[None, :, None]
        dt = 1 / 4
        for step in range(4):
            time = step / 4
            velocity = expected @ matrix.T + bias
            endpoint = expected + (1 - time) * velocity
            residual = (previous - endpoint) * weights
            # For row vectors, J^T @ residual is residual @ J, not residual @ J^T.
            correction = residual @ (identity + (1 - time) * matrix)
            guidance = min(
                (1 - time) / (time + 1e-8) * (time**2 + (1 - time) ** 2) / ((1 - time) ** 2 + 1e-8),
                0.7,
            )
            expected = expected + dt * (velocity + guidance * correction)
            unguided = unguided + dt * (unguided @ matrix.T + bias)
        torch.manual_seed(31)
        with torch.no_grad():
            actual = head.predict_action_realtime(
                [torch.zeros(2, 5, 64)],
                prev_action_chunk=previous,
                inference_delay=1,
                suffix_length=1,
                prefix_attention_schedule="zeros",
                max_guidance_weight=0.7,
            )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[:, 1:], unguided[:, 1:])
        self.assertLess((actual[:, :1] - previous[:, :1]).norm(), (unguided[:, :1] - previous[:, :1]).norm())
        self.assertFalse(actual.requires_grad)


if __name__ == "__main__":
    unittest.main()
