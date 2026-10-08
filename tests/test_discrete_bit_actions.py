import unittest

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.discrete_diffusion.action_binning import ActionBinning
from starVLA.model.modules.action_model.LayerwiseDiscreteDiffusion_ActionHeader import (
    LayerwiseDiscreteDiffusionActionHead,
)


def tiny_head(representation):
    config = OmegaConf.create(
        {
            "framework": {
                "qwenvl": {"num_vl_layers": 2, "vl_hidden_dim": 64},
                "action_model": {
                    "action_dim": 2,
                    "future_action_window_size": 2,
                    "num_inference_steps": 2,
                    "num_bins": 256,
                    "representation": representation,
                    "l1_loss_weight": 0.0,
                    "state_dim": 0,
                    "num_target_vision_tokens": 1,
                    "add_pos_embed": False,
                    "diffusion_model_cfg": {
                        "dropout": 0.0,
                        "final_dropout": False,
                        "positional_embeddings": None,
                    },
                },
            }
        }
    )
    return LayerwiseDiscreteDiffusionActionHead(config)


class BitSamplingTest(unittest.TestCase):
    def test_stochastic_sampling_is_seeded_and_returns_joint_confidence(self):
        binning = ActionBinning(256, 2, representation="bit")
        logits = torch.linspace(-2, 2, 2 * 3 * 2 * 8).reshape(2, 3, 2, 8)
        first = binning.sample_indices_from_logits(
            logits, temperature=0.5, deterministic=False, generator=torch.Generator().manual_seed(42)
        )
        second = binning.sample_indices_from_logits(
            logits, temperature=0.5, deterministic=False, generator=torch.Generator().manual_seed(42)
        )
        indices, confidence = first
        torch.testing.assert_close(indices, second[0])
        torch.testing.assert_close(confidence, second[1])
        self.assertEqual(indices.shape, (2, 3, 2))
        self.assertTrue(((indices >= 0) & (indices < 256)).all())
        bits = ((indices.unsqueeze(-1) >> torch.arange(8)) & 1).float()
        expected = (-F.binary_cross_entropy_with_logits(logits / 0.5, bits, reduction="none").sum(-1)).exp()
        torch.testing.assert_close(confidence, expected)

    def test_temperature_controls_bit_confidence_without_changing_hard_decoding(self):
        binning = ActionBinning(256, 1, representation="bit")
        logits = torch.ones(1, 1, 1, 8)
        for temperature in (0.25, 1.0, 2.0):
            with self.subTest(temperature=temperature):
                indices, confidence = binning.sample_indices_from_logits(logits, temperature=temperature)
                self.assertEqual(indices.item(), 255)
                torch.testing.assert_close(confidence, torch.sigmoid(logits / temperature).prod(-1))
        torch.testing.assert_close(binning.decode_logits(logits), binning.decode(torch.tensor([[[255]]])))

    def test_sampled_bit_frequencies_follow_bernoulli_probabilities(self):
        binning = ActionBinning(256, 1, representation="bit")
        probabilities = torch.linspace(0.1, 0.8, 8)
        logits = torch.logit(probabilities).expand(16384, 1, 1, 8)
        indices, _ = binning.sample_indices_from_logits(
            logits, deterministic=False, generator=torch.Generator().manual_seed(7)
        )
        frequencies = (((indices.unsqueeze(-1) >> torch.arange(8)) & 1).float()).mean((0, 1, 2))
        torch.testing.assert_close(frequencies, probabilities, atol=0.02, rtol=0)


class BitActionHeadTest(unittest.TestCase):
    def setUp(self):
        context = torch.random.fork_rng(devices=[])
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        torch.manual_seed(17)

    def test_masked_bit_loss_and_gradients_match_reference(self):
        head = tiny_head("bit")
        logits = torch.linspace(-2, 2, 2 * 3 * 2 * 8).reshape(2, 3, 2, 8).requires_grad_()
        targets = torch.tensor([[[0, 255], [85, 170], [1, 2]], [[15, 240], [3, 12], [63, 192]]])
        mask = torch.tensor(
            [[[True, False], [False, True], [True, False]], [[False, True], [True, True], [False, True]]]
        )
        loss = head.loss(logits, targets, loss_mask=mask)
        # Independent bit labels, least-significant bit first.
        labels = torch.tensor([[int(bit) for bit in f"{value:08b}"[::-1]] for value in targets.flatten().tolist()])
        labels = labels.reshape_as(logits).float()
        per_action = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(-1)
        expected = torch.stack([per_action[b][mask[b]].mean() for b in range(2)]).mean()
        torch.testing.assert_close(loss, expected)
        actual_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        reference_gradient = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual_gradient, reference_gradient)
        self.assertTrue((actual_gradient[~mask] == 0).all())

    def test_bit_head_can_train_and_decode_actions(self):
        head = tiny_head("bit")
        head.l1_loss_weight = 0.1  # Exercise the configured default as well as the isolated BCE tests.
        embeddings = [torch.randn(2, 3, 64) for _ in range(2)]
        actions = torch.linspace(-0.8, 0.8, 12).reshape(2, 3, 2)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        pred, targets, extra = head(embeddings, actions)
        loss = head.loss(pred, targets, **extra)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradient = head.model.proj_out_2.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        before = head.model.proj_out_2.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, head.model.proj_out_2.weight))

        head.eval()
        # Isolate bit sampling from the stochastic remasking bug tracked in #486.
        result = head.predict_action(embeddings, choice_temperature=0, decode_temperature=0.5)
        realtime = head.predict_action_realtime(
            embeddings,
            prev_action_chunk=actions,
            inference_delay=1,
            execution_horizon=1,
            choice_temperature=0,
            decode_temperature=0.5,
        )
        for output in (result, realtime):
            self.assertEqual(output.shape, actions.shape)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(((output >= -1) & (output <= 1)).all())
        torch.testing.assert_close(realtime[:, :2], head.binning.decode(head.binning.encode(actions))[:, :2])

    def test_empty_bit_loss_mask_has_zero_loss_and_gradient(self):
        head = tiny_head("bit")
        logits = torch.randn(2, 3, 2, 8, requires_grad=True)
        targets = torch.zeros(2, 3, 2, dtype=torch.long)
        loss = head.loss(logits, targets, loss_mask=torch.zeros_like(targets, dtype=torch.bool))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue((logits.grad == 0).all())

    def test_bin_loss_remains_cross_entropy(self):
        head = tiny_head("bin")
        logits = torch.randn(2, 3, 2, 256, requires_grad=True)
        targets = torch.arange(12).reshape(2, 3, 2)
        loss = head.loss(logits, targets)
        expected = F.cross_entropy(logits.reshape(-1, 256), targets.flatten())
        torch.testing.assert_close(loss, expected)


if __name__ == "__main__":
    unittest.main()
