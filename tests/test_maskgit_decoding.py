import unittest

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.discrete_diffusion.mask_git_schedule import (
    mask_by_deterministic_lowest,
    mask_by_random_topk,
)
from starVLA.model.modules.action_model.LayerwiseDiscreteDiffusion_ActionHeader import (
    LayerwiseDiscreteDiffusionActionHead,
)


class MaskGITRankingTest(unittest.TestCase):
    def test_low_temperature_masks_lowest_confidence(self):
        probabilities = torch.tensor([[0.05, 0.3, 0.7, 0.95], [0.9, 0.1, 0.6, 0.2]])
        lengths = torch.tensor([2, 1])
        selected = mask_by_random_topk(
            probabilities, lengths, temperature=1e-6, generator=torch.Generator().manual_seed(0)
        )
        expected = torch.tensor([[True, True, False, False], [False, True, False, False]])
        torch.testing.assert_close(selected, expected)
        torch.testing.assert_close(selected, mask_by_deterministic_lowest(probabilities, lengths))

    def test_committed_tokens_are_never_remasked(self):
        probabilities = torch.tensor([[float("inf"), 0.05, 0.8, float("inf"), 0.4]])
        for temperature in (0.1, 1.0, 10.0):
            with self.subTest(temperature=temperature):
                selected = mask_by_random_topk(
                    probabilities,
                    torch.tensor([2]),
                    temperature=temperature,
                    generator=torch.Generator().manual_seed(7),
                )
                self.assertFalse(selected[0, 0])
                self.assertFalse(selected[0, 3])
                self.assertEqual(selected.sum().item(), 2)

    def test_seeded_sampling_and_mask_counts(self):
        probabilities = torch.tensor([[0.0, 0.2, 0.2, 1.0], [0.25, 0.25, 0.25, 0.25]])
        lengths = torch.tensor([1, 2])
        first = mask_by_random_topk(probabilities, lengths, generator=torch.Generator().manual_seed(11))
        second = mask_by_random_topk(probabilities, lengths, generator=torch.Generator().manual_seed(11))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first.sum(dim=1), lengths)

    def test_zero_and_full_mask_budgets(self):
        selected = mask_by_random_topk(
            torch.ones(2, 4), torch.tensor([0, 4]), generator=torch.Generator().manual_seed(11)
        )
        self.assertFalse(selected[0].any())
        self.assertTrue(selected[1].all())


class MaskGITActionHeadTest(unittest.TestCase):
    def setUp(self):
        self.rng_context = torch.random.fork_rng(devices=[])
        self.rng_context.__enter__()
        self.addCleanup(self.rng_context.__exit__, None, None, None)
        torch.manual_seed(17)
        config = OmegaConf.create(
            {
                "framework": {
                    "qwenvl": {"num_vl_layers": 2, "vl_hidden_dim": 64},
                    "action_model": {
                        "action_dim": 2,
                        "future_action_window_size": 3,
                        "num_inference_steps": 4,
                        "num_bins": 8,
                        "decode_schedule": "linear",
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
        self.head = LayerwiseDiscreteDiffusionActionHead(config).eval()
        self.encodings = [torch.randn(2, 3, 64) for _ in range(2)]
        self.trace = []
        hook = self.head.token_embedding.register_forward_pre_hook(
            lambda module, args: self.trace.append(args[0].detach().clone())
        )
        self.addCleanup(hook.remove)

    def assert_decode_trace(self, expected_counts):
        self.assertEqual(len(self.trace), len(expected_counts))
        for tokens, count in zip(self.trace, expected_counts, strict=True):
            torch.testing.assert_close(
                (tokens == self.head.mask_token_id).sum(dim=1), torch.full((2,), count, dtype=torch.long)
            )
        for previous, following in zip(self.trace, self.trace[1:], strict=False):
            committed = previous != self.head.mask_token_id
            torch.testing.assert_close(previous[committed], following[committed])

    def test_iterative_decode_never_changes_committed_tokens(self):
        actions = self.head.predict_action(self.encodings, choice_temperature=0.1, decode_temperature=0)
        self.assert_decode_trace([8, 6, 4, 2])
        self.assertEqual(actions.shape, (2, 4, 2))
        self.assertTrue(torch.isfinite(actions).all())

    def test_realtime_decode_refines_suffix_and_keeps_prefix(self):
        previous = torch.linspace(-0.8, 0.8, 16).reshape(2, 4, 2)
        actions = self.head.predict_action_realtime(
            self.encodings,
            prev_action_chunk=previous,
            inference_delay=1,
            execution_horizon=2,
            fixed_steps=True,
            early_stop=True,
            choice_temperature=0.1,
            decode_temperature=0,
        )
        self.assert_decode_trace([4, 3, 2, 1])
        quantized_prefix = self.head.binning.decode(self.head.binning.encode(previous))[:, :2]
        torch.testing.assert_close(actions[:, :2], quantized_prefix)
        self.assertTrue(torch.isfinite(actions).all())


if __name__ == "__main__":
    unittest.main()
