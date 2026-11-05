import unittest
from unittest import mock

import torch

from starVLA.model.modules.action_model.OpenPI_ActionHead import GemmaDims, OpenPIGemma
from starVLA.model.modules.vlm.openpi_transformers.gemma import modeling_gemma


class OpenPIGemmaTransformersCompatTest(unittest.TestCase):
    def test_cached_suffix_forward(self):
        config = GemmaDims(width=16, depth=1, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8)
        prefix_embeds = torch.randn(1, 2, config.width)
        suffix_embeds = torch.randn(1, 3, config.width)
        prefix_model = OpenPIGemma(config, use_adarms=False).eval()
        prefix_model.model.config._attn_implementation = "eager"

        with torch.no_grad():
            prefix_output = prefix_model.model.forward(
                inputs_embeds=prefix_embeds,
                attention_mask=torch.zeros(1, 1, 2, 2),
                position_ids=torch.arange(2).unsqueeze(0),
                use_cache=True,
                adarms_cond=None,
            )
        prefix_cache = prefix_output.past_key_values

        for use_adarms in (False, True):
            with self.subTest(use_adarms=use_adarms):
                action_expert = OpenPIGemma(config, use_adarms=use_adarms).eval()
                action_expert.model.config._attn_implementation = "eager"

                with torch.no_grad():
                    suffix_output = action_expert.model.forward(
                        inputs_embeds=suffix_embeds,
                        attention_mask=torch.zeros(1, 1, 3, 5),
                        position_ids=torch.arange(2, 5).unsqueeze(0),
                        past_key_values=prefix_cache,
                        use_cache=False,
                        adarms_cond=torch.zeros(1, config.width) if use_adarms else None,
                    )

                self.assertEqual(prefix_cache.get_seq_length(), prefix_embeds.shape[1])
                self.assertEqual(suffix_output.last_hidden_state.shape, suffix_embeds.shape)
                self.assertTrue(torch.isfinite(suffix_output.last_hidden_state).all())

    def test_tiny_action_expert_forward(self):
        config = GemmaDims(
            width=16,
            depth=1,
            mlp_dim=32,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
        )
        inputs_embeds = torch.randn(1, 3, config.width)
        attention_mask = torch.zeros(1, 1, 3, 3)
        position_ids = torch.arange(3).unsqueeze(0)

        for use_adarms in (False, True):
            with self.subTest(use_adarms=use_adarms):
                action_expert = OpenPIGemma(config, use_adarms=use_adarms).eval()
                action_expert.model.config._attn_implementation = "eager"
                adarms_cond = torch.zeros(1, config.width) if use_adarms else None
                forward_kwargs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "use_cache": False,
                    "adarms_cond": adarms_cond,
                }

                with torch.no_grad():
                    output = action_expert.model.forward(
                        **forward_kwargs,
                        output_hidden_states=True,
                        output_attentions=True,
                    )
                    tuple_output = action_expert.model.forward(**forward_kwargs, return_dict=False)

                self.assertEqual(output.last_hidden_state.shape, inputs_embeds.shape)
                self.assertTrue(torch.isfinite(output.last_hidden_state).all())
                self.assertEqual(len(output.hidden_states), config.depth + 1)
                self.assertEqual(len(output.attentions), config.depth)
                self.assertIsInstance(tuple_output, tuple)

    def test_forward_without_cache_position_mask_parameter(self):
        config = GemmaDims(width=16, depth=1, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8)
        action_expert = OpenPIGemma(config, use_adarms=False).eval()
        action_expert.model.config._attn_implementation = "eager"
        inputs_embeds = torch.randn(1, 3, config.width)
        attention_mask = torch.zeros(1, 1, 3, 3)
        position_ids = torch.arange(3).unsqueeze(0)

        def create_causal_mask_without_cache_position(
            config,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids=None,
        ):
            return attention_mask

        with (
            mock.patch.object(modeling_gemma, "_CREATE_CAUSAL_MASK_SUPPORTS_CACHE_POSITION", False),
            mock.patch.object(
                modeling_gemma,
                "create_causal_mask",
                side_effect=create_causal_mask_without_cache_position,
            ) as create_mask,
            torch.no_grad(),
        ):
            output = action_expert.model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                adarms_cond=None,
            )

        self.assertEqual(output.last_hidden_state.shape, inputs_embeds.shape)
        self.assertNotIn("cache_position", create_mask.call_args.kwargs)
        self.assertIn("past_key_values", create_mask.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
