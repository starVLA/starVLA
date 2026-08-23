import unittest

import torch

from starVLA.model.modules.action_model.OpenPI_ActionHead import GemmaDims, OpenPIGemma


class OpenPIGemmaTransformersCompatTest(unittest.TestCase):
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

                with torch.no_grad():
                    output = action_expert.model.forward(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        adarms_cond=adarms_cond,
                    )

                self.assertEqual(output.last_hidden_state.shape, inputs_embeds.shape)
                self.assertTrue(torch.isfinite(output.last_hidden_state).all())


if __name__ == "__main__":
    unittest.main()
