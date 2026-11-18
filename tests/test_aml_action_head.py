import copy
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.AML_ActionHeader import DiTConfig, get_action_model


def action_config(preset="DiT-B", state_dim=3, add_pos_embed=True, overrides=None):
    dit = {
        "num_layers": 2,
        "cross_attention_dim": 32,
        "output_dim": 24,
        "dropout": 0.0,
        "final_dropout": False,
        "positional_embeddings": None,
        "interleave_self_attention": True,
    }
    if overrides is not None:
        dit.update(overrides)
    return OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "action_model_type": preset,
                    "hidden_size": 32,
                    "action_dim": 2,
                    "state_dim": state_dim,
                    "action_horizon": 4,
                    "num_inference_timesteps": 4,
                    "num_target_vision_tokens": 1,
                    "add_pos_embed": add_pos_embed,
                    "max_seq_len": 16,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "diffusion_model_cfg": dit,
                }
            }
        }
    )


class AMLActionHeadTest(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(11)
        self.hidden = torch.randn(2, 5, 32)
        self.actions = torch.randn(2, 4, 2)
        self.state = torch.randn(2, 1, 3)

    def assert_token_width(self, head, width):
        self.assertEqual(head.model.inner_dim, width)
        self.assertEqual(head.input_embedding_dim, width)
        self.assertEqual(head.action_encoder.hidden_size, width)
        self.assertEqual(head.action_encoder.layer1.out_features, width)
        self.assertEqual(head.action_encoder.layer2.in_features, 2 * width)
        self.assertEqual(head.action_encoder.layer3.out_features, width)
        self.assertEqual(head.future_tokens.embedding_dim, width)
        if head.state_encoder is not None:
            self.assertEqual(head.state_encoder.layer2.out_features, width)
        if head.config.add_pos_embed:
            self.assertEqual(head.position_embedding.embedding_dim, width)
        # The DiT output projection is independently configurable.
        self.assertEqual(head.action_decoder.layer1.in_features, 24)

    def test_attention_geometry_controls_all_token_widths(self):
        for preset in ("DiT-B", "DiT-L"):
            for hint in (None, 64, 999):
                with self.subTest(preset=preset, hint=hint):
                    overrides = {"num_attention_heads": 4, "attention_head_dim": 16}
                    if hint is not None:
                        overrides["input_embedding_dim"] = hint
                    head = get_action_model(action_config(preset, overrides=overrides))
                    self.assert_token_width(head, 64)

    def test_overridden_width_trains_and_samples_with_optional_inputs(self):
        for preset in ("DiT-B", "DiT-L"):
            for with_state, with_position in ((False, False), (False, True), (True, False), (True, True)):
                with self.subTest(preset=preset, state=with_state, position=with_position):
                    config = action_config(
                        preset,
                        state_dim=3 if with_state else 0,
                        add_pos_embed=with_position,
                        overrides={"num_attention_heads": 4, "attention_head_dim": 16},
                    )
                    head = get_action_model(config)
                    state = self.state if with_state else None
                    # Batch contains different padded action dimensions.
                    action_mask = torch.tensor([[True, False], [True, True]])
                    encoder_mask = torch.tensor([[True, True, True, False, False], [True] * 5])
                    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
                    for _ in range(2):
                        optimizer.zero_grad()
                        loss = head(
                            self.hidden,
                            self.actions,
                            state,
                            encoder_attention_mask=encoder_mask,
                            action_mask=action_mask,
                        )
                        self.assertEqual(loss.shape, torch.Size([]))
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        parameters = [
                            head.action_encoder.layer1.weight,
                            head.model.transformer_blocks[0].attn1.to_q.weight,
                            head.action_decoder.layer2.weight,
                            head.future_tokens.weight,
                        ]
                        if with_state:
                            parameters.append(head.state_encoder.layer1.weight)
                        if with_position:
                            parameters.append(head.position_embedding.weight)
                        for parameter in parameters:
                            self.assertIsNotNone(parameter.grad)
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                            self.assertGreater(parameter.grad.abs().sum().item(), 0)
                        optimizer.step()
                    head.eval()
                    prediction = head.predict_action(self.hidden, state)
                    self.assertEqual(prediction.shape, self.actions.shape)
                    self.assertTrue(torch.isfinite(prediction).all())
                    self.assertFalse(prediction.requires_grad)
                    self.assertEqual(prediction.dtype, self.hidden.dtype)

    def test_head_count_only_and_head_size_only_overrides(self):
        for preset in ("DiT-B", "DiT-L"):
            for overrides in ({"num_attention_heads": 2}, {"attention_head_dim": 8}):
                with self.subTest(preset=preset, overrides=overrides):
                    head = get_action_model(action_config(preset, overrides=overrides)).eval()
                    expected_width = overrides.get(
                        "num_attention_heads", DiTConfig[preset]["num_attention_heads"]
                    ) * overrides.get("attention_head_dim", DiTConfig[preset]["attention_head_dim"])
                    self.assert_token_width(head, expected_width)
                    prediction = head.predict_action(self.hidden, self.state)
                    self.assertEqual(prediction.shape, self.actions.shape)
                    self.assertTrue(torch.isfinite(prediction).all())

    def test_default_preset_width_and_checkpoint_shapes_are_preserved(self):
        for preset in ("DiT-B", "DiT-L"):
            with self.subTest(preset=preset):
                config = action_config(preset)
                # Constructor-only legacy shape checks do not need a deep transformer.
                config.framework.action_model.diffusion_model_cfg.num_layers = 0
                head = get_action_model(config)
                width = DiTConfig[preset]["input_embedding_dim"]
                self.assert_token_width(head, width)
                checkpoint = head.state_dict()
                self.assertEqual(checkpoint["action_encoder.layer1.weight"].shape, (width, 2))
                self.assertEqual(checkpoint["state_encoder.layer2.weight"].shape, (width, 32))
                self.assertEqual(checkpoint["future_tokens.weight"].shape, (1, width))
                self.assertEqual(checkpoint["position_embedding.weight"].shape, (16, width))
                loaded = head.load_state_dict(checkpoint, strict=True)
                self.assertEqual(loaded.missing_keys, [])
                self.assertEqual(loaded.unexpected_keys, [])

    def test_config_and_presets_are_not_mutated(self):
        config = action_config(overrides={"num_attention_heads": 4, "attention_head_dim": 16})
        before_config = OmegaConf.to_container(config, resolve=True)
        before_presets = copy.deepcopy(DiTConfig)
        get_action_model(config)
        self.assertEqual(OmegaConf.to_container(config, resolve=True), before_config)
        self.assertEqual(DiTConfig, before_presets)

    def test_real_training_loss_and_gradients_match_velocity_reference(self):
        head = get_action_model(action_config(overrides={"num_attention_heads": 4, "attention_head_dim": 16})).eval()
        times = torch.tensor([0.25, 0.99])
        torch.manual_seed(23)
        noise = torch.randn_like(self.actions)
        noisy = times[:, None, None] * self.actions + (1 - times[:, None, None]) * noise
        timestep = (times * head.num_timestep_buckets).long()
        features = head.action_encoder(noisy, timestep)
        features = features + head.position_embedding(torch.arange(4))[None]
        tokens = torch.cat(
            (head.state_encoder(self.state), head.future_tokens.weight[None].expand(2, -1, -1), features), 1
        )
        encoder_mask = torch.tensor([[True, True, True, False, False], [True] * 5])
        predicted_actions = head.action_decoder(
            head.model(
                hidden_states=tokens,
                encoder_hidden_states=self.hidden,
                timestep=timestep,
                encoder_attention_mask=encoder_mask,
            )
        )[:, -4:]
        # The noisy sample cancels from the difference between predicted and true velocity.
        # The second batch item also exercises the denominator's t_eps clamp.
        residual = (predicted_actions - self.actions) / (1 - times[:, None, None]).clamp_min(head.t_eps)
        action_mask = torch.tensor([[True, False], [True, True]])
        valid = action_mask[:, None, :].expand_as(residual)
        expected = (residual.square() * valid).sum() / (valid.sum() + 1e-8)
        with patch.object(head, "sample_time", return_value=times):
            torch.manual_seed(23)
            actual = head(
                self.hidden, self.actions, self.state, encoder_attention_mask=encoder_mask, action_mask=action_mask
            )
        torch.testing.assert_close(actual, expected)
        parameters = (head.action_encoder.layer1.weight, head.action_decoder.layer2.weight)
        actual_gradients = torch.autograd.grad(actual, parameters)
        expected_gradients = torch.autograd.grad(expected, parameters)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-4, atol=1e-5)

    def test_real_sampler_matches_independent_euler_reference(self):
        head = get_action_model(action_config(overrides={"num_attention_heads": 4, "attention_head_dim": 16})).eval()
        # Independent reconstruction of the existing sample-to-velocity Euler steps.
        with torch.no_grad():
            torch.manual_seed(23)
            expected = torch.randn(2, 4, 2)
            state_features = head.state_encoder(self.state)
            for step in range(4):
                time = step / 4
                timestep = torch.full((2,), int(time * head.num_timestep_buckets))
                action_features = head.action_encoder(expected, timestep)
                action_features = action_features + head.position_embedding(torch.arange(4))[None]
                tokens = torch.cat(
                    (state_features, head.future_tokens.weight[None].expand(2, -1, -1), action_features), 1
                )
                samples = head.action_decoder(
                    head.model(hidden_states=tokens, encoder_hidden_states=self.hidden, timestep=timestep)
                )[:, -4:]
                expected = expected + (samples - expected) / (1 - time) / 4
            torch.manual_seed(23)
            actual = head.predict_action(self.hidden, self.state)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
