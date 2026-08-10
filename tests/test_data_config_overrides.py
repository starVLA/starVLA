import unittest

from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.config_overrides import (
    apply_normalization_mode_overrides,
    build_overridden_data_pipeline,
    resolve_action_horizon,
)
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


class _FakeDataConfig:
    action_keys = ["action.x", "action.gripper"]

    def modality_config(self):
        return {
            "action": ModalityConfig(
                delta_indices=list(range(8)),
                modality_keys=self.action_keys,
            )
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={"action.x": "min_max"},
                )
            ]
        )


class ActionHorizonOverrideTest(unittest.TestCase):
    def test_matching_model_and_data_horizons_are_accepted(self):
        self.assertEqual(
            resolve_action_horizon(
                model_action_horizon=12,
                data_action_horizon=12,
            ),
            12,
        )

    def test_model_horizon_is_used_when_data_value_is_omitted(self):
        self.assertEqual(
            resolve_action_horizon(model_action_horizon=16),
            16,
        )

    def test_mismatched_horizons_fail_before_dataset_construction(self):
        with self.assertRaisesRegex(ValueError, "Action horizon mismatch"):
            resolve_action_horizon(
                model_action_horizon=16,
                data_action_horizon=8,
            )

    def test_dataset_entrypoint_validates_before_touching_dataset_files(self):
        data_cfg = OmegaConf.create(
            {
                "data_root_dir": "/path/that/does/not/exist",
                "data_mix": "libero_all",
                "action_horizon": 8,
            }
        )
        with self.assertRaisesRegex(ValueError, "Action horizon mismatch"):
            get_vla_dataset(data_cfg, model_action_horizon=16)

    def test_pipeline_uses_range_of_resolved_horizon_without_mutating_default(self):
        data_config = _FakeDataConfig()
        original = data_config.modality_config()

        overridden, _ = build_overridden_data_pipeline(
            data_config,
            action_horizon=12,
        )

        self.assertEqual(overridden["action"].delta_indices, list(range(12)))
        self.assertEqual(original["action"].delta_indices, list(range(8)))


class NormalizationModeOverrideTest(unittest.TestCase):
    def _transform(self):
        return _FakeDataConfig().transform()

    def test_scalar_replaces_existing_modes_but_preserves_unnormalized_keys(self):
        transform = apply_normalization_mode_overrides(self._transform(), "q99")
        state_action = transform.transforms[0]

        self.assertEqual(state_action.normalization_modes, {"action.x": "q99"})
        self.assertNotIn("action.gripper", state_action.normalization_modes)

    def test_per_key_mapping_can_enable_a_previously_unnormalized_key(self):
        override = OmegaConf.create(
            {"action": {"x": "mean_std", "gripper": "binary"}}
        )
        transform = apply_normalization_mode_overrides(self._transform(), override)
        state_action = transform.transforms[0]

        self.assertEqual(
            state_action.normalization_modes,
            {"action.x": "mean_std", "action.gripper": "binary"},
        )

    def test_null_per_key_mapping_disables_normalization(self):
        transform = apply_normalization_mode_overrides(
            self._transform(), {"action.x": None}
        )
        self.assertEqual(transform.transforms[0].normalization_modes, {})

    def test_invalid_mode_is_rejected_early(self):
        with self.assertRaisesRegex(ValueError, "Invalid normalization mode"):
            apply_normalization_mode_overrides(self._transform(), "unknown")


if __name__ == "__main__":
    unittest.main()
