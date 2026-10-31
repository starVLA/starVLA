"""Action inverse contracts for the server's training-time transform pipeline."""

import unittest
from typing import ClassVar
from unittest import mock

import numpy as np
import torch

from deployment.model_server import policy_norm_processor
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class _DataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    action_keys: ClassVar = ["action.joints", "action.gripper"]
    action_key_dims: ClassVar = {"action.joints": 2, "action.gripper": 1}
    state_keys: ClassVar = ["state.proprio"]
    state_key_dims: ClassVar = {"state.proprio": 1}

    def __init__(self, with_concat):
        self.with_concat = with_concat

    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={"action.joints": "min_max", "action.gripper": "binary"},
            ),
        ]
        if self.with_concat:
            transforms.append(ConcatTransform(video_concat_order=[], action_concat_order=self.action_keys))
        return ComposedModalityTransform(transforms=transforms)


def _stats(low, high):
    return {"min": low, "max": high, "mean": low, "std": high, "q01": low, "q99": high}


class PolicyNormProcessorTest(unittest.TestCase):
    def _processor(self, with_concat):
        cfg = {"datasets": {"vla_data": {"data_mix": "test_mix"}}}
        stats = {
            "test_robot": {
                "action": _stats([-2.0, 10.0, 0.0], [2.0, 20.0, 1.0]),
                "state": _stats([0.0], [1.0]),
            }
        }
        with (
            mock.patch.object(policy_norm_processor, "read_mode_config", return_value=(cfg, stats)),
            mock.patch.dict(
                policy_norm_processor.DATASET_NAMED_MIXTURES,
                {"test_mix": [("test_dataset", 1.0, "test_robot")]},
            ),
            mock.patch.dict(
                policy_norm_processor.ROBOT_TYPE_CONFIG_MAP,
                {"test_robot": _DataConfig(with_concat)},
            ),
        ):
            return policy_norm_processor.PolicyNormProcessor("unused.pt")

    def test_concat_pipeline_matches_direct_training_inverse(self):
        proc = self._processor(with_concat=True)
        normalized = np.array([[-1.0, 0.0, 0.49], [1.0, 0.5, 0.51]], dtype=np.float32)
        original = normalized.copy()
        reference = proc.transform.unapply({"action": torch.from_numpy(normalized.copy())})
        expected = np.concatenate([reference[k] for k in proc.action_keys], axis=-1)
        np.testing.assert_allclose(expected, [[-2.0, 15.0, 0.0], [2.0, 17.5, 1.0]])

        actual = proc.unapply_actions(normalized)

        np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(normalized, original)

    def test_pipeline_without_concat_preserves_per_key_inverse(self):
        proc = self._processor(with_concat=False)
        normalized = np.array([[-1.0, 0.0, 0.49], [1.0, 0.5, 0.51]], dtype=np.float32)
        actual = proc.unapply_actions(normalized)
        np.testing.assert_allclose(actual, [[-2.0, 15.0, 0.0], [2.0, 17.5, 1.0]])

    def test_training_forward_server_inverse_roundtrip(self):
        physical = np.array([[-2.0, 10.0, 0.0], [0.0, 15.0, 1.0], [2.0, 20.0, 0.0]], dtype=np.float32)
        for with_concat in (False, True):
            with self.subTest(with_concat=with_concat):
                proc = self._processor(with_concat)
                training_input = {"action.joints": physical[:, :2].copy(), "action.gripper": physical[:, 2:].copy()}
                training_output = proc.transform.apply(training_input)
                if with_concat:
                    normalized = training_output["action"].numpy()
                else:
                    normalized = np.concatenate([training_output[key].numpy() for key in proc.action_keys], axis=-1)

                np.testing.assert_allclose(proc.unapply_actions(normalized), physical)

    def test_noncontiguous_action_chunk(self):
        for with_concat in (False, True):
            with self.subTest(with_concat=with_concat):
                proc = self._processor(with_concat)
                normalized = np.zeros((3, 4), dtype=np.float32).T
                self.assertFalse(normalized.flags.c_contiguous)
                actual = proc.unapply_actions(normalized)
                np.testing.assert_allclose(actual, np.tile([0.0, 15.0, 0.0], (4, 1)))

    def test_wrong_action_width_is_rejected(self):
        for with_concat in (False, True):
            for width in (2, 4):
                with self.subTest(with_concat=with_concat, width=width):
                    proc = self._processor(with_concat)
                    with self.assertRaisesRegex(ValueError, "Sum of per-key dims"):
                        proc.unapply_actions(np.zeros((2, width), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
