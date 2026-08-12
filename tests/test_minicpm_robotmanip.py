import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.model.framework.VLM4A.MiniCPMRobotManip import (
    EE6D_SLOT_END,
    EE6D_SLOT_START,
    MiniCPM_RobotManip,
)


class MiniCPMRobotManipActionTest(unittest.TestCase):
    def setUp(self):
        self.framework = MiniCPM_RobotManip.__new__(MiniCPM_RobotManip)
        self.framework.action_dim = 80
        self.framework.xyz_loss_scale = 500.0
        self.framework.rot6d_loss_scale = 10.0

    def test_ee6d_is_mapped_to_left_arm_slot(self):
        ee6d = np.arange(2 * 3 * 10, dtype=np.float32).reshape(2, 3, 10)

        action, valid = self.framework._to_80d(ee6d, torch.device("cpu"))

        self.assertEqual(tuple(action.shape), (2, 3, 80))
        torch.testing.assert_close(action[:, :, EE6D_SLOT_START:EE6D_SLOT_END], torch.from_numpy(ee6d))
        self.assertEqual(int(valid.sum().item()), 2 * 3 * 10)
        self.assertEqual(int(torch.count_nonzero(action[:, :, :EE6D_SLOT_START])), 0)
        self.assertEqual(int(torch.count_nonzero(action[:, :, EE6D_SLOT_END:])), 0)

    def test_group_loss_ignores_masked_channels(self):
        target = torch.zeros(1, 2, 80)
        valid = torch.zeros_like(target)
        valid[:, :, EE6D_SLOT_START:EE6D_SLOT_END] = 1
        pred = torch.zeros_like(target)
        pred[:, :, :EE6D_SLOT_START] = 1000

        losses = self.framework._group_weighted_action_loss(pred, target, valid)

        self.assertEqual(losses["action_loss"].item(), 0.0)


class MiniCPMRobotManipDataPackingTest(unittest.TestCase):
    def test_configurable_dtype_and_image_size(self):
        dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
        dataset.data_cfg = {
            "image_size": 448,
            "include_state": True,
            "state_action_dtype": "float32",
        }
        dataset._modality_keys = {
            "video": ["video.primary", "video.wrist"],
            "language": ["annotation.task"],
            "action": ["action.eef"],
            "state": ["state.eef"],
        }
        dataset.tag = "franka"
        data = {
            "video.primary": [np.zeros((16, 24, 3), dtype=np.uint8)],
            "video.wrist": [np.zeros((12, 20, 3), dtype=np.uint8)],
            "annotation.task": ["pick up the object"],
            "action.eef": np.zeros((30, 10), dtype=np.float64),
            "state.eef": np.zeros((1, 10), dtype=np.float64),
        }

        sample = dataset._pack_sample(data)

        self.assertTrue(all(isinstance(image, Image.Image) for image in sample["image"]))
        self.assertTrue(all(image.size == (448, 448) for image in sample["image"]))
        self.assertEqual(sample["action"].dtype, np.float32)
        self.assertEqual(sample["state"].dtype, np.float32)

    def test_incomplete_action_chunks_are_never_sampled(self):
        from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset

        class StubDataset:
            dataset_name = "stub"

            def __init__(self):
                self.trajectory_lengths = np.array([35])
                self.trajectory_ids = np.array([7])
                self.modality_keys = {"action": ["action.eef"]}
                self.delta_indices = {"action.eef": np.arange(1, 31)}

            def __len__(self):
                return 35

        with mock.patch.object(LeRobotMixtureDataset, "update_metadata"):
            mixture = LeRobotMixtureDataset(
                [(StubDataset(), 1.0)],
                mode="test",
                data_cfg={"drop_incomplete_action_chunks": True},
            )

        sampled = [mixture.sample_step(index)[2] for index in range(100)]
        self.assertTrue(all(0 <= base_index < 5 for base_index in sampled))


if __name__ == "__main__":
    unittest.main()
