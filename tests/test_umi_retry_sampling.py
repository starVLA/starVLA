import unittest

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from starVLA.dataloader.umi_datasets import UMISampleAdapter, UMISamplePolicy


class RecordingDataset(Dataset):
    def __init__(self, length, bad_indices):
        self.length = length
        self.bad_indices = set(bad_indices)
        self.visited = []
        self.image = Image.new("RGB", (2, 2))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.visited.append(index)
        value = np.nan if index in self.bad_indices else 1.0
        return {"action": np.full((2, 2), value), "image": [self.image], "lang": "pick"}


def adapter_for(dataset, retries, seed=42):
    return UMISampleAdapter(dataset, UMISamplePolicy(action_horizon=2, action_dim=2, retry_bad_samples=retries), seed)


class UMIRetrySamplingTest(unittest.TestCase):
    def test_recovers_outside_the_original_short_cycle(self):
        dataset = RecordingDataset(6, bad_indices={1, 4})
        adapter = adapter_for(dataset, retries=5)

        result = adapter[1]

        self.assertEqual(dataset.visited[0], 1)
        self.assertNotIn(dataset.visited[-1], dataset.bad_indices)
        self.assertLessEqual(len(dataset.visited), 6)
        self.assertTrue(np.isfinite(result["action"]).all())
        self.assertEqual(adapter.rejected_samples, len(dataset.visited) - 1)

    def test_full_retry_cycle_visits_each_index_and_is_repeatable(self):
        for length in (1, 2, 6, 9, 10, 15, 21):
            for seed in (0, 42):
                for epoch in (0, 3):
                    for index in {0, length // 2, length - 1}:
                        with self.subTest(length=length, seed=seed, epoch=epoch, index=index):
                            dataset = RecordingDataset(length, bad_indices=range(length))
                            adapter = adapter_for(dataset, retries=length - 1, seed=seed)
                            adapter.set_epoch(epoch)
                            with self.assertRaises(RuntimeError) as caught:
                                adapter[index]
                            first_visits = dataset.visited.copy()
                            self.assertEqual(len(first_visits), length)
                            self.assertEqual(set(first_visits), set(range(length)))
                            self.assertEqual(first_visits[0], (index + epoch) % length)
                            self.assertIsInstance(caught.exception.__cause__, ValueError)
                            dataset.visited.clear()
                            with self.assertRaises(RuntimeError):
                                adapter[index]
                            self.assertEqual(dataset.visited, first_visits)

    def test_valid_first_candidate_is_unchanged(self):
        dataset = RecordingDataset(6, bad_indices=set())
        adapter = adapter_for(dataset, retries=5)
        adapter.set_epoch(3)
        adapter[1]
        self.assertEqual(dataset.visited, [4])
        self.assertEqual(adapter.rejected_samples, 0)

    def test_existing_coprime_retry_order_is_unchanged(self):
        dataset = RecordingDataset(8, bad_indices={0, 1})
        adapter_for(dataset, retries=7)[1]
        self.assertEqual(dataset.visited, [1, 0, 7])

    def test_retry_limit_and_failure_cause_are_preserved(self):
        for retries in (0, 2, 8):
            with self.subTest(retries=retries):
                dataset = RecordingDataset(6, bad_indices=range(6))
                adapter = adapter_for(dataset, retries=retries)
                with self.assertRaisesRegex(RuntimeError, f"after {retries + 1} attempts") as caught:
                    adapter[1]
                self.assertEqual(len(dataset.visited), retries + 1)
                self.assertEqual(adapter.rejected_samples, retries + 1)
                self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_empty_dataset_still_raises(self):
        with self.assertRaisesRegex(IndexError, "empty UMI dataset"):
            adapter_for(RecordingDataset(0, bad_indices=set()), retries=5)[0]


if __name__ == "__main__":
    unittest.main()
