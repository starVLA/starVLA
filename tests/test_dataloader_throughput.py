import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from starVLA.dataloader.dataloader_options import DATALOADER_DEFAULTS, build_dataloader_kwargs
from starVLA.training.trainer_utils.throughput import (
    build_step_performance_metrics,
    count_batch_samples,
    count_batches_samples,
)


class TinyDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return index


class DataloaderOptionsTest(unittest.TestCase):
    def test_defaults_match_legacy_behavior(self):
        kwargs = build_dataloader_kwargs({})

        self.assertEqual(kwargs["num_workers"], DATALOADER_DEFAULTS["num_workers"])
        self.assertEqual(kwargs["prefetch_factor"], DATALOADER_DEFAULTS["prefetch_factor"])
        self.assertFalse(kwargs["pin_memory"])
        self.assertFalse(kwargs["persistent_workers"])
        self.assertFalse(kwargs["drop_last"])
        self.assertEqual(kwargs["timeout"], 0)

    def test_overrides_accept_dict_and_string_booleans(self):
        kwargs = build_dataloader_kwargs(
            {
                "num_workers": "8",
                "pin_memory": "true",
                "persistent_workers": "yes",
                "prefetch_factor": "3",
                "drop_last": "on",
                "timeout": "5",
            }
        )

        self.assertEqual(
            kwargs,
            {
                "num_workers": 8,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 3,
                "drop_last": True,
                "timeout": 5,
            },
        )

    def test_num_workers_zero_omits_prefetch_factor(self):
        kwargs = build_dataloader_kwargs(SimpleNamespace(num_workers=0, prefetch_factor="ignored"))

        self.assertEqual(kwargs["num_workers"], 0)
        self.assertNotIn("prefetch_factor", kwargs)

    def test_persistent_workers_requires_positive_num_workers(self):
        with self.assertRaisesRegex(ValueError, "persistent_workers=True"):
            build_dataloader_kwargs({"num_workers": 0, "persistent_workers": True})

    def test_invalid_values_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "num_workers"):
            build_dataloader_kwargs({"num_workers": -1})
        with self.assertRaisesRegex(ValueError, "pin_memory"):
            build_dataloader_kwargs({"pin_memory": "maybe"})
        with self.assertRaisesRegex(ValueError, "prefetch_factor"):
            build_dataloader_kwargs({"prefetch_factor": 0})

    def test_real_dataloader_iterates_with_num_workers_zero(self):
        kwargs = build_dataloader_kwargs({"num_workers": 0, "drop_last": True})
        loader = DataLoader(TinyDataset(), batch_size=2, **kwargs)

        self.assertEqual([batch.tolist() for batch in loader], [[0, 1], [2, 3]])

    def test_real_dataloader_accepts_worker_overrides(self):
        kwargs = build_dataloader_kwargs(
            {
                "num_workers": 1,
                "persistent_workers": True,
                "prefetch_factor": 3,
                "timeout": 1,
            }
        )
        loader = DataLoader(TinyDataset(), batch_size=2, **kwargs)

        self.assertEqual(loader.num_workers, 1)
        self.assertTrue(loader.persistent_workers)
        self.assertEqual(loader.prefetch_factor, 3)
        self.assertEqual(loader.timeout, 1)

    def test_core_training_configs_expose_default_loader_options(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_paths = [
            repo_root / "starVLA/config/training/starvla_train_adapter.yaml",
            repo_root / "starVLA/config/training/starvla_cotrain_libero.yaml",
            repo_root / "starVLA/config/training/starvla_cotrain_oxe.yaml",
        ]
        expected = {
            "num_workers": 4,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "drop_last": False,
            "timeout": 0,
        }

        for config_path in config_paths:
            cfg = OmegaConf.load(config_path)
            for section_name in ("vla_data", "vlm_data"):
                if section_name not in cfg.datasets:
                    continue
                with self.subTest(config=config_path.name, section=section_name):
                    kwargs = build_dataloader_kwargs(cfg.datasets[section_name])
                    self.assertEqual(kwargs, expected)


class ThroughputMetricsTest(unittest.TestCase):
    def test_count_batch_samples_for_vla_and_vlm_batches(self):
        self.assertEqual(count_batch_samples([{"action": 1}, {"action": 2}]), 2)
        self.assertEqual(count_batch_samples({"input_ids": torch.zeros(4, 8)}), 4)
        self.assertEqual(count_batches_samples([1, 2], {"input_ids": torch.zeros(3, 8)}), 5)

    def test_count_batch_samples_for_flattened_vlm_batch(self):
        batch = {
            "input_ids": torch.zeros(1, 12),
            "attention_mask": torch.tensor([0, 3, 7, 12], dtype=torch.int32),
        }

        self.assertEqual(count_batch_samples(batch), 3)

    def test_metrics_include_timing_throughput_and_no_cuda_memory_keys(self):
        class NoCuda:
            @staticmethod
            def is_available():
                return False

        metrics = build_step_performance_metrics(data_time=0.25, model_time=0.75, sample_count=4, cuda=NoCuda)

        self.assertEqual(metrics["timing/step"], 1.0)
        self.assertEqual(metrics["throughput/samples_per_sec"], 4.0)
        self.assertEqual(metrics["throughput/model_samples_per_sec"], 4 / 0.75)
        self.assertEqual(metrics["throughput/data_wait_ratio"], 0.25)
        self.assertEqual(metrics["memory/gpu_allocated_gb"], 0.0)
        self.assertEqual(metrics["memory/gpu_reserved_gb"], 0.0)

    def test_metrics_report_cuda_memory_in_gb(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def memory_allocated():
                return 2 * 1024**3

            @staticmethod
            def memory_reserved():
                return 3 * 1024**3

        metrics = build_step_performance_metrics(data_time=0.0, model_time=2.0, sample_count=4, cuda=FakeCuda)

        self.assertEqual(metrics["memory/gpu_allocated_gb"], 2.0)
        self.assertEqual(metrics["memory/gpu_reserved_gb"], 3.0)


if __name__ == "__main__":
    unittest.main()
