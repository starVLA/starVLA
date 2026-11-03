import os
import subprocess
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn

from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


class SingleProcessDistSafetyTest(unittest.TestCase):
    def test_trainer_utils_safe_without_process_group(self):
        self.assertFalse(dist.is_initialized())

        model = nn.Linear(2, 3)
        num_params, num_trainable = TrainerUtils.print_trainable_parameters(model)
        self.assertEqual(num_params, num_trainable)

        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint_file:
            torch.save(model.state_dict(), checkpoint_file.name)
            loaded_model = TrainerUtils.load_pretrained_backbones(nn.Linear(2, 3), checkpoint_file.name)

        self.assertIsInstance(loaded_model, nn.Module)

    def _run_subprocess(self, code: str, disable_deepspeed: bool = True):
        env = os.environ.copy()
        if disable_deepspeed:
            env["STARVLA_DISABLE_DEEPSPEED"] = "1"
        else:
            env.pop("STARVLA_DISABLE_DEEPSPEED", None)
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)

    def test_training_entrypoints_keep_deepspeed_enabled_by_default(self):
        for module_name in (
            "starVLA.training.train_starvla",
            "starVLA.training.train_starvlm",
            "starVLA.training.train_starvla_cotrain",
        ):
            with self.subTest(module_name=module_name):
                self._run_subprocess(
                    f"""
import importlib
import types
from unittest import mock

plugin = object()
fake_accelerator = types.SimpleNamespace(state=None, print=lambda *args: None)
with (
    mock.patch("accelerate.DeepSpeedPlugin", return_value=plugin),
    mock.patch("accelerate.Accelerator", return_value=fake_accelerator),
):
    module = importlib.import_module("{module_name}")
assert module.deepspeed_plugin is plugin
""",
                    disable_deepspeed=False,
                )

    def test_train_starvla_prepare_data_safe_without_process_group(self):
        self._run_subprocess(
            """
import importlib
import types
from unittest import mock

module = importlib.import_module("starVLA.training.train_starvla")
assert module.deepspeed_plugin is None
cfg = types.SimpleNamespace(
    datasets=types.SimpleNamespace(
        vla_data=types.SimpleNamespace(data_mix="dummy", dataset_py="dummy")
    )
)
fake_accelerator = types.SimpleNamespace(dataloader_config=types.SimpleNamespace(dispatch_batches=None))
with mock.patch.object(module, "build_dataloader", return_value=[1, 2, 3]):
    dataloader = module.prepare_data(cfg, fake_accelerator, output_dir=None)
assert dataloader == [1, 2, 3]
""",
        )

    def test_train_starvlm_prepare_data_safe_without_process_group(self):
        self._run_subprocess(
            """
import importlib
import types
from unittest import mock

module = importlib.import_module("starVLA.training.train_starvlm")
assert module.deepspeed_plugin is None
cfg = types.SimpleNamespace(
    datasets=types.SimpleNamespace(
        vlm_data=types.SimpleNamespace(dataset_use="dummy", dataset_py="dummy")
    )
)
fake_accelerator = types.SimpleNamespace(dataloader_config=types.SimpleNamespace(dispatch_batches=None))
with mock.patch.object(module, "build_dataloader", return_value=[1]):
    dataloader = module.prepare_data(cfg, fake_accelerator, output_dir=None)
assert dataloader == [1]
""",
        )

    def test_train_starvla_cotrain_prepare_data_safe_without_process_group(self):
        self._run_subprocess(
            """
import importlib
import types
from unittest import mock

module = importlib.import_module("starVLA.training.train_starvla_cotrain")
assert module.deepspeed_plugin is None
cfg = types.SimpleNamespace(
    datasets=types.SimpleNamespace(
        vla_data=types.SimpleNamespace(data_mix="dummy", dataset_py="dummy"),
        vlm_data=types.SimpleNamespace(dataset_py="dummy"),
    )
)
fake_accelerator = types.SimpleNamespace(dataloader_config=types.SimpleNamespace(dispatch_batches=None))
with mock.patch.object(module, "build_dataloader", side_effect=[[1], [2]]):
    vla_dataloader, vlm_dataloader = module.prepare_data(cfg, fake_accelerator, output_dir=None)
assert vla_dataloader == [1]
assert vlm_dataloader == [2]
""",
        )


if __name__ == "__main__":
    unittest.main()
