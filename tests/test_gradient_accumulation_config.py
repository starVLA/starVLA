import importlib
import unittest
from types import SimpleNamespace
from unittest import mock

TRAINING_MODULES = (
    ("starVLA.training.train_starvla", "VLATrainer"),
    ("starVLA.training.train_starvla_cotrain", "VLAMTrainer"),
    ("starVLA.training.train_starvlm", "VLAMTrainer"),
)


class GradientAccumulationConfigTest(unittest.TestCase):
    def test_shared_builder_forwards_configured_accumulation_steps(self):
        trainer_tools = importlib.import_module("starVLA.training.trainer_utils.trainer_tools")
        cfg = SimpleNamespace(trainer=SimpleNamespace(gradient_accumulation_steps=4))
        accelerator = mock.Mock(state=object())

        with (
            mock.patch.object(trainer_tools, "DeepSpeedPlugin", return_value="plugin") as plugin_cls,
            mock.patch.object(trainer_tools, "Accelerator", return_value=accelerator) as accelerator_cls,
        ):
            result = trainer_tools.build_accelerator(cfg)

        plugin_cls.assert_called_once_with()
        accelerator_cls.assert_called_once_with(gradient_accumulation_steps=4, deepspeed_plugin="plugin")
        accelerator.print.assert_called_once_with(accelerator.state)
        self.assertIs(result, accelerator)

    def test_all_entrypoints_build_accelerator_from_wrapped_config(self):
        for module_name, trainer_name in TRAINING_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                original_cfg = object()
                wrapped_cfg = SimpleNamespace(trainer=SimpleNamespace(gradient_accumulation_steps=7))
                accelerator = mock.Mock()
                model = mock.Mock()
                optimizer = mock.Mock()
                scheduler = mock.Mock()
                trainer = mock.Mock()
                dataloader = ("vla_dataloader", "vlm_dataloader") if "cotrain" in module_name else "dataloader"

                with (
                    mock.patch.object(module, "wrap_config", return_value=wrapped_cfg) as wrap_config,
                    mock.patch.object(module, "logger"),
                    mock.patch.object(module, "build_accelerator", return_value=accelerator) as build_accelerator,
                    mock.patch.object(module, "setup_directories", return_value="output"),
                    mock.patch.object(module, "build_framework", return_value=model),
                    mock.patch.object(module, "prepare_data", return_value=dataloader),
                    mock.patch.object(module, "setup_optimizer_and_scheduler", return_value=(optimizer, scheduler)),
                    mock.patch.object(module, trainer_name, return_value=trainer) as trainer_cls,
                ):
                    module.main(original_cfg)

                wrap_config.assert_called_once_with(original_cfg)
                build_accelerator.assert_called_once_with(wrapped_cfg)
                self.assertIs(trainer_cls.call_args.kwargs["cfg"], wrapped_cfg)
                self.assertIs(trainer_cls.call_args.kwargs["accelerator"], accelerator)
                trainer.prepare_training.assert_called_once_with()
                trainer.train.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
