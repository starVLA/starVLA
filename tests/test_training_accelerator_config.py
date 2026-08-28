import ast
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAINING_ENTRYPOINTS = [
    REPO_ROOT / "starVLA" / "training" / "train_starvla.py",
    REPO_ROOT / "starVLA" / "training" / "train_starvla_cotrain.py",
    REPO_ROOT / "starVLA" / "training" / "train_starvlm.py",
]


class AcceleratorConfigTest(unittest.TestCase):
    def test_create_accelerator_uses_trainer_gradient_accumulation_steps(self):
        try:
            accelerator_utils = importlib.import_module("starVLA.training.accelerator_utils")
        except ImportError as exc:
            self.fail(f"starVLA.training.accelerator_utils is missing: {exc}")

        cfg = SimpleNamespace(trainer=SimpleNamespace(gradient_accumulation_steps=7))
        fake_plugin = object()
        fake_accelerator = object()
        plugin_cls = mock.Mock(return_value=fake_plugin)
        accelerator_cls = mock.Mock(return_value=fake_accelerator)

        with mock.patch.object(
            accelerator_utils,
            "load_accelerate_classes",
            return_value=(accelerator_cls, plugin_cls),
        ):
            accelerator = accelerator_utils.create_accelerator(cfg)

        self.assertIs(accelerator, fake_accelerator)
        accelerator_cls.assert_called_once_with(
            deepspeed_plugin=fake_plugin,
            gradient_accumulation_steps=7,
        )

    def test_training_entrypoints_create_accelerator_after_config_load(self):
        for script_path in TRAINING_ENTRYPOINTS:
            with self.subTest(script=script_path.name):
                tree = ast.parse(script_path.read_text(encoding="utf-8"))
                top_level_accelerator_calls = []
                main_create_calls = 0

                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == "accelerator":
                                if isinstance(node.value, ast.Call):
                                    func = node.value.func
                                    if isinstance(func, ast.Name) and func.id == "Accelerator":
                                        top_level_accelerator_calls.append(node.lineno)
                    if isinstance(node, ast.FunctionDef) and node.name == "main":
                        for child in ast.walk(node):
                            if isinstance(child, ast.Call):
                                func = child.func
                                if isinstance(func, ast.Name) and func.id == "create_accelerator":
                                    main_create_calls += 1

                self.assertEqual(top_level_accelerator_calls, [])
                self.assertEqual(main_create_calls, 1)

    def test_deepspeed_config_defers_gradient_accumulation_to_accelerate(self):
        ds_config_path = REPO_ROOT / "starVLA" / "config" / "deepseeds" / "ds_config.yaml"
        self.assertIn('"gradient_accumulation_steps": "auto"', ds_config_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
