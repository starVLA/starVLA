import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER_TOOLS_PATH = REPO_ROOT / "starVLA/training/trainer_utils/trainer_tools.py"


class _DeepSpeedPlugin:
    pass


class _Accelerator:
    def __init__(self, *, gradient_accumulation_steps, deepspeed_plugin):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.deepspeed_plugin = deepspeed_plugin
        self.state = object()
        self.printed_state = None

    def print(self, state):
        self.printed_state = state


def _load_build_accelerator():
    path = TRAINER_TOOLS_PATH
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_accelerator"
    )
    namespace = {
        "Accelerator": _Accelerator,
        "DeepSpeedPlugin": _DeepSpeedPlugin,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["build_accelerator"]


class GradientAccumulationConfigTest(unittest.TestCase):
    def test_shared_builder_forwards_configured_accumulation_steps(self):
        cfg = SimpleNamespace(trainer=SimpleNamespace(gradient_accumulation_steps=4))
        accelerator = _load_build_accelerator()(cfg)

        self.assertEqual(accelerator.gradient_accumulation_steps, 4)
        self.assertIsInstance(accelerator.deepspeed_plugin, _DeepSpeedPlugin)
        self.assertIs(accelerator.printed_state, accelerator.state)


if __name__ == "__main__":
    unittest.main()
