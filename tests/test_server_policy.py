import subprocess
import sys
import unittest
from unittest import mock
import importlib
from pathlib import Path

from deployment.model_server import server_policy_utils

REPO_ROOT = Path(__file__).resolve().parents[1]


class _DummyPolicy:
    pass


class ServerPolicyTests(unittest.TestCase):
    def _run_script_help(self, relative_script_path: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / relative_script_path), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_server_policy_module_imports_without_optional_framework_dependencies(self):
        module = importlib.import_module("deployment.model_server.server_policy")
        self.assertTrue(hasattr(module, "main"))

    def test_server_policy_help_runs_as_script(self):
        proc = self._run_script_help("deployment/model_server/server_policy.py")
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--ckpt_path", proc.stdout)

    def test_benchmark_policy_server_help_runs_as_script(self):
        proc = self._run_script_help("deployment/model_server/tools/benchmark_policy_server.py")
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--image", proc.stdout)

    def test_debug_server_policy_help_runs_as_script(self):
        proc = self._run_script_help("deployment/model_server/tools/debug_server_policy.py")
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--instruction", proc.stdout)

    def test_resolve_server_device_uses_cpu_when_auto_and_cuda_missing(self):
        with mock.patch.object(server_policy_utils.torch.cuda, "is_available", return_value=False):
            self.assertEqual("cpu", server_policy_utils.resolve_server_device("auto"))

    def test_resolve_server_device_uses_cuda_when_auto_and_cuda_present(self):
        with mock.patch.object(server_policy_utils.torch.cuda, "is_available", return_value=True):
            self.assertEqual("cuda", server_policy_utils.resolve_server_device("auto"))

    def test_resolve_server_device_raises_for_unavailable_cuda(self):
        with mock.patch.object(server_policy_utils.torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                server_policy_utils.resolve_server_device("cuda:0")

    def test_build_server_metadata_includes_cache_support_and_checkpoint_name(self):
        policy = _DummyPolicy()
        policy.get_inference_cache_stats = lambda session_id=None: {}

        metadata = server_policy_utils.build_server_metadata(
            vla=policy,
            ckpt_path="D:/models/checkpoints/demo_model.pt",
            device="cpu",
        )

        self.assertEqual("cpu", metadata["device"])
        self.assertEqual("demo_model.pt", metadata["checkpoint"])
        self.assertEqual("_DummyPolicy", metadata["framework"])
        self.assertTrue(metadata["supports_vlm_cache"])


if __name__ == "__main__":
    unittest.main()
