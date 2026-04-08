import unittest
from unittest import mock
import importlib

from deployment.model_server import server_policy_utils


class _DummyPolicy:
    pass


class ServerPolicyTests(unittest.TestCase):
    def test_server_policy_module_imports_without_optional_framework_dependencies(self):
        module = importlib.import_module("deployment.model_server.server_policy")
        self.assertTrue(hasattr(module, "main"))

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
