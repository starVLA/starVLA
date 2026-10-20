"""Device selection tests: mocked host detection, dispatch, and one XPU parity check.

Run from the repository root:
    python -m pytest -q tests/test_device_agnostic.py
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from starVLA.model import device as dev

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


@contextmanager
def fake_host(xpu=False, cuda=False):
    with mock.patch.object(dev, "is_available", side_effect=lambda t: {"xpu": xpu, "cuda": cuda}.get(t, t == "cpu")):
        yield


class DeviceModuleTest(unittest.TestCase):
    def test_default_prefers_xpu_then_cuda_then_cpu(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
        with fake_host(cuda=True):
            self.assertEqual(dev.default_device_type(), "cuda")
        with fake_host():
            self.assertEqual(dev.default_device_type(), "cpu")

    def test_explicit_device_wins(self):
        with fake_host(xpu=True):
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda", 1))
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("cuda"), torch.device("cuda", 0))
            self.assertEqual(dev.resolve_device(None), torch.device("xpu", 0))
        with fake_host(cuda=True):
            self.assertEqual(dev.resolve_device(None), torch.device("cuda", 0))

    def test_autocast_is_disabled_on_cpu_and_for_float32_on_xpu(self):
        with dev.autocast("cpu", torch.bfloat16):
            self.assertFalse(torch.is_autocast_enabled("cpu"))
        with dev.autocast("xpu", torch.float32):
            self.assertFalse(torch.is_autocast_enabled("cpu"))

    def test_cast_like_only_acts_off_cuda(self):
        head = nn.Linear(2, 2)  # float32
        x = torch.ones(1, 2, dtype=torch.bfloat16)
        self.assertEqual(dev.cast_like(head, x, "cuda").dtype, torch.bfloat16)
        self.assertEqual(dev.cast_like(head, x, "xpu").dtype, torch.float32)
        self.assertEqual([t.dtype for t in dev.cast_like(head, [x, x], "cpu")], [torch.float32] * 2)
        self.assertIsNone(dev.cast_like(head, None, "xpu"))

    def test_manual_seed_all_seeds_the_present_backends(self):
        with fake_host(xpu=True), mock.patch.object(torch.xpu, "manual_seed_all") as seed:
            dev.manual_seed_all(3)
        seed.assert_called_once_with(3)


class DeploymentEntryPointsTest(unittest.TestCase):
    def test_server_device_defaults_to_detection(self):
        from deployment.model_server.server_policy import build_argparser

        self.assertIsNone(build_argparser().parse_args([]).device)
        self.assertEqual(build_argparser().parse_args(["--device", "xpu"]).device, "xpu")

    def _load_wrapper(self, device, **host):
        from deployment.model_server import policy_wrapper

        framework = mock.MagicMock()
        framework.to.return_value = framework
        framework.eval.return_value = framework
        cfg = {"framework": {"action_model": {"action_horizon": 8}}}
        with fake_host(**host), \
                mock.patch.object(policy_wrapper.baseframework, "from_pretrained", return_value=framework), \
                mock.patch.object(policy_wrapper, "read_mode_config", return_value=(cfg, {"a": {}})), \
                mock.patch.object(policy_wrapper, "merge_config_overrides", return_value=cfg), \
                mock.patch.object(policy_wrapper, "PolicyNormProcessor"):
            policy_wrapper.PolicyServerWrapper("ckpt.pt", device=device)
        return framework.to.call_args.args[0]

    def test_wrapper_places_the_model_on_the_resolved_device(self):
        self.assertEqual(self._load_wrapper(None, xpu=True), torch.device("xpu", 0))
        self.assertEqual(self._load_wrapper(None, cuda=True), torch.device("cuda", 0))
        self.assertEqual(self._load_wrapper(None), torch.device("cpu"))
        self.assertEqual(self._load_wrapper("cpu", xpu=True), torch.device("cpu"))

    def test_seed_helper_routes_through_the_device_module(self):
        from deployment.model_server import seed_utils

        with mock.patch.object(seed_utils, "manual_seed_all") as seed:
            seed_utils.set_seed_everywhere(5)
        seed.assert_called_once_with(5)


class _TinyBackbone(nn.Sequential):
    def __init__(self, hidden, vocab):
        super().__init__(nn.Embedding(vocab, hidden), nn.Linear(hidden, hidden))
        self.config = SimpleNamespace(hidden_size=hidden)

    @property
    def device(self):
        return self[0].weight.device


class _TinyQwenInterface(nn.Module):
    """Stand-in for the Qwen wrapper: token embeddings, one layer, hidden states out."""

    def __init__(self, hidden=32, vocab=16):
        super().__init__()
        self.model = _TinyBackbone(hidden, vocab)

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        device = self.model[0].weight.device
        ids = torch.tensor([[1, 2, 3] + [7] * 4 + [4]] * len(instructions), device=device)
        return {"input_ids": ids}

    def forward(self, input_ids, **kwargs):
        hidden = torch.tanh(self.model[1](self.model[0](input_ids)))
        return SimpleNamespace(hidden_states=[hidden])


def _tiny_oft_framework(dtype):
    from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
    from starVLA.model.modules.action_model.MLP_ActionHeader import L1RegressionActionHead

    fw = Qwenvl_OFT.__new__(Qwenvl_OFT)
    nn.Module.__init__(fw)
    fw.qwen_vl_interface = _TinyQwenInterface()
    fw.action_model = L1RegressionActionHead(input_dim=32, hidden_dim=64, action_dim=7, NUM_ACTIONS_CHUNK=4)
    fw.config = SimpleNamespace(datasets=SimpleNamespace(vla_data=SimpleNamespace(obs_image_size=None)))
    fw.chunk_len, fw.action_horizon = 4, 4
    fw.action_token, fw.action_token_id = "x", 7
    with torch.no_grad():  # no zero-initialised layer may hide a difference
        for p in fw.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    return fw.to(dtype).eval()


class OftPredictActionTest(unittest.TestCase):
    def _predict(self, fw):
        with mock.patch("starVLA.model.framework.VLM4A.QwenOFT.to_pil_preserve", side_effect=lambda x: x):
            return fw.predict_action([{"image": [object()], "lang": "pick"}])["normalized_actions"]

    def test_bf16_model_on_cpu_returns_a_float32_numpy_chunk(self):
        out = self._predict(_tiny_oft_framework(torch.bfloat16))
        self.assertEqual(out.shape, (1, 4, 7))
        self.assertEqual(out.dtype, "float32")

    @unittest.skipUnless(HAS_XPU, "needs an Intel XPU")
    def test_predict_action_on_xpu_matches_cpu(self):
        torch.manual_seed(0)
        fw = _tiny_oft_framework(torch.bfloat16)
        ref = self._predict(fw)
        out = self._predict(fw.to("xpu"))
        self.assertGreater(float(ref.std()), 0.0)
        torch.testing.assert_close(torch.from_numpy(out), torch.from_numpy(ref), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
