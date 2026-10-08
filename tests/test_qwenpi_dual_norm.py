"""CPU regression checks for the normalization primitives and call sites.

Extract the small source definitions to avoid loading VLM/video dependencies.
Run from the repository root: python -m unittest discover -s tests -p test_qwenpi_dual_norm.py
"""
import ast
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

SOURCE = Path(__file__).resolve().parents[1] / 'starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py'
TREE = ast.parse(SOURCE.read_text())
MLP_NODE = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'MLP')
HEAD = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'LayerwiseFlowmatchingActionHead')
NORMALIZE = next(n for n in HEAD.body if isinstance(n, ast.FunctionDef) and n.name == '_normalize_vl_features')
namespace = {'torch': torch, 'nn': nn, 'F': F}
exec(compile(ast.Module(body=[MLP_NODE, NORMALIZE], type_ignores=[]), str(SOURCE), 'exec'), namespace)
MLP = namespace['MLP']


class FusionProbe(nn.Module):
    _normalize_vl_features = namespace['_normalize_vl_features']

    def __init__(self, width, enabled):
        super().__init__()
        self.fusion_input_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6) if enabled else nn.Identity()


class TestDualNorm(unittest.TestCase):
    def test_decoder_disabled_preserves_outputs_and_gradients(self):
        torch.manual_seed(42)
        module = MLP(8, 12, 3)
        x = torch.randn(2, 5, 8, requires_grad=True)
        y = module(x)
        reference = module.layer2(F.relu(module.layer1(x)))
        torch.testing.assert_close(y, reference, rtol=0, atol=0)
        g1 = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
        g2 = torch.autograd.grad(reference.sum(), x)[0]
        torch.testing.assert_close(g1, g2, rtol=0, atol=0)

    def test_decoder_enabled_matches_runtime_hook(self):
        module = MLP(8, 12, 3, use_input_norm=True)
        x = torch.randn(2, 5, 8, requires_grad=True) * 100
        reference = module.layer2(F.relu(module.layer1(F.layer_norm(x, (8,), eps=1e-6))))
        torch.testing.assert_close(module(x), reference)
        self.assertTrue(torch.isfinite(torch.autograd.grad(module(x).sum(), x)[0]).all())

    def test_checkpoint_keys_and_roundtrip(self):
        old, new = MLP(8, 12, 3), MLP(8, 12, 3, use_input_norm=True)
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        new.load_state_dict(old.state_dict(), strict=True)
        restored = MLP(8, 12, 3, use_input_norm=True)
        restored.load_state_dict(new.state_dict(), strict=True)
        x = torch.randn(2, 5, 8)
        torch.testing.assert_close(new(x), restored(x))

    def test_fusion_each_token_and_layer_independently(self):
        x = torch.randn(2, 5, 8, requires_grad=True)
        other = torch.randn(2, 3, 8) * 1000
        enabled = FusionProbe(8, True)
        out = enabled._normalize_vl_features([x, other])
        torch.testing.assert_close(out[0], F.layer_norm(x, (8,), eps=1e-6))
        torch.testing.assert_close(out[1], F.layer_norm(other, (8,), eps=1e-6))
        changed = x.detach().clone(); changed[:, 1:] *= 99
        torch.testing.assert_close(enabled._normalize_vl_features([changed])[0][:, 0], out[0][:, 0])
        self.assertTrue(torch.isfinite(torch.autograd.grad(out[0].square().sum(), x)[0]).all())
        self.assertIs(FusionProbe(8, False)._normalize_vl_features([x])[0], x)
        self.assertEqual(len(enabled.state_dict()), 0)

    def test_training_and_sampling_paths_normalize_context(self):
        counts = {}
        for method in HEAD.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'model'
                     and isinstance(n.func.value, ast.Name) and n.func.value.id == 'self']
            for call in calls:
                context = next(k.value for k in call.keywords if k.arg == 'encoder_hidden_states')
                self.assertIsInstance(context, ast.Call)
                self.assertEqual(context.func.attr, '_normalize_vl_features')
            if calls:
                counts[method.name] = len(calls)
        self.assertEqual(counts, {'forward': 1, 'predict_action': 1, 'predict_action_realtime': 2})


if __name__ == '__main__':
    unittest.main()
