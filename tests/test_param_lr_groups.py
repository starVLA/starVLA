import unittest

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = nn.Sequential(nn.Linear(2, 2))
        self.vlm.head = nn.Linear(2, 2)
        self.action = nn.Linear(2, 2)


def _cfg(learning_rate):
    return OmegaConf.create({"trainer": {"learning_rate": learning_rate}})


class BuildParamLrGroupsTest(unittest.TestCase):
    def test_nested_module_keeps_its_own_lr(self):
        model = _Model()
        cfg = _cfg({"base": 1e-4, "vlm": 1e-5, "vlm.head": 1e-3})
        groups = build_param_lr_groups(model, cfg)
        torch.optim.AdamW(groups, lr=1e-4)

        by_name = {g["name"]: g for g in groups}
        self.assertEqual(by_name["vlm.head"]["lr"], 1e-3)
        self.assertEqual(len(by_name["vlm.head"]["params"]), 2)
        self.assertEqual(by_name["vlm"]["lr"], 1e-5)
        self.assertEqual(len(by_name["vlm"]["params"]), 2)
        self.assertEqual(by_name["base"]["lr"], 1e-4)
        self.assertEqual(len(by_name["base"]["params"]), 2)

        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), len(list(model.parameters())))


if __name__ == "__main__":
    unittest.main()
