"""Test all framework DefaultConfig classes."""
from dataclasses import asdict
from omegaconf import OmegaConf

from starVLA.model.framework.QwenOFT import QwenOFTDefaultConfig
from starVLA.model.framework.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.QwenPI import QwenPIDefaultConfig
from starVLA.model.framework.QwenFast import QwenFastDefaultConfig
from starVLA.model.framework.QwenAdapter import QwenAdapterDefaultConfig
from starVLA.model.framework.QwenDual import QwenDualDefaultConfig
from starVLA.model.framework.LangForce import LangForceDefaultConfig
from starVLA.model.framework.M1 import InternVLA_M1DefaultConfig
from starVLA.model.framework.ABot_M0 import ABot_M0DefaultConfig
from starVLA.model.framework.share_tools import merge_framework_config

configs = [
    QwenOFTDefaultConfig, QwenGR00TDefaultConfig, QwenPIDefaultConfig,
    QwenFastDefaultConfig, QwenAdapterDefaultConfig, QwenDualDefaultConfig,
    LangForceDefaultConfig, InternVLA_M1DefaultConfig, ABot_M0DefaultConfig,
]

print("=== Test 1: All imports OK ===")

for cls in configs:
    inst = cls()
    d = asdict(inst)
    omega = OmegaConf.create(d)
    assert omega.name == inst.name
    print(f"  {cls.__name__}: name={inst.name}, keys={list(d.keys())}")

print("=== Test 2: All DefaultConfigs OK ===")

for cls in configs:
    inst = cls()
    yaml_cfg = OmegaConf.create({"framework": {"name": inst.name}})
    merged = merge_framework_config(cls, yaml_cfg)
    assert hasattr(merged.framework, "qwenvl"), f"{cls.__name__} missing qwenvl"
    print(f"  merge OK: {cls.__name__}")

print("=== Test 3: All merges OK ===")

# Test 4: YAML override works
yaml_cfg = OmegaConf.create({"framework": {"name": "QwenOFT", "action_model": {"action_dim": 14}}})
merged = merge_framework_config(QwenOFTDefaultConfig, yaml_cfg)
assert merged.framework.action_model.action_dim == 14
assert merged.framework.action_model.future_action_window_size == 15
print("=== Test 4: Override + default preservation OK ===")

# Test 5: LangForce top-level fields
yaml_cfg = OmegaConf.create({"framework": {"name": "LangForce", "kl_weight": 0.5}})
merged = merge_framework_config(LangForceDefaultConfig, yaml_cfg)
assert merged.framework.kl_weight == 0.5
assert merged.framework.prior_loss_weight == 0.3
print("=== Test 5: LangForce top-level fields OK ===")

# Test 6: Extra YAML key preservation
yaml_cfg = OmegaConf.create({"framework": {"name": "QwenGR00T", "custom_key": 42}})
merged = merge_framework_config(QwenGR00TDefaultConfig, yaml_cfg)
assert merged.framework.custom_key == 42
print("=== Test 6: Extra YAML key preservation OK ===")

print("\nALL TESTS PASSED ✓")
