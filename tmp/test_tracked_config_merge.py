"""
Regression test: merge_framework_config must invalidate AccessTrackedConfig
child cache so that newly-merged keys (e.g. action_model) are visible.
"""
from omegaconf import OmegaConf

from starVLA.model.framework.QwenOFT import QwenOFTDefaultConfig
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig

# Simulate the YAML that the LIBERO training script uses:
# framework section has name + qwenvl only (no action_model)
raw_cfg = OmegaConf.create({
    "framework": {
        "name": "QwenOFT",
        "qwenvl": {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
        },
    },
    "trainer": {"epochs": 1},
})

# Wrap in AccessTrackedConfig (same as train_starvla.py does)
tracked = AccessTrackedConfig(raw_cfg)

# merge_framework_config reads tracked.framework internally,
# caching a child.  It then writes the merged result back.
merged = merge_framework_config(QwenOFTDefaultConfig, tracked)

# The bug: after merge, tracked.framework still returned the STALE cache
# that had no action_model.  Now it should be fixed.
fw = merged.framework
print(f"framework keys: {list(fw._cfg.keys()) if hasattr(fw, '_cfg') else dir(fw)}")

assert hasattr(fw, "action_model"), "action_model missing after merge!"
am = fw.action_model
assert hasattr(am, "action_dim"), "action_dim missing in action_model!"
assert am.action_dim == 7
assert am.future_action_window_size == 15
print(f"action_model.action_dim = {am.action_dim}")
print(f"action_model.future_action_window_size = {am.future_action_window_size}")

# Also verify we can SET a value (QwenOFT.__init__ does this)
fw.action_model.action_hidden_dim = 9999
assert fw.action_model.action_hidden_dim == 9999
print(f"action_model.action_hidden_dim (set) = {fw.action_model.action_hidden_dim}")

print("\n=== All AccessTrackedConfig merge tests PASSED ===")

# bash 运行训练看看，确认没有问题