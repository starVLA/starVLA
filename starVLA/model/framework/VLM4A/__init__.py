"""VLM4A — VLM-for-Action frameworks.

All VLA frameworks that use a Vision-Language Model (Qwen-VL, Florence, etc.)
as the perception backbone live here.

The registry decorator on each class registers them in FRAMEWORK_REGISTRY,
so `build_framework(cfg)` can find them after auto-import.
"""
