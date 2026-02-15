"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module (e.g., M1.py, QwenFast.py) should register itself:
    from starVLA.model.framework.framework_registry import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("InternVLA-M1")
    def build_model_framework(config):
        return InternVLA_M1(config=config)
"""

import pkgutil
import importlib
from starVLA.model.tools import FRAMEWORK_REGISTRY

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# Auto-import all framework submodules to trigger registration
if pkg_path is not None:
    for _, module_name, _ in pkgutil.iter_modules(pkg_path):
        if module_name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{module_name}")
        except Exception as e:
            logger.warning(f"Failed to auto-import framework submodule `{module_name}`: {e}")
        
def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string (e.g. "InternVLA-M1")
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg, "framework"):
        raise ValueError("Missing `cfg.framework` in configuration.")

    framework_id = getattr(cfg.framework, "name", None)
    if not framework_id:
        framework_id = getattr(cfg.framework, "framework_py", None)  # Backward compatibility for legacy config yaml
        if framework_id:
            cfg.framework.name = framework_id

    if not framework_id:
        raise ValueError("Missing framework identifier. Set `cfg.framework.name` (or legacy `framework_py`).")

    registry = FRAMEWORK_REGISTRY.list()
    if framework_id not in registry:
        raise NotImplementedError(
            f"Framework `{framework_id}` is not implemented. "
            "Make sure its module is importable and registers via FRAMEWORK_REGISTRY."
        )

    model_class = FRAMEWORK_REGISTRY[framework_id]
    return model_class(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
