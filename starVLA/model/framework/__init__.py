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


try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# Auto-import all framework submodules to trigger registration
if pkg_path is not None:
    try:
        for _, module_name, _ in pkgutil.iter_modules(pkg_path):
            importlib.import_module(f"{__name__}.{module_name}")
    except Exception as e:
        print(f"Warning: Failed to auto-import framework submodules: {e}")
        
def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string (e.g. "InternVLA-M1")
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"): 
        cfg.framework.name = cfg.framework.framework_py  # Backward compatibility for legacy config yaml
        
    if cfg.framework.name == "QwenOFT":
        from starVLA.model.framework.QwenOFT import Qwenvl_OFT
        return Qwenvl_OFT(cfg)
    elif cfg.framework.name == "QwenFast":
        from starVLA.model.framework.QwenFast import Qwenvl_Fast
        return Qwenvl_Fast(cfg)

    
    # auto detect from registry
    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {cfg.framework.name} is not implemented.")
    
    MODLE_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODLE_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
