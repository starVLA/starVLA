# starVLA/training/trainer_utils/config_tracker.py

from omegaconf import OmegaConf
from typing import Set, Any, Optional
import json
from pathlib import Path


class AccessTrackedConfig:
    """
    Wrapper for OmegaConf to track accessed parameters.
    Only saves configuration items that were actually accessed during execution.
    """
    
    _original_cfg_snapshot: Optional[OmegaConf] = None
    
    def __init__(self, cfg: OmegaConf, parent: 'AccessTrackedConfig' = None, key_path: str = ""):
        object.__setattr__(self, '_cfg', cfg)
        object.__setattr__(self, '_parent', parent)
        object.__setattr__(self, '_key_path', key_path)
        object.__setattr__(self, '_local_accessed', set())
        object.__setattr__(self, '_children', {})
        
        if parent is None:
            AccessTrackedConfig._original_cfg_snapshot = OmegaConf.create(
                OmegaConf.to_container(cfg, resolve=True)
            )
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        self._local_accessed.add(name)
        # Use safe access: for hasattr() semantics, raise AttributeError on missing keys
        try:
            value = self._cfg[name]
        except Exception:
            raise AttributeError(f"Config has no attribute '{name}'")
        
        if OmegaConf.is_config(value):
            new_path = f"{self._key_path}.{name}" if self._key_path else name
            if name not in self._children:
                self._children[name] = AccessTrackedConfig(value, parent=self, key_path=new_path)
            return self._children[name]
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)
    
    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._cfg[name] = value
    
    def __setitem__(self, key: str, value: Any):
        self._cfg[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator"""
        return key in self._cfg
    
    def __len__(self) -> int:
        """Return number of keys"""
        return len(self._cfg)
    
    def __iter__(self):
        """Support iteration (required for dict unpacking {**cfg})"""
        return iter(self._cfg)
    
    def keys(self):
        """Return config keys (required for dict unpacking)"""
        return self._cfg.keys()
    
    def values(self):
        """Return config values"""
        for key in self._cfg.keys():
            yield self.get(key)
    
    def items(self):
        """Return config items"""
        for key in self._cfg.keys():
            yield key, self.get(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback"""
        self._local_accessed.add(key)
        value = self._cfg.get(key, default)
        
        if value is not default and OmegaConf.is_config(value):
            new_path = f"{self._key_path}.{key}" if self._key_path else key
            if key not in self._children:
                self._children[key] = AccessTrackedConfig(value, parent=self, key_path=new_path)
            return self._children[key]
        
        return value
    
    def update(self, other: Any = None, **kwargs):
        """Update config with values from another dict/config"""
        if other is not None:
            # Handle different input types
            if isinstance(other, AccessTrackedConfig):
                other = OmegaConf.to_container(other._cfg, resolve=True)
            elif OmegaConf.is_config(other):
                other = OmegaConf.to_container(other, resolve=True)
            elif not isinstance(other, dict):
                # Try to convert to dict if possible
                other = dict(other)
            
            for key, value in other.items():
                self._local_accessed.add(key)
                self._cfg[key] = value
                # Invalidate child cache if exists
                if key in self._children:
                    del self._children[key]
        
        for key, value in kwargs.items():
            self._local_accessed.add(key)
            self._cfg[key] = value
            if key in self._children:
                del self._children[key]
    
    def pop(self, key: str, *args):
        """Remove and return a value"""
        self._local_accessed.add(key)
        if key in self._children:
            del self._children[key]
        if args:
            return self._cfg.pop(key, args[0])
        return self._cfg.pop(key)
    
    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set default value if key doesn't exist"""
        self._local_accessed.add(key)
        if key not in self._cfg:
            self._cfg[key] = default
        return self.get(key)
    
    def copy(self) -> 'AccessTrackedConfig':
        """Return a shallow copy"""
        new_cfg = OmegaConf.create(OmegaConf.to_container(self._cfg, resolve=True))
        return AccessTrackedConfig(new_cfg)
    
    def unwrap(self) -> OmegaConf:
        """Get the underlying OmegaConf object"""
        return self._cfg
    
    def get_root(self) -> 'AccessTrackedConfig':
        """Get root config object"""
        current = self
        while current._parent is not None:
            current = current._parent
        return current
    
    def _collect_all_paths(self, node: 'AccessTrackedConfig' = None, prefix: str = "") -> Set[str]:
        """Recursively collect all accessed paths"""
        if node is None:
            node = self.get_root()
        
        paths = set()
        for key in node._local_accessed:
            current_path = f"{prefix}.{key}" if prefix else key
            paths.add(current_path)
            if key in node._children:
                paths.update(self._collect_all_paths(node._children[key], current_path))
        return paths
    
    def _filter_leaf_paths(self, paths: Set[str]) -> Set[str]:
        """Filter to only leaf paths (no sub-paths)"""
        if not paths:
            return set()
        
        leaf_paths = set()
        for path in paths:
            if not any(other.startswith(f"{path}.") for other in paths if other != path):
                leaf_paths.add(path)
        return leaf_paths
    
    @staticmethod
    def _get_nested_value(cfg: OmegaConf, path: str) -> Any:
        """Get nested value through dot-separated path"""
        value = cfg
        for key in path.split('.'):
            value = value[key]
        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    
    @staticmethod
    def _set_nested_value(d: dict, path: str, value: Any):
        """Set nested value through dot-separated path"""
        keys = path.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    
    def export_accessed_config(self, use_original_values: bool = True) -> dict:
        """Export accessed configuration as dictionary (only leaf values)"""
        all_paths = self._collect_all_paths()
        leaf_paths = self._filter_leaf_paths(all_paths)
        source_cfg = AccessTrackedConfig._original_cfg_snapshot if use_original_values else self.get_root()._cfg
        
        result = {}
        for path in sorted(leaf_paths):
            try:
                value = self._get_nested_value(source_cfg, path)
                self._set_nested_value(result, path, value)
            except Exception:
                if use_original_values:
                    try:
                        value = self._get_nested_value(self.get_root()._cfg, path)
                        self._set_nested_value(result, path, value)
                    except Exception:
                        pass
        return result
    
    def save_accessed_config(self, filepath: Path, use_original_values: bool = True):
        """Save accessed configuration to file"""
        accessed_config = self.export_accessed_config(use_original_values=use_original_values)
        filepath = Path(filepath)
        
        with open(filepath, 'w') as f:
            if filepath.suffix == '.json':
                json.dump(accessed_config, f, indent=2)
            elif filepath.suffix == '.yaml':
                OmegaConf.save(OmegaConf.create(accessed_config), f)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def get_access_summary(self) -> dict:
        """Get summary of accessed configuration"""
        all_paths = self._collect_all_paths()
        leaf_paths = self._filter_leaf_paths(all_paths)
        
        return {
            "total_accessed_keys": len(all_paths),
            "leaf_accessed_keys": len(leaf_paths),
            "leaf_accessed_paths": sorted(leaf_paths),
            "top_level_keys": sorted(self.get_root()._local_accessed)
        }


def wrap_config(cfg: OmegaConf) -> AccessTrackedConfig:
    """Wrap OmegaConf configuration to enable access tracking"""
    return AccessTrackedConfig(cfg)


def unwrap_config(cfg) -> OmegaConf:
    """Unwrap AccessTrackedConfig to get underlying OmegaConf object"""
    return cfg.unwrap() if isinstance(cfg, AccessTrackedConfig) else cfg


# ========== Monkey Patch OmegaConf for Compatibility ==========

_original_to_container = OmegaConf.to_container
_original_save = OmegaConf.save
_original_to_yaml = OmegaConf.to_yaml
_original_is_config = OmegaConf.is_config


def _patched_to_container(cfg, resolve=True, enum_to_str=False, structured_config_mode=None):
    """Patched OmegaConf.to_container that handles AccessTrackedConfig"""
    if isinstance(cfg, AccessTrackedConfig):
        cfg = cfg.unwrap()
    
    try:
        if structured_config_mode is not None:
            return _original_to_container(cfg, resolve=resolve, enum_to_str=enum_to_str, 
                                         structured_config_mode=structured_config_mode)
        else:
            return _original_to_container(cfg, resolve=resolve, enum_to_str=enum_to_str)
    except TypeError:
        return _original_to_container(cfg, resolve=resolve)


def _patched_save(config, f, resolve=False):
    """Patched OmegaConf.save that handles AccessTrackedConfig"""
    if isinstance(config, AccessTrackedConfig):
        config = config.unwrap()
    return _original_save(config, f, resolve=resolve)


def _patched_to_yaml(cfg, resolve=False, sort_keys=False):
    """Patched OmegaConf.to_yaml that handles AccessTrackedConfig"""
    if isinstance(cfg, AccessTrackedConfig):
        cfg = cfg.unwrap()
    
    try:
        return _original_to_yaml(cfg, resolve=resolve, sort_keys=sort_keys)
    except TypeError:
        return _original_to_yaml(cfg, resolve=resolve)


def _patched_is_config(obj):
    """Patched OmegaConf.is_config that handles AccessTrackedConfig"""
    return True if isinstance(obj, AccessTrackedConfig) else _original_is_config(obj)


# Apply patches
OmegaConf.to_container = _patched_to_container
OmegaConf.save = _patched_save
OmegaConf.to_yaml = _patched_to_yaml
OmegaConf.is_config = _patched_is_config