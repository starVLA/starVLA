import json
from pathlib import Path
from typing import Any, Optional, Set, Union

from omegaconf import DictConfig, ListConfig, OmegaConf


class AccessTrackedConfig:
    """
    Wrapper for OmegaConf to track accessed parameters.
    Only saves configuration items that were actually accessed during execution.

    Tracking rules:
      - __getattr__ / __getitem__ / get()  →  marks key as accessed (leaf read)
      - __setattr__ / __setitem__ / update()  →  marks key as accessed (write)
      - __iter__ on DictConfig / keys()  →  does NOT mark (navigation only)
      - __contains__  →  does NOT mark (existence check)
      - OmegaConf.to_container()  →  marks entire subtree (bulk consumption)
    """

    _original_cfg_snapshot: Optional[OmegaConf] = None
    _cli_overrides: list = []  # CLI override dotlist (e.g. ["framework.name=X"])

    def __init__(
        self,
        cfg: Union[DictConfig, ListConfig],
        parent: "AccessTrackedConfig" = None,
        key_path: str = "",
        cli_overrides: list = None,
    ):
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_key_path", key_path)
        object.__setattr__(self, "_local_accessed", set())
        object.__setattr__(self, "_children", {})

        if parent is None:
            AccessTrackedConfig._original_cfg_snapshot = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            AccessTrackedConfig._cli_overrides = list(cli_overrides) if cli_overrides else []

    def _is_list_config(self) -> bool:
        """Check if underlying config is a ListConfig"""
        return isinstance(self._cfg, ListConfig)

    def _is_dict_config(self) -> bool:
        """Check if underlying config is a DictConfig"""
        return isinstance(self._cfg, DictConfig)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        # Access underlying config first — raise AttributeError on missing keys
        # so hasattr() works correctly WITHOUT marking non-existent keys.
        try:
            value = self._cfg[name]
        except Exception:
            raise AttributeError(f"Config has no attribute '{name}'")

        # Mark AFTER successful access so only existing & read keys are tracked.
        self._local_accessed.add(name)

        if OmegaConf.is_config(value):
            new_path = f"{self._key_path}.{name}" if self._key_path else name
            if name not in self._children:
                self._children[name] = AccessTrackedConfig(value, parent=self, key_path=new_path)
            return self._children[name]

        return value

    def __getitem__(self, key) -> Any:
        """Support both dict-style and list-style access"""
        if isinstance(key, int):
            # List-style access
            self._local_accessed.add(f"[{key}]")
            value = self._cfg[key]
            if OmegaConf.is_config(value):
                new_path = f"{self._key_path}[{key}]" if self._key_path else f"[{key}]"
                cache_key = f"[{key}]"
                if cache_key not in self._children:
                    self._children[cache_key] = AccessTrackedConfig(value, parent=self, key_path=new_path)
                return self._children[cache_key]
            return value
        else:
            # Dict-style access
            return self.__getattr__(key)

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._local_accessed.add(name)
            self._cfg[name] = value
            # Invalidate child cache if exists
            if name in self._children:
                del self._children[name]

    def __setitem__(self, key, value: Any):
        """Support both dict-style and list-style setting"""
        if isinstance(key, int):
            self._local_accessed.add(f"[{key}]")
            self._cfg[key] = value
            cache_key = f"[{key}]"
            if cache_key in self._children:
                del self._children[cache_key]
        else:
            self._local_accessed.add(key)
            self._cfg[key] = value
            if key in self._children:
                del self._children[key]

    def __contains__(self, key) -> bool:
        """Support 'in' operator — does NOT mark as accessed (existence check only)."""
        return key in self._cfg

    def __len__(self) -> int:
        """Return number of keys/items"""
        return len(self._cfg)

    def __iter__(self):
        """Support iteration for both DictConfig and ListConfig.

        DictConfig: yields key names only — does NOT mark as accessed.
            Keys are marked later when their values are actually read
            via __getattr__ / __getitem__.
        ListConfig: yields element values — marks all indices as accessed
            because the caller is consuming every element.
        """
        if self._is_list_config():
            for i in range(len(self._cfg)):
                self._local_accessed.add(f"[{i}]")
            return iter(self._cfg)
        else:
            # Navigation only — don't mark
            return iter(self._cfg)

    def __repr__(self) -> str:
        """String representation"""
        if self._is_list_config():
            return f"AccessTrackedConfig({self._key_path or 'root'}, list_len={len(self._cfg)})"
        return f"AccessTrackedConfig({self._key_path or 'root'}, keys={list(self._cfg.keys())})"

    def __str__(self) -> str:
        """String representation"""
        return OmegaConf.to_yaml(self._cfg)

    def __bool__(self) -> bool:
        """Boolean evaluation - True if config has any keys/items"""
        return len(self._cfg) > 0

    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if isinstance(other, AccessTrackedConfig):
            return self._cfg == other._cfg
        elif OmegaConf.is_config(other):
            return self._cfg == other
        elif isinstance(other, (dict, list)):
            return OmegaConf.to_container(self._cfg, resolve=True) == other
        return False

    def keys(self):
        """Return config keys — does NOT mark as accessed (navigation only).
        Keys are tracked when their values are actually consumed via __getattr__ etc.
        """
        if self._is_list_config():
            raise TypeError("ListConfig does not support keys()")
        return self._cfg.keys()

    def values(self):
        """Return config values — marks via get()/getitem() when yielded."""
        if self._is_list_config():
            for i in range(len(self._cfg)):
                self._local_accessed.add(f"[{i}]")
                yield self[i]
        else:
            for key in self._cfg.keys():
                yield self.get(key)  # get() marks the key

    def items(self):
        """Return config items — marks via get() when yielded."""
        if self._is_list_config():
            raise TypeError("ListConfig does not support items()")
        for key in self._cfg.keys():
            yield key, self.get(key)  # get() marks the key

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
        if self._is_list_config():
            raise TypeError("ListConfig does not support update()")

        if other is not None:
            # Handle different input types
            if isinstance(other, AccessTrackedConfig):
                other = OmegaConf.to_container(other._cfg, resolve=True)
            elif OmegaConf.is_config(other):
                other = OmegaConf.to_container(other, resolve=True)
            elif hasattr(other, "items"):
                # Dict-like object
                other = dict(other.items())
            elif hasattr(other, "__iter__"):
                # Iterable of key-value pairs
                other = dict(other)
            else:
                raise TypeError(f"Cannot update from {type(other)}")

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

    def pop(self, key, *args):
        """Remove and return a value"""
        if isinstance(key, int):
            self._local_accessed.add(f"[{key}]")
            cache_key = f"[{key}]"
        else:
            self._local_accessed.add(key)
            cache_key = key

        if cache_key in self._children:
            del self._children[cache_key]
        if args:
            return self._cfg.pop(key, args[0])
        return self._cfg.pop(key)

    def append(self, value: Any):
        """Append value to list (only for ListConfig)"""
        if not self._is_list_config():
            raise TypeError("append() only supported for ListConfig")
        self._cfg.append(value)
        idx = len(self._cfg) - 1
        self._local_accessed.add(f"[{idx}]")

    def extend(self, values):
        """Extend list with values (only for ListConfig)"""
        if not self._is_list_config():
            raise TypeError("extend() only supported for ListConfig")
        start_idx = len(self._cfg)
        self._cfg.extend(values)
        for i in range(start_idx, len(self._cfg)):
            self._local_accessed.add(f"[{i}]")

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set default value if key doesn't exist"""
        if self._is_list_config():
            raise TypeError("ListConfig does not support setdefault()")
        self._local_accessed.add(key)
        if key not in self._cfg:
            self._cfg[key] = default
        return self.get(key)

    def copy(self) -> "AccessTrackedConfig":
        """Return a shallow copy (does not copy access tracking state)"""
        new_cfg = OmegaConf.create(OmegaConf.to_container(self._cfg, resolve=True))
        return AccessTrackedConfig(new_cfg)

    def deepcopy(self) -> "AccessTrackedConfig":
        """Return a deep copy (does not copy access tracking state)"""
        new_cfg = OmegaConf.create(OmegaConf.to_container(self._cfg, resolve=True))
        return AccessTrackedConfig(new_cfg)

    def merge_with(self, *others) -> "AccessTrackedConfig":
        """Merge with other configs and return new tracked config"""
        configs = [self._cfg]
        for other in others:
            if isinstance(other, AccessTrackedConfig):
                configs.append(other._cfg)
            elif OmegaConf.is_config(other):
                configs.append(other)
            else:
                configs.append(OmegaConf.create(other))

        merged = OmegaConf.merge(*configs)
        return AccessTrackedConfig(merged)

    def to_dict(self, resolve: bool = True) -> dict:
        """Convert to plain dictionary or list"""
        return OmegaConf.to_container(self._cfg, resolve=resolve)

    def to_yaml(self, resolve: bool = False) -> str:
        """Convert to YAML string"""
        return OmegaConf.to_yaml(self._cfg, resolve=resolve)

    def unwrap(self) -> Union[DictConfig, ListConfig]:
        """Get the underlying OmegaConf object"""
        return self._cfg

    def get_root(self) -> "AccessTrackedConfig":
        """Get root config object"""
        current = self
        while current._parent is not None:
            current = current._parent
        return current

    def _mark_all_accessed(self):
        """Recursively mark every key/index in this subtree as accessed.

        Called automatically by the patched ``OmegaConf.to_container()`` so that
        bulk-consumption patterns like ``OmegaConf.to_container(cfg.trainer.loss_scale)``
        correctly record all leaf parameters.
        """
        if self._is_dict_config():
            for key in self._cfg.keys():
                self._local_accessed.add(key)
                value = self._cfg[key]
                if OmegaConf.is_config(value):
                    new_path = f"{self._key_path}.{key}" if self._key_path else key
                    if key not in self._children:
                        self._children[key] = AccessTrackedConfig(value, parent=self, key_path=new_path)
                    self._children[key]._mark_all_accessed()
        elif self._is_list_config():
            for i in range(len(self._cfg)):
                cache_key = f"[{i}]"
                self._local_accessed.add(cache_key)
                value = self._cfg[i]
                if OmegaConf.is_config(value):
                    new_path = f"{self._key_path}{cache_key}" if self._key_path else cache_key
                    if cache_key not in self._children:
                        self._children[cache_key] = AccessTrackedConfig(value, parent=self, key_path=new_path)
                    self._children[cache_key]._mark_all_accessed()

    def _collect_all_paths(self, node: "AccessTrackedConfig" = None, prefix: str = "") -> Set[str]:
        """Recursively collect all accessed paths"""
        if node is None:
            node = self.get_root()

        paths = set()
        for key in node._local_accessed:
            current_path = (
                f"{prefix}.{key}" if prefix and not key.startswith("[") else f"{prefix}{key}" if prefix else key
            )
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
            # Check if any other path starts with this path followed by . or [
            is_leaf = True
            for other in paths:
                if other != path:
                    if other.startswith(f"{path}.") or other.startswith(f"{path}["):
                        is_leaf = False
                        break
            if is_leaf:
                leaf_paths.add(path)
        return leaf_paths

    @staticmethod
    def _get_nested_value(cfg, path: str) -> Any:
        """Get nested value through dot-separated path with bracket notation support"""
        import re

        value = cfg
        # Split by . but keep bracket notation together
        parts = re.split(r"\.(?![^\[]*\])", path)
        for part in parts:
            # Handle bracket notation like [0]
            bracket_match = re.match(r"\[(\d+)\]", part)
            if bracket_match:
                idx = int(bracket_match.group(1))
                value = value[idx]
            elif "[" in part:
                # Handle cases like "key[0]"
                key_part, rest = part.split("[", 1)
                if key_part:
                    value = value[key_part]
                indices = re.findall(r"\[(\d+)\]", "[" + rest)
                for idx_str in indices:
                    value = value[int(idx_str)]
            else:
                value = value[part]

        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

    @staticmethod
    def _set_nested_value(d: dict, path: str, value: Any):
        """Set nested value through dot-separated path"""
        import re

        parts = re.split(r"\.(?![^\[]*\])", path)

        for i, part in enumerate(parts[:-1]):
            bracket_match = re.match(r"\[(\d+)\]", part)
            if bracket_match:
                idx = int(bracket_match.group(1))
                while len(d) <= idx:
                    d.append({})
                d = d[idx]
            elif "[" in part:
                key_part, rest = part.split("[", 1)
                if key_part:
                    d = d.setdefault(key_part, {})
                indices = re.findall(r"\[(\d+)\]", "[" + rest)
                for idx_str in indices:
                    idx = int(idx_str)
                    if isinstance(d, list):
                        while len(d) <= idx:
                            d.append({})
                        d = d[idx]
                    else:
                        d = d.setdefault(idx, {})
            else:
                d = d.setdefault(part, {})

        # Set final value
        last_part = parts[-1]
        bracket_match = re.match(r"\[(\d+)\]", last_part)
        if bracket_match:
            idx = int(bracket_match.group(1))
            while len(d) <= idx:
                d.append(None)
            d[idx] = value
        elif "[" in last_part:
            key_part, rest = last_part.split("[", 1)
            if key_part:
                d = d.setdefault(key_part, [])
            indices = re.findall(r"\[(\d+)\]", "[" + rest)
            for idx_str in indices[:-1]:
                idx = int(idx_str)
                while len(d) <= idx:
                    d.append([])
                d = d[idx]
            final_idx = int(indices[-1])
            while len(d) <= final_idx:
                d.append(None)
            d[final_idx] = value
        else:
            d[last_part] = value

    def export_accessed_config(self, use_original_values: bool = True) -> dict:
        """Export accessed configuration as dictionary (only leaf values).

        Includes:
        - All leaf parameters that were read via __getattr__ / __getitem__ / get()
        - All leaf parameters inside subtrees consumed via OmegaConf.to_container()
        - All CLI override paths (always included regardless of access tracking)
        """
        all_paths = self._collect_all_paths()
        leaf_paths = self._filter_leaf_paths(all_paths)

        # Always include CLI-overridden parameters
        for override in AccessTrackedConfig._cli_overrides:
            if "=" in override:
                path = override.split("=", 1)[0]
                leaf_paths.add(path)

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

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            if filepath.suffix == ".json":
                json.dump(accessed_config, f, indent=2)
            elif filepath.suffix in (".yaml", ".yml"):
                OmegaConf.save(OmegaConf.create(accessed_config), f)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def save_full_config(self, filepath: Path):
        """Save the complete configuration (all parameters) to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        source = AccessTrackedConfig._original_cfg_snapshot
        if source is None:
            source = self.get_root()._cfg
        with open(filepath, "w") as f:
            if filepath.suffix == ".json":
                json.dump(_original_to_container(source, resolve=True), f, indent=2)
            elif filepath.suffix in (".yaml", ".yml"):
                _original_save(source, f, resolve=True)
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
            "top_level_keys": sorted(self.get_root()._local_accessed),
        }

    def print_access_summary(self):
        """Print a formatted summary of accessed configuration"""
        summary = self.get_access_summary()
        print(f"\n{'='*60}")
        print("Configuration Access Summary")
        print(f"{'='*60}")
        print(f"Total accessed keys: {summary['total_accessed_keys']}")
        print(f"Leaf accessed keys: {summary['leaf_accessed_keys']}")
        print(f"\nTop-level keys accessed: {summary['top_level_keys']}")
        print("\nLeaf paths accessed:")
        for path in summary["leaf_accessed_paths"]:
            print(f"  - {path}")
        print(f"{'='*60}\n")


def wrap_config(cfg: OmegaConf, cli_overrides: list = None) -> AccessTrackedConfig:
    """Wrap OmegaConf configuration to enable access tracking.

    Args:
        cfg: The OmegaConf config (already merged with CLI overrides).
        cli_overrides: Optional list of CLI dotlist strings (e.g. ["framework.name=X"]).
            These paths are always included in the exported snapshot.
    """
    if isinstance(cfg, AccessTrackedConfig):
        return cfg  # Already wrapped — idempotent
    return AccessTrackedConfig(cfg, cli_overrides=cli_overrides)


def unwrap_config(cfg) -> OmegaConf:
    """Unwrap AccessTrackedConfig to get underlying OmegaConf object"""
    return cfg.unwrap() if isinstance(cfg, AccessTrackedConfig) else cfg


# ========== Monkey Patch OmegaConf for Compatibility ==========

_original_to_container = OmegaConf.to_container
_original_save = OmegaConf.save
_original_to_yaml = OmegaConf.to_yaml
_original_is_config = OmegaConf.is_config
_original_merge = OmegaConf.merge


def _patched_to_container(cfg, resolve=True, enum_to_str=False, structured_config_mode=None):
    """Patched OmegaConf.to_container that handles AccessTrackedConfig.

    When called on a tracked config, marks the entire subtree as accessed
    because to_container consumes all values (e.g. for passing to external code).
    """
    if isinstance(cfg, AccessTrackedConfig):
        cfg._mark_all_accessed()
        cfg = cfg.unwrap()

    try:
        if structured_config_mode is not None:
            return _original_to_container(
                cfg, resolve=resolve, enum_to_str=enum_to_str, structured_config_mode=structured_config_mode
            )
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


def _patched_merge(*configs):
    """Patched OmegaConf.merge that handles AccessTrackedConfig"""
    unwrapped_configs = []
    for cfg in configs:
        if isinstance(cfg, AccessTrackedConfig):
            unwrapped_configs.append(cfg.unwrap())
        else:
            unwrapped_configs.append(cfg)
    return _original_merge(*unwrapped_configs)


# Apply patches
OmegaConf.to_container = _patched_to_container
OmegaConf.save = _patched_save
OmegaConf.to_yaml = _patched_to_yaml
OmegaConf.is_config = _patched_is_config
OmegaConf.merge = _patched_merge
