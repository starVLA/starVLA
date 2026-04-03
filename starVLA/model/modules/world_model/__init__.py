def get_world_model(config):
    """Factory for world model backends.

    Routes to the correct world-model wrapper based on
    ``config.framework.world_model.base_wm`` (or falls back to
    ``config.framework.qwenvl.base_vlm`` for backward compatibility).

    Every world-model wrapper exposes the **same interface** as VLM
    wrappers (``forward``, ``generate``, ``build_qwenvl_inputs``),
    so frameworks can swap VLM ↔ WM transparently.
    """

    # Prefer explicit world_model config; fall back to qwenvl for compat
    wm_cfg = config.framework.get("world_model", None)
    if wm_cfg is not None:
        wm_name = wm_cfg.get("base_wm", "")
    else:
        wm_name = config.framework.qwenvl.base_vlm

    if "cosmos-reason2" in wm_name.lower():
        from .CosmosReason2 import _CosmosReason2_Interface

        return _CosmosReason2_Interface(config)
    else:
        raise NotImplementedError(f"World model {wm_name} not implemented")
