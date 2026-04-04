def get_world_model(config):
    """Factory for world model backends.

    Routes to the correct world-model wrapper based on
    ``config.framework.world_model.base_wm`` (or falls back to
    ``config.framework.qwenvl.base_vlm`` for backward compatibility).

    Every world-model wrapper exposes:
      - ``forward(**kwargs)`` → model outputs with hidden_states
      - ``build_inputs(images, instructions)`` → dict of tensors
      - ``generate(**kwargs)`` → generation (optional)
    """

    # Prefer explicit world_model config; fall back to qwenvl for compat
    wm_cfg = config.framework.get("world_model", None)
    if wm_cfg is not None:
        wm_name = wm_cfg.get("base_wm", "")
    else:
        wm_name = config.framework.qwenvl.base_vlm

    if "cosmos-reason2" in wm_name.lower():
        from ..vlm.CosmosReason2 import _CosmosReason2_Interface

        return _CosmosReason2_Interface(config)
    elif "cosmos-predict2" in wm_name.lower() or "cosmos-predict2" in wm_name.lower():
        from .CosmoPredict2 import _CosmoPredict2_Interface

        return _CosmoPredict2_Interface(config)
    elif "wan2" in wm_name.lower() or "ti2v" in wm_name.lower():
        from .Wan2 import _Wan2_Interface

        return _Wan2_Interface(config)
    else:
        raise NotImplementedError(f"World model {wm_name} not implemented")
