def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.get("template")
    if not vlm_name:
        vlm_name = config.framework.qwenvl.get("base_vlm", "")
    if not vlm_name:
        vlm_name = "Qwen3-VL"
    vlm_name_lower = vlm_name.lower()

    if "qwen2.5-vl" in vlm_name_lower or "qwen2.5vl" in vlm_name_lower or "nora" in vlm_name_lower:  # temp for some ckpt
        from .QWen2_5 import _QWen_VL_Interface

        return _QWen_VL_Interface(config)
    elif "qwen3-vl" in vlm_name_lower or "qwen3vl" in vlm_name_lower:
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    elif "qwen3.5" in vlm_name_lower:
        from .QWen3_5 import _QWen3_5_VL_Interface

        return _QWen3_5_VL_Interface(config)
    elif "gemma-4" in vlm_name_lower or "gemma4" in vlm_name_lower:
        from .Gemma4 import _Gemma4_VL_Interface

        return _Gemma4_VL_Interface(config)
    elif "molmo2" in vlm_name_lower:
        from .Molmo2 import _Molmo2_VL_Interface

        return _Molmo2_VL_Interface(config)
    elif "minicpm-v" in vlm_name_lower or "minicpmv" in vlm_name_lower:
        from .MiniCPM_V import _MiniCPM_VL_Interface

        return _MiniCPM_VL_Interface(config)
    elif "florence" in vlm_name_lower:  # temp for some ckpt
        from .Florence2 import _Florence_Interface

        return _Florence_Interface(config)
    elif "cosmos-reason2" in vlm_name_lower:
        # Cosmos-Reason2 is architecturally Qwen3-VL (VLM), but implemented
        # in world_model/ for historical reasons. Import directly.
        from starVLA.model.modules.vlm.CosmosReason2 import _CosmosReason2_Interface

        return _CosmosReason2_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")
