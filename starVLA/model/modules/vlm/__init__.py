def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm

    if "Qwen2.5-VL" in vlm_name or "nora" in vlm_name.lower():  # temp for some ckpt
        from .QWen2_5 import _QWen_VL_Interface

        return _QWen_VL_Interface(config)
    elif "Qwen3-VL" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    elif "Qwen3.5" in vlm_name:
        from .QWen3_5 import _QWen3_5_VL_Interface

        return _QWen3_5_VL_Interface(config)
    elif "florence" in vlm_name.lower():  # temp for some ckpt
        from .Florence2 import _Florence_Interface

        return _Florence_Interface(config)
    elif "cosmos-reason2" in vlm_name.lower():
        # Cosmos-Reason2 is a world model; route through world_model module
        from starVLA.model.modules.world_model import get_world_model

        return get_world_model(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")
