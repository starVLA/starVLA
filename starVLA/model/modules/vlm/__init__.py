import importlib

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# Speedups that leave every result bitwise identical (in deterministic mode, where the original code reproduces
# itself): work moved off the critical path, or work whose result nobody reads left out. Each one checks the model it
# is given, most of them with a probe against the original code on the running device, and returns False where it
# does not apply, so nothing is configured per recipe. Applied in this order to every backbone interface.
BACKBONE_SPEEDUPS = (
    "starVLA.model.modules.vlm.monkey_patch:patch_attention_mask",
    "starVLA.model.modules.vlm.monkey_patch:skip_unused_lm_head",
    "starVLA.model.modules.vlm.monkey_patch:batch_vision_attention",
    "starVLA.model.modules.vlm.monkey_patch:cache_vision_positions",
    "starVLA.dataloader.worker_prep:prepare_inputs_in_workers",
    "starVLA.dataloader.worker_prep:prepare_videos_in_workers",
)
_sped_up = set()  # ids of the Hugging Face models that went through apply_backbone_speedups


def run_speedups(target, speedups):
    """Call each "module:function" on `target`; returns the (applied, not applicable) function names."""
    applied, skipped = [], []
    for speedup in speedups:
        module, name = speedup.split(":")
        try:
            ok = getattr(importlib.import_module(module), name)(target)
        except Exception as error:  # a speedup must never stop a run that works without it
            logger.warning(f"{name}: left off ({type(error).__name__}: {error})")
            ok = False
        (applied if ok else skipped).append(name)
    if applied:
        logger.info(f"{type(target).__name__}: {', '.join(applied)}")
    return applied, skipped


class InnerBackbone:
    """A framework's own backbone model, presented as an interface (`.model`)."""

    def __init__(self, model):
        self.model = model


def apply_backbone_speedups(vlm):
    """BACKBONE_SPEEDUPS for a backbone interface (anything whose `.model` is the Hugging Face model). Returns `vlm`."""
    _sped_up.add(id(getattr(vlm, "model", vlm)))
    run_speedups(vlm, BACKBONE_SPEEDUPS)
    return vlm


def get_vlm_model(config):
    """Build the backbone interface for `config.framework.qwenvl.base_vlm`."""
    return apply_backbone_speedups(_build_vlm_model(config))


def _build_vlm_model(config):

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
    elif "gemma-4" in vlm_name.lower() or "gemma4" in vlm_name.lower():
        from .Gemma4 import _Gemma4_VL_Interface

        return _Gemma4_VL_Interface(config)
    elif "molmo2" in vlm_name.lower():
        from .Molmo2 import _Molmo2_VL_Interface

        return _Molmo2_VL_Interface(config)
    elif "minicpm-v" in vlm_name.lower() or "minicpmv" in vlm_name.lower():
        from .MiniCPM_V import _MiniCPM_VL_Interface

        return _MiniCPM_VL_Interface(config)
    elif "florence" in vlm_name.lower():  # temp for some ckpt
        from .Florence2 import _Florence_Interface

        return _Florence_Interface(config)
    elif "cosmos-reason2" in vlm_name.lower():
        # Cosmos-Reason2 is architecturally Qwen3-VL (VLM), but implemented
        # in world_model/ for historical reasons. Import directly.
        from starVLA.model.modules.vlm.CosmosReason2 import _CosmosReason2_Interface

        return _CosmosReason2_Interface(config)
    elif "egovla" in vlm_name.lower() or "ego_vla" in vlm_name.lower() or "vila" in vlm_name.lower():
        # EgoVLA (VILA): SigLIP-384 + Qwen2-1.5B + mlp_downsample projector.
        from starVLA.model.modules.vlm.VILA import _EgoVLA_VILA_Interface

        return _EgoVLA_VILA_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")
