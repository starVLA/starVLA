"""Memoized position tables for the Qwen vision towers (and, with transformers 5, every tower built on vision_utils).

The vision models of Qwen2-VL, Qwen2.5-VL and Qwen3-VL compute, in every forward and with a Python loop over the
images that reads GPU scalars back to the host, things that depend only on the grid sizes (grid_thw), not on any
weight:

  rot_pos_emb(grid_thw)                 the rotary position table of every patch (a lookup in a fixed buffer)
  get_window_index(grid_thw)            Qwen2.5-VL: the window order of the patches and the window boundaries
  fast_pos_embed_interpolate(grid_thw)  Qwen3-VL: the four neighbour indices and bilinear weights of every patch

In training every batch has the same grids (all views are resized to one resolution), so each is built once per
distinct grid_thw and reused. Of fast_pos_embed_interpolate only the indices and weights are reused: the part that
touches the trainable position embedding (lookup, weighting, the sum of the four neighbours, the per-image
reordering) runs every time exactly as upstream, so its values and gradients are the upstream ones. About 27 ms of
host time per step at 32 images on Qwen3-VL.
"""
import sys

import torch

_installed = set()
_MAX_ENTRIES = 16


def _grid_key(grid_thw):
    return tuple(tuple(row) for row in grid_thw.tolist())  # one host read per forward


def _remember(cache, key, value):
    if len(cache) >= _MAX_ENTRIES:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


def _memoize_rot_pos_emb(model_class):
    original_rot = model_class.rot_pos_emb
    rot_cache = {}

    def rot_pos_emb(self, grid_thw):
        inv_freq = self.rotary_pos_emb.inv_freq
        key = (_grid_key(grid_thw), str(inv_freq.device), str(inv_freq.dtype), id(self))
        hit = rot_cache.get(key)
        if hit is None:
            hit = _remember(rot_cache, key, original_rot(self, grid_thw).detach())
        return hit

    rot_pos_emb._starvla_original = original_rot
    model_class.rot_pos_emb = rot_pos_emb


def _memoize_window_index(model_class):
    original = model_class.get_window_index
    cache = {}

    def get_window_index(self, grid_thw):
        key = (_grid_key(grid_thw), id(self))
        hit = cache.get(key)
        if hit is None:
            hit = _remember(cache, key, original(self, grid_thw))
        window_index, cu_window_seqlens = hit
        return window_index, list(cu_window_seqlens)

    get_window_index._starvla_original = original
    model_class.get_window_index = get_window_index


def _install_qwen3(qwen3):
    _memoize_rot_pos_emb(qwen3.Qwen3VLVisionModel)
    interp_cache = {}

    def interp_indices(self, grid):
        """idx_tensor and weight_tensor of the upstream function, built from host integers (same arithmetic)."""
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in grid:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        return idx_tensor, weight_tensor

    def fast_pos_embed_interpolate(self, grid_thw):
        grid = _grid_key(grid_thw)
        weight = self.pos_embed.weight
        key = (grid, str(weight.device), str(weight.dtype), self.num_grid_per_side)
        hit = interp_cache.get(key)
        if hit is None:
            hit = _remember(interp_cache, key, interp_indices(self, grid))
        idx_tensor, weight_tensor = hit

        # from here on: upstream, with host integers for the grid sizes
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for _, h, w in grid])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, (t, h, w) in zip(patch_pos_embeds, grid):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    fast_pos_embed_interpolate._starvla_original = qwen3.Qwen3VLVisionModel.fast_pos_embed_interpolate
    qwen3.Qwen3VLVisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate


def _install_qwen2_5(qwen2_5):
    _memoize_rot_pos_emb(qwen2_5.Qwen2_5_VisionTransformerPretrainedModel)
    _memoize_window_index(qwen2_5.Qwen2_5_VisionTransformerPretrainedModel)


def _install_qwen2(qwen2):
    _memoize_rot_pos_emb(qwen2.Qwen2VisionTransformerPretrainedModel)


_FAMILIES = {
    "transformers.models.qwen3_vl.modeling_qwen3_vl": _install_qwen3,
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl": _install_qwen2_5,
    "transformers.models.qwen2_vl.modeling_qwen2_vl": _install_qwen2,
}


# ---- transformers 5: the vision towers take these tables from module-level helpers (transformers.vision_utils), each
# of which returns a precomputed value instead when the caller passes one. Qwen3-VL, Qwen3.5 and MiniCPM-V 4.6 / 4.7
# call them in every forward: per-image Python loops (position ids), dozens of small GPU ops (interpolation indices,
# cumulative lengths) and host reads. Memoizing the helpers per grid gives the same tensors without that work.
_PRECOMPUTED = frozenset({"cu_seqlens", "max_seqlen", "position_ids", "interp_indices", "interp_weights",
                          "bilinear_indices", "bilinear_weights", "window_index", "cu_window_seqlens", "merged_shape"})
_last_grid = [None, None, None]  # the grid tensor last read, its version counter, its key


def _frozen(value):
    return tuple(_frozen(v) for v in value) if isinstance(value, list) else value


def _tensor_key(tensor):
    """The contents of a small index tensor (grid_thw, target sizes): one host read, shared by the helpers of a forward.

    The key holds whether inference mode is on: a table made under it is an inference tensor, which autograd refuses
    to save for a training step's backward."""
    mode = torch.is_inference_mode_enabled()
    if tensor.is_inference():  # no version counter: read it every time
        return (_frozen(tensor.tolist()), tensor.dtype, str(tensor.device), mode)
    if _last_grid[0] is tensor and _last_grid[1] == tensor._version:
        return _last_grid[2] + (mode,)
    key = (_frozen(tensor.tolist()), tensor.dtype, str(tensor.device))
    _last_grid[:] = [tensor, tensor._version, key]
    return key + (mode,)


def _argument_key(value):
    if isinstance(value, torch.Tensor):
        return _tensor_key(value)
    if hasattr(value, "_attn_implementation"):  # a config: only the attention implementation matters (max_seqlen)
        return ("config", id(value), value._attn_implementation)
    hash(value)  # TypeError for anything else unhashable: no memoization
    return value


def _memoized_helper(helper):
    cache = {}

    def memoized(grid, *args, kwargs=None, **options):
        if kwargs and _PRECOMPUTED & kwargs.keys():  # the caller passes its own table: upstream pops and returns it
            return helper(grid, *args, kwargs=kwargs, **options)
        try:
            key = (_argument_key(grid), tuple(_argument_key(a) for a in args),
                   tuple(sorted((k, _argument_key(v)) for k, v in options.items())))
        except TypeError:
            return helper(grid, *args, kwargs=kwargs, **options)
        if key not in cache:
            _remember(cache, key, helper(grid, *args, kwargs=kwargs, **options))
        return cache[key]

    memoized._starvla_original = helper
    return memoized


def _install_vision_utils(modeling):
    """Memoize the transformers.vision_utils helpers that this modeling module calls. False if it calls none."""
    try:
        from transformers import vision_utils
    except ImportError:
        return False
    found = False
    for name, value in list(vars(modeling).items()):
        if name.startswith("get_") and callable(value) and getattr(vision_utils, name, None) is value:
            setattr(modeling, name, _memoized_helper(value))
            found = True
        elif getattr(value, "_starvla_original", None) is not None and name.startswith("get_"):
            found = True
    return found


def _vision_tower(vlm):
    core = getattr(getattr(vlm, "model", None), "model", None)
    for name in ("visual", "vision_tower"):
        tower = getattr(core, name, None)
        if tower is not None:
            return tower
    return None


def apply(vlm):
    """Memoize the position tables if the backbone's vision tower is one of the Qwen families (transformers 4), or
    builds them with the helpers of transformers.vision_utils (transformers 5)."""
    visual = _vision_tower(vlm)
    if visual is None:
        return False
    family = type(visual).__module__
    modeling = sys.modules.get(family)
    if family in _installed:
        return True
    if modeling is not None and _install_vision_utils(modeling):
        _installed.add(family)
        return True
    if family not in _FAMILIES:
        return False
    try:
        _FAMILIES[family](modeling)
    except AttributeError as error:  # another transformers version, whose classes lack these methods
        print(f"[starVLA] vision_pos: {family} has another layout here ({error}); left alone", flush=True)
        return False
    _installed.add(family)
    return True
