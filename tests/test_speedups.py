"""The bitwise-identical speedups on tiny random-weight backbones (CPU, seconds): every backbone
speedup leaves hidden states, logits and gradients bitwise identical on Qwen3-VL and Qwen2.5-VL. With
STARVLA_TEST_VLMS set to model directories, also worker_prep with their real processors.

    python tests/test_speedups.py        # or: pytest tests/test_speedups.py
"""

import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

import torch

BACKBONE_PATHS = ("patch_attention_mask", "batch_vision_attention", "cache_vision_positions")


def _tiny(family):
    torch.manual_seed(0)
    text = dict(
        vocab_size=160,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]},
    )  # head_dim 16 = 2 * (4 + 2 + 2)
    if family == "qwen3_5":  # transformers 5: three linear-attention layers and one full-attention layer
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        text = dict(
            vocab_size=160,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            max_position_embeddings=512,
            pad_token_id=0,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [1, 1, 0],
                "mrope_interleaved": True,
                "partial_rotary_factor": 0.25,
            },
        )
        vision = dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=64,
            num_position_embeddings=36,
        )
        config = Qwen3_5Config(
            text_config=text,
            vision_config=vision,
            image_token_id=150,
            video_token_id=151,
            vision_start_token_id=152,
            vision_end_token_id=153,
        )
        return Qwen3_5ForConditionalGeneration._from_config(config, attn_implementation="sdpa").float()
    if family == "qwen3":
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        vision = dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=64,
            num_position_embeddings=36,
            deepstack_visual_indexes=[0],
        )
        config = Qwen3VLConfig(
            text_config=text,
            vision_config=vision,
            image_token_id=150,
            video_token_id=151,
            vision_start_token_id=152,
            vision_end_token_id=153,
        )
        model = Qwen3VLForConditionalGeneration._from_config(config, attn_implementation="sdpa")
    else:
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

        vision = dict(
            depth=3,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=64,
            window_size=16,
            fullatt_block_indexes=[1],
            spatial_patch_size=4,
        )
        config = Qwen2_5_VLConfig(
            text_config=text,
            vision_config=vision,
            image_token_id=150,
            video_token_id=151,
            vision_start_token_id=152,
            vision_end_token_id=153,
        )
        model = Qwen2_5_VLForConditionalGeneration._from_config(config, attn_implementation="sdpa")
    return model.float()


def _inputs(model, grids):
    """Two samples; `grids` lists the (h, w) patch grid of each image, split evenly between the samples."""
    torch.manual_seed(1)
    vision = model.config.vision_config
    patch_dim = 3 * vision.temporal_patch_size * vision.patch_size**2
    merge = vision.spatial_merge_size**2
    per_sample = len(grids) // 2
    rows = []
    for sample in range(2):
        ids = [5, 6]
        for h, w in grids[sample * per_sample : (sample + 1) * per_sample]:
            ids += [152] + [150] * (h * w // merge) + [153]
        ids += list(range(10, 14 + 3 * sample))  # different text lengths: left padding on the shorter sample
        rows.append(ids)
    length = max(len(r) for r in rows)
    input_ids = torch.tensor([[0] * (length - len(r)) + r for r in rows])
    attention_mask = torch.tensor([[0] * (length - len(r)) + [1] * len(r) for r in rows])
    inputs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=torch.randn(sum(h * w for h, w in grids), patch_dim),
        image_grid_thw=torch.tensor([[1, h, w] for h, w in grids]),
    )
    if "mm_token_type_ids" in inspect.signature(type(model).forward).parameters:
        inputs["mm_token_type_ids"] = (input_ids == 150).long()  # transformers 5: the modality of every token
    return inputs


def _vision_attention_calls(model, inputs):
    """Number of scaled_dot_product_attention calls inside one forward of the vision tower."""
    import torch.nn.functional as F

    with torch.no_grad():  # let vision_attn probe this shape before the calls are counted
        model.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    calls, original = [0], F.scaled_dot_product_attention

    def counting(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    F.scaled_dot_product_attention = counting
    try:
        with torch.no_grad():
            model.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    finally:
        F.scaled_dot_product_attention = original
    return calls[0]


def _run(model, inputs):
    model.zero_grad()
    outputs = model(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1]
    hidden.float().pow(2).mean().backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    return hidden.detach().clone(), grads, outputs


def _partition(model):
    """Leave every parameter empty outside its own module's forward, as DeepSpeed ZeRO-3 does with the parameters it
    partitions (param.data = torch.empty(0) after the module's forward hook). Returns the function that undoes it."""
    data = {p: p.data for p in model.parameters()}
    active = dict.fromkeys(data, 0)
    for p in data:
        p.data = torch.empty(0, dtype=p.dtype)

    def gather(module, args):
        for p in module.parameters(recurse=False):
            active[p] += 1
            p.data = data[p]

    def release(module, args, output):
        for p in module.parameters(recurse=False):
            active[p] -= 1
            if not active[p]:
                p.data = torch.empty(0, dtype=p.dtype)

    handles = []
    for module in model.modules():
        if next(module.parameters(recurse=False), None) is not None:
            handles += [module.register_forward_pre_hook(gather), module.register_forward_hook(release)]

    def undo():
        for handle in handles:
            handle.remove()
        for p, tensor in data.items():
            p.data = tensor

    return undo


def _partitioned_forward(model, inputs):
    """Hidden states and logits of a forward on partitioned parameters (the logits are read after the forward)."""
    undo = _partition(model)
    try:
        outputs = model(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
        return outputs.hidden_states[-1].detach().clone(), outputs.logits.detach().clone()
    finally:
        undo()


def _generate(model, inputs):
    """Greedy tokens of generate(), and the head inputs it made (one row of positions per step when the model's
    forward is seen to accept logits_to_keep)."""
    shapes = []
    handle = model.lm_head.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape)))
    model.eval()
    try:
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    finally:
        handle.remove()
        model.train()
    return tokens, shapes


def _picklable(model):
    import io

    try:
        torch.save(model, io.BytesIO())
        return True
    except Exception:
        return False


def _check_copies(family, model, inputs, reference, picklable):
    """A deep copy (an EMA or teacher copy) runs on its own weights, and the model can be pickled."""
    import copy
    import io

    twin = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in twin.parameters():
            parameter.zero_()
    kwargs = dict(output_attentions=False, output_hidden_states=True, return_dict=True)
    with torch.no_grad():
        ours, theirs = model(**inputs, **kwargs), twin(**inputs, **kwargs)
    assert torch.equal(ours.hidden_states[-1], reference), f"{family}: the model changed when it was copied"
    assert not torch.equal(theirs.hidden_states[-1], reference), f"{family}: a deep copy runs the original's weights"
    assert torch.equal(theirs.logits, torch.zeros_like(theirs.logits)), f"{family}: a deep copy uses the original's head"
    if not picklable:  # transformers 5 keeps a local function on the model: it cannot be pickled, patched or not
        return
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    with torch.no_grad():
        again = loaded(**inputs, **kwargs)
    assert type(again) is type(ours) and torch.equal(again.hidden_states[-1], reference)
    assert torch.equal(again.logits, ours.logits), f"{family}: the pickled model's logits differ"


def _check_mask_builder():
    """The registered mask builder against the one it replaces, also where the two are called with a padding mask for
    another batch size (the original raises, or takes the first rows)."""
    from transformers import masking_utils

    ours = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    theirs = getattr(ours, "_starvla_original", None)
    if theirs is None:  # not registered here (another transformers version)
        return

    def outcome(builder, **arguments):
        try:
            return builder(**arguments)
        except Exception as error:
            return type(error)

    generator = torch.Generator().manual_seed(0)
    for batch, rows, queries, offset in (
        (3, 3, 5, 0),
        (3, 1, 5, 0),
        (2, 4, 6, 0),
        (1, 3, 4, 2),
        (4, 4, 1, 7),
        (2, 2, 3, 3),
    ):
        kv_length = queries + offset
        mask = (torch.rand(rows, kv_length, generator=generator) > 0.3).long()
        for skip in (True, False):
            arguments = dict(
                batch_size=batch,
                cache_position=torch.arange(offset, offset + queries),
                kv_length=kv_length,
                attention_mask=mask,
                allow_is_causal_skip=skip,
            )
            a, b = outcome(ours, **arguments), outcome(theirs, **arguments)
            same = (
                (a is b)
                if not torch.is_tensor(a) or not torch.is_tensor(b)
                else (a.shape == b.shape and torch.equal(a, b))
            )
            assert same, f"mask builder differs for batch {batch}, {rows} mask rows: {a} / {b}"


def _check(family):
    from starVLA.model.modules import vlm

    model = _tiny(family).train()
    inference_first, no_windows = "first seen under inference_mode", "window layout unavailable"
    cases = {
        "uniform": [(8, 8)] * 4,
        "mixed sizes": [(8, 8), (8, 12), (4, 8), (8, 8)],
        inference_first: [(4, 12)] * 4,
        no_windows: [(12, 8)] * 4,
    }
    reference = {}
    calls_before = _vision_attention_calls(model, _inputs(model, cases["uniform"]))
    for name, grids in cases.items():
        hidden, grads, outputs = _run(model, _inputs(model, grids))
        reference[name] = (hidden, grads, outputs.logits.detach().clone())
        output_class = type(outputs)
    signature = inspect.signature(model.forward)
    picklable = _picklable(model)  # before anything is patched
    generated = _generate(model, _inputs(model, cases["uniform"]))
    # the emulation of partitioned parameters, where the upstream code itself runs on it (Qwen3.5 reads a child
    # module's weight in its parent's forward)
    try:
        hidden, logits = _partitioned_forward(model, _inputs(model, cases["uniform"]))
        partitioned = True
    except Exception:
        partitioned = False
    assert not partitioned or (
        torch.equal(hidden, reference["uniform"][0]) and torch.equal(logits, reference["uniform"][2])
    )

    applied, skipped = vlm.run_speedups(
        SimpleNamespace(model=model), [f"starVLA.model.modules.vlm.monkey_patch:{p}" for p in BACKBONE_PATHS]
    )
    # transformers 5 builds its masks differently: attn_mask checks its builder against the installed one first
    expected = [p for p in BACKBONE_PATHS if not (p == "patch_attention_mask" and p in skipped)]
    assert sorted(applied) == sorted(expected) and set(skipped) <= {"patch_attention_mask"}, (applied, skipped)

    # DeepSpeed ZeRO-3: parameters exist only inside their module's forward.
    # First, so that the patched forward meets these grids with nothing cached.
    for name in ("uniform", "mixed sizes") if partitioned else ():
        hidden, logits = _partitioned_forward(model, _inputs(model, cases[name]))
        assert torch.equal(hidden, reference[name][0]), f"{family} / {name}: hidden state differs when partitioned"
        assert torch.equal(logits, reference[name][2]), f"{family} / {name}: logits differ when partitioned"
    from starVLA.model.modules.vlm import monkey_patch

    def unavailable(*args):  # transformers >= 5.17: the Qwen2.5-VL vision model has no get_window_index
        raise AttributeError("no get_window_index")

    for name, grids in cases.items():
        if name == inference_first:  # predict_action() meets a grid before training does: nothing it caches may
            with torch.inference_mode():  # be an inference tensor, which autograd refuses to save for backward
                model(**_inputs(model, grids), output_hidden_states=True, return_dict=True)
        window_layout = monkey_patch._window_layout
        if name == no_windows:  # the layout cannot be derived for this grid: the forward runs as upstream
            monkey_patch._window_layout = unavailable
        try:
            hidden, grads, outputs = _run(model, _inputs(model, grids))
        finally:
            monkey_patch._window_layout = window_layout
        ref_hidden, ref_grads, ref_logits = reference[name]
        assert type(outputs) is output_class, type(outputs)  # the upstream output object, whatever a caller does with it
        assert torch.equal(hidden, ref_hidden), f"{family} / {name}: hidden state differs"
        assert grads.keys() == ref_grads.keys()
        for key in grads:
            assert torch.equal(grads[key], ref_grads[key]), f"{family} / {name}: gradient of {key} differs"
        assert torch.equal(outputs.logits, ref_logits), f"{family} / {name}: logits differ"
    _check_mask_builder()
    # what was memoized is kept on the model it was made for (not under its address, which another model may get)
    assert set(model.model.visual.__dict__["_starvla_tables"]) >= {"layouts"}, model.model.visual.__dict__.keys()
    # generate() reads the signature of forward (position_ids, logits_to_keep, attention_mask): same tokens, and
    # the same head inputs
    assert inspect.signature(model.forward) == signature, f"{family}: the signature of forward changed"
    tokens, head_inputs = _generate(model, _inputs(model, cases["uniform"]))
    assert torch.equal(tokens, generated[0]), f"{family}: generate() returns other tokens"
    assert head_inputs == generated[1], f"{family}: generate() calls the head on {head_inputs}, was {generated[1]}"
    _check_copies(family, model, _inputs(model, cases["uniform"]), reference["uniform"][0], picklable)
    # a call with labels takes the upstream path
    inputs = _inputs(model, cases["uniform"])
    assert model(**inputs, labels=inputs["input_ids"], output_hidden_states=True).loss is not None
    # the batched vision attention really ran on the uniform case: one attention call per vision layer
    calls_after = _vision_attention_calls(model, inputs)
    depth = len(model.model.visual.blocks)
    # (transformers >= 5.17 has no get_window_index: there the window-attention layers of Qwen2.5-VL stay as upstream)
    every_layer = family != "qwen2_5" or hasattr(type(model.model.visual), "get_window_index")
    assert depth <= calls_after < calls_before and (calls_after == depth or not every_layer), (calls_before, calls_after)
    from starVLA.model.modules.vlm import monkey_patch as vision_attn

    assert vision_attn.decisions and all(vision_attn.decisions.values()), vision_attn.decisions  # probes ran, passed
    print(
        f"ok {family}: {len(grads)} gradients, hidden states and logits bitwise identical "
        f"({', '.join(cases)}; speedups: {', '.join(applied)}); "
        f"vision attention calls per forward {calls_before} -> {calls_after}"
    )


def _check_flash_attention():
    """A backbone loaded with flash-attention keeps it, and the SDPA-only speedups report "not applicable"
    instead of failing in the first training step."""
    from starVLA.model.modules import vlm
    from starVLA.model.modules.vlm import monkey_patch as vision_attn

    model = _tiny("qwen2_5")
    for config in (model.config, model.model.visual.config, model.model.language_model.config):
        config._attn_implementation_internal = "flash_attention_2"  # cannot be loaded this way without a GPU
    applied, skipped = vlm.run_speedups(SimpleNamespace(model=model), vlm.BACKBONE_SPEEDUPS)
    assert applied == ["cache_vision_positions"], applied
    assert skipped == [
        "patch_attention_mask",
        "batch_vision_attention",
        "prepare_inputs_in_workers",
        "prepare_videos_in_workers",
    ], skipped
    assert vision_attn._attention_implementation(model.model.visual, model) == "flash_attention_2"
    print(f"ok flash-attention: kept ({', '.join(applied)}; not applicable: {', '.join(skipped)})")


class _Samples:
    """Six samples the way the LeRobot datasets return them: two views, an instruction, a state, an action chunk."""

    def __init__(self, count=6):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(index)
        return {
            "image": [Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)],
            "lang": "put the bowl on the plate" + " and then close the drawer" * (index % 3),
            "state": rng.uniform(-1, 1, (1, 8)).astype(np.float32),
            "action": rng.normal(size=(8, 7)),
        }


def _same_batch(batch):
    return batch


class _Shell(torch.nn.Module):
    """Stands for the Hugging Face model: its device, its inner model, and a forward that shows what it was given."""

    device = property(lambda self: torch.device("cpu"))

    def __init__(self, core):
        super().__init__()
        self.model, self.seen = core, None

    def forward(self, **kwargs):
        self.seen = kwargs


def _check_worker_prep(model_dir):
    """worker_prep with the real processor of `model_dir` (no weights are loaded): what the interface returns for a
    batch whose inputs a data-loader worker built is what it returns without workers, for a framework that passes the
    instruction unchanged and for two that edit it; any other call is answered by the interface itself."""
    import torch.nn as nn
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoProcessor

    from starVLA.model.modules import vlm
    from starVLA.dataloader import worker_prep
    from starVLA.model.framework.share_tools import add_discretized_state_to_instruction

    if "Qwen2.5-VL" in model_dir:
        from transformers import Qwen2_5_VLModel as core_class

        from starVLA.model.modules.vlm.QWen2_5 import _QWen_VL_Interface as interface_class
    else:
        from transformers import Qwen3VLModel as core_class

        from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface as interface_class
    cfg = OmegaConf.create(
        {
            "framework": {"name": "QwenGR00T", "qwenvl": {"base_vlm": model_dir}, "action_model": {"action_horizon": 8}},
            "datasets": {"vla_data": {}},
        }
    )
    core = core_class.__new__(core_class)  # get_rope_index reads only the config
    nn.Module.__init__(core)
    core.config = AutoConfig.from_pretrained(model_dir)

    def new_interface():
        interface = interface_class.__new__(interface_class)
        nn.Module.__init__(interface)
        interface.processor = AutoProcessor.from_pretrained(model_dir)
        interface.processor.tokenizer.padding_side = "left"
        interface.config, interface.model = cfg, _Shell(core)
        return interface

    interface = new_interface()
    shell = interface.model
    original = interface.build_qwenvl_inputs

    applied, skipped = vlm.run_speedups(interface, ["starVLA.dataloader.worker_prep:prepare_inputs_in_workers"])
    assert applied == ["prepare_inputs_in_workers"] and not skipped, (applied, skipped)

    suffix = f" Please predict the next 8 robot actions: <action>{'🔍' * 8}<action>."
    with_state = lambda batch: add_discretized_state_to_instruction(
        [e["lang"] for e in batch], [e["state"] for e in batch]
    )
    prompts = {
        "QwenGR00T": lambda batch: [e["lang"] for e in batch],
        "QwenPI_v3": with_state,
        "QwenOFT": lambda batch: [instruction + suffix for instruction in with_state(batch)],
    }
    for framework, prompt in prompts.items():
        cfg.framework.name = framework
        loader = DataLoader(
            _Samples(),
            batch_size=2,
            num_workers=2,
            collate_fn=worker_prep.inputs_collate(cfg, _same_batch),
            multiprocessing_context="spawn" if framework == "QwenOFT" else None,
        )  # must pickle
        used = worker_prep.input_stats["used"]
        for batch in loader:
            images, instructions = [e["image"] for e in batch], prompt(batch)
            got = interface.build_qwenvl_inputs(images=images, instructions=instructions)
            reference = original(images=images, instructions=instructions)
            position_ids, _ = core.get_rope_index(
                input_ids=reference["input_ids"],
                image_grid_thw=reference["image_grid_thw"],
                attention_mask=reference["attention_mask"],
            )
            # the keys are the interface's own; the workers' position ids reach a forward on exactly these tensors
            assert type(got) is type(reference) and set(got.keys()) == set(reference.keys())
            for key in reference.keys():
                assert torch.equal(got[key], reference[key]) and got[key].dtype == reference[key].dtype, (framework, key)
            shell(**got, output_hidden_states=True)
            assert torch.equal(shell.seen["position_ids"], position_ids), framework
            for changed in (
                {"input_ids": got["input_ids"].clone()},  # other tensors (generate(), a framework that edits them)
                {"attention_mask": got["attention_mask"].clone()},
                {"position_ids": position_ids + 1},  # the caller's own
                {"inputs_embeds": torch.zeros(1)},
            ):
                shell(**{**got, **changed})
                assert shell.seen.get("position_ids") is changed.get("position_ids"), (framework, list(changed))
            got["attention_mask"][0, 0] = got["attention_mask"][0, 0].clone()  # modified in place since
            shell(**got)
            assert "position_ids" not in shell.seen, framework
        assert worker_prep.input_stats["used"] == used + 3, (framework, worker_prep.input_stats)
    # one batch per case from here on (what the workers built is handed over once)
    cfg.framework.name = "QwenOFT"
    loader = DataLoader(
        _Samples(40), batch_size=2, num_workers=1, collate_fn=worker_prep.inputs_collate(cfg, _same_batch)
    )
    batches = iter(loader)
    stats = worker_prep.input_stats

    def case():
        batch = next(batches)
        return batch, [e["image"] for e in batch], prompts["QwenOFT"](batch)

    def identical(got, images, instructions):
        reference = original(images=images, instructions=instructions)
        return set(got.keys()) == set(reference.keys()) and all(
            torch.equal(got[k], reference[k]) for k in reference.keys()
        )

    # a prompt the workers did not expect: the interface answers itself
    batch, images, instructions = case()
    other = interface.build_qwenvl_inputs(images=images, instructions=["something else"] * 2)
    shell(**other)
    assert "position_ids" not in shell.seen and stats["other prompt"] == 1 and stats["used"] == 9
    # the same call twice: the second is answered by the interface, with the same tensors
    first = interface.build_qwenvl_inputs(images=images, instructions=instructions)
    again = interface.build_qwenvl_inputs(images=images, instructions=instructions)
    assert stats["used"] == 10 and all(torch.equal(first[k], again[k]) for k in first.keys())
    # copies of the images with the same pixels (a framework that resizes to the size they have)
    batch, images, instructions = case()
    copies = interface.build_qwenvl_inputs(
        images=[[im.copy() for im in sample] for sample in images], instructions=instructions
    )
    assert stats["used"] == 11 and identical(copies, images, instructions)
    # an image edited in place after the workers saw it: built from what it is now
    batch, images, instructions = case()
    images[0][0].paste(0, (0, 0, 64, 64))
    edited = interface.build_qwenvl_inputs(images=images, instructions=instructions)
    assert stats["used"] == 11 and identical(edited, images, instructions), "an image edited in place went unnoticed"
    # the processor or the config changed after the DataLoader was made: built from what they are now
    for change, undo in (
        (lambda: interface.processor.tokenizer.add_tokens(["<|test_token|>"]), None),
        (lambda: setattr(interface.processor.image_processor, "max_pixels", 96 * 96), None),
        (
            lambda: OmegaConf.update(
                cfg, "datasets.vla_data.CoT_prompt", "Task: {instruction} Think first.", force_add=True
            ),
            None,
        ),
    ):
        batch, images, instructions = case()
        changed_before = stats["changed since"]
        change()
        late = interface.build_qwenvl_inputs(images=images, instructions=instructions)
        assert stats["used"] == 11 and stats["changed since"] == changed_before + 1, stats
        assert identical(late, images, instructions)
    # a DataLoader made now carries the interface as it is now
    loader = DataLoader(
        _Samples(40), batch_size=2, num_workers=1, collate_fn=worker_prep.inputs_collate(cfg, _same_batch)
    )
    batches = iter(loader)
    batch, images, instructions = case()
    now = interface.build_qwenvl_inputs(images=images, instructions=instructions)
    assert stats["used"] == 12 and identical(now, images, instructions)
    # under inference mode (predict_action) the worker's inputs are used; tensors made under it (a copy to the GPU)
    # carry no position ids, the model derives them
    batch, images, instructions = case()
    with torch.inference_mode():
        inferred = interface.build_qwenvl_inputs(images=images, instructions=instructions)
        assert stats["used"] == 13 and identical(inferred, images, instructions)
        shell(**{key: value.clone() for key, value in inferred.items()})
    assert "position_ids" not in shell.seen
    # a deep copy (an EMA or teacher copy) and a pickle have their own builder; the worker's inputs fit a copy too
    import copy
    import pickle

    twin = copy.deepcopy(interface)
    assert twin.build_qwenvl_inputs.owner is twin and interface.build_qwenvl_inputs.owner is interface
    batch, images, instructions = case()
    taken = twin.build_qwenvl_inputs(images=images, instructions=instructions)
    assert stats["used"] == 14 and identical(taken, images, instructions)
    shell.seen = None
    assert pickle.loads(pickle.dumps(interface)).build_qwenvl_inputs.key == interface.build_qwenvl_inputs.key
    # a second interface in the process (a teacher, another checkpoint): each takes only what was built for it
    second = new_interface()
    assert vlm.run_speedups(second, ["starVLA.dataloader.worker_prep:prepare_inputs_in_workers"])[0]
    loader = DataLoader(_Samples(), batch_size=2, num_workers=1, collate_fn=worker_prep.inputs_collate(cfg, _same_batch))
    batch = next(iter(loader))
    images, instructions = [e["image"] for e in batch], prompts["QwenOFT"](batch)
    mine = interface.build_qwenvl_inputs(images=images, instructions=instructions)
    assert stats["used"] == 14, "an interface took inputs that were built for another one"
    theirs = second.build_qwenvl_inputs(images=images, instructions=instructions)
    assert stats["used"] == 15 and set(mine.keys()) == set(theirs.keys())
    # a framework whose training call the workers cannot make (QwenFast passes `solutions`): nothing is built
    cfg.framework.name = "QwenFast"
    assert worker_prep.inputs_collate(cfg, _same_batch) is _same_batch
    # a DataLoader for another recipe than the model that was set up: nothing is built
    cfg.framework.name = "QwenGR00T"
    other_recipe = OmegaConf.create({"framework": {"name": "QwenGR00T", "qwenvl": {"base_vlm": "another/model"}}})
    assert worker_prep.inputs_collate(other_recipe, _same_batch) is _same_batch
    # too little shared memory for the workers' tensors: the batches are left as they are
    assert isinstance(worker_prep.worker_collate(cfg, _same_batch), worker_prep.Collate)
    worker_prep._MIN_SHARED_MEMORY = 1 << 62
    assert worker_prep.worker_collate(cfg, _same_batch) is _same_batch
    print(
        f"ok worker_prep {os.path.basename(model_dir)}: inputs built in workers are identical for "
        f"{', '.join(prompts)} ({', '.join(sorted(reference.keys()))}; position ids through the forward hook)"
    )


class _PixelProcessor:
    """A processor without files: token ids from the prompt's characters, pixels from the images (left padding)."""

    def __call__(self, prompts, images):
        from transformers import BatchFeature

        width = max(len(p) for p in prompts)
        ids = torch.tensor([[0] * (width - len(p)) + [ord(c) % 256 for c in p] for p in prompts])
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts])
        pixels = torch.stack(
            [torch.stack([torch.as_tensor(np.asarray(im), dtype=torch.float32) for im in views]) for views in images]
        )
        return BatchFeature(data={"input_ids": ids, "attention_mask": mask, "pixel_values": pixels})


class _OwnBuilderFramework(torch.nn.Module):
    """A framework that builds its backbone's inputs itself, as MiniCPM-RobotManip does."""

    def __init__(self):
        super().__init__()
        self.processor = _PixelProcessor()
        self.config = None
        self.template = "The robot is a test arm. {instruction}"
        self.model = torch.nn.Linear(2, 2)  # stands for the backbone + head
        self.model.device = torch.device("cpu")

    @staticmethod
    def _strip(text):
        return text.strip()

    def _format_instruction(self, instruction):
        return self.template.format(instruction=self._strip(instruction))

    def _build_vlm_inputs(self, images, instructions):
        return self.processor([self._format_instruction(i) for i in instructions], images).to(self.model.device)


def test_worker_prep_framework():
    """worker_prep_framework: a framework's own _build_vlm_inputs runs in the data-loader workers on a stand-in (its
    processor, plain settings and methods); the framework gets identical inputs, and any other call is built by the
    framework itself."""
    from torch.utils.data import DataLoader

    from starVLA.model.modules import vlm
    from starVLA.dataloader import worker_prep

    framework = _OwnBuilderFramework()
    original = framework._build_vlm_inputs
    applied, _ = vlm.run_speedups(framework, ["starVLA.dataloader.worker_prep:prepare_framework_inputs_in_workers"])
    assert applied == ["prepare_framework_inputs_in_workers"], applied
    cfg = SimpleNamespace(framework={"name": "MiniCPMRobotManip"})
    collate = worker_prep.inputs_collate(cfg, _same_batch)
    assert worker_prep.inputs_collate(cfg, collate) is collate  # wrapped once
    loader = DataLoader(_Samples(), batch_size=2, num_workers=2, collate_fn=collate, multiprocessing_context="spawn")
    used = worker_prep.input_stats["used"]
    for batch in loader:
        images, instructions = [e["image"] for e in batch], [e["lang"] for e in batch]
        got, reference = framework._build_vlm_inputs(images, instructions), original(images, instructions)
        assert got.keys() == reference.keys() and all(torch.equal(got[k], reference[k]) for k in reference.keys())
    assert worker_prep.input_stats["used"] == used + 3, worker_prep.input_stats
    other = framework._build_vlm_inputs(images, ["something else"] * 2)  # built by the framework itself
    assert torch.equal(other["input_ids"], original(images, ["something else"] * 2)["input_ids"])
    print("ok worker_prep_framework: a framework's own input construction in the workers, identical inputs")


class _VideoBackbone(torch.nn.Module):
    """The image encoding loop of starVLA's Wan interface, without the VAE."""

    size = (480, 832)

    def __init__(self):
        super().__init__()
        from diffusers.video_processor import VideoProcessor

        self.video_processor = VideoProcessor(vae_scale_factor=16)

    def _encode_images_vae(self, images, num_frames=None):
        height, width = 480, 832
        preprocessed = []
        for sample_imgs in images:
            video_tensor = self.video_processor.preprocess_video(sample_imgs, height=height, width=width)
            preprocessed.append(video_tensor)
        return preprocessed

    encode = _encode_images_vae


class _CosmosBackbone(torch.nn.Module):
    """The same loop as starVLA's Cosmos-Predict2 interface has it: another method name, another size."""

    size = (320, 576)

    def __init__(self):
        super().__init__()
        from diffusers.video_processor import VideoProcessor

        self.video_processor = VideoProcessor(vae_scale_factor=8)

    def _encode_images(self, images, num_frames=None):
        height, width = 320, 576
        preprocessed = []
        for sample_imgs in images:
            video_tensor = self.video_processor.preprocess_video(sample_imgs, height=height, width=width)
            preprocessed.append(video_tensor)
        return preprocessed

    encode = _encode_images


def test_video_prep():
    """video_prep: a world model's per-sample image preprocessing in the data-loader workers; the interface's loop
    gets identical tensors for the batch's own images and computes any other call itself."""
    try:
        from diffusers.video_processor import VideoProcessor
    except ImportError:
        print("skipped video_prep: no diffusers")
        return
    import copy
    import pickle

    from torch.utils.data import DataLoader

    from starVLA.dataloader import worker_prep as world_model
    from starVLA.model.modules import vlm

    stats = world_model.video_stats
    for backbone_class in (_VideoBackbone, _CosmosBackbone):
        backbone = backbone_class()
        height, width = backbone.size
        reference = VideoProcessor(vae_scale_factor=backbone.video_processor.config.vae_scale_factor)
        applied, _ = vlm.run_speedups(backbone, ["starVLA.dataloader.worker_prep:prepare_videos_in_workers"])
        assert applied == ["prepare_videos_in_workers"], (backbone_class.__name__, applied)

        def same(tensors, images):
            return all(
                torch.equal(got, reference.preprocess_video(sample, height=height, width=width))
                for got, sample in zip(tensors, images, strict=True)
            )

        collate = world_model.video_collate(SimpleNamespace(framework={"name": "WanGR00T"}), _same_batch)
        loader = DataLoader(
            _Samples(12), batch_size=2, num_workers=2, collate_fn=collate, multiprocessing_context="spawn"
        )
        batches = iter(loader)
        used = stats["used"]
        for _ in range(3):
            batch = next(batches)  # alive while it is used, as in a training step
            images = [e["image"] for e in batch]
            assert same(backbone.encode(images), images)
        assert stats["used"] == used + 6, stats
        # the same call again (handed over once), and copies of the images: computed here, the same tensors
        copies = [[im.copy() for im in sample] for sample in images]
        assert same(backbone.encode(images), images) and same(backbone.encode(copies), images)
        assert stats["used"] == used + 6, stats
        # an image edited in place after the workers saw it: computed from what it is now
        batch = next(batches)
        images = [e["image"] for e in batch]
        images[0][0].paste(0, (0, 0, 64, 64))
        assert same(backbone.encode(images), images) and stats["used"] == used + 7, stats  # the second sample only
        # the processor's settings changed after the DataLoader was made: computed here
        batch = next(batches)
        images = [e["image"] for e in batch]
        setting = backbone.video_processor.config.resample
        backbone.video_processor._processor._internal_dict = type(backbone.video_processor.config)(
            {**backbone.video_processor.config, "resample": "bilinear" if setting != "bilinear" else "lanczos"}
        )
        changed = backbone.encode(images)
        assert stats["used"] == used + 7, stats
        backbone.video_processor._processor._internal_dict = type(backbone.video_processor.config)(
            {**backbone.video_processor.config, "resample": setting}
        )
        assert not same(changed, images) or setting == "bilinear"  # it did use the changed setting
        # a deep copy and a pickle of the backbone work, and a copy takes what was built for the original
        twin = copy.deepcopy(backbone)
        assert pickle.loads(pickle.dumps(backbone.video_processor)).config == backbone.video_processor.config
        batch = next(batches)
        images = [e["image"] for e in batch]
        assert same(twin.encode(images), images) and stats["used"] == used + 9, stats
        del loader, batches
    print("ok video_prep: a world model's image preprocessing in the workers (Wan, Cosmos-Predict2), identical tensors")


def test_speedup_lists():
    """Every listed speedup names a function that exists."""
    import importlib

    from starVLA.model.framework import base_framework
    from starVLA.model.modules import vlm

    for speedup in vlm.BACKBONE_SPEEDUPS + base_framework.FRAMEWORK_SPEEDUPS:
        module, name = speedup.split(":")
        assert callable(getattr(importlib.import_module(module), name)), speedup


def _in_subprocess(family):
    # the vision patches are installed on transformers classes for the life of the process: one process per family
    result = subprocess.run([sys.executable, __file__, family], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())


def test_qwen3_vl_bitwise():
    _in_subprocess("qwen3")


def test_qwen2_5_vl_bitwise():
    _in_subprocess("qwen2_5")


def test_qwen3_5_bitwise():
    """Qwen3.5 needs transformers 5; skipped on the pinned 4.57. It runs in a process without
    flash-linear-attention (its Triton kernels need GPU tensors; the tiny models are on the CPU)."""
    try:
        import transformers.models.qwen3_5  # noqa: F401
    except ImportError:
        print("skipped qwen3_5: this transformers has no Qwen3.5")
        return
    _in_subprocess("qwen3_5_torch")


def test_flash_attention_backbone():
    _in_subprocess("flash")


def test_worker_prep():
    """Needs real processors: STARVLA_TEST_VLMS=<dir of Qwen3-VL-4B-Instruct>,<dir of Qwen2.5-VL-3B-Instruct>."""
    model_dirs = [d for d in os.environ.get("STARVLA_TEST_VLMS", "").split(",") if d]
    if not model_dirs:
        print("skipped worker_prep: STARVLA_TEST_VLMS is not set")
    for model_dir in model_dirs:
        _in_subprocess(model_dir)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if len(sys.argv) > 1:
        if os.path.isdir(sys.argv[1]):
            _check_worker_prep(sys.argv[1])
        elif sys.argv[1] == "qwen3_5_torch":
            sys.modules["fla"] = None  # transformers then uses its torch implementation of the linear attention
            _check("qwen3_5")
        elif sys.argv[1] == "flash":
            _check_flash_attention()
        else:
            _check(sys.argv[1])
    else:
        test_speedup_lists()
        print("ok speedup lists")
        test_qwen3_vl_bitwise()
        test_qwen2_5_vl_bitwise()
        test_qwen3_5_bitwise()
        test_flash_attention_backbone()
        test_worker_prep()
        test_worker_prep_framework()
        test_video_prep()
