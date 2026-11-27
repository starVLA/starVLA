"""The bitwise-identical speedups on tiny random-weight backbones (CPU, seconds): every backbone
speedup leaves hidden states, logits and gradients bitwise identical on Qwen3-VL and Qwen2.5-VL. With
STARVLA_TEST_VLMS set to model directories, also worker_prep with their real processors.

    python tests/test_speedups.py        # or: pytest tests/test_speedups.py
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

import torch

BACKBONE_PATHS = ("patch_attention_mask", "skip_unused_lm_head", "batch_vision_attention", "cache_vision_positions")


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
    if model.config.model_type == "qwen3_5":  # transformers 5 processors return the modality of every token
        inputs["mm_token_type_ids"] = (input_ids == 150).long()
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


def _check(family):
    from starVLA.model.modules import vlm

    model = _tiny(family).train()
    cases = {"uniform": [(8, 8)] * 4, "mixed sizes": [(8, 8), (8, 12), (4, 8), (8, 8)]}
    reference = {}
    calls_before = _vision_attention_calls(model, _inputs(model, cases["uniform"]))
    for name, grids in cases.items():
        hidden, grads, outputs = _run(model, _inputs(model, grids))
        reference[name] = (hidden, grads, outputs.logits.detach().clone())

    applied, skipped = vlm.run_speedups(
        SimpleNamespace(model=model), [f"starVLA.model.modules.vlm.monkey_patch:{p}" for p in BACKBONE_PATHS]
    )
    # transformers 5 builds its masks differently: attn_mask checks its builder against the installed one first
    expected = [p for p in BACKBONE_PATHS if not (family == "qwen3_5" and p == "patch_attention_mask" and p in skipped)]
    assert sorted(applied) == sorted(expected) and not set(skipped) - set(BACKBONE_PATHS) - set(expected) - {
        "patch_attention_mask"
    }, (applied, skipped)

    for name, grids in cases.items():
        hidden, grads, outputs = _run(model, _inputs(model, grids))
        ref_hidden, ref_grads, ref_logits = reference[name]
        assert type(outputs).__name__ == "HiddenOnlyOutput", type(outputs)
        assert torch.equal(hidden, ref_hidden), f"{family} / {name}: hidden state differs"
        assert grads.keys() == ref_grads.keys()
        for key in grads:
            assert torch.equal(grads[key], ref_grads[key]), f"{family} / {name}: gradient of {key} differs"
        assert torch.equal(outputs.logits, ref_logits), f"{family} / {name}: lazily computed logits differ"
    # a call with labels takes the upstream path
    inputs = _inputs(model, cases["uniform"])
    assert model(**inputs, labels=inputs["input_ids"], output_hidden_states=True).loss is not None
    # the batched vision attention really ran on the uniform case: one attention call per vision layer
    calls_after = _vision_attention_calls(model, inputs)
    depth = len(model.model.visual.blocks)
    assert calls_after == depth < calls_before, (calls_before, calls_after, depth)
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
    assert applied == ["skip_unused_lm_head", "cache_vision_positions"], applied
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

    def __len__(self):
        return 6

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
    interface = interface_class.__new__(interface_class)
    nn.Module.__init__(interface)
    interface.processor = AutoProcessor.from_pretrained(model_dir)
    interface.processor.tokenizer.padding_side = "left"
    interface.config = cfg
    interface.__dict__["model"] = SimpleNamespace(device=torch.device("cpu"), model=core)
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
            assert type(got) is type(reference) and set(got.keys()) == set(reference.keys()) | {"position_ids"}
            for key in reference.keys():
                assert torch.equal(got[key], reference[key]) and got[key].dtype == reference[key].dtype, (framework, key)
            assert torch.equal(got["position_ids"], position_ids), framework
        assert worker_prep.input_stats["used"] == used + 3, (framework, worker_prep.input_stats)
    # a prompt the workers did not expect, and a call with other images: the interface answers itself
    batch = next(iter(loader))
    images = [e["image"] for e in batch]
    other = interface.build_qwenvl_inputs(images=images, instructions=["something else"] * 2)
    assert "position_ids" not in other and worker_prep.input_stats["other prompt"] == 1
    fresh = interface.build_qwenvl_inputs(
        images=[list(sample) for sample in images][::-1], instructions=prompts["QwenOFT"](batch)[::-1]
    )
    assert "position_ids" not in fresh and worker_prep.input_stats["used"] == 9
    print(
        f"ok worker_prep {os.path.basename(model_dir)}: inputs built in workers are identical for "
        f"{', '.join(prompts)} ({', '.join(sorted(reference.keys()))} + position_ids)"
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


def test_video_prep():
    """video_prep: a world model's per-sample image preprocessing in the data-loader workers; the interface's loop
    gets identical tensors for the batch's own images and computes any other call itself."""
    try:
        from diffusers.video_processor import VideoProcessor
    except ImportError:
        print("skipped video_prep: no diffusers")
        return
    from torch.utils.data import DataLoader

    from starVLA.model.modules import vlm
    from starVLA.dataloader import worker_prep as world_model

    backbone = _VideoBackbone()
    applied, _ = vlm.run_speedups(backbone, ["starVLA.dataloader.worker_prep:prepare_videos_in_workers"])
    assert applied == ["prepare_videos_in_workers"], applied
    reference = VideoProcessor(vae_scale_factor=16)
    collate = world_model.video_collate(SimpleNamespace(framework={"name": "WanGR00T"}), _same_batch)
    loader = DataLoader(_Samples(), batch_size=2, num_workers=2, collate_fn=collate, multiprocessing_context="spawn")
    used = world_model.video_stats["used"]
    for batch in loader:
        images = [e["image"] for e in batch]
        for got, sample in zip(backbone._encode_images_vae(images), images):
            assert torch.equal(got, reference.preprocess_video(sample, height=480, width=832))
    assert world_model.video_stats["used"] == used + 6, world_model.video_stats
    copies = [[im.copy() for im in sample] for sample in images]  # other image objects: computed here
    assert all(
        torch.equal(a, b) for a, b in zip(backbone._encode_images_vae(copies), backbone._encode_images_vae(images))
    )
    assert world_model.video_stats["used"] == used + 8
    print("ok video_prep: a world model's image preprocessing in the workers, identical tensors")


def test_grad_norm_identity():
    """What grad_norm relies on: vector_norm(g, dtype=float64) is vector_norm(g.double()) bit for bit."""
    generator = torch.Generator().manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        for shape in ((7,), (64, 33), (3, 5, 17), (1024, 257)):
            g = torch.randn(shape, generator=generator).to(dtype)
            fused = torch.linalg.vector_norm(g, ord=2.0, dtype=torch.float64)
            assert torch.equal(fused, torch.linalg.vector_norm(g.double(), ord=2.0)), (dtype, shape)
    print("ok grad_norm: the fused fp64 norm is bitwise the norm of the fp64 copy")


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
        test_grad_norm_identity()
        test_qwen3_vl_bitwise()
        test_qwen2_5_vl_bitwise()
        test_qwen3_5_bitwise()
        test_flash_attention_backbone()
        test_worker_prep()
        test_worker_prep_framework()
        test_video_prep()
