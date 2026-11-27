"""Input construction in the data-loader workers. A training step otherwise spends CPU time in the training process,
with the GPU waiting, on work that depends on the samples only: the VLM processor, a world model's image preprocessing,
the PI0 / PI05 sample preparation. The workers have the samples first, so they do that work with a copy of the same
processor and send the tensors along with the batch. Every path is probed once against the original code and gives
the same tensors; a call the workers did not prepare (inference, an edited batch) runs in the training process as
before.

    prepare_inputs_in_workers(interface)             the VLM interface's build_qwenvl_inputs
    prepare_framework_inputs_in_workers(framework)   a framework that builds its backbone inputs itself
    prepare_videos_in_workers(backbone)              a world model's VideoProcessor (Wan, Cosmos-Predict2)
    prepare_examples_in_workers(framework)           PI0 / PI05 `_prepare_examples`
    worker_collate(cfg, collate_fn)                  the collate function that does the workers' part
"""

import inspect
import json
import pickle
import re
import shutil
import threading
import types
import uuid
import weakref
import zlib
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.share_tools import add_discretized_state_to_instruction
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


# ---------------------------------------------------------------------------------------------------------------------
# VLM inputs
# ---------------------------------------------------------------------------------------------------------------------
# Build the VLM inputs of a training batch in the data-loader workers.
#
# A framework's forward() hands the images and prompts of a batch to its backbone interface:
#
#     qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
#
# That call is the VLM's processor: chat template, tokenization, image resize / normalization / patchification. It is
# CPU work (about 0.19 s per step at batch 16 x 2 images) done in the training process while the GPU waits. The
# data-loader workers have the samples first and spare time, so they make the same call and send the result along
# with the batch. In the training process the interface then returns what the worker built.
#
# No framework and no interface contains code for this:
#
#   * the worker runs the interface's own `build_qwenvl_inputs`, on a stand-in for the interface that has the
#     processor and the config but no model. `prepare_inputs_in_workers` checks once, on a small batch, that the
#     stand-in returns exactly what the interface returns; an interface whose input construction needs more than
#     that is left alone.
#   * the worker's result is used only for a call with the images of that batch (the same objects, or copies with the
#     same pixels) and the very prompts the worker used. Any other call (inference, a framework that edits the images,
#     a prompt the workers did not expect) is answered by the interface as before.
#
# The one thing a worker cannot see is how a framework edits the prompt before the call. Most frameworks pass the
# sample's "lang" unchanged, which is the default; the others register their edit with @prompt("<framework name>")
# below. A missing or wrong registration costs the speedup, never correctness, and is reported once.
#
# Workers also compute the multimodal rotary position ids (Qwen2.5-VL, Qwen3-VL), which the model otherwise derives
# at the start of every forward with a per-sample host loop. They are not part of what `build_qwenvl_inputs` returns
# (its keys stay the interface's own, e.g. for generate()): they ride on the returned `input_ids` tensor, and a
# forward pre-hook of the model passes them on only when it is called with exactly the returned tensors, unchanged.
#
# Everything set on an interface or a framework here is an instance of a module-level class that holds its owner, not
# a closure, so a deep copy of the model (an EMA or teacher copy) gets its own, and the model stays picklable. What the
# workers build carries the key of the interface it was built for, and only that interface (or a copy of it) takes it.


# ---------------------------------------------------------------------------------------------- prompts

_PROMPTS = {}  # framework name -> factory(cfg) -> function(examples) -> the instructions the framework will pass


def prompt(framework_name):
    """Register how a framework edits the prompt: a factory that gets the training config and returns a picklable
    function from the samples of a batch to the instructions of the framework's `build_qwenvl_inputs` call (a tuple
    of them for a framework that makes several calls per step)."""

    def register(factory):
        _PROMPTS[framework_name] = factory
        return factory

    return register


# Frameworks whose training forward calls build_qwenvl_inputs with more than images and instructions (QwenFast passes
# the action tokens as `solutions`): the workers' inputs would never be used in training, so none are built.
_OTHER_CALL_IN_TRAINING = {"QwenFast"}


def _lang(examples):
    return [example["lang"] for example in examples]


def _lang_with_state(examples):
    instructions = _lang(examples)
    if "state" in examples[0]:
        instructions = add_discretized_state_to_instruction(instructions, [example["state"] for example in examples])
    return instructions


def _with_action_slots(examples, instructions_of, chunk_len, slots):
    suffix = f" Please predict the next {chunk_len} robot actions: <action>{'🔍' * slots}<action>."
    return [instruction + suffix for instruction in instructions_of(examples)]


def _query_before_and_after(examples, query):
    return [query + example["lang"] for example in examples], [example["lang"] + query for example in examples]


@prompt("QwenOFT")
def _qwen_oft(cfg):
    chunk_len = int(cfg.framework.action_model.action_horizon)
    return partial(_with_action_slots, instructions_of=_lang_with_state, chunk_len=chunk_len, slots=chunk_len)


@prompt("QwenAdapter")
def _qwen_adapter(cfg):
    action_model = cfg.framework.action_model
    return partial(
        _with_action_slots,
        instructions_of=_lang,
        chunk_len=action_model.get("num_actions_chunk", None),
        slots=action_model.get("action_query_num", 64),
    )


@prompt("QwenPI_v3")
def _qwen_pi_v3(cfg):
    return _lang_with_state


@prompt("LangForce")
def _lang_force(cfg):
    queries = int(cfg.framework.qwenvl.get("num_latent_action_query", 64))
    return partial(_query_before_and_after, query="".join(f"<|action_{i}|>" for i in range(queries)))


# ---------------------------------------------------------------------------------------------- worker side


class _InterfaceStandIn(SimpleNamespace):
    """A framework without its model: its processor, config and plain settings (strings, numbers), and the methods of
    its class bound to the stand-in (MiniCPM-RobotManip's `_format_instruction`)."""

    def __getattr__(self, name):
        cls = self.__dict__.get("_class")
        method = getattr(cls, name, None) if cls is not None and not name.startswith("__") else None
        if isinstance(inspect.getattr_static(cls, name, None), (staticmethod, classmethod)):
            return method
        if callable(method):
            return types.MethodType(method, self)
        raise AttributeError(name)


def _plain(owner):
    """An object's plain settings: what a stand-in for it carries besides the processor and the config."""
    return {k: v for k, v in vars(owner).items() if not k.startswith("_") and isinstance(v, (str, int, float, bool))}


def _unwrap(config):
    inner = vars(config).get("_cfg") if hasattr(config, "__dict__") else None  # the trainer's access-tracking wrapper
    return inner if OmegaConf.is_config(inner) else config


def _config_state(config):
    config = _unwrap(config)
    return repr(OmegaConf.to_container(config, resolve=False)) if OmegaConf.is_config(config) else type(config).__name__


def _processor_state(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    state = [
        type(processor).__name__,
        getattr(processor, "chat_template", None),
        getattr(tokenizer, "chat_template", None),
        getattr(tokenizer, "padding_side", None),
        getattr(tokenizer, "model_max_length", None),
        str(getattr(tokenizer, "added_tokens_decoder", None)),  # add_tokens / add_special_tokens
    ]
    for name in ("image_processor", "video_processor"):
        part = getattr(processor, name, None)
        state.append(json.dumps(part.to_dict(), sort_keys=True, default=str) if hasattr(part, "to_dict") else None)
    return state


def _state(owner, method):
    """What the input construction reads besides its arguments, as one number: the processor's settings, the added
    tokens, the config and (for a framework's own builder) its plain settings. The workers got a copy of all of it
    when the DataLoader was made; a change made afterwards must send the call back to the training process."""
    state = [_processor_state(getattr(owner, "processor", None)), _config_state(getattr(owner, "config", None))]
    if method != "build_qwenvl_inputs":
        state.append(sorted(_plain(owner).items()))
    return zlib.crc32(repr(state).encode())


def _content(*groups):
    """A checksum of the contents of images, arrays and strings (nested lists of them). The workers send it with what
    they built from these objects: a batch whose objects were edited in place afterwards no longer matches. None for
    an object it cannot read."""
    crc = 0
    stack = [groups]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            crc = zlib.crc32(b"[%d" % len(item), crc)
            stack.extend(reversed(item))
        elif isinstance(item, Image.Image):
            crc = zlib.crc32(item.tobytes(), zlib.crc32(f"{item.mode}{item.size}".encode(), crc))
        elif isinstance(item, np.ndarray):
            crc = zlib.crc32(np.ascontiguousarray(item).tobytes(), zlib.crc32(f"{item.dtype}{item.shape}".encode(), crc))
        elif torch.is_tensor(item):
            stack.append(item.detach().cpu().numpy())
        elif item is None or isinstance(item, (str, int, float, bool)):
            crc = zlib.crc32(repr(item).encode(), crc)
        else:
            return None
    return crc


class Builder:
    """What a worker needs to make an interface's `build_qwenvl_inputs` call (or a framework's own
    `_build_vlm_inputs`) without the model. Pickled when the DataLoader is made and unpickled in each worker, which
    gives every worker its own processor."""

    def __init__(self, interface, method="build_qwenvl_inputs"):
        self.interface_class = type(interface)
        self.method = method
        self.key = uuid.uuid4().hex  # of this interface: what a worker builds with it is for it alone
        self.use_rope = True
        self._owner = weakref.ref(interface)
        self.snapshot()

    def __getstate__(self):  # the workers get the snapshot, not the interface
        return {k: v for k, v in self.__dict__.items() if k != "_owner"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._owner = None

    def snapshot(self):
        """Take the interface's processor, config and settings as they are now. False if the interface is gone."""
        interface = self._owner() if self._owner is not None else None
        if interface is None:
            return False
        self.rope = None
        self.state = _state(interface, self.method)
        if self.method != "build_qwenvl_inputs":  # a framework's own builder
            self.stand_in = _InterfaceStandIn(
                _class=type(interface),
                processor=interface.processor,
                config=interface.config,
                model=SimpleNamespace(device="cpu"),
                **_plain(interface),
            )
            return True
        self.stand_in = SimpleNamespace(
            processor=interface.processor, config=interface.config, model=SimpleNamespace(device="cpu")
        )
        core = getattr(interface.model, "model", None)  # e.g. Qwen3VLModel inside Qwen3VLForConditionalGeneration
        if (
            self.use_rope
            and isinstance(interface.model, torch.nn.Module)  # the position ids reach it through a forward pre-hook
            and callable(getattr(type(core), "get_rope_index", None))
            and getattr(core, "config", None) is not None
        ):
            self.rope = (type(core), core.config)
        return True

    def __call__(self, images, instructions):
        inputs = getattr(self.interface_class, self.method)(self.stand_in, images, instructions)
        if self.rope is not None and "position_ids" not in inputs and "pixel_values_videos" not in inputs:
            model_class, config = self.rope
            # get_rope_index reads only the model config and these inputs
            inputs[_POSITION_IDS], _ = model_class.get_rope_index(
                SimpleNamespace(config=config),
                input_ids=inputs["input_ids"],
                image_grid_thw=inputs.get("image_grid_thw"),
                attention_mask=inputs["attention_mask"],
            )
        return inputs


_POSITION_IDS = "_starvla_position_ids"  # the workers' position ids: a private key, never returned to a caller
_inputs_arrived = weakref.WeakValueDictionary()  # id(first image of a batch) -> its Prepared, in the consuming process
_inputs_lock = threading.Lock()  # batches arrive in the DataLoader's pinning thread


def _same_picture(a, b):
    """Two PIL images with the same pixels (a framework that resizes to the size the images already have)."""
    try:
        return a.mode == b.mode == "RGB" and a.size == b.size and a.tobytes() == b.tobytes()
    except AttributeError:
        return False


class Prepared:
    """The inputs a worker built for one batch: one [instructions, inputs] pair per call the framework will make.
    It travels as batch[0]["_prepared"] and, where the batch is consumed, is found again through the batch's images."""

    def __init__(self, images, calls, key, state):
        self.images = [tuple(sample) for sample in images]  # the same objects as in the batch, also after pickling
        self.calls = calls
        self.key, self.state = key, state  # of the Builder that made them, and of what it had read
        self.content = _content(self.images)  # of the images they were made from
        self._announce()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._announce()

    def _announce(self):
        if self.images and self.images[0]:
            with _inputs_lock:
                _inputs_arrived[id(self.images[0][0])] = self

    def pin_memory(self):  # called by the DataLoader's pinning thread
        for _, inputs in self.calls:
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.pin_memory()
        return self

    def same_images(self, images):
        """These are my images (the same objects, or copies with the same pixels), with the contents the worker saw."""
        return (
            len(images) == len(self.images)
            and all(
                len(sample) == len(mine)
                and all(a is b or _same_picture(a, b) for a, b in zip(sample, mine, strict=False))
                for sample, mine in zip(images, self.images, strict=False)
            )
            and self.content is not None
            and _content([tuple(sample) for sample in images]) == self.content
        )


class Collate:
    """`collate_fn`, followed by the interface's input construction for the batch it returns."""

    def __init__(self, collate_fn, builder, instructions_of):
        self.collate_fn = collate_fn
        self.instructions_of = instructions_of
        builder.snapshot()  # the interface as it is now, when the DataLoader is made
        self.payload = pickle.dumps(builder)
        self._builder = None

    def __getstate__(self):
        return {**self.__dict__, "_builder": None}

    def __call__(self, batch):
        batch = self.collate_fn(batch)
        if not (
            isinstance(batch, list)
            and batch
            and isinstance(batch[0], dict)
            and "image" in batch[0]
            and "lang" in batch[0]
        ):
            return batch
        if self._builder is None:  # once per worker process
            self._builder = pickle.loads(self.payload)
        images = [example["image"] for example in batch]
        calls = self.instructions_of(batch)
        calls = calls if isinstance(calls, tuple) else (calls,)
        batch[0]["_prepared"] = Prepared(
            images,
            [[list(instructions), self._builder(images, instructions)] for instructions in calls],
            self._builder.key,
            self._builder.state,
        )
        return batch


# ---------------------------------------------------------------------------------------------- training process

_builder = None  # the Builder of the interface most recently accelerated in this process
input_stats = {"used": 0, "other prompt": 0, "changed since": 0}
_reported = threading.Event()


def _take_inputs(images, instructions, key, state):
    """The inputs a worker built for exactly this call of the interface with this key, or None. They are handed
    over once: a second identical call for the same batch is answered by the interface."""
    try:
        with _inputs_lock:
            waiting = _inputs_arrived.get(id(images[0][0]))  # the batch's own image objects
            waiting = [waiting] if waiting is not None else list(_inputs_arrived.values())  # or copies of them
    except (IndexError, KeyError, TypeError):
        return None
    prepared = next((c for c in waiting if c.key == key and c.same_images(images)), None)
    if prepared is None:
        return None
    if prepared.state != state:  # the processor, the tokens or the config changed after the DataLoader was made
        input_stats["changed since"] += 1
        return None
    instructions = list(instructions)
    for call in prepared.calls:
        if call[0] == instructions and call[1] is not None:
            input_stats["used"] += 1
            inputs, call[1] = call[1], None
            return inputs
    if any(call[0] == instructions for call in prepared.calls):
        return None  # taken already
    input_stats["other prompt"] += 1
    if not _reported.is_set():
        _reported.set()
        logger.info(
            "worker_prep: this framework edits the prompt in a way the data-loader workers do not "
            "know, so the inputs are built in the training process. Register the edit with @prompt in "
            f"starVLA/dataloader/worker_prep.py.\n  framework: {instructions[0]!r}\n  workers:   "
            f"{prepared.calls[0][0][0]!r}"
        )
    return None


def _same(a, b):
    return a.keys() == b.keys() and all(
        torch.equal(a[key].cpu(), b[key].cpu()) and a[key].dtype == b[key].dtype for key in a.keys()
    )


def _probe_inputs(interface, builder):
    """Does a worker's Builder return what the interface (or framework) returns? Two samples of different prompt
    length, two views."""
    rng = np.random.default_rng(0)
    images = [[Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)] for _ in range(2)]
    instructions = ["pick up the block", "open the top drawer and put the bowl inside"]
    theirs = getattr(interface, builder.method)(images, instructions)
    ours = pickle.loads(pickle.dumps(builder))(images, instructions)
    if _POSITION_IDS in ours:
        position_ids, _ = interface.model.model.get_rope_index(
            input_ids=theirs["input_ids"],
            image_grid_thw=theirs.get("image_grid_thw"),
            attention_mask=theirs["attention_mask"],
        )
        if not torch.equal(position_ids.cpu(), ours[_POSITION_IDS]):
            return False
        ours = {key: value for key, value in ours.items() if key != _POSITION_IDS}
    return _same(dict(theirs), dict(ours))


def prepare_inputs_in_workers(interface):
    """Check that workers can build this interface's inputs, and let it return what they built."""
    return _install(interface, "build_qwenvl_inputs")


def prepare_framework_inputs_in_workers(framework):
    """The same for a framework that builds its backbone's inputs itself, with its own processor
    (`_build_vlm_inputs(images, instructions)`: MiniCPM-RobotManip). The workers run the framework's method on a
    stand-in without the model; the same probe decides."""
    return _install(framework, "_build_vlm_inputs")


def _versions(*tensors):
    """In-place modification counters; None for a missing tensor or one made under inference mode (which has none)."""
    return tuple(None if t is None or t.is_inference() else t._version for t in tensors)


def _attach_position_ids(inputs, position_ids):
    """Leave the workers' position ids on the `input_ids` tensor just returned, with what they were derived from."""
    ids, mask, grid = inputs.get("input_ids"), inputs.get("attention_mask"), inputs.get("image_grid_thw")
    if ids is None or mask is None or any(t.is_inference() for t in (ids, mask, position_ids)):
        return  # e.g. predict_action under inference mode: the model derives them, as before
    ids._starvla_position_ids = (position_ids, mask, grid, _versions(ids, mask, grid))


def _give_position_ids(module, args, kwargs):
    """Forward pre-hook of the backbone's model. A plain forward on exactly the tensors `build_qwenvl_inputs`
    returned, none of them modified since, gets the position ids the workers derived from them. Every other call
    (generate(), a framework that edits or re-creates the inputs, position ids given by the caller) is untouched."""
    ids = kwargs.get("input_ids")
    stash = getattr(ids, "_starvla_position_ids", None)
    if stash is None or args:
        return None
    if any(
        kwargs.get(k) is not None for k in ("position_ids", "inputs_embeds", "past_key_values", "pixel_values_videos")
    ):
        return None
    position_ids, mask, grid, versions = stash
    if kwargs.get("attention_mask") is not mask or kwargs.get("image_grid_thw") is not grid:
        return None
    if _versions(ids, mask, grid) != versions:
        return None
    return args, {**kwargs, "position_ids": position_ids}


class _BuildInputs:
    """An interface's `build_qwenvl_inputs` (or a framework's `_build_vlm_inputs`), replaced on the instance: the
    inputs a worker built for exactly this call, or the method it replaces."""

    def __init__(self, owner, method, key):
        self.owner, self.method, self.key = owner, method, key
        self.replaced = vars(owner).get(method)  # a method someone else had set on the instance, if any

    def original(self, *args, **kwargs):
        if self.replaced is not None:
            return self.replaced(*args, **kwargs)
        return getattr(type(self.owner), self.method)(self.owner, *args, **kwargs)

    @property
    def __signature__(self):
        return inspect.signature(
            self.replaced if self.replaced is not None else getattr(type(self.owner), self.method).__get__(self.owner)
        )

    def __call__(self, images, instructions, solutions=None, **kwargs):
        prepared = None
        if solutions is None and not kwargs and _inputs_arrived:
            prepared = _take_inputs(images, instructions, self.key, _state(self.owner, self.method))
        if prepared is None:
            if solutions is not None:
                kwargs["solutions"] = solutions
            return self.original(images, instructions, **kwargs)
        device = self.owner.model.device
        data = {key: value.to(device) for key, value in prepared.items()}
        position_ids = data.pop(_POSITION_IDS, None)
        inputs = type(prepared)(data=data)  # with the keys the interface itself returns
        if position_ids is not None:
            _attach_position_ids(inputs, position_ids)
        return inputs


def _install(interface, method):
    global _builder
    original = getattr(interface, method, None)
    if not callable(original) or getattr(interface, "processor", None) is None or not hasattr(interface, "model"):
        return False
    if isinstance(original, _BuildInputs):
        return True
    try:
        builder = Builder(interface, method)
        try:
            works = _probe_inputs(interface, builder)
        except Exception:
            works = False
        if not works and builder.rope is not None:  # position ids this model derives differently: only the processor
            builder.use_rope = False
            builder.snapshot()
            works = _probe_inputs(interface, builder)
        if not works:
            return False
    except Exception as error:  # an interface that builds its inputs differently: leave it alone
        logger.info(f"worker_prep: {type(interface).__name__} not supported ({type(error).__name__}: {error})")
        return False

    setattr(interface, method, _BuildInputs(interface, method, builder.key))
    if builder.rope is not None and _give_position_ids not in interface.model._forward_pre_hooks.values():
        interface.model.register_forward_pre_hook(_give_position_ids, with_kwargs=True)
    _builder = builder
    return True


def _recipe(config):
    """The framework and backbones a config names: a DataLoader made for another recipe than the model that was set
    up here (a second model in the process) gets no worker-side construction."""
    try:
        framework = _unwrap(config).framework
        return tuple(
            str(OmegaConf.select(framework, key, default=None))
            for key in ("name", "qwenvl.base_vlm", "world_model.base_wm")
        )
    except Exception:
        return None


def inputs_collate(cfg, collate_fn):
    """The collate function for the workers; `collate_fn` itself when no interface was set up."""
    name = str(cfg.framework.get("name", ""))
    if _builder is None or isinstance(collate_fn, Collate) or name in _OTHER_CALL_IN_TRAINING:
        return collate_fn  # nothing to build, wrapped already, or nothing the framework would use
    interface = _builder._owner() if _builder._owner is not None else None
    if interface is None or _recipe(getattr(interface, "config", None)) not in (None, _recipe(cfg)):
        return collate_fn  # that model is gone, or this DataLoader is for another one
    factory = _PROMPTS.get(name)
    return Collate(collate_fn, _builder, factory(cfg) if factory is not None else _lang)


# ---------------------------------------------------------------------------------------------------------------------
# World-model videos
# ---------------------------------------------------------------------------------------------------------------------
# video_prep: the observation images of a world-model backbone resized and normalized in the data-loader
# workers. Every step the Wan / Cosmos-Predict2 interface passes each sample's views through diffusers' VideoProcessor
# (resize to 480 x 832 for Wan, 320 x 576 for Cosmos-Predict2, then scale to [-1, 1]) on the CPU of the training process
# while the GPU waits. The workers have the samples first: they make the same call with a copy of the same processor
# and send the tensors with the batch. The interface's call for a sample whose images are the batch's own objects gets
# the worker's tensor; any other call (inference, a framework that edits the images) runs as before.


_SIZE = re.compile(r"height, width = (\d+), (\d+)")
_videos = {"builder": None}  # (VideoProcessor, height, width, key, state) of the backbone most recently set up
_video_arrived = weakref.WeakValueDictionary()  # id(first image of a sample) -> the PreparedVideos holding it
_video_lock = threading.Lock()
video_stats = {"used": 0}


def _video_size(backbone):
    for name in ("_encode_images_vae", "_encode_images"):  # Wan, Cosmos-Predict2
        try:
            source = inspect.getsource(getattr(type(backbone), name))
        except (AttributeError, OSError, TypeError):
            continue
        match = _SIZE.search(source)
        if match and "self.video_processor.preprocess_video(sample_imgs, height=height, width=width)" in source:
            return int(match.group(1)), int(match.group(2))
    return None


def _video_state(processor):
    """The settings of a diffusers VideoProcessor, as one number."""
    return zlib.crc32(repr(sorted(dict(getattr(processor, "config", {})).items())).encode())


class PreparedVideos:
    """The preprocessed views of one batch, one tensor per sample. Found again through each sample's first image."""

    def __init__(self, images, videos, size, key, state):
        self.images, self.videos, self.size = [tuple(sample) for sample in images], videos, size
        self.key, self.state = key, state  # of the backbone they are for, and of the processor that made them
        self.contents = [_content(sample) for sample in self.images]  # of the images they were made from
        self._announce()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._announce()

    def _announce(self):
        with _video_lock:
            for sample in self.images:
                if sample:
                    _video_arrived[id(sample[0])] = self

    def pin_memory(self):
        self.videos = [video.pin_memory() for video in self.videos]
        return self

    def take(self, sample, size, key, state):
        """The tensor for this sample: for this backbone, with its processor's settings unchanged, the sample's own
        image objects with the contents the worker saw. Handed over once."""
        if size != self.size or key != self.key or state != self.state:
            return None
        for index, mine in enumerate(self.images):
            if len(mine) == len(sample) and all(a is b for a, b in zip(mine, sample, strict=False)):
                if self.contents[index] is None or _content(tuple(sample)) != self.contents[index]:
                    return None
                video, self.videos[index] = self.videos[index], None
                return video
        return None


class VideoCollate:
    """`collate_fn`, followed by the backbone's preprocessing of each sample's views."""

    def __init__(self, collate_fn, builder):
        self.collate_fn, self.builder = collate_fn, builder

    def __call__(self, batch):
        batch = self.collate_fn(batch)
        if not (isinstance(batch, list) and batch and isinstance(batch[0], dict) and "image" in batch[0]):
            return batch
        processor, height, width, key, state = self.builder
        images = [
            sample["image"] if isinstance(sample["image"], (list, tuple)) else [sample["image"]] for sample in batch
        ]
        videos = [processor.preprocess_video(list(sample), height=height, width=width) for sample in images]
        batch[0]["_videos"] = PreparedVideos(images, videos, (height, width), key, state)
        return batch


class _VideoProcessor:
    """The backbone's VideoProcessor, answering with the workers' tensor where there is one."""

    def __init__(self, processor, key):
        self._processor = processor
        self._key = key  # what the workers preprocess with this backbone's processor is for it alone

    def __getattr__(self, name):
        processor = self.__dict__.get("_processor")
        if processor is None:  # an empty instance, while it is copied or unpickled
            raise AttributeError(name)
        return getattr(processor, name)

    def preprocess_video(self, video, height=None, width=None, **kwargs):
        if not kwargs and isinstance(video, (list, tuple)) and video:
            with _video_lock:
                prepared = _video_arrived.get(id(video[0]))
            found = None
            if prepared is not None:
                found = prepared.take(video, (height, width), self._key, _video_state(self._processor))
            if found is not None:
                video_stats["used"] += 1
                return found
        return self._processor.preprocess_video(video, height=height, width=width, **kwargs)


def prepare_videos_in_workers(backbone):
    """A world-model interface whose `_encode_images_vae` preprocesses each sample with its
    VideoProcessor at a fixed size gets the tensors the data-loader workers made."""
    import pickle

    processor = getattr(backbone, "video_processor", None)
    size = _video_size(backbone)
    if processor is None or size is None:
        return False
    if isinstance(processor, _VideoProcessor):
        return True
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    sample = [Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)]
    try:
        copy = pickle.loads(pickle.dumps(processor))  # what a worker has
        if not torch.equal(
            processor.preprocess_video(sample, height=size[0], width=size[1]),
            copy.preprocess_video(sample, height=size[0], width=size[1]),
        ):
            return False
    except Exception:  # a processor that cannot travel to the workers: leave it alone
        return False
    key = uuid.uuid4().hex
    backbone.video_processor = _VideoProcessor(processor, key)
    _videos["builder"] = (copy, size[0], size[1], key, _video_state(processor))
    return True


def video_collate(cfg, collate_fn):
    """The collate function for the workers; `collate_fn` itself when no backbone was set up."""
    if _videos["builder"] is None or isinstance(collate_fn, VideoCollate):
        return collate_fn
    return VideoCollate(collate_fn, _videos["builder"])


# ---------------------------------------------------------------------------------------------------------------------
# PI0 / PI05 examples
# ---------------------------------------------------------------------------------------------------------------------
# worker_examples: the sample preparation of the OpenPI frameworks (PI0, PI05) in the data-loader workers.
#
# These frameworks do not use a `get_vlm_model` backbone, so this acts on the framework and recognises them by
# what it patches. The framework's `_prepare_examples` (every view resized and padded to the model's resolution and
# scaled, missing views filled in, state and actions padded: CPU work on the batch as the data set returns it) runs in
# the data-loader workers, on a stand-in for the framework that has its settings and methods but no model.
# `prepare_examples_in_workers` checks once that the stand-in returns exactly what the framework returns. In the
# training process the framework
# gets the worker's tensors for a batch whose images, prompts, states and actions are the objects the worker saw;
# any other call (inference, an edited batch) is prepared as before. Bitwise identical.


_SETTINGS = (bool, int, float, str, tuple)
_examples_arrived = (
    weakref.WeakValueDictionary()
)  # id(first image of a batch) -> its PreparedExamples, where it is consumed
_examples_lock = threading.Lock()  # batches arrive in the DataLoader's pinning thread
example_stats = {"used": 0}
_framework_stand_in = None  # of the framework most recently accelerated in this process


def _settings(framework):
    """A framework's configs and plain settings: what `_prepare_examples` can read without the model."""
    settings = {}
    for name, value in vars(framework).items():
        if name.startswith("_"):
            continue
        inner = _unwrap(value)
        if OmegaConf.is_config(inner):
            settings[name] = inner
        elif isinstance(value, _SETTINGS):
            settings[name] = value
    return settings


def _settings_state(settings):
    return zlib.crc32(
        repr(sorted((k, _config_state(v) if OmegaConf.is_config(v) else repr(v)) for k, v in settings.items())).encode()
    )


class FrameworkStandIn:
    """A framework's plain settings and its class's methods, without the model: enough for `_prepare_examples`."""

    def __init__(self, framework):
        self.framework_class = type(framework)
        self.key = uuid.uuid4().hex  # of this framework: what a worker prepares with it is for it alone
        self._owner = weakref.ref(framework)
        self.snapshot()

    def __getstate__(self):  # the workers get the snapshot, not the framework
        return {k: v for k, v in self.__dict__.items() if k != "_owner"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._owner = None

    def snapshot(self):
        """Take the framework's settings as they are now. False if the framework is gone."""
        framework = self._owner() if self._owner is not None else None
        if framework is None:
            return False
        self.settings = _settings(framework)
        self.state = _settings_state(self.settings)
        return True

    def __getattr__(self, name):
        state = self.__dict__
        if "settings" not in state:  # while unpickling
            raise AttributeError(name)
        if name in state["settings"]:
            return state["settings"][name]
        method = getattr(state["framework_class"], name)
        if not isinstance(method, types.FunctionType):
            raise AttributeError(name)
        return types.MethodType(method, self)


def _parts(example):
    """The objects of a sample that `_prepare_examples` reads."""
    images = example["image"]
    images = tuple(images) if isinstance(images, (list, tuple)) else (images,)
    return (
        images,
        example.get("lang", example.get("language")),
        example.get("state"),
        example.get("action"),
        example.get("image_mask"),
    )


def _same_parts(mine, theirs):
    (images, lang, *arrays), (their_images, their_lang, *their_arrays) = mine, theirs
    return (
        len(images) == len(their_images)
        and all(a is b for a, b in zip(images, their_images, strict=False))
        and lang == their_lang
        and all(a is b for a, b in zip(arrays, their_arrays, strict=False))
    )


class PreparedExamples:
    """What a worker's `_prepare_examples` returned for one batch. It travels as batch[0]["_prepared_examples"]."""

    def __init__(self, batch, tensors, key, state):
        self.parts = [_parts(example) for example in batch]  # the same objects as in the batch, also after pickling
        self.tensors = tensors
        self.key, self.state = key, state  # of the stand-in that made them, and of the settings it had
        self.content = _content(self.parts)  # of the images, prompts, states and actions they were made from
        self._announce()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._announce()

    def _announce(self):
        with _examples_lock:
            _examples_arrived[id(self.parts[0][0][0])] = self

    def pin_memory(self):  # called by the DataLoader's pinning thread
        self.tensors = {
            key: value.pin_memory() if torch.is_tensor(value) else value for key, value in self.tensors.items()
        }
        return self


class ExamplesCollate:
    """`collate_fn`, followed by the framework's `_prepare_examples` for the batch it returns."""

    def __init__(self, collate_fn, stand_in):
        stand_in.snapshot()  # the framework as it is now, when the DataLoader is made
        self.collate_fn, self.stand_in = collate_fn, stand_in

    def __call__(self, batch):
        batch = self.collate_fn(batch)
        if (
            isinstance(batch, list)
            and batch
            and isinstance(batch[0], dict)
            and "image" in batch[0]
            and "action" in batch[0]
        ):
            batch[0]["_prepared_examples"] = PreparedExamples(
                batch, self.stand_in._prepare_examples(batch, True), self.stand_in.key, self.stand_in.state
            )
        return batch


def _take_examples(examples, key, state):
    """What a worker prepared for exactly this batch of the framework with this key, or None: the batch's own
    objects, with the contents the worker saw, and the framework's settings unchanged. Handed over once."""
    try:
        with _examples_lock:
            prepared = _examples_arrived.get(id(_parts(examples[0])[0][0]))
        if (
            prepared is not None
            and prepared.key == key
            and prepared.state == state
            and prepared.tensors is not None
            and len(examples) == len(prepared.parts)
            and all(_same_parts(mine, _parts(example)) for mine, example in zip(prepared.parts, examples, strict=False))
            and prepared.content is not None
            and _content([_parts(example) for example in examples]) == prepared.content
        ):
            example_stats["used"] += 1
            if example_stats["used"] == 1:
                logger.info("worker_examples: training batches arrive prepared by the data-loader workers")
            tensors, prepared.tensors = prepared.tensors, None
            return dict(tensors)
    except (IndexError, KeyError, TypeError):
        pass
    return None


def _probe_examples(framework, stand_in):
    import pickle

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    examples = [
        {
            "image": [
                Image.fromarray(rng.integers(0, 256, (200 + 24 * views, 224, 3), dtype=np.uint8)) for _ in range(views)
            ],
            "lang": "pick up the block",
            "state": rng.uniform(-1, 1, (1, 5)).astype(np.float32),
            "action": rng.uniform(-1, 1, (min(3, framework.action_horizon), min(5, framework.action_dim))).astype(
                np.float32
            ),
        }
        for views in (1, 2)
    ]
    theirs = framework._prepare_examples(examples, True)
    ours = pickle.loads(pickle.dumps(stand_in))._prepare_examples(examples, True)
    return theirs.keys() == ours.keys() and all(
        torch.equal(theirs[key], ours[key]) and theirs[key].dtype == ours[key].dtype
        if torch.is_tensor(theirs[key])
        else theirs[key] == ours[key]
        for key in theirs
    )


class _PrepareExamples:
    """A framework's `_prepare_examples`, replaced on the instance: what a worker prepared for exactly this batch, or
    the method it replaces."""

    def __init__(self, owner, key):
        self.owner, self.key = owner, key
        self.replaced = vars(owner).get("_prepare_examples")  # a method someone else had set on the instance, if any

    def original(self, *args, **kwargs):
        if self.replaced is not None:
            return self.replaced(*args, **kwargs)
        return type(self.owner)._prepare_examples(self.owner, *args, **kwargs)

    def __call__(self, examples, include_actions, *args, **kwargs):
        prepared = None
        if include_actions and not args and not kwargs and _examples_arrived:
            prepared = _take_examples(examples, self.key, _settings_state(_settings(self.owner)))
        return prepared if prepared is not None else self.original(examples, include_actions, *args, **kwargs)


def prepare_examples_in_workers(framework):
    """Check that workers can run this framework's `_prepare_examples`, and let it return theirs."""
    global _framework_stand_in
    original = getattr(framework, "_prepare_examples", None)
    if not callable(original):
        return False
    if isinstance(original, _PrepareExamples):
        return True
    try:
        stand_in = FrameworkStandIn(framework)
        if not _probe_examples(framework, stand_in):
            return False
    except Exception as error:  # a framework that prepares its samples differently: leave it alone
        logger.info(f"worker_examples: {type(framework).__name__} not supported ({type(error).__name__}: {error})")
        return False

    framework._prepare_examples = _PrepareExamples(framework, stand_in.key)
    _framework_stand_in = stand_in
    return True


def examples_collate(cfg, collate_fn):
    """The collate function for the workers; `collate_fn` itself when no framework was set up."""
    stand_in = _framework_stand_in
    framework = stand_in._owner() if stand_in is not None and stand_in._owner is not None else None
    if framework is None or isinstance(collate_fn, ExamplesCollate):
        return collate_fn  # no framework was set up (or it is gone), or wrapped already
    if _recipe(getattr(framework, "config", None)) not in (None, _recipe(cfg)):
        return collate_fn  # this DataLoader is for another model
    return ExamplesCollate(collate_fn, stand_in)


_MIN_SHARED_MEMORY = 4 << 30


def _shared_memory_is_small():
    """Workers send tensors through shared memory (/dev/shm), where the samples themselves (PIL images, arrays)
    travel through a pipe: tens to hundreds of MiB per batch in flight. A container with the default 64 MiB would
    fail with a bus error, so with less than 4 GiB free the batches are left as they are."""
    try:
        return shutil.disk_usage("/dev/shm").free < _MIN_SHARED_MEMORY
    except OSError:  # no /dev/shm: not Linux, nothing to conclude
        return False


def worker_collate(cfg, collate_fn):
    """`collate_fn`, followed by the input construction that was set up above when the model was built. Each wrapper
    is `collate_fn` itself when nothing was set up."""
    wrapped = collate_fn
    for wrap in (inputs_collate, video_collate, examples_collate):
        wrapped = wrap(cfg, wrapped)
    if wrapped is not collate_fn and _shared_memory_is_small():
        logger.info(
            "worker_prep: /dev/shm has less than 4 GiB free, so the data-loader workers do not build the model "
            "inputs (their tensors travel through shared memory). Raise it (docker run --shm-size) to enable."
        )
        return collate_fn
    return wrapped
