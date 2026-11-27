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
import pickle
import re
import threading
import types
import weakref
from functools import partial
from types import SimpleNamespace

import torch

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
# at the start of every forward with a per-sample host loop.


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


class Builder:
    """What a worker needs to make an interface's `build_qwenvl_inputs` call (or a framework's own
    `_build_vlm_inputs`) without the model. Pickled once in the training process and unpickled in each worker, which
    gives every worker its own processor."""

    def __init__(self, interface, method="build_qwenvl_inputs"):
        self.interface_class = type(interface)
        self.method = method
        self.rope = None
        if method != "build_qwenvl_inputs":  # a framework's own builder
            plain = {
                k: v
                for k, v in vars(interface).items()
                if not k.startswith("_") and isinstance(v, (str, int, float, bool))
            }
            self.stand_in = _InterfaceStandIn(
                _class=type(interface),
                processor=interface.processor,
                config=interface.config,
                model=SimpleNamespace(device="cpu"),
                **plain,
            )
            return
        self.stand_in = SimpleNamespace(
            processor=interface.processor, config=interface.config, model=SimpleNamespace(device="cpu")
        )
        core = getattr(interface.model, "model", None)  # e.g. Qwen3VLModel inside Qwen3VLForConditionalGeneration
        if callable(getattr(type(core), "get_rope_index", None)) and getattr(core, "config", None) is not None:
            self.rope = (type(core), core.config)

    def __call__(self, images, instructions):
        inputs = getattr(self.interface_class, self.method)(self.stand_in, images, instructions)
        if self.rope is not None and "position_ids" not in inputs and "pixel_values_videos" not in inputs:
            model_class, config = self.rope
            # get_rope_index reads only the model config and these inputs
            inputs["position_ids"], _ = model_class.get_rope_index(
                SimpleNamespace(config=config),
                input_ids=inputs["input_ids"],
                image_grid_thw=inputs.get("image_grid_thw"),
                attention_mask=inputs["attention_mask"],
            )
        return inputs


_inputs_arrived = weakref.WeakValueDictionary()  # id(first image of a batch) -> its Prepared, in the consuming process
_inputs_lock = threading.Lock()  # batches arrive in the DataLoader's pinning thread


def _same_picture(a, b):
    """Two PIL images with the same pixels (a framework that resizes to the size the images already have)."""
    try:
        return a.mode == b.mode == "RGB" and a.size == b.size and a.tobytes() == b.tobytes()
    except AttributeError:
        return False


class Prepared:
    """The inputs a worker built for one batch: one (instructions, inputs) pair per call the framework will make.
    It travels as batch[0]["_prepared"] and, where the batch is consumed, is found again through the batch's images."""

    def __init__(self, images, calls):
        self.images = [tuple(sample) for sample in images]  # the same objects as in the batch, also after pickling
        self.calls = calls
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
        return len(images) == len(self.images) and all(
            len(sample) == len(mine) and all(a is b or _same_picture(a, b) for a, b in zip(sample, mine, strict=False))
            for sample, mine in zip(images, self.images, strict=False)
        )


class Collate:
    """`collate_fn`, followed by the interface's input construction for the batch it returns."""

    def __init__(self, collate_fn, builder, instructions_of):
        self.collate_fn = collate_fn
        self.instructions_of = instructions_of
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
            images, [(list(instructions), self._builder(images, instructions)) for instructions in calls]
        )
        return batch


# ---------------------------------------------------------------------------------------------- training process

_builder = None  # the Builder of the interface most recently accelerated in this process
input_stats = {"used": 0, "other prompt": 0}
_reported = threading.Event()


def _take_inputs(images, instructions):
    """The inputs a worker built for exactly this call, or None."""
    try:
        with _inputs_lock:
            waiting = _inputs_arrived.get(id(images[0][0]))  # the batch's own image objects
            waiting = [waiting] if waiting is not None else list(_inputs_arrived.values())  # or copies of them
    except (IndexError, KeyError, TypeError):
        return None
    prepared = next((candidate for candidate in waiting if candidate.same_images(images)), None)
    if prepared is None:
        return None
    instructions = list(instructions)
    for theirs, inputs in prepared.calls:
        if theirs == instructions:
            input_stats["used"] += 1
            return inputs
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
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    images = [[Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)] for _ in range(2)]
    instructions = ["pick up the block", "open the top drawer and put the bowl inside"]
    theirs = getattr(interface, builder.method)(images, instructions)
    ours = pickle.loads(pickle.dumps(builder))(images, instructions)
    if "position_ids" in ours and "position_ids" not in theirs:
        position_ids, _ = interface.model.model.get_rope_index(
            input_ids=theirs["input_ids"],
            image_grid_thw=theirs.get("image_grid_thw"),
            attention_mask=theirs["attention_mask"],
        )
        if not torch.equal(position_ids.cpu(), ours["position_ids"]):
            return False
        ours = {key: value for key, value in ours.items() if key != "position_ids"}
    return _same(dict(theirs), dict(ours))


def prepare_inputs_in_workers(interface):
    """Check that workers can build this interface's inputs, and let it return what they built."""
    return _install(interface, "build_qwenvl_inputs")


def prepare_framework_inputs_in_workers(framework):
    """The same for a framework that builds its backbone's inputs itself, with its own processor
    (`_build_vlm_inputs(images, instructions)`: MiniCPM-RobotManip). The workers run the framework's method on a
    stand-in without the model; the same probe decides."""
    return _install(framework, "_build_vlm_inputs")


def _install(interface, method):
    global _builder
    original = getattr(interface, method, None)
    if not callable(original) or getattr(interface, "processor", None) is None or not hasattr(interface, "model"):
        return False
    if getattr(original, "_starvla_worker_prep", False):
        return True
    try:
        builder = Builder(interface, method)
        try:
            works = _probe_inputs(interface, builder)
        except Exception:
            works = False
        if not works and builder.rope is not None:  # position ids this model derives differently: only the processor
            builder.rope = None
            works = _probe_inputs(interface, builder)
        if not works:
            return False
    except Exception as error:  # an interface that builds its inputs differently: leave it alone
        logger.info(f"worker_prep: {type(interface).__name__} not supported ({type(error).__name__}: {error})")
        return False

    def build_inputs(images, instructions, solutions=None, **kwargs):
        prepared = _take_inputs(images, instructions) if solutions is None and not kwargs else None
        if prepared is None:
            if solutions is not None:
                kwargs["solutions"] = solutions
            return original(images, instructions, **kwargs)
        device = interface.model.device
        return type(prepared)(data={key: value.to(device) for key, value in prepared.items()})  # a new one per call

    build_inputs._starvla_worker_prep = True
    setattr(interface, method, build_inputs)
    _builder = builder
    return True


def inputs_collate(cfg, collate_fn):
    """The collate function for the workers; `collate_fn` itself when no interface was set up."""
    if _builder is None or isinstance(collate_fn, Collate):  # nothing to build, or wrapped already
        return collate_fn
    factory = _PROMPTS.get(str(cfg.framework.get("name", "")))
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
_videos = {"builder": None}  # (VideoProcessor, height, width) of the backbone most recently set up
_video_arrived = weakref.WeakValueDictionary()  # id(first image of a sample) -> the PreparedVideos holding it
_video_lock = threading.Lock()
video_stats = {"used": 0}


def _video_size(backbone):
    encode = getattr(type(backbone), "_encode_images_vae", None)
    try:
        source = inspect.getsource(encode)
    except (OSError, TypeError):
        return None
    match = _SIZE.search(source)
    if match is None or "self.video_processor.preprocess_video(sample_imgs, height=height, width=width)" not in source:
        return None
    return int(match.group(1)), int(match.group(2))


class PreparedVideos:
    """The preprocessed views of one batch, one tensor per sample. Found again through each sample's first image."""

    def __init__(self, images, videos, size):
        self.images, self.videos, self.size = [tuple(sample) for sample in images], videos, size
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

    def take(self, sample, size):
        if size != self.size:
            return None
        for mine, video in zip(self.images, self.videos, strict=False):
            if len(mine) == len(sample) and all(a is b for a, b in zip(mine, sample, strict=False)):
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
        processor, height, width = self.builder
        images = [
            sample["image"] if isinstance(sample["image"], (list, tuple)) else [sample["image"]] for sample in batch
        ]
        videos = [processor.preprocess_video(list(sample), height=height, width=width) for sample in images]
        batch[0]["_videos"] = PreparedVideos(images, videos, (height, width))
        return batch


class _VideoProcessor:
    """The backbone's VideoProcessor, answering with the workers' tensor where there is one."""

    def __init__(self, processor):
        self._processor = processor

    def __getattr__(self, name):
        return getattr(self.__dict__["_processor"], name)

    def preprocess_video(self, video, height=None, width=None, **kwargs):
        if not kwargs and isinstance(video, (list, tuple)) and video:
            with _video_lock:
                prepared = _video_arrived.get(id(video[0]))
            found = prepared.take(video, (height, width)) if prepared is not None else None
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
    backbone.video_processor = _VideoProcessor(processor)
    _videos["builder"] = (copy, size[0], size[1])
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


class FrameworkStandIn:
    """A framework's plain settings and its class's methods, without the model: enough for `_prepare_examples`."""

    def __init__(self, framework):
        from omegaconf import OmegaConf

        self.framework_class = type(framework)
        self.settings = {}
        for name, value in vars(framework).items():
            if name.startswith("_"):
                continue
            inner = (
                vars(value).get("_cfg") if hasattr(value, "__dict__") else None
            )  # the trainer's access-tracking wrapper
            if OmegaConf.is_config(inner):
                self.settings[name] = inner
            elif isinstance(value, _SETTINGS) or OmegaConf.is_config(value):
                self.settings[name] = value

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

    def __init__(self, batch, tensors):
        self.parts = [_parts(example) for example in batch]  # the same objects as in the batch, also after pickling
        self.tensors = tensors
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
            batch[0]["_prepared_examples"] = PreparedExamples(batch, self.stand_in._prepare_examples(batch, True))
        return batch


def _take_examples(examples):
    try:
        with _examples_lock:
            prepared = _examples_arrived.get(id(_parts(examples[0])[0][0]))
        if (
            prepared is not None
            and len(examples) == len(prepared.parts)
            and all(_same_parts(mine, _parts(example)) for mine, example in zip(prepared.parts, examples, strict=False))
        ):
            example_stats["used"] += 1
            if example_stats["used"] == 1:
                logger.info("worker_examples: training batches arrive prepared by the data-loader workers")
            return dict(prepared.tensors)
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


def prepare_examples_in_workers(framework):
    """Check that workers can run this framework's `_prepare_examples`, and let it return theirs."""
    global _framework_stand_in
    original = getattr(framework, "_prepare_examples", None)
    if not callable(original):
        return False
    if getattr(original, "_starvla_worker_examples", False):
        return True
    try:
        stand_in = FrameworkStandIn(framework)
        if not _probe_examples(framework, stand_in):
            return False
    except Exception as error:  # a framework that prepares its samples differently: leave it alone
        logger.info(f"worker_examples: {type(framework).__name__} not supported ({type(error).__name__}: {error})")
        return False

    def _prepare_examples(self, examples, include_actions):
        prepared = _take_examples(examples) if include_actions else None
        return prepared if prepared is not None else original(examples, include_actions)

    _prepare_examples._starvla_worker_examples = True
    framework._prepare_examples = types.MethodType(_prepare_examples, framework)
    _framework_stand_in = stand_in
    return True


def examples_collate(cfg, collate_fn):
    """The collate function for the workers; `collate_fn` itself when no framework was set up."""
    return collate_fn if _framework_stand_in is None else ExamplesCollate(collate_fn, _framework_stand_in)


def worker_collate(cfg, collate_fn):
    """`collate_fn`, followed by the input construction that was set up above when the model was built. Each wrapper
    is `collate_fn` itself when nothing was set up."""
    for wrap in (inputs_collate, video_collate, examples_collate):
        collate_fn = wrap(cfg, collate_fn)
    return collate_fn
