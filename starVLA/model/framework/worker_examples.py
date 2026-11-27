"""worker_examples: the sample preparation of the OpenPI frameworks (PI0, PI05) in the data-loader workers.

These frameworks do not use a `get_vlm_model` backbone, so this acts on the framework and recognises them by
what it patches. The framework's `_prepare_examples` (every view resized and padded to the model's resolution and
scaled, missing views filled in, state and actions padded: CPU work on the batch as the data set returns it) runs in
the data-loader workers, on a stand-in for the framework that has its settings and methods but no model. `apply`
checks once that the stand-in returns exactly what the framework returns. In the training process the framework
gets the worker's tensors for a batch whose images, prompts, states and actions are the objects the worker saw;
any other call (inference, an edited batch) is prepared as before. Bitwise identical.
"""
import threading
import types
import weakref

import torch

_SETTINGS = (bool, int, float, str, tuple)
_arrived = weakref.WeakValueDictionary()  # id(first image of a batch) -> its PreparedExamples, where it is consumed
_lock = threading.Lock()  # batches arrive in the DataLoader's pinning thread
stats = {"used": 0}
_stand_in = None  # of the framework most recently accelerated in this process


class StandIn:
    """A framework's plain settings and its class's methods, without the model: enough for `_prepare_examples`."""

    def __init__(self, framework):
        from omegaconf import OmegaConf

        self.framework_class = type(framework)
        self.settings = {}
        for name, value in vars(framework).items():
            if name.startswith("_"):
                continue
            inner = vars(value).get("_cfg") if hasattr(value, "__dict__") else None  # the trainer's access-tracking wrapper
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
    return images, example.get("lang", example.get("language")), example.get("state"), example.get("action"), \
        example.get("image_mask")


def _same_parts(mine, theirs):
    (images, lang, *arrays), (their_images, their_lang, *their_arrays) = mine, theirs
    return (len(images) == len(their_images) and all(a is b for a, b in zip(images, their_images))
            and lang == their_lang and all(a is b for a, b in zip(arrays, their_arrays)))


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
        with _lock:
            _arrived[id(self.parts[0][0][0])] = self

    def pin_memory(self):  # called by the DataLoader's pinning thread
        self.tensors = {key: value.pin_memory() if torch.is_tensor(value) else value for key, value in self.tensors.items()}
        return self


class ExamplesCollate:
    """`collate_fn`, followed by the framework's `_prepare_examples` for the batch it returns."""

    def __init__(self, collate_fn, stand_in):
        self.collate_fn, self.stand_in = collate_fn, stand_in

    def __call__(self, batch):
        batch = self.collate_fn(batch)
        if isinstance(batch, list) and batch and isinstance(batch[0], dict) and "image" in batch[0] and "action" in batch[0]:
            batch[0]["_prepared_examples"] = PreparedExamples(batch, self.stand_in._prepare_examples(batch, True))
        return batch


def _take(examples):
    try:
        with _lock:
            prepared = _arrived.get(id(_parts(examples[0])[0][0]))
        if prepared is not None and len(examples) == len(prepared.parts) and all(
                _same_parts(mine, _parts(example)) for mine, example in zip(prepared.parts, examples)):
            stats["used"] += 1
            if stats["used"] == 1:
                print("[starVLA] worker_examples: training batches arrive prepared by the data-loader workers", flush=True)
            return dict(prepared.tensors)
    except (IndexError, KeyError, TypeError):
        pass
    return None


def _probe(framework, stand_in):
    import pickle

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    examples = [{"image": [Image.fromarray(rng.integers(0, 256, (200 + 24 * views, 224, 3), dtype=np.uint8)) for _ in range(views)],
                 "lang": "pick up the block", "state": rng.uniform(-1, 1, (1, 5)).astype(np.float32),
                 "action": rng.uniform(-1, 1, (min(3, framework.action_horizon), min(5, framework.action_dim))).astype(np.float32)}
                for views in (1, 2)]
    theirs = framework._prepare_examples(examples, True)
    ours = pickle.loads(pickle.dumps(stand_in))._prepare_examples(examples, True)
    return theirs.keys() == ours.keys() and all(
        torch.equal(theirs[key], ours[key]) and theirs[key].dtype == ours[key].dtype if torch.is_tensor(theirs[key])
        else theirs[key] == ours[key] for key in theirs)


def apply(framework):
    """Check that workers can run this framework's `_prepare_examples`, and let it return theirs."""
    global _stand_in
    original = getattr(framework, "_prepare_examples", None)
    if not callable(original):
        return False
    if getattr(original, "_starvla_worker_examples", False):
        return True
    try:
        stand_in = StandIn(framework)
        if not _probe(framework, stand_in):
            return False
    except Exception as error:  # a framework that prepares its samples differently: leave it alone
        print(f"[starVLA] worker_examples: {type(framework).__name__} not supported ({type(error).__name__}: {error})",
              flush=True)
        return False

    def _prepare_examples(self, examples, include_actions):
        prepared = _take(examples) if include_actions else None
        return prepared if prepared is not None else original(examples, include_actions)

    _prepare_examples._starvla_worker_examples = True
    framework._prepare_examples = types.MethodType(_prepare_examples, framework)
    _stand_in = stand_in
    return True


def collate(cfg, collate_fn):
    """The collate function for the workers; `collate_fn` itself when no framework was set up."""
    return collate_fn if _stand_in is None else ExamplesCollate(collate_fn, _stand_in)
