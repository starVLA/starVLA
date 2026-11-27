"""video_prep: the observation images of a world-model backbone resized and normalized in the data-loader
workers. Every step the Wan / Cosmos-Predict2 interface passes each sample's views through diffusers' VideoProcessor
(resize to 480 x 832 for Wan, 320 x 576 for Cosmos-Predict2, then scale to [-1, 1]) on the CPU of the training process
while the GPU waits. The workers have the samples first: they make the same call with a copy of the same processor
and send the tensors with the batch. The interface's call for a sample whose images are the batch's own objects gets
the worker's tensor; any other call (inference, a framework that edits the images) runs as before.
"""
import inspect
import re
import threading
import weakref

import torch

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
        for mine, video in zip(self.images, self.videos):
            if len(mine) == len(sample) and all(a is b for a, b in zip(mine, sample)):
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
        images = [sample["image"] if isinstance(sample["image"], (list, tuple)) else [sample["image"]] for sample in batch]
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


def apply(backbone):
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
        if not torch.equal(processor.preprocess_video(sample, height=size[0], width=size[1]),
                           copy.preprocess_video(sample, height=size[0], width=size[1])):
            return False
    except Exception:  # a processor that cannot travel to the workers: leave it alone
        return False
    backbone.video_processor = _VideoProcessor(processor)
    _videos["builder"] = (copy, size[0], size[1])
    return True


def collate(cfg, collate_fn):
    """The collate function for the workers; `collate_fn` itself when no backbone was set up."""
    if _videos["builder"] is None or isinstance(collate_fn, VideoCollate):
        return collate_fn
    return VideoCollate(collate_fn, _videos["builder"])
