import json
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch.distributed as dist
from PIL import Image
from accelerate.logging import get_logger
from torch.utils.data import IterableDataset

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except ImportError:
    tf = None
    tfds = None

logger = get_logger(__name__)


def _require_tfds():
    if tf is None or tfds is None:
        raise ImportError(
            "RLDS LIBERO dataloader requires tensorflow and tensorflow-datasets. "
            "Please install: pip install tensorflow-cpu tensorflow-datasets"
        )


def _setup_tf_runtime():
    if tf is None:
        return
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except Exception:
        pass


DATA_MIX_TO_SUITES = {
    "libero_goal": ["libero_goal_no_noops"],
    "libero_object": ["libero_object_no_noops"],
    "libero_spatial": ["libero_spatial_no_noops"],
    "libero_10": ["libero_10_no_noops"],
    "libero_all": [
        "libero_spatial_no_noops",
        "libero_object_no_noops",
        "libero_goal_no_noops",
        "libero_10_no_noops",
    ],
}


def _dist_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _dist_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _decode_text(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    if value is None:
        return ""
    return str(value)


def _format_state(state: np.ndarray, decimals: int) -> str:
    return np.array2string(
        np.asarray(state, dtype=np.float32),
        precision=decimals,
        separator=", ",
        suppress_small=False,
        floatmode="fixed",
    )


def _extract_images(observation: dict) -> List[Image.Image]:
    images = []
    for key in ("image", "wrist_image"):
        value = observation.get(key)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[-1] < 3:
            continue
        images.append(Image.fromarray(array[..., :3].astype(np.uint8)))
    return images


def resolve_suite_names(data_root_dir: Path, data_mix: str) -> List[str]:
    if data_mix in DATA_MIX_TO_SUITES:
        return DATA_MIX_TO_SUITES[data_mix]

    candidates = [item.strip() for item in str(data_mix).split(",") if item.strip()]
    resolved = []
    for name in candidates:
        if (data_root_dir / name).exists():
            resolved.append(name)
        elif (data_root_dir / f"{name}_no_noops").exists():
            resolved.append(f"{name}_no_noops")
        else:
            raise FileNotFoundError(f"Cannot resolve RLDS suite '{name}' under {data_root_dir}")
    if not resolved:
        raise ValueError(f"Empty RLDS data_mix: {data_mix}")
    return resolved


def _load_num_transitions(version_dir: Path, split: str) -> int:
    for stats_path in sorted(version_dir.glob("dataset_statistics_*.json")):
        try:
            stats = json.loads(stats_path.read_text())
        except Exception:
            continue
        split_key = f"num_{split}_transitions"
        if split_key in stats:
            return int(stats[split_key])
        if "num_transitions" in stats:
            return int(stats["num_transitions"])

    info_path = version_dir / "dataset_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        for split_info in info.get("splits", []):
            if split_info.get("name") == split:
                return sum(int(x) for x in split_info.get("shardLengths", []))
    return 1


class LiberoRLDSPiFastDataset(IterableDataset):
    def __init__(self, cfg, split: str = "train"):
        super().__init__()
        _require_tfds()
        _setup_tf_runtime()

        self.cfg = cfg
        self.vla_cfg = cfg.datasets.vla_data
        self.split = split
        self.data_root_dir = Path(self.vla_cfg.data_root_dir)
        self.suite_names = resolve_suite_names(self.data_root_dir, self.vla_cfg.data_mix)
        self.horizon = int(cfg.framework.action_model.future_action_window_size) + 1
        self.rank = _dist_rank()
        self.world_size = _dist_world_size()
        self.shuffle_buffer_size = int(getattr(self.vla_cfg, "shuffle_buffer_size", 2048))
        self.shuffle_buffer_warmup = int(getattr(self.vla_cfg, "shuffle_buffer_warmup", 512))
        self.state_decimal_places = int(getattr(self.vla_cfg, "state_decimal_places", 4))
        self.use_state_in_prompt = bool(getattr(self.vla_cfg, "use_state_in_prompt", True))

        self.datasets = []
        self.total_transitions = 0
        for suite_name in self.suite_names:
            version_dir = self.data_root_dir / suite_name / "1.0.0"
            if not version_dir.exists():
                version_dir = self.data_root_dir / suite_name
            if not version_dir.exists():
                raise FileNotFoundError(f"RLDS suite not found: {suite_name}")
            builder = tfds.builder_from_directory(str(version_dir))
            dataset = builder.as_dataset(split=split, shuffle_files=True).repeat()
            self.datasets.append(dataset)
            self.total_transitions += _load_num_transitions(version_dir, split)

        logger.info(
            "Loaded RLDS LIBERO suites %s with ~%s transitions for split %s",
            self.suite_names,
            self.total_transitions,
            split,
        )

    def __len__(self) -> int:
        samples_per_rank = max(1, self.total_transitions // max(1, self.world_size))
        return samples_per_rank

    def _build_instruction(self, task_text: str, state: Optional[np.ndarray]) -> str:
        task_text = task_text.strip() or "complete the manipulation task"
        if self.use_state_in_prompt and state is not None:
            state_text = _format_state(state, self.state_decimal_places)
            return (
                f"task: {task_text}\n"
                f"current observation.state: {state_text}\n"
                f"predict next {self.horizon} timestep action.\n"
            )
        return f"task: {task_text}\npredict next {self.horizon} timestep action.\n"

    def _build_action_chunk(self, actions: List[np.ndarray], index: int) -> np.ndarray:
        last = len(actions) - 1
        chunk = [actions[min(index + offset, last)] for offset in range(self.horizon)]
        return np.asarray(chunk, dtype=np.float32)

    def _iter_suite(self, dataset: Iterable):
        for episode in tfds.as_numpy(dataset):
            steps = list(episode["steps"])
            if not steps:
                continue

            actions = [np.asarray(step["action"], dtype=np.float32) for step in steps]
            episode_task = _decode_text(episode.get("language_instruction", ""))

            for index, step in enumerate(steps):
                observation = step.get("observation", {})
                images = _extract_images(observation)
                if not images:
                    continue

                task_text = _decode_text(step.get("language_instruction", episode_task))
                state = observation.get("state")
                state_array = None if state is None else np.asarray(state, dtype=np.float32)

                yield {
                    "image": images,
                    "lang": self._build_instruction(task_text, state_array),
                    "action": self._build_action_chunk(actions, index),
                    "state": state_array,
                }

    def __iter__(self):
        suite_iters = [self._iter_suite(dataset) for dataset in self.datasets]
        shuffle_buffer = []
        seen = 0
        warmup = max(1, min(self.shuffle_buffer_size, self.shuffle_buffer_warmup))

        while True:
            suite_idx = random.randrange(len(suite_iters))
            try:
                sample = next(suite_iters[suite_idx])
            except StopIteration:
                suite_iters[suite_idx] = self._iter_suite(self.datasets[suite_idx])
                continue

            if seen % self.world_size != self.rank:
                seen += 1
                continue
            seen += 1

            shuffle_buffer.append(sample)
            if len(shuffle_buffer) < warmup:
                continue
            if len(shuffle_buffer) > self.shuffle_buffer_size:
                shuffle_buffer.pop(random.randrange(len(shuffle_buffer)))
            yield shuffle_buffer.pop(random.randrange(len(shuffle_buffer)))


def collate_fn(batch):
    return batch


def get_vla_dataset(cfg):
    split = getattr(cfg.datasets.vla_data, "split", "train")
    return LiberoRLDSPiFastDataset(cfg=cfg, split=split)
