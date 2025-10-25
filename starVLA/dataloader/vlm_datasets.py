import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
import transformers
from omegaconf import OmegaConf
from starVLA.dataloader.qwenvl_llavajson.qwen_data_config import data_list
from starVLA.dataloader.qwenvl_llavajson.rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>\n"
DEFAULT_VIDEO_TOKEN = "<video>\n"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>" * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if data["data_path"] != "":
                    ann["data_path"] = data["data_path"]
                elif "raw_data" in ann.keys():
                    ann["data_path"] = ann["raw_data"]["data_root"]
            list_data_dict += annotations

        list_data_dict = self.pre_filter_long_case(list_data_dict, max_words=tokenizer.max_len_single_sentence)
        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        rank0_print(f"Total training samples: {len(self.list_data_dict)}")
        rank0_print("Formatting inputs...Skip in lazy mode")

        # self.data_args.image_processor.max_pixels = data_args.max_pixels
        # self.data_args.image_processor.min_pixels = data_args.min_pixels
        # self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        # self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    def pre_filter_long_case(self, list_data_dict, max_words=1024):
        """filter out conversations with total words exceeding max_words"""

        def count_total_words(convs):
            total = 0
            for entry in convs:
                value = entry.get("value", "")
                total += len(value.strip().split())
            return total

        return [item for item in list_data_dict if count_total_words(item.get("conversations", [])) <= max_words]

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if ("images" in sample) or ("videos" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")
        # if fix image size?
        if getattr(self.data_args, "fix_image_size", None) is not None:
            image = image.resize(
                self.data_args.fix_image_size,
                resample=Image.BICUBIC,
            )
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(images=None, videos=video, return_tensors="pt")
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [self.data_args.image_processor.temporal_patch_size / fps] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        video = None
        if "images" in sources[0] and len(sources[0]["images"]):
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["images"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [os.path.join(image_folder, file) for file in image_file]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2 for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image")
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),  # (1,16,16)
            )
        elif "videos" in sources[0] and len(sources[0]["videos"]):
            video_file = self.list_data_dict[i]["videos"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [os.path.join(video_folder, file) for file in video_file]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2 for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video")
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
        else:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=grid_thw_merged)
            position_ids = torch.arange(0, data_dict["input_ids"].size(1)).view(1, -1).unsqueeze(0).expand(3, -1, -1)

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                position_ids=position_ids,
            )
        if "images" in self.list_data_dict[i]:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
        # video exist in the data
        elif "videos" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw

        max_len = self.tokenizer.max_len_single_sentence
        if data_dict["input_ids"].shape[0] > max_len:
            data_dict["input_ids"] = data_dict["input_ids"][:max_len]
            data_dict["labels"] = data_dict["labels"][:max_len]
            data_dict["position_ids"] = position_ids[:, :, :max_len]

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side=self.tokenizer.padding_side,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side=self.tokenizer.padding_side
        )
        position_ids = pad_and_cat(position_ids)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[..., : self.tokenizer.model_max_length]  # 3,bs,length

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            itertools.chain(*(instance["pixel_values"] for instance in instances if "pixel_values" in instance))
        )
        videos = list(
            itertools.chain(
                *(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(*(instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance))
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(*(instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance))
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids")
        )

        seq_lens = torch.tensor([0] + [len(seq) for seq in input_ids], dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            itertools.chain(*(instance["pixel_values"] for instance in instances if "pixel_values" in instance))
        )
        videos = list(
            itertools.chain(
                *(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(*(instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance))
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(*(instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance))
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # load training dataset
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    # load evaluation dataset (if specified eval dataset path)
    eval_dataset = None
    if hasattr(data_args, "eval_dataset") and data_args.eval_dataset:
        eval_data_args = copy.deepcopy(data_args)
        eval_data_args.dataset_use = data_args.eval_dataset
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=eval_data_args)

    # select appropriate collator based on whether data needs to be flattened
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def make_vlm_dataloader(cfg):
    data_args = cfg.datasets.vlm_data
    image_processor = AutoProcessor.from_pretrained(
        cfg.framework.qwenvl.base_vlm,
    ).image_processor

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.framework.qwenvl.base_vlm,
        model_max_length=data_args.model_max_length,
        padding_side="left",  # flash Attention version of Qwen2.5_VL. Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input.
        use_fast=False,
    )

    # avoid processing these in dataset
    image_processor.max_pixels = int(data_args.max_pixels)
    image_processor.min_pixels = int(data_args.min_pixels)
    image_processor.size["longest_edge"] = int(data_args.max_pixels)
    image_processor.size["shortest_edge"] = int(data_args.min_pixels)
    data_args.model_type = "qwen2.5vl"
    data_args_ns = SimpleNamespace(**OmegaConf.to_container(data_args, resolve=True))
    data_args_ns.image_processor = image_processor  # TODO later remove the logic bound to model
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args_ns)

    #
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.datasets.vlm_data.per_device_batch_size,
        collate_fn=data_collator,
        num_workers=4,
    )

    return {
        "train_dataloader": train_dataloader,
    }


from transformers import AutoTokenizer, AutoProcessor

if __name__ == "__main__":
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    
    data_args = cfg.datasets.vlm_data
    image_processor = AutoProcessor.from_pretrained(
        cfg.framework.qwenvl.base_vlm,
    ).image_processor

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.framework.qwenvl.base_vlm,
        model_max_length=data_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    # avoid processing these in dataset
    image_processor.max_pixels = data_args.max_pixels
    image_processor.min_pixels = data_args.min_pixels
    image_processor.size["longest_edge"] = data_args.max_pixels
    image_processor.size["shortest_edge"] = data_args.min_pixels
    data_args.model_type = "qwen2.5vl"
    data_args_ns = SimpleNamespace(**OmegaConf.to_container(data_args, resolve=True))
    data_args_ns.image_processor = image_processor
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args_ns)

    #
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.datasets.vlm_data.per_device_batch_size,
        collate_fn=data_collator,
    )
    batchs = iter(train_dataloader)
    batch_samples = next(batchs)
    # skip the first 99 batches, get the 100th batch
    from itertools import islice

    # batch_samples = next(islice(batchs, 99, 100))
    count = 0
    while count < 100:
        batch_samples = next(batchs)  # for debug
        print(count)
        count += 1
    pass
