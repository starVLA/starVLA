import re

from pathlib import Path

# You can add multimodal datasets here and register a short nickname to ${data_dict}.
# The data format should follow the general multimodal VLM format, for example:
# https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md

json_root = f"./playground/Datasets/LLaVA-OneVision-COCO/llava_jsons"
image_root = f"./playground/Datasets/LLaVA-OneVision-COCO/images"

vlnce_root = "./playground/Datasets/VLN-CE"

SHAREGPT4V_COCO = {
    "annotation_path": f"{json_root}/sharegpt4v_coco.json",
    "data_path": f"{image_root}/",
}

R2R = {
    "annotation_path": f"{vlnce_root}/R2R/annotations_qwenvl.json",
    "data_path": f"{vlnce_root}/R2R/train/",
}

RXR = {
    "annotation_path": f"{vlnce_root}/RxR/annotations_qwenvl.json",
    "data_path": f"{vlnce_root}/RxR/train/",
}


data_dict = {
    "sharegpt4v_coco": SHAREGPT4V_COCO,
    "r2r": R2R,
    "rxr": RXR,
}

# Additional dataset definitions adapted from InternVLA-M1:
# https://github.com/InternRobotics/InternVLA-M1/blob/21e6e8f4bd42bee269dd021a118133501e9d8ede/InternVLA/dataloader/qwenvl_llavajson/qwen_data_config.py
# Local dataset roots below are examples; point them to your prepared data.
# Existing StarVLA dataset paths are kept unchanged.
# MIT License
#
# Copyright (c) InternVLA Team.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# General vision-language data (LLaVA-OneVision format).

onevision_json_root = f"./playground/Datasets/LLaVA-OneVision-Data/decoders/llava_format"

onevision_image_root = f"./playground/Datasets/LLaVA-OneVision-Data/decoders/visualData"

MAPQA_MATHV360K = {
    "annotation_path": f"{onevision_json_root}/MapQA_MathV360K.json",
    "data_path": f"{onevision_image_root}/",
}

AOKVQA_CCAULDRON_LLAVA = {
    "annotation_path": f"{onevision_json_root}/aokvqa_cauldron_llava_format.json",
    "data_path": f"{onevision_image_root}/",
}

SHAREGPT4V_KNOWLEDGE = {
    "annotation_path": f"{onevision_json_root}/sharegpt4v_knowledge.json",
    "data_path": f"{onevision_image_root}/",
}

SHAREGPT4V_LLAVA = {
    "annotation_path": f"{onevision_json_root}/sharegpt4v_llava.json",
    "data_path": f"{onevision_image_root}/",
}

SHAREGPT4V_SAM = {
    "annotation_path": f"{onevision_json_root}/sharegpt4v_sam.json",
    "data_path": f"{onevision_image_root}/",
}

VISUALWEBINSTRUCT_FILTERED = {
    "annotation_path": f"{onevision_json_root}/VisualWebInstruct_filtered.json",
    "data_path": f"{onevision_image_root}/",
}

# Text-only definitions are left unregistered, as in the source configuration.

MAGPIE_PRO_L3_80B_MT = {
    "annotation_path": f"{onevision_json_root}/magpie_pro_l3_80b_mt.json",
    "data_path": "",
}

MAGPIE_PRO_L3_80B_ST = {
    "annotation_path": f"{onevision_json_root}/magpie_pro_l3_80b_st.json",
    "data_path": "",
}

VISUAL7W_CCAULDRON_LLAVA = {
    "annotation_path": f"{onevision_json_root}/visual7w_cauldron_llava_format.json",
    "data_path": f"{onevision_image_root}/",
}

VISUALMRC_CCAULDRON = {
    "annotation_path": f"{onevision_json_root}/visualmrc_cauldron.json",
    "data_path": f"{onevision_image_root}/",
}

VSR_CCAULDRON_LLAVA = {
    "annotation_path": f"{onevision_json_root}/vsr_cauldron_llava_format.json",
    "data_path": f"{onevision_image_root}/",
}

# GenManip embodied reasoning data.

genmanip_json_root = f"./playground/Datasets/Genmanip_vlm_v5"

genmanip_object_object_root = f"{genmanip_json_root}/object_container"

GENMANIP_v5_object_container = {
    "annotation_path": f"{genmanip_object_object_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_object_object_root}",
}

genmanip_object_object_root = f"{genmanip_json_root}/object_object"

GENMANIP_v5_object_object = {
    "annotation_path": f"{genmanip_object_object_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_object_object_root}",
}

genmanip_kitchen_5plus5 = f"{genmanip_json_root}/Kitchen/5x5"

GENMANIP_v5_kitchen_5plus5 = {
    "annotation_path": f"{genmanip_kitchen_5plus5}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_5plus5}",
}

genmanip_kitchen_5plus5_more = f"{genmanip_json_root}/Kitchen/5x5_more"

GENMANIP_v5_kitchen_5plus5_more = {
    "annotation_path": f"{genmanip_kitchen_5plus5_more}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_5plus5_more}",
}

genmanip_kitchen_apple_root = f"{genmanip_json_root}/Kitchen/apple_plate"

GENMANIP_v5_kitchen_apple = {
    "annotation_path": f"{genmanip_kitchen_apple_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_apple_root}",
}

genmanip_kitchen_banana_root = f"{genmanip_json_root}/Kitchen/banana_plate"

GENMANIP_v5_kitchen_banana = {
    "annotation_path": f"{genmanip_kitchen_banana_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_banana_root}",
}

genmanip_kitchen_lemon_root = f"{genmanip_json_root}/Kitchen/lemon_plate"

GENMANIP_v5_kitchen_lemon = {
    "annotation_path": f"{genmanip_kitchen_lemon_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_lemon_root}",
}

genmanip_kitchen_sandwich_root = f"{genmanip_json_root}/Kitchen/sandwich_plate"

GENMANIP_v5_kitchen_sandwich = {
    "annotation_path": f"{genmanip_kitchen_sandwich_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_sandwich_root}",
}

# COCO captioning, question answering, OCR, and grounding data.

llava_format_root = "./playground/Datasets/COCO/coco_internvl3/qwen_224_format_minp_3136_maxp_12845056"

data_root = "./playground/Datasets/COCO"

asv2_conversation_en = {
    "annotation_path": f"{llava_format_root}/asv2_conversation_en.jsonl",
    "data_path": f"{data_root}",  # Images may be referenced inside jsonl
}

asv2_detailed_description_en = {
    "annotation_path": f"{llava_format_root}/asv2_detailed_description_en.jsonl",
    "data_path": f"{data_root}",
}

asv2_region_captioning_en = {
    "annotation_path": f"{llava_format_root}/asv2_region_captioning_en.jsonl",
    "data_path": f"{data_root}",
}

coco_internvl_longcap_en = {
    "annotation_path": f"{llava_format_root}/coco_internvl_longcap_en.jsonl",
    "data_path": f"{data_root}",
}

coco_karpathy_train_567_en = {
    "annotation_path": f"{llava_format_root}/coco_karpathy_train_567_en.jsonl",
    "data_path": f"{data_root}",
}

coco_negative_gpt4o_en = {
    "annotation_path": f"{llava_format_root}/coco_negative_gpt4o_en.jsonl",
    "data_path": f"{data_root}",
}

coco_poetry_zh = {
    "annotation_path": f"{llava_format_root}/coco_poetry_zh.jsonl",
    "data_path": f"{data_root}",
}

coco_rem_en_zh = {
    "annotation_path": f"{llava_format_root}/coco_rem_en_zh.jsonl",
    "data_path": f"{data_root}",
}

cocorem_exist_yorn_en = {
    "annotation_path": f"{llava_format_root}/cocorem_exist_yorn_en.jsonl",
    "data_path": f"{data_root}",
}

cocotextv2_en = {
    "annotation_path": f"{llava_format_root}/cocotextv2_en.jsonl",
    "data_path": f"{data_root}",
}

cocotextv2_gpt4o_en = {
    "annotation_path": f"{llava_format_root}/cocotextv2_gpt4o_en.jsonl",
    "data_path": f"{data_root}",
}

okvqa_en = {"annotation_path": f"{llava_format_root}/okvqa_en.jsonl", "data_path": f"{data_root}"}

refcoco_grounding_aug_en = {
    "annotation_path": f"{llava_format_root}/refcoco_grounding_aug_en.jsonl",
    "data_path": f"{data_root}",
}

refcoco_grounding_en = {
    "annotation_path": f"{llava_format_root}/refcoco_grounding_en.jsonl",
    "data_path": f"{data_root}",
}

tallyqa_coco_en = {
    "annotation_path": f"{llava_format_root}/tallyqa_coco_en.jsonl",
    "data_path": f"{data_root}",
}

toloka_grounding_aug_en = {
    "annotation_path": f"{llava_format_root}/toloka_grounding_aug_en.jsonl",
    "data_path": f"{data_root}",
}

vqav2_en = {
    "annotation_path": f"{llava_format_root}/vqav2_en.jsonl",
    "data_path": f"{data_root}",
}

vsr_en = {
    "annotation_path": f"{llava_format_root}/vsr_en.jsonl",
    "data_path": f"{data_root}",
}

data_dict.update(
    {
        "MapQA_MathV360K": MAPQA_MATHV360K,
        "aokvqa_cauldron_llava_format": AOKVQA_CCAULDRON_LLAVA,
        "sharegpt4v_knowledge": SHAREGPT4V_KNOWLEDGE,
        "sharegpt4v_llava": SHAREGPT4V_LLAVA,
        "sharegpt4v_sam": SHAREGPT4V_SAM,
        "VisualWebInstruct_filtered": VISUALWEBINSTRUCT_FILTERED,
        "visual7w_cauldron_llava_format": VISUAL7W_CCAULDRON_LLAVA,
        "visualmrc_cauldron": VISUALMRC_CCAULDRON,
        "vsr_cauldron_llava_format": VSR_CCAULDRON_LLAVA,
        "GENMANIP_v5_object_object": GENMANIP_v5_object_object,
        "GENMANIP_v5_object_container": GENMANIP_v5_object_container,
        "GENMANIP_v5_kitchen_5plus5": GENMANIP_v5_kitchen_5plus5,
        "GENMANIP_v5_kitchen_5plus5_more": GENMANIP_v5_kitchen_5plus5_more,
        "GENMANIP_v5_kitchen_apple": GENMANIP_v5_kitchen_apple,
        "GENMANIP_v5_kitchen_banana": GENMANIP_v5_kitchen_banana,
        "GENMANIP_v5_kitchen_lemon": GENMANIP_v5_kitchen_lemon,
        "GENMANIP_v5_kitchen_sandwich": GENMANIP_v5_kitchen_sandwich,
        "asv2_conversation_en": asv2_conversation_en,
        "asv2_detailed_description_en": asv2_detailed_description_en,
        "asv2_region_captioning_en": asv2_region_captioning_en,
        "coco_internvl_longcap_en": coco_internvl_longcap_en,
        "coco_karpathy_train_567_en": coco_karpathy_train_567_en,
        "coco_negative_gpt4o_en": coco_negative_gpt4o_en,
        "coco_poetry_zh": coco_poetry_zh,
        "coco_rem_en_zh": coco_rem_en_zh,
        "cocorem_exist_yorn_en": cocorem_exist_yorn_en,
        "cocotextv2_en": cocotextv2_en,
        "cocotextv2_gpt4o_en": cocotextv2_gpt4o_en,
        "okvqa_en": okvqa_en,
        "refcoco_grounding_aug_en": refcoco_grounding_aug_en,
        "refcoco_grounding_en": refcoco_grounding_en,
        "tallyqa_coco_en": tallyqa_coco_en,
        "toloka_grounding_aug_en": toloka_grounding_aug_en,
        "vqav2_en": vqav2_en,
        "vsr_en": vsr_en,
    }
)


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0

def data_list(dataset_names):
    if dataset_names == ["all"]:
        dataset_names = list(data_dict.keys())
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list

if __name__ == "__main__":
    print(data_list)
    
