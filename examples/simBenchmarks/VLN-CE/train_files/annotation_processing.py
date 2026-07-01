import json
import argparse
import numpy as np


QUESTION = (
    "Imagine you are an autonomous robot performing a vision-language navigation task.\n"
    "You are given a sequence of historical observations {history_images} "
    "and the current observation <image>\n\n"
    "Your goal is: \"{instruction}\".\n\n"
    "Based on the history and current view, analyze the environment and "
    "decide the best next action to safely reach the goal.\n\n"
)


def process_annotations(data_path: str, dataset_name: str, max_frames: int = 8):
    data = json.load(open(data_path, "r"))
    print(f"\n{dataset_name} 数据总量: {len(data)}")
    print(f"{dataset_name} 数据示例:", data[0])

    qwen_anno = []
    for item in data:
        frames = item["frames"]
        latest_frame = frames[-1]

        if len(frames) <= max_frames:
            sampled_frames = frames
        else:
            sampled_indices = np.linspace(
                0, len(frames) - 1, num=max_frames - 1, endpoint=False, dtype=int
            )
            sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]

        new_item = {
            "image": sampled_frames,
            "conversations": [
                {
                    "from": "human",
                    "value": QUESTION.format(
                        history_images="<image>\n" * (len(sampled_frames) - 1),
                        instruction=item["q"],
                    ),
                },
                {
                    "from": "gpt",
                    "value": item["a"],
                },
            ],
        }
        qwen_anno.append(new_item)

    print(f"{dataset_name} 处理后的数据示例:", qwen_anno[0])

    output_path = "/".join(data_path.split("/")[:-1]) + "/annotations.json"
    with open(output_path, "w") as f:
        json.dump(qwen_anno, f)

    print(f"{dataset_name} 数据处理完成，保存至: {output_path}")
    return qwen_anno


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="annotations.json 的路径")
    parser.add_argument("--dataset", type=str, required=True, choices=["R2R", "RxR"], help="数据集名称")
    parser.add_argument("--max_frames", type=int, default=8)
    args = parser.parse_args()

    process_annotations(args.data_path, args.dataset, args.max_frames)
