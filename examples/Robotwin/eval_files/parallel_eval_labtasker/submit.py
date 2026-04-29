#!/usr/bin/env python3
"""
Submit RoboTwin evaluation tasks to Labtasker.

Each task = one (task_name, mode) pair.
Worker command (one per GPU):
    CUDA_VISIBLE_DEVICES=<gpu> labtasker loop -- \\
        ${STARVLA_PYTHON} examples/Robotwin/eval_files/parallel_eval_labtasker/run.py \\
            '%(task_name)' '%(mode)' '%(ckpt)' '%(policy_name)'
"""

import itertools

import labtasker


# ===== USER CONFIG =====
CKPT = "/path/to/starVLA/.cache/huggingface/hub/models--StarVLA--Qwen3-VL-OFT-RoboTwin2-All/snapshots/2d8a0487b501c15bc96569f4ec7c09c8aba76820/checkpoints/steps_140000_pytorch_model.pt"
POLICY_NAME = "Qwen3-VL-OFT-RoboTwin2-All"

MODES = [
    "demo_clean",
    "demo_randomized",
]

TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle_horizontally",
    "shake_bottle",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]

MAX_RETRIES = 3
PRIORITY = 10  # 0-20
# =======================


def main():
    if not CKPT:
        raise ValueError("Set CKPT to the checkpoint path before submitting.")
    if not POLICY_NAME:
        raise ValueError("Set POLICY_NAME before submitting.")

    submitted = 0
    for task_name, mode in itertools.product(TASKS, MODES):
        result = labtasker.submit_task(
            task_name=f"{POLICY_NAME}_{task_name}_{mode}",
            args={
                "task_name": task_name,
                "mode": mode,
                "ckpt": CKPT,
                "policy_name": POLICY_NAME,
            },
            metadata={
                "benchmark": "RoboTwin",
                "task_name": task_name,
                "mode": mode,
                "policy_name": POLICY_NAME,
            },
            max_retries=MAX_RETRIES,
            priority=PRIORITY,
        )
        print(f"  submitted  {task_name} x {mode}  =>  task_id={result.task_id}")
        submitted += 1

    print(f"\nDone. {submitted} tasks submitted.")


if __name__ == "__main__":
    main()
