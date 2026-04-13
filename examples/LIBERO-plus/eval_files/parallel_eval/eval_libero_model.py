import dataclasses
import json
import logging
import math
import os
import pathlib
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2 as cv
import draccus
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import read_mode_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions - 1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()

        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def get_logger(file):

    logger = logging.getLogger("dual_logger")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224, 224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    log_path: str = "experiments/libero/logs"

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"

    use_bf16: bool = True

    start_idx: int = -1
    end_idx: int = -1
    output_dir: str = "./output"


class PolicyModel:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble=True,
        action_ensemble_horizon: Optional[int] = 3,  # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10095,
        use_bf16=True,
    ) -> None:

        # build client to connect server policy
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        vla = baseframework.from_pretrained(  # TODO should auto detect framework from model path
            policy_ckpt_path,
        )

        if use_bf16:  # False
            vla = vla.to(torch.bfloat16)
        self.vla = vla.to("cuda").eval()

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon  # 0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(self, example: dict, step: int = 0, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param image: Input image in the format (H, W, 3), type uint8
        :param task_description: Task description text
        :return: (raw action, processed action)
        """

        task_description = example.get("lang", None)
        images = example["image"]  # list of images for history

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        example["image"] = images
        vla_input = {
            # "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0:
            response = self.vla.predict_action(example, **vla_input)
            normalized_actions = response["normalized_actions"]  # B, chunk, D

            normalized_actions = normalized_actions[0]

            if normalized_actions.shape[1] > 7:
                normalized_actions = normalized_actions[:, -7:]

            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
            )

        raw_actions = self.raw_actions[step % action_chunk_size][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        return {"raw_action": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = PolicyModel._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        # import ipdb; ipdb.set_trace()
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key


@draccus.wrap()
def eval_libero(args: Args) -> None:

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    print(f"🌍 Rank {rank}/{world_size} | GPU: {local_rank}")
    torch.cuda.set_device(local_rank)

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    # if args.start_idx != -1:
    #     num_tasks_in_suite = args.end_idx - args.start_idx
    # patch_num = num_tasks_in_suite // world_size
    # if rank == world_size - 1:
    #     start_idx = rank * patch_num
    #     end_idx = num_tasks_in_suite
    # else:
    #     start_idx = rank * patch_num
    #     end_idx = start_idx + patch_num
    if args.start_idx == -1:
        args.start_idx = 0
        args.end_idx = num_tasks_in_suite
    # args.start_idx = start_idx
    # args.end_idx = end_idx
    print(f"processing tasks from {args.start_idx} to {args.end_idx}")
    # args.video_out_path = f"{date_base}+{args.job_name}"
    log_path = os.path.join(args.output_dir, f"logs/{args.task_suite_name}")
    log_file = os.path.join(log_path, f"{args.start_idx}_{args.end_idx}.log")
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_file)
    logger.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    video_out_path = os.path.join(args.output_dir, args.task_suite_name)
    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = PolicyModel(
        policy_ckpt_path=args.pretrained_path,  # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        use_bf16=args.use_bf16,
    )

    disturb_res = {}
    LIBERO_HOME = os.environ.get("LIBERO_HOME", "path_to_LIBERO-plus")
    with open(os.path.join(LIBERO_HOME, "libero/libero/benchmark/task_classification.json")) as f:
        TASK_MAPPING = json.load(f)[args.task_suite_name]

    ID2CATEGORY = {}
    for item in TASK_MAPPING:
        category = item["category"]
        item_name = item["name"]
        ID2CATEGORY[item["id"]] = (category, item_name)
        if category not in disturb_res:
            disturb_res[category] = {"total_count": 0, "success_count": 0}

    # Start evaluation

    total_episodes, total_successes = 0, 0
    print(
        f"*****************num tasks in {args.task_suite_name}: {num_tasks_in_suite}****************, processing from{args.start_idx} to {args.end_idx}"
    )
    # for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
    for task_id in tqdm.tqdm(range(args.start_idx, args.end_idx)):

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):

            logger.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []

            logger.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:

                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = {  #
                    "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                    # "state": observation["observation.state"],
                }

                start_time = time.time()

                # response = client_model.step(example=example_dict)
                response = client_model.step(example=example_dict, step=step)

                end_time = time.time()
                # print(f"time: {end_time - start_time}")

                # #
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logger.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    disturb_res[ID2CATEGORY[task_id + 1][0]]["success_count"] += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1
            disturb_res[ID2CATEGORY[task_id + 1][0]]["total_count"] += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")

            imageio.mimwrite(
                pathlib.Path(video_out_path) / f"rollout_{ID2CATEGORY[task_id+1][1]}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=25,
            )

            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logger.info(f"Success: {done}")
            logger.info(f"# episodes completed so far: {total_episodes}")
            logger.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logger.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logger.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
    with open(os.path.join(log_path, f"{args.start_idx}_to_{args.end_idx}.json"), "w", encoding="utf-8") as f:
        json.dump(disturb_res, f)
    logger.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logger.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    eval_libero()
