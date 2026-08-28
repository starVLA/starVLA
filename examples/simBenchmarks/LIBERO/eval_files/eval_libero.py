import dataclasses
import json
import logging
import math
import os
import pathlib
import time
from collections import deque

import imageio
import numpy as np
from PIL import Image
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def _pack_eval_multiview(primary_image: np.ndarray, wrist_image: np.ndarray, pack_mode: str | None) -> np.ndarray:
    if pack_mode in (None, "none", "None", ""):
        return primary_image
    if pack_mode in ("primary_only", "primary", "first_view", "single_view"):
        return primary_image
    if pack_mode in ("horizontal_by_time", "horizontal"):
        return np.concatenate([primary_image, wrist_image], axis=1)
    if pack_mode in ("vertical_by_time", "vertical"):
        return np.concatenate([primary_image, wrist_image], axis=0)
    raise ValueError(f"Unsupported multiview_pack for LIBERO eval: {pack_mode}")


def _build_policy_image_history(
    image_history: deque,
    primary_image: np.ndarray,
    wrist_image: np.ndarray,
    history_len: int,
    pack_mode: str | None,
) -> list[np.ndarray]:
    if pack_mode in (None, "none", "None", ""):
        return [primary_image, wrist_image]
    fused_current = _pack_eval_multiview(primary_image, wrist_image, pack_mode)
    image_history.append(fused_current)
    while len(image_history) < history_len:
        image_history.appendleft(fused_current.copy())
    return list(image_history)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.
    task_start: int = 0  # First task id to evaluate.
    task_count: int = -1  # Number of tasks to evaluate from task_start. -1 = all remaining tasks.
    trial_start: int = 0  # First initial-state / episode index to evaluate for each task.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    save_videos: bool = True
    image_views: str = "primary+wrist"  # primary+wrist matches the LIBERO report eval; also supports auto | primary.
    policy_image_size: int = 0  # 0 keeps render size; QwenVAR stage2 training used 224x224 PIL images.
    validate_inputs: bool = True
    min_image_mean: float = 2.0
    min_image_std: float = 1.0
    strict_trial_count: bool = True

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True
    constrain_to_action_tokens: bool | None = None
    max_new_tokens: int | None = None

    # Tianyi-compatible policy image adapter. "auto" reads server metadata
    # and falls back to the legacy primary+wrist image list when unset.
    image_history: int = -1
    multiview_pack: str = "auto"

    job_name: str = "test"


def _select_policy_images(args: Args, client_model: ModelClient, primary_img: np.ndarray, wrist_img: np.ndarray) -> list:
    if args.image_views == "primary":
        return [primary_img]
    if args.image_views == "primary+wrist":
        return [primary_img, wrist_img]
    if args.image_views == "wrist+primary":
        return [wrist_img, primary_img]
    if args.image_views != "auto":
        raise ValueError(
            f"Unsupported image_views={args.image_views!r}; "
            "use auto, primary, primary+wrist, or wrist+primary"
        )

    # Align eval input with the checkpoint's training observation config.
    # LIBERO report checkpoints advertise obs: [image_0], so they should receive
    # the front agentview only. Multi-view checkpoints can opt in through config.
    obs_keys = set(getattr(client_model, "vla_obs", []) or [])
    if any(key in obs_keys for key in ("wrist_image", "image_1", "video.wrist_image")):
        return [primary_img, wrist_img]
    return [primary_img]


def _resize_policy_images(images: list, size: int) -> list:
    if size <= 0:
        return images
    resized = []
    for image in images:
        arr = np.asarray(image)
        if arr.shape[:2] == (size, size):
            resized.append(np.ascontiguousarray(arr))
            continue
        resized.append(np.asarray(Image.fromarray(arr).resize((size, size))).copy())
    return resized


def _orient_libero_image(image: np.ndarray) -> np.ndarray:
    orientation = os.getenv("LIBERO_IMAGE_ORIENTATION", "rot180")
    if orientation == "raw":
        return np.ascontiguousarray(image)
    if orientation == "flipud":
        return np.ascontiguousarray(image[::-1])
    if orientation == "fliplr":
        return np.ascontiguousarray(image[:, ::-1])
    if orientation == "rot180":
        return np.ascontiguousarray(image[::-1, ::-1])
    raise ValueError(
        f"Unsupported LIBERO_IMAGE_ORIENTATION={orientation!r}; "
        "use raw, flipud, fliplr, or rot180"
    )


def _image_summary(image: np.ndarray) -> dict:
    arr = np.asarray(image)
    return {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _validate_policy_images(images: list, args: Args, *, task_id: int, episode_idx: int, step: int) -> None:
    if not args.validate_inputs:
        return
    for view_idx, image in enumerate(images):
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise RuntimeError(
                f"Invalid policy image shape at task={task_id} episode={episode_idx} step={step} "
                f"view={view_idx}: shape={arr.shape}"
            )
        if arr.dtype != np.uint8:
            raise RuntimeError(
                f"Invalid policy image dtype at task={task_id} episode={episode_idx} step={step} "
                f"view={view_idx}: dtype={arr.dtype}"
            )
        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"Non-finite policy image at task={task_id} episode={episode_idx} step={step} view={view_idx}"
            )
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        max_value = float(np.max(arr))
        if mean < args.min_image_mean or std < args.min_image_std or max_value <= args.min_image_mean:
            raise RuntimeError(
                f"Degenerate policy image at task={task_id} episode={episode_idx} step={step} "
                f"view={view_idx}: mean={mean:.3f}, std={std:.3f}, max={max_value:.3f}"
            )


def _validate_state(state: np.ndarray, *, task_id: int, episode_idx: int, step: int) -> None:
    if state.shape != (8,):
        raise ValueError(f"Unexpected LIBERO state shape {state.shape}; expected (8,)")
    if not np.isfinite(state).all():
        raise RuntimeError(
            f"Non-finite LIBERO state at task={task_id} episode={episode_idx} step={step}: {state}"
        )


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

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

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        constrain_to_action_tokens=args.constrain_to_action_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    server_meta = getattr(client_model, "server_metadata", {}) or {}
    eval_multiview_pack = args.multiview_pack
    inferred_singleview = "singleview" in str(args.pretrained_path).lower()
    if eval_multiview_pack == "auto":
        eval_multiview_pack = server_meta.get("multiview_pack", "none")
        if eval_multiview_pack in (None, "", "none", "None") and inferred_singleview:
            eval_multiview_pack = "primary_only"
    eval_image_history = args.image_history
    if eval_image_history < 0:
        if inferred_singleview and eval_multiview_pack in ("primary_only", "primary", "first_view", "single_view"):
            eval_image_history = 1
        else:
            eval_image_history = int(
                server_meta.get(
                    "policy_image_history",
                    4 if eval_multiview_pack not in (None, "", "none", "None") else 0,
                )
            )
    logging.info(
        "Policy image adapter: multiview_pack=%s, image_history=%s, server_meta=%s",
        eval_multiview_pack,
        eval_image_history,
        server_meta,
    )

    if args.task_start < 0 or args.task_start >= num_tasks_in_suite:
        raise ValueError(f"task_start must be in [0, {num_tasks_in_suite}), got {args.task_start}")
    if args.trial_start < 0:
        raise ValueError(f"trial_start must be >= 0, got {args.trial_start}")

    # Optional smoke-test caps (still useful for quick verification with -1 = full run).
    remaining_tasks = num_tasks_in_suite - args.task_start
    task_limit = remaining_tasks if args.task_count <= 0 else min(args.task_count, remaining_tasks)
    if args.max_tasks > 0:
        task_limit = min(task_limit, args.max_tasks)
    task_ids = list(range(args.task_start, args.task_start + task_limit))
    logging.info(
        f"Evaluating {len(task_ids)} of {num_tasks_in_suite} tasks "
        f"(task_start={args.task_start}, task_count={args.task_count}, max_tasks={args.max_tasks})"
    )

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        try:
            # Start episodes
            task_episodes, task_successes = 0, 0
            trial_end = min(args.trial_start + args.num_trials_per_task, len(initial_states))
            if args.strict_trial_count and trial_end - args.trial_start != args.num_trials_per_task:
                raise RuntimeError(
                    f"Requested {args.num_trials_per_task} trials from trial_start={args.trial_start}, "
                    f"but task={task_id} only has {len(initial_states)} initial states"
                )
            logged_policy_input = False
            for episode_idx in tqdm.tqdm(range(args.trial_start, trial_end)):
                logging.info(f"\nTask id: {task_id}")
                logging.info(f"Task: {task_description}")

                # Reset environment
                client_model.reset(task_description=task_description)  # Reset the client connection
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []
                full_actions = []
                policy_image_history = deque(maxlen=max(eval_image_history, 1))

                logging.info(f"Starting episode {episode_idx + 1}...")
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

                    img = _orient_libero_image(obs["agentview_image"])
                    wrist_img = _orient_libero_image(obs["robot0_eye_in_hand_image"])

                    # Save preprocessed image for replay video
                    if args.save_videos:
                        replay_images.append(img)

                    gripper_q = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
                    state = np.concatenate(
                        (
                            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                            _quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32).reshape(-1),
                            gripper_q[:2],
                        )
                    ).astype(np.float32)
                    if args.validate_inputs:
                        _validate_state(state, task_id=task_id, episode_idx=episode_idx, step=step)

                    observation = {  #
                        "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                        "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                        "observation.state": np.expand_dims(state, axis=0),
                        "instruction": [str(task_description)],
                    }

                    if eval_multiview_pack in (None, "", "none", "None"):
                        # Legacy feihong path: image_views controls primary/wrist selection.
                        policy_images = _select_policy_images(
                            args,
                            client_model,
                            observation["observation.primary"][0],
                            observation["observation.wrist_image"][0],
                        )
                    else:
                        # Tianyi path: pack multiview frames and maintain policy image history.
                        policy_images = _build_policy_image_history(
                            policy_image_history,
                            observation["observation.primary"][0],
                            observation["observation.wrist_image"][0],
                            max(eval_image_history, 1),
                            eval_multiview_pack,
                        )
                    policy_images = _resize_policy_images(policy_images, args.policy_image_size)
                    _validate_policy_images(
                        policy_images,
                        args,
                        task_id=task_id,
                        episode_idx=episode_idx,
                        step=step,
                    )
                    example_dict = {
                        "image": policy_images,
                        "vggt_image": [
                            observation["observation.primary"][0],
                            observation["observation.wrist_image"][0],
                        ],
                        "lang": observation["instruction"][0],
                        "state": observation["observation.state"],
                    }
                    if not logged_policy_input:
                        logging.info(
                            "Policy input check: image_views=%s, multiview_pack=%s, image_history=%s, policy_image_size=%s, image_summaries=%s, num_images=%d, state_shape=%s, server_vla_obs=%s",
                            args.image_views,
                            eval_multiview_pack,
                            eval_image_history,
                            args.policy_image_size,
                            [_image_summary(image) for image in example_dict["image"]],
                            len(example_dict["image"]),
                            tuple(example_dict["state"].shape),
                            getattr(client_model, "vla_obs", []),
                        )
                        logged_policy_input = True

                    start_time = time.time()

                    response = client_model.step(example=example_dict, step=step)

                    end_time = time.time()
                    # print(f"time: {end_time - start_time}")

                    # #
                    raw_action = response["raw_action"]

                    world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                    rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                    open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                    gripper = _binarize_gripper_open(open_gripper)

                    if step == 0:
                        logging.info(
                            "First raw server action: world=%s, rotation=%s, open_gripper=%s",
                            np.array2string(world_vector_delta, precision=6, suppress_small=False),
                            np.array2string(rotation_delta, precision=6, suppress_small=False),
                            np.array2string(open_gripper, precision=6, suppress_small=False),
                        )

                    if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                        logging.warning(
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
                    if step == 0:
                        logging.info(
                            "First env action: finite=%s, min=%.6f, max=%.6f, values=%s",
                            bool(np.isfinite(delta_action).all()),
                            float(np.nanmin(delta_action)),
                            float(np.nanmax(delta_action)),
                            np.array2string(delta_action, precision=6, suppress_small=False),
                        )

                    # __import__("ipdb").set_trace()
                    # see ../robosuite/controllers/controller_factory.py
                    obs, reward, done, info = env.step(delta_action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    step += 1

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                if args.save_videos:
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )

                full_actions = np.stack(full_actions)
                # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)

                # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
                # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Log final results
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        finally:
            env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info("EVAL_CHUNK_OK")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
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


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
