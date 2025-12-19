import os

# from IPython import embed; embed()
from examples.Behavior.custom_argparse import get_args
# from examples.Behavior.model2behavior_interface_depth import M1Inference
# from examples.Behavior.model2behavior_interface import M1Inference
from examples.Behavior.model2behavior_interface_hack import M1Inference

from omnigibson.learning.eval import Evaluator
from omnigibson.macros import gm
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
from omnigibson.learning.utils.obs_utils import create_video_writer
from omegaconf import DictConfig, OmegaConf
import numpy as np
import logging
from pathlib import Path
import json
import csv


def load_task_description(task_name: str, tasks_jsonl_path: Path = None) -> str:
    """
    Load task description from tasks.jsonl file based on task_name.
    """
    # Load tasks.jsonl and create mapping
    task_name_to_description = {}
    with open(tasks_jsonl_path, "r") as f:
        for line in f:
            task_data = json.loads(line)
            task_name_to_description[task_data["task_name"]] = task_data["task"]
    
    if task_name not in task_name_to_description:
        raise KeyError(
            f"Task name '{task_name}' not found in tasks.jsonl. "
            f"Available tasks: {list(task_name_to_description.keys())}"
        )
    
    return task_name_to_description[task_name]

# Module-specific constants
NUM_EVAL_EPISODES = 1
NUM_TRAIN_INSTANCES = 200 # 200 human-collected demonstrations
NUM_EVAL_INSTANCES = 10 # 20 extra configuration instances

# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

# create module logger
logger = logging.getLogger("evaluator")
logger.setLevel(20)  # info

if __name__ == "__main__":
    args = get_args()
    args.eval_instance_ids = [int(x) for x in args.eval_instance_ids.split()]
    # print("args.eval_instance_ids",args.eval_instance_ids)

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Load task description from tasks.jsonl using task_name
    task_description = load_task_description(args.task_name,tasks_jsonl_path=os.path.join(args.behaviro_data_path, "meta", "tasks.jsonl"))
    logger.info(f"Loaded task description for '{args.task_name}': {task_description}")

    model = M1Inference(
        policy_ckpt_path=args.ckpt_path, # to get unnormalization stats
        policy_setup=args.policy_setup,
        port=args.port,
        task_description=task_description,
    )

    # policy model creation; update this if you are using a new policy model
    # run real-to-sim evaluation
    # set headless mode
    gm.HEADLESS = args.headless

    # Set behavior data path, should be set to the path to the 2025-challenge-task-instances folder
    gm.DATA_PATH = args.behaviro_data_path
    # set video path
    if args.write_video:
        video_path = Path(args.logging_dir).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
    # get run instances
    if args.eval_on_train_instances:
        logger.info(
            "You are evaluating on training instances, set eval_on_train_instances to False for test instances."
        )
        task_idx = TASK_NAMES_TO_INDICES[args.task_name]
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        instances_to_run = []
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                instances_to_run.append(str(int((episode["episode_index"] // 10) % 1e3)))
        if args.eval_instance_ids:
            assert set(args.eval_instance_ids).issubset(
                set(range(NUM_TRAIN_INSTANCES))
            ), f"eval instance ids must be in range({NUM_TRAIN_INSTANCES}), currently {args.eval_instance_ids}"
            instances_to_run = [instances_to_run[i] for i in args.eval_instance_ids]
    else:
        instances_to_run = (
            args.eval_instance_ids if args.eval_instance_ids is not None else set(range(NUM_EVAL_INSTANCES))
        )
        assert set(instances_to_run).issubset(
            set(range(NUM_EVAL_INSTANCES))
        ), f"eval instance ids must be in range({NUM_EVAL_INSTANCES})"
        # load csv file
        task_instance_csv_path = os.path.join(
            gm.DATA_PATH, "2025-challenge-task-instances","metadata", "test_instances.csv"
        )
        with open(task_instance_csv_path, "r") as f:
            lines = list(csv.reader(f))[1:]
        assert (
            lines[TASK_NAMES_TO_INDICES[args.task_name]][1] == args.task_name
        ), f"Task name from config {args.task_name} does not match task name from csv {lines[TASK_NAMES_TO_INDICES[args.task_name]][1]}"
        test_instances = lines[TASK_NAMES_TO_INDICES[args.task_name]][2].strip().split(",")
        instances_to_run = [int(test_instances[i]) for i in instances_to_run]
    # establish metrics
    metrics = {}
    metrics_path = Path(args.logging_dir).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "env_wrapper": {"_target_": f"omnigibson.learning.wrappers.{args.wrappers}"},
        "task": {"name": args.task_name},
        "robot": {"type": "R1Pro", "controllers": None}, # TODO: add controllers
        "partial_scene_load": args.partial_scene_load, 
        "max_steps": args.max_steps,
        "write_video": args.write_video,
        "model": {
            # "_target_": "examples.Behavior.model2behavior_interface_depth.M1Inference", 
            # "_target_": "examples.Behavior.model2behavior_interface.M1Inference", 
            "_target_": "examples.Behavior.model2behavior_interface_hack.M1Inference", 
            "policy_ckpt_path": args.ckpt_path,
            "policy_setup": args.policy_setup,
            "port": args.port,
            "task_description": task_description,
            "use_state": args.use_state,
        },
        "policy_name": args.policy_model, # It's just a name
    }
    
    # Convert dictionary to OmegaConf DictConfig to support attribute access
    config = OmegaConf.create(config_dict)
    
    with Evaluator(cfg=config) as evaluator:
        logger.info("Starting evaluation...")

        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(idx)
            logger.info(f"Starting task instance {idx} for evaluation...")
            for epi in range(NUM_EVAL_EPISODES):
                evaluator.reset()
                done = False
                if config.write_video:
                    video_name = str(video_path) + f"/{config.task.name}_{idx}_{epi}.mp4"
                    evaluator.video_writer = create_video_writer(
                        fpath=video_name,
                        resolution=(448, 672),
                    )
                # run metric start callbacks
                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)
                while not done:
                    terminated, truncated = evaluator.step()
                    if terminated or truncated:
                        done = True
                    if config.write_video:
                        evaluator._write_video()
                    if evaluator.env._current_step % 1000 == 0:
                        logger.info(f"Current step: {evaluator.env._current_step}")
                # run metric end callbacks
                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)
                logger.info(f"Evaluation finished at step {evaluator.env._current_step}.")
                logger.info(f"Evaluation exit state: {terminated}, {truncated}")
                logger.info(f"Total trials: {evaluator.n_trials}")
                logger.info(f"Total success trials: {evaluator.n_success_trials}")
                # gather metric results and write to file
                for metric in evaluator.metrics:
                    metrics.update(metric.gather_results())
                with open(metrics_path / f"{config.task.name}_{idx}_{epi}.json", "w") as f:
                    json.dump(metrics, f)
                # reset video writer
                if config.write_video:
                    evaluator.video_writer = None
                    logger.info(f"Saved video to {video_name}")
                else:
                    logger.warning("No observations were recorded.")