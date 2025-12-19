# copy from SimplerEnv/simpler_env/evaluation/argparse.py 

import argparse

import numpy as np


def parse_range_tuple(t):
    return np.linspace(t[0], t[1], int(t[2]))


def get_args():
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-model",
        type=str,
        default="rt1",
        help="Policy model type; e.g., 'rt1', 'octo-base', 'octo-small'",
    )
    parser.add_argument("--policy-setup", type=str, default="franka", help="Policy setup")
    parser.add_argument("--ckpt-path", type=str, default=None)

    parser.add_argument("--eval-on-train-instances", type=lambda x: x.lower() == "true",default=True)
    parser.add_argument("--logging-dir", type=str, default="./results")
    parser.add_argument("--tf-memory-limit", type=int, default=3072, help="Tensorflow memory limit")
    parser.add_argument("--async-freq", type=int, default=1)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Octo init rng seed")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--headless", type=bool, default=True, help="Whether to run in headless mode (no GUI)")
    parser.add_argument("--write-video", type=bool, default=True, help="Whether to save videos of the evaluation")
    parser.add_argument("--task-name", type=str, required=True, help="Task name")
    parser.add_argument("--eval-instance-ids", type=str, default=None, help="Instance ids to evaluate on, if None, evaluate on all instances")
    parser.add_argument("--behaviro-data-path", type=str, required=True, help="2025-challenge-task-instances data path")
    parser.add_argument("--partial-scene-load", type=bool, default=False, help="Whether to only load task-relevant rooms for a specific task")
    parser.add_argument("--wrappers", type=str, default="RGBLowResWrapper",choices=["RGBLowResWrapper", "DefaultWrapper", "RichObservationWrapper"], help="List of wrappers to apply to the environment")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per rollout episode, setting to null will use the default value (2 x average human demo completion steps)")
    parser.add_argument("--use-state", default=False, type=lambda x: x.lower() == "true",help="Whether to use state as part of the observation")
    
    args = parser.parse_args()

    return args