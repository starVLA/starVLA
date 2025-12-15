import os

# from IPython import embed; embed()
from examples.SimplerEnv.eval_files.custom_argparse import get_args
from examples.SimplerEnv.eval_files.model2simpler_interface import ModelClient
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

import numpy as np


def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    model = ModelClient(
        policy_ckpt_path=args.ckpt_path, # to get unnormalization stats
        policy_setup=args.policy_setup,
        port=args.port,
        action_scale=args.action_scale,
        cfg_scale=1.5                  # cfg from 1.5 to 7 also performs well
    )

    # policy model creation; update this if you are using a new policy model
    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
