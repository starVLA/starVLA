import json
import os

# from IPython import embed; embed()
from examples.simBenchmarks.SimplerEnv.eval_files.custom_argparse import get_args
from examples.simBenchmarks.SimplerEnv.eval_files.evaluation_utils import (
    build_evaluation_summary,
    set_seed_everywhere,
    validate_server_metadata,
    write_evaluation_summary,
)


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
    set_seed_everywhere(args.seed)

    # Import simulation and model dependencies after seeding the process.
    from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

    from examples.simBenchmarks.SimplerEnv.eval_files.model2simpler_interface import ModelClient

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if os.getenv("DEBUG", False):
        start_debugpy_once()
    model = ModelClient(
        policy_ckpt_path=args.ckpt_path,  # to get unnormalization stats
        policy_setup=args.policy_setup,
        port=args.port,
        action_scale=args.action_scale,
        cfg_scale=1.5,  # cfg from 1.5 to 7 also performs well
    )
    validate_server_metadata(args, model.server_metadata)

    # policy model creation; update this if you are using a new policy model
    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    summary = build_evaluation_summary(args, success_arr, model.server_metadata)
    print(args)
    print(" " * 10, "Average success", summary["success_rate"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.results_file is not None:
        write_evaluation_summary(args.results_file, summary)
