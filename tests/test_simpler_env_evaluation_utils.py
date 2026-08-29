import random
from types import SimpleNamespace

import numpy as np
import torch

from examples.simBenchmarks.SimplerEnv.eval_files.evaluation_utils import (
    build_evaluation_summary,
    set_seed_everywhere,
    write_evaluation_summary,
)


def test_set_seed_everywhere_repeats_python_numpy_and_torch_streams():
    set_seed_everywhere(123)
    first = (random.random(), np.random.rand(3), torch.rand(3))

    set_seed_everywhere(123)
    second = (random.random(), np.random.rand(3), torch.rand(3))

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2])


def test_set_seed_everywhere_rejects_seeds_numpy_cannot_represent():
    try:
        set_seed_everywhere(np.iinfo(np.uint32).max + 1)
    except ValueError as exc:
        assert "seed must be between" in str(exc)
    else:
        raise AssertionError("expected an out-of-range seed to be rejected")


def test_build_and_write_evaluation_summary(tmp_path):
    args = SimpleNamespace(
        seed=7,
        policy_model="qwen",
        policy_setup="widowx_bridge",
        ckpt_path="checkpoint.pt",
        env_name="PickCube-v1",
        scene_name="bridge_table_1_v1",
        robot="widowx",
        obj_variation_mode="episode",
        obj_episode_range=[0, 3],
        max_episode_steps=120,
    )

    summary = build_evaluation_summary(args, [True, False, True])
    assert summary["num_episodes"] == 3
    assert summary["num_successes"] == 2
    assert summary["success_rate"] == 2 / 3
    assert summary["task"] == "PickCube-v1"

    output = tmp_path / "nested" / "summary.json"
    write_evaluation_summary(output, summary)
    assert output.exists()
    assert '"seed": 7' in output.read_text()
