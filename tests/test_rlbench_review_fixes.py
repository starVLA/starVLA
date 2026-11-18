"""Regression tests for RLBench integration, data configuration, and evaluation.

The tests cover the following contracts:

* Git tracking: verifies that the joint-velocity environment configuration
  can be committed and is not excluded by Git ignore rules.
* State dimensions: verifies that the RLBench joint-velocity state consists
  of 53 dimensions, including the expected joint-position and gripper-matrix
  widths.
* Dynamic data configuration: verifies that the policy normalizer instantiates
  a dynamic data configuration with the action horizon specified by the
  checkpoint configuration.
* Dataset registry: verifies that the nine-task joint-velocity dataset
  registry entry matches the documented conversion directory.
* Discrete gripper actions: verifies that discrete gripper commands are
  binarized before being passed to RLBench without modifying the input action.
* Evaluation observation contract: verifies that the RLBench adapter produces
  the configured camera inputs and a state tensor with the expected shape.
* Relative action modes: verifies that the dataset converter rejects relative
  joint-position action configurations that are incompatible with its native
  action representation.
* Episode success criteria: verifies that termination alone does not indicate
  success and that a positive total reward combined with termination is
  recognized as successful.
"""

import subprocess
import unittest
from pathlib import Path

import numpy as np

from examples.simBenchmarks.RLbench.configs.train.data_config import (
    RLBenchJointVelocityDataConfig,
)
from examples.simBenchmarks.RLbench.configs.train.data_registry import (
    DATASET_NAMED_MIXTURES,
)
from examples.simBenchmarks.RLbench.conversion.rlbench_to_lerobot import (
    validate_conversion_action_mode,
)
from examples.simBenchmarks.RLbench.eval_files.eval_rlbench import rollout_success
from examples.simBenchmarks.RLbench.eval_files.rlbench_adapter import RLBenchAdapter
from deployment.model_server.policy_norm_processor import _instantiate_data_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class _Mode:
    action_shape = (8,)


class _Task:
    def reset(self):
        return ["reach"], self._observation()

    def step(self, action):
        return self._observation(), 0.0, False

    @staticmethod
    def _observation():
        observation = {"front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)}
        for key, dim in RLBenchJointVelocityDataConfig.state_key_dims.items():
            observation[key.removeprefix("state.")] = np.arange(dim, dtype=np.float32)
        return observation


class _Environment:
    def __init__(self):
        self.task = _Task()

    def get_task(self, task):
        return self.task


class ReviewFixesTest(unittest.TestCase):
    def test_env_configs_are_not_ignored_by_git(self):
        config = "examples/simBenchmarks/RLbench/configs/env/joint_velocity.yaml"
        result = subprocess.run(
            ["git", "check-ignore", "-q", config], cwd=REPO_ROOT, check=False
        )
        self.assertNotEqual(result.returncode, 0, f"{config} must be committable")

    def test_rlbench_state_dimensions_are_explicit_and_sum_to_53(self):
        config = RLBenchJointVelocityDataConfig(action_horizon=2)
        self.assertEqual(sum(config.state_key_dims.values()), 53)
        self.assertEqual(config.state_key_dims["state.joint_positions"], 7)
        self.assertEqual(config.state_key_dims["state.gripper_matrix"], 16)

    def test_policy_normalizer_instantiates_dynamic_data_config_with_checkpoint_horizon(self):
        class DynamicConfig:
            def __init__(self, action_horizon):
                self.action_horizon = action_horizon

        config = _instantiate_data_config(
            DynamicConfig,
            {"framework": {"action_model": {"action_horizon": 6}}},
        )

        self.assertIsInstance(config, DynamicConfig)
        self.assertEqual(config.action_horizon, 6)

    def test_dataset_registry_matches_documented_conversion_directory(self):
        datasets = DATASET_NAMED_MIXTURES["rlbench_joint_velocity_9task"]
        self.assertEqual(datasets[0][0], "joint_velocity_9task")

    def test_discrete_gripper_action_is_binarized_before_step(self):
        adapter = RLBenchAdapter(_Environment(), _Mode())
        adapter.reset("ReachTarget")
        action = np.zeros(8, dtype=np.float32)
        action[-1] = 0.6

        native = adapter.action_for_step(action, gripper_type="Discrete")

        self.assertEqual(native[-1], 1.0)
        self.assertAlmostEqual(float(action[-1]), 0.6)

    def test_eval_example_uses_configured_state_and_camera_contract(self):
        config = RLBenchJointVelocityDataConfig(action_horizon=2)
        adapter = RLBenchAdapter(
            _Environment(),
            _Mode(),
            camera_names=("front_rgb",),
            state_keys=config.state_keys,
        )
        descriptions, observation = adapter.reset("ReachTarget")

        example = adapter.policy_example(observation, descriptions)

        self.assertEqual(len(example["image"]), 1)
        self.assertEqual(example["state"].shape, (1, 53))

    def test_relative_action_modes_are_rejected_by_converter(self):
        with self.assertRaisesRegex(ValueError, "relative action labels"):
            validate_conversion_action_mode(
                {
                    "type": "MoveArmThenGripper",
                    "arm": {"type": "JointPosition", "kwargs": {"absolute_mode": False}},
                    "gripper": {"type": "Discrete"},
                }
            )

    def test_termination_is_not_itself_a_success_signal(self):
        self.assertFalse(rollout_success(total_reward=0.0, terminate=True))
        self.assertTrue(rollout_success(total_reward=1.0, terminate=True))


if __name__ == "__main__":
    unittest.main()
