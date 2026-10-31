"""Contract tests for the RLBench benchmark adapter.

These tests intentionally use small stand-ins for RLBench classes.  They
exercise the adapter contract without requiring CoppeliaSim or a display.
"""

import unittest

import numpy as np

from examples.simBenchmarks.RLbench.eval_files import rlbench_adapter as adapter


class _FakeArm:
    def __init__(self, shape, **kwargs):
        self.action_shape = shape
        self.kwargs = kwargs


class _FakeJointVelocity(_FakeArm):
    def __init__(self, **kwargs):
        super().__init__((7,), **kwargs)


class _FakeJointPosition(_FakeArm):
    def __init__(self, **kwargs):
        super().__init__((7,), **kwargs)


class _FakeJointTorque(_FakeArm):
    def __init__(self, **kwargs):
        super().__init__((7,), **kwargs)


class _FakeEndEffectorPoseViaIK(_FakeArm):
    def __init__(self, **kwargs):
        super().__init__((7,), **kwargs)


class _FakeEndEffectorPoseViaPlanning(_FakeArm):
    def __init__(self, **kwargs):
        super().__init__((7,), **kwargs)


class _FakeDiscrete(_FakeArm):
    def __init__(self, **kwargs):
        super().__init__((1,), **kwargs)


class _FakeContinuous(_FakeArm):
    def __init__(self, **kwargs):
        super().__init__((1,), **kwargs)


class _FakeMoveArmThenGripper:
    def __init__(self, arm_action_mode, gripper_action_mode):
        self.arm_action_mode = arm_action_mode
        self.gripper_action_mode = gripper_action_mode
        self.action_shape = (arm_action_mode.action_shape[0] + gripper_action_mode.action_shape[0],)


def _fake_modules():
    return adapter.RLBenchModules(
        move_arm_then_gripper=_FakeMoveArmThenGripper,
        arm_modes={
            "JointVelocity": _FakeJointVelocity,
            "JointPosition": _FakeJointPosition,
            "JointTorque": _FakeJointTorque,
            "EndEffectorPoseViaIK": _FakeEndEffectorPoseViaIK,
            "EndEffectorPoseViaPlanning": _FakeEndEffectorPoseViaPlanning,
        },
        gripper_modes={"Discrete": _FakeDiscrete, "Continuous": _FakeContinuous},
    )


class _FakeTask:
    def __init__(self):
        self.actions = []

    def reset(self):
        return ["reach the target"], {"front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)}

    def step(self, action):
        self.actions.append(action)
        return {"front_rgb": np.ones((2, 2, 3), dtype=np.uint8)}, 0.0, False


class _FakeEnvironment:
    def __init__(self):
        self.task = _FakeTask()
        self.requested_task = None

    def get_task(self, task):
        self.requested_task = task
        return self.task


class ActionModeFactoryTest(unittest.TestCase):
    def test_joint_velocity_dimension_is_taken_from_rlbench_action_mode(self):
        mode = adapter.build_action_mode(
            {
                "type": "MoveArmThenGripper",
                "arm": {"type": "JointVelocity"},
                "gripper": {"type": "Discrete"},
            },
            modules=_fake_modules(),
        )

        self.assertIsInstance(mode.arm_action_mode, _FakeJointVelocity)
        self.assertIsInstance(mode.gripper_action_mode, _FakeDiscrete)
        self.assertEqual(adapter.action_spec(mode).action_dim, 8)

    def test_pose_planning_keeps_native_constructor_options_and_dimension(self):
        mode = adapter.build_action_mode(
            {
                "type": "MoveArmThenGripper",
                "arm": {"type": "EndEffectorPoseViaPlanning", "kwargs": {"absolute_mode": False}},
                "gripper": {"type": "Continuous"},
            },
            modules=_fake_modules(),
        )

        self.assertIsInstance(mode.arm_action_mode, _FakeEndEffectorPoseViaPlanning)
        self.assertEqual(mode.arm_action_mode.kwargs, {"absolute_mode": False})
        self.assertIsInstance(mode.gripper_action_mode, _FakeContinuous)
        self.assertEqual(adapter.action_spec(mode).action_shape, (8,))

    def test_unknown_action_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported RLBench arm action mode"):
            adapter.build_action_mode(
                {
                    "type": "MoveArmThenGripper",
                    "arm": {"type": "CartesianDelta"},
                    "gripper": {"type": "Discrete"},
                },
                modules=_fake_modules(),
            )


class EnvironmentAdapterTest(unittest.TestCase):
    def setUp(self):
        self.mode = adapter.build_action_mode(
            {
                "type": "MoveArmThenGripper",
                "arm": {"type": "JointTorque"},
                "gripper": {"type": "Discrete"},
            },
            modules=_fake_modules(),
        )
        self.environment = _FakeEnvironment()
        self.adapter = adapter.RLBenchAdapter(self.environment, self.mode)

    def test_step_forwards_native_action_without_reencoding(self):
        self.adapter.reset("ReachTarget")
        raw_action = np.arange(8, dtype=np.float32)

        self.adapter.step(raw_action)

        forwarded = self.environment.task.actions[-1]
        self.assertIs(forwarded, raw_action)
        np.testing.assert_array_equal(forwarded, raw_action)

    def test_step_rejects_an_action_with_wrong_native_dimension(self):
        self.adapter.reset("ReachTarget")

        with self.assertRaisesRegex(ValueError, "expected action shape"):
            self.adapter.step(np.zeros(7, dtype=np.float32))

    def test_observation_and_language_are_model_agnostic(self):
        descriptions, observation = self.adapter.reset("ReachTarget")
        example = self.adapter.policy_example(observation, descriptions)

        self.assertEqual(example["lang"], "reach the target")
        self.assertEqual(len(example["image"]), 1)
        self.assertEqual(example["image"][0].shape, (2, 2, 3))

    def test_policy_language_uses_all_ordered_descriptions_like_conversion(self):
        example = self.adapter.policy_example(
            {"front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            ["open the drawer", "pull its handle", "open the drawer"],
        )

        self.assertEqual(
            example["lang"],
            "open the drawer | pull its handle | open the drawer",
        )


class PolicyOutputAdapterTest(unittest.TestCase):
    def test_first_predicted_action_is_checked_against_native_action_dimension(self):
        mode = adapter.build_action_mode(
            {
                "type": "MoveArmThenGripper",
                "arm": {"type": "EndEffectorPoseViaIK"},
                "gripper": {"type": "Discrete"},
            },
            modules=_fake_modules(),
        )
        output = adapter.extract_first_action({"data": {"actions": np.arange(16).reshape(1, 2, 8)}}, mode)

        np.testing.assert_array_equal(output, np.arange(8))

    def test_policy_output_dimension_mismatch_is_rejected(self):
        mode = adapter.build_action_mode(
            {
                "type": "MoveArmThenGripper",
                "arm": {"type": "JointPosition"},
                "gripper": {"type": "Discrete"},
            },
            modules=_fake_modules(),
        )

        with self.assertRaisesRegex(ValueError, "expected action shape"):
            adapter.extract_first_action({"data": {"actions": np.zeros((1, 1, 7))}}, mode)

    def test_policy_adapter_forwards_unnorm_key_without_action_reencoding(self):
        class FakeClient:
            def __init__(self):
                self.request = None

            def predict_action(self, request):
                self.request = request
                return {"data": {"actions": np.arange(8, dtype=np.float32).reshape(1, 1, 8)}}

        mode = adapter.build_action_mode(
            {
                "type": "MoveArmThenGripper",
                "arm": {"type": "JointPosition"},
                "gripper": {"type": "Discrete"},
            },
            modules=_fake_modules(),
        )
        client = FakeClient()
        policy = adapter.StarVLAPolicyAdapter(client, mode, unnorm_key="rlbench_joint_position")

        action = policy.predict({"image": [np.zeros((2, 2, 3), dtype=np.uint8)], "lang": "move"})

        self.assertEqual(client.request["unnorm_key"], "rlbench_joint_position")
        np.testing.assert_array_equal(action, np.arange(8, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
