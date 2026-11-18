"""RLBench dataset registry.

The central registry instantiates the configuration class with the active
training config's action horizon.
"""

from examples.simBenchmarks.RLbench.configs.train.data_config import (
    RLBenchJointVelocityDataConfig,
)


ROBOT_TYPE_CONFIG_MAP = {
    "rlbench_joint_velocity": RLBenchJointVelocityDataConfig,
}

DATASET_NAMED_MIXTURES = {
    "rlbench_joint_velocity_9task": [
        (
            "joint_velocity_9task",
            1.0,
            "rlbench_joint_velocity",
        ),
    ],
}