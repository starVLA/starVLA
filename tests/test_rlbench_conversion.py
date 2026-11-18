"""TDD contracts for the RLBench to StarVLA LeRobot conversion."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from examples.simBenchmarks.RLbench.conversion.rlbench_reader import RLBenchRawDataset
from examples.simBenchmarks.RLbench.conversion.rlbench_schema import (
    ActionSpec,
    assert_action_dimension_contract,
)
from examples.simBenchmarks.RLbench.conversion.rlbench_to_lerobot import (
    _episode_parquet_path,
    convert,
    validate_conversion_action_mode,
)
from examples.simBenchmarks.RLbench.configs.train.data_config import (
    RLBenchJointVelocityDataConfig,
)
from examples.simBenchmarks.RLbench.configs.train.data_registry import (
    ROBOT_TYPE_CONFIG_MAP,
)
from examples.simBenchmarks.RLbench.configs.train.data_config import (
    RLBenchJointVelocityDataConfig,
)
from examples.simBenchmarks.RLbench.configs.train.data_registry import (
    ROBOT_TYPE_CONFIG_MAP,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


def _write_fixture(root: Path) -> None:
    """The structure of the fixture:
        raw/
        └── push_buttons/
            └── variation0/
                ├── variation_descriptions.pkl
                └── episodes/
                    └── episode0/
                        ├── low_dim_obs.pkl
                        ├── front_rgb/
                        │   ├── 0.png
                        │   └── 1.png
                        └── wrist_rgb/
                            ├── 0.png
                            └── 1.png
    """
    episode = root / "push_buttons" / "variation0" / "episodes" / "episode0"
    episode.mkdir(parents=True)
    observations = []
    # Two frames of data, with incrementing values
    # for each frame to ensure that the conversion preserves the data correctly.
    for frame in range(2):
        observations.append(
            SimpleNamespace(
                joint_positions=np.arange(7, dtype=np.float64) + frame,
                joint_velocities=np.arange(7, dtype=np.float64) + 10 + frame,
                joint_forces=np.arange(7, dtype=np.float64) + 20 + frame,
                gripper_open=float(frame),
                gripper_joint_positions=np.array([0.1, 0.2], dtype=np.float64),
                gripper_touch_forces=np.arange(6, dtype=np.float64),
                gripper_pose=np.arange(7, dtype=np.float64),
                gripper_matrix=np.eye(4, dtype=np.float64),
                task_low_dim_state=np.arange(3, dtype=np.float64),
            )
        )
        # Two cameras, front_rgb and wrist_rgb,
        # with incrementing pixel values for each frame to ensure
        # that the conversion preserves the data correctly.
        for camera in ("front_rgb", "wrist_rgb"):
            camera_dir = episode / camera
            camera_dir.mkdir(exist_ok=True)
            Image.new("RGB", (3, 2), (frame, 2, 3)).save(camera_dir / f"{frame}.png")
    with (episode / "low_dim_obs.pkl").open("wb") as stream:
        pickle.dump(observations, stream)
    with (root / "push_buttons" / "variation0" / "variation_descriptions.pkl").open("wb") as stream:
        pickle.dump(["push the buttons", "press all of the buttons"], stream)


class RLBenchConversionTest(unittest.TestCase):
    def test_registry_stores_config_class_and_horizon_is_injected(self):
        config_class = ROBOT_TYPE_CONFIG_MAP["rlbench_joint_velocity"]
        self.assertIs(config_class, RLBenchJointVelocityDataConfig)
        config = config_class(action_horizon=5)
        # RLBench's official imitation-learning example supervises each
        # observation with that observation's joint_velocities. The loader
        # adds these offsets to the current observation index.
        self.assertEqual(config.action_indices, [0, 1, 2, 3, 4])

    def test_episode_parquet_paths_follow_lerobot_chunk_size(self):
        root = Path("/tmp/rlbench-conversion")
        self.assertEqual(
            _episode_parquet_path(root, 999),
            root / "data/chunk-000/episode_000999.parquet",
        )
        self.assertEqual(
            _episode_parquet_path(root, 1000),
            root / "data/chunk-001/episode_001000.parquet",
        )

    def test_video_path_is_required_metadata_but_not_the_png_decode_path(self):
        """
        Test that the video_path pattern, which is a path of png files, 
        is required metadata for the dataset,
        but that the actual video frames are read from the PNG files in the dataset.
        """
        dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
        dataset._lerobot_info_meta = {
            "video_path": "unused/video/{video_key}.mp4",
        }
        self.assertEqual(
            dataset._get_video_path_pattern(),
            "unused/video/{video_key}.mp4",
        )

        dataset._lerobot_info_meta = {}
        with self.assertRaises(KeyError):
            dataset._get_video_path_pattern()

        with tempfile.TemporaryDirectory() as temporary:
            dataset_root = Path(temporary)
            image_path = dataset_root / "images/front_rgb/episode_000000/000000.png"
            image_path.parent.mkdir(parents=True)
            # np.array[0]->1 frame, (3,2)->2 height 3 weight, (1,2,3) is RGB channel
            Image.new("RGB", (3, 2), (1, 2, 3)).save(image_path)

            dataset._dataset_path = dataset_root
            dataset._delta_indices = {"video.front_rgb": np.array([0])}
            dataset._trajectory_lengths = [1]
            dataset.get_trajectory_index = lambda trajectory_id: 0
            dataset._lerobot_modality_meta = SimpleNamespace(
                video={
                    "front_rgb": SimpleNamespace(
                        original_key="observation.images.front_rgb"
                    )
                }
            )
            dataset.curr_traj_data = pd.DataFrame(
                {
                    "observation.images.front_rgb": [
                        [{"path": "images/front_rgb/episode_000000/000000.png"}][0]
                    ]
                }
            )

            frames = dataset.get_video(0, "video.front_rgb", 0)

        self.assertEqual(frames.shape, (1, 2, 3, 3))
        # frames[0,0,0] is the left-up point
        np.testing.assert_array_equal(frames[0, 0, 0], np.array([1, 2, 3]))

    def test_action_dimension_contract_is_shared_by_all_layers(self):
        """Test that the action dimension contract is shared by all layers of the conversion."""
        spec = ActionSpec(
            action_mode={
                "type": "MoveArmThenGripper",
                "arm": {"type": "JointVelocity"},
                "gripper": {"type": "Discrete"},
            },
            action_shape=(8,),
        )
        assert_action_dimension_contract(
            raw_action_dim=8,
            environment_action_shape=(8,),
            lerobot_action_shape=(4, 8),
            framework_action_dim=8,
            spec=spec,
        )

    def test_conversion_preserves_native_action_images_state_and_metadata(self):
        """Test that the conversion from RLBench to LeRobot preserves the native action, images, state, and metadata correctly."""
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            output_root = Path(temporary) / "lerobot"
            _write_fixture(raw_root)

            result = convert(
                raw_root,
                output_root,
                action_mode={
                    "type": "MoveArmThenGripper",
                    "arm": {"type": "JointVelocity"},
                    "gripper": {"type": "Discrete"},
                },
                environment_action_shape=(8,),
                framework_action_dim=8,
                fps=20,
            )

            self.assertEqual(result.total_frames, 2)
            self.assertEqual(result.action_dim, 8)
            self.assertEqual(result.state_dim, 53)
            action_spec = json.loads((output_root / "meta/action_spec.json").read_text())
            self.assertEqual(action_spec["source"], "RLBench")
            self.assertEqual(action_spec["action_shape"], [8])
            self.assertEqual(action_spec["action_dim"], 8)
            info = json.loads((output_root / "meta/info.json").read_text())
            self.assertEqual(info["action_spec"]["action_dim"], 8)
            self.assertEqual(info["total_tasks"], 1)
            self.assertEqual(info["total_chunks"], 1)
            modality = json.loads((output_root / "meta/modality.json").read_text())
            self.assertEqual(
                modality["video"]["front_rgb"]["original_key"],
                "observation.images.front_rgb",
            )
            self.assertEqual(modality["state"]["joint_positions"]["start"], 0)
            self.assertEqual(modality["state"]["joint_positions"]["end"], 7)

            parquet_files = list((output_root / "data").rglob("*.parquet"))
            self.assertEqual(len(parquet_files), 1)
            import pandas as pd

            frame = pd.read_parquet(parquet_files[0])
            self.assertEqual(np.asarray(frame["action"].iloc[0]).shape, (8,))
            self.assertEqual(np.asarray(frame["observation.state"].iloc[0]).shape, (53,))
            self.assertEqual(
                np.asarray(frame["observation.task_low_dim_state"].iloc[0]).shape,
                (3,),
            )
            language_instruction = "push the buttons | press all of the buttons"
            self.assertEqual(frame["task"].tolist(), [language_instruction, language_instruction])
            self.assertEqual(frame["task_index"].tolist(), [0, 0])
            tasks = [
                json.loads(line)
                for line in (output_root / "meta/tasks.jsonl").read_text().splitlines()
            ]
            self.assertEqual(tasks, [{"task_index": 0, "task": language_instruction}])
            self.assertTrue(
                (output_root / "images" / "observation.images.front_rgb" / "episode_000000").exists()
            )

    def test_conversion_rejects_a_nonempty_output_root_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            output_root = Path(temporary) / "lerobot"
            _write_fixture(raw_root)
            output_root.mkdir()
            (output_root / "old-output.txt").write_text("stale", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "not empty"):
                convert(
                    raw_root,
                    output_root,
                    action_mode={
                        "type": "MoveArmThenGripper",
                        "arm": {"type": "JointVelocity"},
                        "gripper": {"type": "Discrete"},
                    },
                    environment_action_shape=(8,),
                    framework_action_dim=8,
                    fps=20,
                )

    def test_default_pose_modes_are_absolute_in_the_installed_rlbench_version(self):
        for arm_type in (
            "JointPosition",
            "EndEffectorPoseViaIK",
            "EndEffectorPoseViaPlanning",
        ):
            validate_conversion_action_mode(
                {
                    "type": "MoveArmThenGripper",
                    "arm": {"type": arm_type, "kwargs": {}},
                    "gripper": {"type": "Discrete"},
                }
            )
            with self.assertRaisesRegex(ValueError, "relative action labels"):
                validate_conversion_action_mode(
                    {
                        "type": "MoveArmThenGripper",
                        "arm": {"type": arm_type, "kwargs": {"absolute_mode": False}},
                        "gripper": {"type": "Discrete"},
                    }
                )

    def test_reader_derives_joint_velocity_action_without_reencoding(self):
        """Test that the RLBenchRawDataset reader derives the joint velocity action without re-encoding."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            episode = next(iter(RLBenchRawDataset(root)))
            self.assertEqual(
                episode.task_descriptions,
                ("push the buttons", "press all of the buttons"),
            )
            np.testing.assert_array_equal(
                episode.frames[0].action,
                np.array([10, 11, 12, 13, 14, 15, 16, 0], dtype=np.float32),
            )
            self.assertEqual(episode.frames[0].images["front_rgb"].dtype, np.uint8)
            self.assertEqual(episode.frames[0].state.shape, (53,))


if __name__ == "__main__":
    unittest.main()
