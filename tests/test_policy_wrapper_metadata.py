"""Tests for policy-server training metadata helpers."""

import sys
import types

from deployment.model_server.metadata_utils import (
    build_image_metadata_from_model_config,
    training_video_keys_from_model_config,
)


def _install_fake_data_config(monkeypatch, robot_configs, mixtures_by_name):
    data_config = types.ModuleType("starVLA.dataloader.gr00t_lerobot.data_config")
    data_config.ROBOT_TYPE_CONFIG_MAP = robot_configs

    mixtures = types.ModuleType("starVLA.dataloader.gr00t_lerobot.mixtures")
    mixtures.DATASET_NAMED_MIXTURES = mixtures_by_name

    monkeypatch.setitem(sys.modules, data_config.__name__, data_config)
    monkeypatch.setitem(sys.modules, mixtures.__name__, mixtures)


def test_libero_mix_metadata_uses_data_config_video_keys(monkeypatch):
    class FakeRobotConfig:
        video_keys = ["video.primary_image", "video.wrist_image"]

    _install_fake_data_config(
        monkeypatch,
        {"libero_franka": FakeRobotConfig()},
        {
            "libero_all": [
                ("libero_object", 1.0, "libero_franka"),
                ("libero_goal", 1.0, "libero_franka"),
            ]
        },
    )

    model_cfg = {
        "datasets": {
            "vla_data": {
                "data_mix": "libero_all",
                "obs": ["image_0"],
            }
        }
    }

    assert training_video_keys_from_model_config(model_cfg) == [
        "video.primary_image",
        "video.wrist_image",
    ]


def test_missing_data_mix_has_no_video_key_metadata():
    model_cfg = {"datasets": {"vla_data": {"obs": ["image_0"]}}}

    assert training_video_keys_from_model_config(model_cfg) is None


def test_unknown_data_mix_has_no_video_key_metadata(monkeypatch):
    _install_fake_data_config(monkeypatch, {}, {})
    model_cfg = {"datasets": {"vla_data": {"data_mix": "unknown_mix"}}}

    assert training_video_keys_from_model_config(model_cfg) is None


def test_mixed_video_key_orders_omit_video_key_metadata(monkeypatch):
    class PrimaryOnlyConfig:
        video_keys = ["video.primary_image"]

    class DualViewConfig:
        video_keys = ["video.primary_image", "video.wrist_image"]

    _install_fake_data_config(
        monkeypatch,
        {
            "primary_only": PrimaryOnlyConfig(),
            "dual_view": DualViewConfig(),
        },
        {
            "mixed_views": [
                ("primary_dataset", 1.0, "primary_only"),
                ("dual_dataset", 1.0, "dual_view"),
            ]
        },
    )
    model_cfg = {"datasets": {"vla_data": {"data_mix": "mixed_views"}}}

    assert training_video_keys_from_model_config(model_cfg) is None


def test_image_metadata_warns_when_derived_video_keys_override_raw_obs_count(monkeypatch, caplog):
    class FakeRobotConfig:
        video_keys = ["video.primary_image", "video.wrist_image"]

    _install_fake_data_config(
        monkeypatch,
        {"libero_franka": FakeRobotConfig()},
        {"libero_all": [("libero_goal", 1.0, "libero_franka")]},
    )
    model_cfg = {
        "datasets": {
            "vla_data": {
                "data_mix": "libero_all",
                "obs": ["image_0"],
            }
        }
    }

    metadata = build_image_metadata_from_model_config(model_cfg)

    assert metadata == {
        "vla_video_keys": ["video.primary_image", "video.wrist_image"],
        "vla_data_obs": ["image_0"],
    }
    assert "Using derived vla_video_keys over raw vla_data_obs" in caplog.text
