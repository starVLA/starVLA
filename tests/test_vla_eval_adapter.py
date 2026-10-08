from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("vla_eval")

from vla_eval.cli.config_loader import load_config

from deployment.vla_eval import model_server as adapter


class _FakePolicy:
    metadata: ClassVar = {
        "action_chunk_size": 2,
        "default_unnorm_key": "test",
    }

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls = []

    def predict_action(self, **kwargs):
        self.calls.append(kwargs)
        batch = len(kwargs["examples"])
        action_dim = 7 if len(kwargs["examples"][0]["image"]) == 2 else 14
        actions = np.zeros((batch, 2, action_dim), dtype=np.float32)
        if action_dim == 7:
            actions[..., 6] = 1.0
        else:
            actions[..., 6] = 1.0
            actions[..., 12] = 12.0
            actions[..., 13] = 13.0
        return {"actions": actions}


@pytest.fixture
def fake_policy(monkeypatch):
    monkeypatch.setattr(adapter, "resolve_checkpoint", lambda checkpoint: "/tmp/run/checkpoints/model.pt")
    monkeypatch.setattr(adapter, "_build_policy", _FakePolicy)


def test_configured_missing_camera_fails_fast(fake_policy):
    server = adapter.StarVLAHarnessServer("unused", image_keys=["agentview", "wrist"])
    with pytest.raises(KeyError, match="wrist"):
        server._make_example(
            {
                "images": {"agentview": np.zeros((2, 2, 3), dtype=np.uint8)},
                "task_description": "test",
            }
        )


def test_checkpoint_directory_selects_latest_numeric_step(tmp_path):
    (tmp_path / "config.yaml").write_text("{}")
    (tmp_path / "dataset_statistics.json").write_text("{}")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "steps_9000.pt").touch()
    latest = checkpoint_dir / "steps_10000.safetensors"
    latest.touch()

    assert adapter.resolve_checkpoint(str(tmp_path)) == str(latest)


def test_checkpoint_symlink_preserves_colocated_metadata(tmp_path):
    root = tmp_path / "snapshot"
    (root / "checkpoints").mkdir(parents=True)
    (root / "config.yaml").write_text("{}")
    (root / "dataset_statistics.json").write_text("{}")
    blob = tmp_path / "blob-without-extension"
    blob.touch()
    checkpoint = root / "checkpoints/model.pt"
    checkpoint.symlink_to(blob)

    assert adapter.resolve_checkpoint(str(checkpoint)) == str(checkpoint)


@pytest.mark.parametrize("missing", ["config.yaml", "dataset_statistics.json"])
def test_checkpoint_requires_colocated_metadata(tmp_path, missing):
    (tmp_path / "checkpoints").mkdir()
    checkpoint = tmp_path / "checkpoints/model.pt"
    checkpoint.touch()
    for name in {"config.yaml", "dataset_statistics.json"} - {missing}:
        (tmp_path / name).write_text("{}")

    with pytest.raises(FileNotFoundError, match=missing):
        adapter.resolve_checkpoint(str(checkpoint))


@pytest.mark.parametrize("unnorm_key", [None, "alternate"])
def test_checkpoint_statistics_key_forwarded_to_wrapper_and_inference(fake_policy, unnorm_key):
    server = adapter.StarVLAHarnessServer("unused", unnorm_key=unnorm_key, image_keys=["agentview", "wrist"])
    obs = {"images": {key: np.zeros((2, 2, 3), dtype=np.uint8) for key in ["wrist", "agentview"]}}
    server.predict_batch([obs], [SimpleNamespace()])

    assert server._policy.init_kwargs["unnorm_key"] == unnorm_key
    assert server._policy.calls[0]["unnorm_key"] == (unnorm_key or "test")


def test_ambiguous_checkpoint_statistics_fail_at_startup(fake_policy, monkeypatch):
    monkeypatch.setattr(
        _FakePolicy,
        "metadata",
        {
            "action_chunk_size": 2,
            "available_unnorm_keys": ["first", "second"],
            "default_unnorm_key": None,
        },
    )
    with pytest.raises(ValueError, match="multiple statistics keys"):
        adapter.StarVLAHarnessServer("unused")
    server = adapter.StarVLAHarnessServer("unused", unnorm_key="second")
    assert server._unnorm_key == "second"
    with pytest.raises(ValueError, match="Unknown unnorm_key"):
        adapter.StarVLAHarnessServer("unused", unnorm_key="missing")


def test_gripper_threshold_and_joint_values_are_preserved():
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    actions[:, -1] = [0.0, 0.5, 1.0]
    original = actions.copy()
    result = adapter.postprocess_actions(actions, gripper_transform="open01_to_close_positive")
    np.testing.assert_array_equal(result[:, -1], [1.0, 1.0, -1.0])
    np.testing.assert_array_equal(result[:, :6], original[:, :6])
    np.testing.assert_array_equal(actions, original)


@pytest.mark.parametrize("profile", ["libero", "robotwin"])
def test_shipped_model_configs_match_native_contract(fake_policy, profile):
    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root / f"examples/vla_eval/model_servers/{profile}_qwen3_oft.yaml"))
    server = adapter.StarVLAHarnessServer(**config["args"])
    keys = ["agentview", "wrist"] if profile == "libero" else ["head_camera", "left_camera", "right_camera"]
    obs = {"images": {key: np.full((3, 4, 3), index, dtype=np.uint8) for index, key in reversed(list(enumerate(keys)))}}
    actions = server.predict_batch([obs], [SimpleNamespace()])[0]["actions"]
    example = server._policy.calls[0]["examples"][0]
    assert [np.asarray(image)[0, 0, 0] for image in example["image"]] == list(range(len(keys)))
    assert all(image.size == (224, 224) for image in example["image"])
    assert "state" not in example
    assert server.chunk_size == 2
    assert server._policy.init_kwargs["config_overrides"] == ["framework.qwenvl.base_vlm=Qwen/Qwen3-VL-4B-Instruct"]
    assert server.get_observation_params() == ({"send_wrist_image": True} if profile == "libero" else {})
    if profile == "libero":
        np.testing.assert_array_equal(actions[:, 6], -1.0)
    else:
        np.testing.assert_array_equal(actions[0], [0, 0, 0, 0, 0, 0, 12, 1, 0, 0, 0, 0, 0, 13])
        assert server._policy.calls[0]["unnorm_key"] == "new_embodiment"
