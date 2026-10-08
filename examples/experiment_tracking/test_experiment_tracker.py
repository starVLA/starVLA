"""Test tracking without launching Accelerate, importing torch, or using a GPU."""

import ast
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from omegaconf import OmegaConf

from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.experiment_tracker import ExperimentTracker

ROOT = Path(__file__).resolve().parents[2]


def make_config(**tracking):
    cfg = OmegaConf.create(
        {
            "run_id": "test-run",
            "run_root_dir": "/tmp/starvla",
            "output_dir": "${run_root_dir}/${run_id}",
            "wandb_project": "legacy-project",
            "wandb_entity": "legacy-entity",
            "trainer": {"logging_frequency": 10},
        }
    )
    if tracking:
        cfg.tracking = tracking
    return cfg


class ExperimentTrackerTest(unittest.TestCase):
    def setUp(self):
        self.swanlab = mock.Mock()
        self.wandb = mock.Mock()
        self.modules = mock.patch.dict(sys.modules, {"swanlab": self.swanlab, "wandb": self.wandb})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.environment = mock.patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop("WANDB_MODE", None)
        os.environ.pop("WANDB_DISABLED", None)

    def test_default_backend_preserves_wandb_arguments_and_lifecycle(self):
        tracker = ExperimentTracker(make_config())
        tracker.log({"loss": 0.5}, step=10)
        tracker.finish()

        self.wandb.init.assert_called_once_with(
            name="test-run",
            dir="/tmp/starvla/test-run/wandb",
            project="legacy-project",
            entity="legacy-entity",
            group="vla-train",
        )
        self.wandb.log.assert_called_once_with({"loss": 0.5}, step=10)
        self.wandb.finish.assert_called_once_with()
        self.swanlab.init.assert_not_called()

    def test_swanlab_handles_wrapped_config_and_resolves_interpolation(self):
        cfg = make_config(backend="swanlab", project="test-project", workspace="test-team", mode="offline")
        del cfg.wandb_project
        del cfg.wandb_entity
        tracker = ExperimentTracker(wrap_config(cfg))
        tracker.log({"loss": 0.25, "learning_rate/base": 1e-5}, step=20)
        tracker.finish()

        self.swanlab.init.assert_called_once_with(
            project="test-project",
            workspace="test-team",
            experiment_name="test-run",
            config=OmegaConf.to_container(cfg, resolve=True),
            logdir="/tmp/starvla/test-run/swanlog",
            mode="offline",
        )
        self.swanlab.log.assert_called_once_with({"loss": 0.25, "learning_rate/base": 1e-5}, step=20)
        self.swanlab.finish.assert_called_once_with()
        self.wandb.init.assert_not_called()

    def test_swanlab_defaults(self):
        ExperimentTracker(make_config(backend="swanlab"))
        kwargs = self.swanlab.init.call_args.kwargs
        self.assertEqual(kwargs["project"], "starVLA")
        self.assertIsNone(kwargs["workspace"])
        self.assertEqual(kwargs["mode"], "online")

    def test_wandb_disable_flags_do_not_disable_swanlab(self):
        os.environ.update(WANDB_MODE="disabled", WANDB_DISABLED="true")
        tracker = ExperimentTracker(make_config(backend="swanlab", mode="offline"))
        self.assertTrue(tracker.enabled)
        self.swanlab.init.assert_called_once()

    def test_disabled_wandb_does_not_import_or_call_sdk(self):
        for environment in ({"WANDB_MODE": "disabled"}, {"WANDB_DISABLED": "TRUE"}):
            with self.subTest(environment=environment), mock.patch.dict(os.environ, environment):
                with mock.patch.dict(sys.modules, {"wandb": None}):
                    tracker = ExperimentTracker(make_config())
                    self.assertFalse(tracker.enabled)
                    tracker.log({"loss": 1}, step=1)
                    tracker.finish()
        self.wandb.init.assert_not_called()

    def test_unused_backends_are_not_imported(self):
        with mock.patch.dict(sys.modules, {"swanlab": None}):
            ExperimentTracker(make_config())
        with mock.patch.dict(sys.modules, {"wandb": None}):
            ExperimentTracker(make_config(backend="swanlab", mode="offline"))

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported tracking backend"):
            ExperimentTracker(make_config(backend="unknown"))
        self.wandb.init.assert_not_called()
        self.swanlab.init.assert_not_called()


class TrainerTrackingTest(unittest.TestCase):
    def setUp(self):
        # Load only the actual tracking methods: importing the trainer would
        # initialize a global Accelerator and require the full GPU stack.
        path = ROOT / "starVLA/training/train_starvla.py"
        tree = ast.parse(path.read_text())
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VLATrainer")
        methods = [node for node in trainer.body if getattr(node, "name", None) in {"_init_tracker", "_log_metrics"}]
        self.logger = mock.Mock()
        self.tracker_factory = mock.Mock()
        self.dist = mock.Mock()
        self.dist.is_initialized.return_value = False
        namespace = {"ExperimentTracker": self.tracker_factory, "logger": self.logger, "dist": self.dist}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
        self.trainer = SimpleNamespace(
            config=make_config(),
            accelerator=mock.Mock(is_main_process=True),
            completed_steps=10,
            optimizer=SimpleNamespace(param_groups=[{"name": "base"}]),
            lr_scheduler=mock.Mock(),
            vla_train_dataloader=[None] * 20,
        )
        self.trainer.lr_scheduler.get_last_lr.return_value = [1e-5]
        self.init_tracker = namespace["_init_tracker"]
        self.log_metrics = namespace["_log_metrics"]

    def test_main_rank_initializes_and_synchronizes(self):
        self.tracker_factory.return_value.enabled = True
        self.init_tracker(self.trainer)
        self.assertTrue(self.trainer._tracker_enabled)
        self.tracker_factory.assert_called_once_with(self.trainer.config)
        self.trainer.accelerator.wait_for_everyone.assert_called_once_with()

    def test_non_main_rank_only_synchronizes(self):
        self.trainer.accelerator.is_main_process = False
        self.init_tracker(self.trainer)
        self.tracker_factory.assert_not_called()
        self.assertFalse(self.trainer._tracker_enabled)
        self.trainer.accelerator.wait_for_everyone.assert_called_once_with()

    def test_init_failure_preserves_rank_rendezvous(self):
        self.tracker_factory.side_effect = RuntimeError("tracker unavailable")
        self.init_tracker(self.trainer)
        self.assertFalse(self.trainer._tracker_enabled)
        self.trainer.accelerator.wait_for_everyone.assert_called_once_with()
        self.logger.warning.assert_called_once()

    def test_metrics_include_learning_rate_epoch_and_global_step(self):
        self.init_tracker(self.trainer)
        self.log_metrics(self.trainer, {"loss": 0.5})
        self.trainer.tracker.log.assert_called_once_with(
            {"loss": 0.5, "learning_rate/base": 1e-5, "epoch": 0.5}, step=10
        )

    def test_logging_failure_disables_further_logging(self):
        self.init_tracker(self.trainer)
        self.trainer.tracker.log.side_effect = RuntimeError("network unavailable")
        self.log_metrics(self.trainer, {"loss": 1})
        self.log_metrics(self.trainer, {"loss": 0.5})
        self.assertFalse(self.trainer._tracker_enabled)
        self.trainer.tracker.log.assert_called_once()
        self.logger.warning.assert_called_once()

    def test_non_main_rank_does_not_log(self):
        self.dist.is_initialized.return_value = True
        self.dist.get_rank.return_value = 1
        self.init_tracker(self.trainer)
        self.log_metrics(self.trainer, {"loss": 1})
        self.trainer.tracker.log.assert_not_called()

    def test_disabled_backend_does_not_log(self):
        self.tracker_factory.return_value.enabled = False
        self.init_tracker(self.trainer)
        self.log_metrics(self.trainer, {"loss": 1})
        self.trainer.tracker.log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
