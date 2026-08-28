import contextlib
import importlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType

TRAINERS = (
    ("starVLA.training.train_starvla", "vla"),
    ("starVLA.training.train_starvla_cotrain", "cotrain"),
    ("starVLA.training.train_starvlm", "vlm"),
)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return {"action_loss": self.weight * batch}

    def qwen_vl_interface(self, value):
        return SimpleNamespace(loss=self.weight * value)


class _FakeDeepSpeedEngine:
    def __init__(self, model, boundaries):
        self.module = model
        self.boundaries = iter(boundaries)
        self.backward_calls = 0
        self.step_calls = 0
        self._boundary = False

    def forward(self, batch):
        return self.module(batch)

    def backward(self, loss):
        self.backward_calls += 1
        loss.backward()

    def is_gradient_accumulation_boundary(self):
        self._boundary = next(self.boundaries)
        return self._boundary

    def step(self):
        self.step_calls += 1


def _trainer_config():
    return SimpleNamespace(
        trainer=SimpleNamespace(
            gradient_accumulation_steps=2,
            gradient_clipping=100.0,
            loss_scale=SimpleNamespace(vlm=1.0),
            max_train_steps=1,
            eval_interval=1,
            save_interval=1,
        )
    )


def _make_trainer(module, kind, model, optimizer, scheduler, accelerator):
    trainer_cls = module.VLATrainer if kind == "vla" else module.VLAMTrainer
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = _trainer_config()
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.accelerator = accelerator
    return trainer


class TrainingGradientAccumulationTest(unittest.TestCase):
    def test_two_microbatches_contribute_to_each_trainer_update(self):
        for module_name, kind in TRAINERS:
            with self.subTest(trainer=kind):
                module = importlib.import_module(module_name)
                accelerator = Accelerator(gradient_accumulation_steps=2)
                model = _ToyModel()
                base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                optimizer_step = mock.patch.object(base_optimizer, "step", wraps=base_optimizer.step)
                optimizer_zero_grad = mock.patch.object(base_optimizer, "zero_grad", wraps=base_optimizer.zero_grad)
                model, optimizer = accelerator.prepare(model, base_optimizer)
                scheduler = mock.Mock()
                trainer = _make_trainer(module, kind, model, optimizer, scheduler, accelerator)

                with (
                    optimizer_step as effective_step,
                    optimizer_zero_grad as effective_zero_grad,
                    mock.patch.object(
                        accelerator,
                        "clip_grad_norm_",
                        wraps=accelerator.clip_grad_norm_,
                    ) as clip_grad_norm,
                    mock.patch.object(module.torch, "autocast", return_value=contextlib.nullcontext()),
                ):
                    if kind == "vla":
                        first = trainer._train_step(torch.tensor(1.0))
                        second = trainer._train_step(torch.tensor(3.0))
                        expected = 0.8
                    elif kind == "cotrain":
                        first = trainer._train_step(torch.tensor(1.0), {"value": torch.tensor(2.0)})
                        second = trainer._train_step(torch.tensor(3.0), {"value": torch.tensor(4.0)})
                        expected = 0.5
                    else:
                        first = trainer._train_step({"value": torch.tensor(1.0)})
                        second = trainer._train_step({"value": torch.tensor(3.0)})
                        expected = 0.8

                self.assertFalse(first["_optimizer_step"])
                self.assertTrue(second["_optimizer_step"])
                self.assertAlmostEqual(accelerator.unwrap_model(model).weight.item(), expected, places=6)
                self.assertEqual(effective_step.call_count, 1)
                self.assertEqual(effective_zero_grad.call_count, 1)
                clip_grad_norm.assert_called_once()
                scheduler.step.assert_called_once_with()

    def test_outer_loop_side_effects_run_only_on_optimizer_boundary(self):
        for module_name, kind in TRAINERS:
            with self.subTest(trainer=kind):
                module = importlib.import_module(module_name)
                trainer_cls = module.VLATrainer if kind == "vla" else module.VLAMTrainer
                trainer = trainer_cls.__new__(trainer_cls)
                trainer.config = _trainer_config()
                trainer.completed_steps = 0
                trainer.accelerator = SimpleNamespace(
                    sync_gradients=False,
                    is_local_main_process=False,
                )
                trainer._log_training_config = mock.Mock()
                trainer._create_data_iterators = mock.Mock()
                trainer._get_next_batch = mock.Mock(return_value=(object(), object()) if kind == "cotrain" else object())
                trainer._train_step = mock.Mock(
                    side_effect=[{"_optimizer_step": False}, {"loss": 1.0, "_optimizer_step": True}]
                )
                trainer.eval_action_model = mock.Mock(side_effect=lambda metrics: metrics)
                trainer._log_metrics = mock.Mock()
                trainer._save_checkpoint = mock.Mock()
                trainer._finalize_training = mock.Mock()

                progress_bar = mock.Mock()
                with mock.patch.object(module, "tqdm", return_value=progress_bar):
                    trainer.train()

                self.assertEqual(trainer._train_step.call_count, 2)
                progress_bar.update.assert_called_once_with(1)
                trainer.eval_action_model.assert_called_once()
                trainer._log_metrics.assert_called_once()
                trainer._save_checkpoint.assert_called_once_with()
                trainer._finalize_training.assert_called_once_with()

    def test_deepspeed_engine_owns_accumulation_boundaries(self):
        for module_name, kind in TRAINERS:
            with self.subTest(trainer=kind):
                module = importlib.import_module(module_name)
                model = _ToyModel()
                engine = _FakeDeepSpeedEngine(model, [False, True])
                accelerator = mock.Mock()
                accelerator.distributed_type = DistributedType.DEEPSPEED
                accelerator.unwrap_model.return_value = model
                scheduler = mock.Mock()
                optimizer = mock.Mock()
                trainer = _make_trainer(module, kind, engine, optimizer, scheduler, accelerator)

                with mock.patch.object(module.torch, "autocast", return_value=contextlib.nullcontext()):
                    results = []
                    for value in (1.0, 3.0):
                        if kind == "vla":
                            result = trainer._train_step(torch.tensor(value))
                        elif kind == "cotrain":
                            result = trainer._train_step(torch.tensor(value), {"value": torch.tensor(value)})
                        else:
                            result = trainer._train_step({"value": torch.tensor(value)})
                        results.append(result)

                self.assertEqual([result["_optimizer_step"] for result in results], [False, True])
                self.assertEqual(engine.backward_calls, 4 if kind == "cotrain" else 2)
                self.assertEqual(engine.step_calls, 2)
                scheduler.step.assert_called_once_with()
                accelerator.accumulate.assert_not_called()
                accelerator.backward.assert_not_called()
                accelerator.clip_grad_norm_.assert_not_called()
                optimizer.step.assert_not_called()
                optimizer.zero_grad.assert_not_called()

    def test_outer_loop_requires_optimizer_step_metadata(self):
        for module_name, kind in TRAINERS:
            with self.subTest(trainer=kind):
                module = importlib.import_module(module_name)
                trainer_cls = module.VLATrainer if kind == "vla" else module.VLAMTrainer
                trainer = trainer_cls.__new__(trainer_cls)
                trainer.config = _trainer_config()
                trainer.completed_steps = 0
                trainer.accelerator = SimpleNamespace(is_local_main_process=False)
                trainer._log_training_config = mock.Mock()
                trainer._create_data_iterators = mock.Mock()
                trainer._get_next_batch = mock.Mock(return_value=(object(), object()) if kind == "cotrain" else object())
                trainer._train_step = mock.Mock(return_value={})

                with (
                    mock.patch.object(module, "tqdm", return_value=mock.Mock()),
                    self.assertRaises(KeyError),
                ):
                    trainer.train()


if __name__ == "__main__":
    unittest.main()
