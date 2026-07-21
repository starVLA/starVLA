"""Real DeepSpeed gradient-accumulation smoke test.

Run this file through ``accelerate launch`` with a DeepSpeed ZeRO-2 or ZeRO-3
configuration. The same command works with one or multiple processes.
"""

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from starVLA.training.train_starvla import VLATrainer
from starVLA.training.trainer_utils.trainer_tools import build_accelerator


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.fill_(1.0)

    def forward(self, value):
        return {"action_loss": self.projection(value).sum()}


class _Scheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def main():
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            gradient_accumulation_steps=2,
            gradient_clipping=None,
        )
    )
    accelerator = build_accelerator(config)
    model = _ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    values = torch.tensor([[0.25], [0.75]]).repeat(accelerator.num_processes, 1)
    dataloader = DataLoader(TensorDataset(values), batch_size=1)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    engine = model
    assert hasattr(engine, "is_gradient_accumulation_boundary")
    assert hasattr(engine, "backward")
    assert accelerator.gradient_accumulation_steps == 2
    assert engine.gradient_accumulation_steps() == 2

    scheduler = _Scheduler()
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = config
    trainer.model = engine
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.accelerator = accelerator

    boundaries = []
    initial_global_steps = engine.global_steps
    for (value,) in dataloader:
        value = value.to(dtype=engine.module.projection.weight.dtype)
        metrics = trainer._train_step(value)
        boundaries.append(metrics["_optimizer_step"])

    assert boundaries == [False, True]
    assert engine.global_steps == initial_global_steps + 1
    assert scheduler.steps == 1
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(engine)
    if accelerator.is_main_process:
        final_weight = state_dict["projection.weight"].float().item()
        assert abs(final_weight - 0.95) < 0.006
    accelerator.print("DeepSpeed gradient accumulation smoke passed")
    accelerator.state.destroy_process_group()


if __name__ == "__main__":
    main()
