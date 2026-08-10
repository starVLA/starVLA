"""Real DeepSpeed gradient-accumulation smoke test.

Run this file through ``accelerate launch`` with a DeepSpeed ZeRO-2 or ZeRO-3
configuration and ``--trainer {vla,vlm,cotrain}``. The same command works with
one or multiple processes.
"""

import argparse
from types import SimpleNamespace

import torch
from accelerate.utils import DistributedType
from torch.utils.data import DataLoader, TensorDataset

from starVLA.training.train_starvla import VLATrainer
from starVLA.training.train_starvla_cotrain import VLAMTrainer as CoTrainTrainer
from starVLA.training.train_starvlm import VLAMTrainer as VLMTrainer
from starVLA.training.trainer_utils.trainer_tools import build_accelerator


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.fill_(1.0)

    def forward(self, value):
        return {"action_loss": self.projection(value).sum()}

    def qwen_vl_interface(self, value):
        return SimpleNamespace(loss=self.projection(value).sum())


class _Scheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def main(trainer_kind):
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            gradient_accumulation_steps=2,
            gradient_clipping=None,
            loss_scale=SimpleNamespace(vlm=1.0),
        )
    )
    accelerator = build_accelerator(config)
    model = _ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    values = torch.tensor([[0.25], [0.75]]).repeat(accelerator.num_processes, 1)
    dataloader = DataLoader(TensorDataset(values), batch_size=1)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    engine = model
    assert accelerator.distributed_type == DistributedType.DEEPSPEED
    assert accelerator.gradient_accumulation_steps == 2
    assert engine.gradient_accumulation_steps() == 2

    scheduler = _Scheduler()
    trainer_cls = {
        "vla": VLATrainer,
        "vlm": VLMTrainer,
        "cotrain": CoTrainTrainer,
    }[trainer_kind]
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = config
    trainer.model = engine
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.accelerator = accelerator

    boundaries = []
    initial_global_steps = engine.global_steps
    for (value,) in dataloader:
        value = value.to(dtype=engine.module.projection.weight.dtype)
        if trainer_kind == "vla":
            metrics = trainer._train_step(value)
        elif trainer_kind == "vlm":
            metrics = trainer._train_step({"value": value})
        else:
            metrics = trainer._train_step(value, {"value": value})
        boundaries.append(metrics["_optimizer_step"])

    assert boundaries == [False, True]
    assert engine.global_steps == initial_global_steps + 1
    assert scheduler.steps == 1
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(engine)
    if accelerator.is_main_process:
        final_weight = state_dict["projection.weight"].float().item()
        expected_weight = 0.90 if trainer_kind == "cotrain" else 0.95
        assert abs(final_weight - expected_weight) < 0.006
    accelerator.print(f"DeepSpeed {trainer_kind} gradient accumulation smoke passed")
    accelerator.state.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", choices=("vla", "vlm", "cotrain"), required=True)
    args = parser.parse_args()
    main(args.trainer)
