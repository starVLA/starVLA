from accelerate import Accelerator, DeepSpeedPlugin


def build_accelerator(cfg) -> Accelerator:
    deepspeed_plugin = DeepSpeedPlugin()
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
    )
    accelerator.print(accelerator.state)
    return accelerator
