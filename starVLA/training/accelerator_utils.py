def _config_get(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def get_gradient_accumulation_steps(cfg) -> int:
    value = _config_get(_config_get(cfg, "trainer"), "gradient_accumulation_steps", 1)
    try:
        steps = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("trainer.gradient_accumulation_steps must be a positive integer") from exc
    if steps < 1:
        raise ValueError("trainer.gradient_accumulation_steps must be a positive integer")
    return steps


def load_accelerate_classes():
    from accelerate import Accelerator, DeepSpeedPlugin

    return Accelerator, DeepSpeedPlugin


def create_accelerator(cfg):
    accelerator_cls, deepspeed_plugin_cls = load_accelerate_classes()
    deepspeed_plugin = deepspeed_plugin_cls()
    return accelerator_cls(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=get_gradient_accumulation_steps(cfg),
    )
