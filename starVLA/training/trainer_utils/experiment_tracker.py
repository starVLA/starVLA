"""Experiment logging backends for the VLA trainer."""

import os

from omegaconf import OmegaConf


class ExperimentTracker:
    def __init__(self, cfg):
        tracking = cfg.get("tracking", {})
        self.backend = tracking.get("backend", "wandb")
        self.client = None

        if self.backend == "swanlab":
            import swanlab

            full_cfg = cfg.unwrap() if hasattr(cfg, "unwrap") else cfg
            swanlab.init(
                project=tracking.get("project", "starVLA"),
                workspace=tracking.get("workspace", None),
                experiment_name=cfg.run_id,
                config=OmegaConf.to_container(full_cfg, resolve=True),
                logdir=os.path.join(cfg.output_dir, "swanlog"),
                mode=tracking.get("mode", "online"),
            )
            self.client = swanlab
        elif self.backend == "wandb":
            if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
                "1",
                "true",
                "yes",
            }:
                return

            import wandb

            wandb.init(
                name=cfg.run_id,
                dir=os.path.join(cfg.output_dir, "wandb"),
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group="vla-train",
            )
            self.client = wandb
        else:
            raise ValueError(f"Unsupported tracking backend: {self.backend}")

    @property
    def enabled(self):
        return self.client is not None

    def log(self, metrics, step):
        if self.enabled:
            self.client.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            self.client.finish()
