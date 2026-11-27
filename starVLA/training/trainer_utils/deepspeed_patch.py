"""Two changes to the DeepSpeed ZeRO-2 optimizer step that leave every gradient and update bitwise identical. They act
on the engine and are independent of the model. Written against DeepSpeed 0.16.9 (the version in requirements.txt).

grad norm    the clipping norm as vector_norm(g, dtype=float64) instead of vector_norm(g.double()), per parameter as
             before: the same value bit for bit, without materializing every gradient in fp64 (18 GB written and
             read per step for a 4.5 B-parameter model on two ranks).
all-reduce   each gradient bucket is all-reduced in place (zero_optimization.reduce_scatter = false) instead of being
             sliced per parameter and owner, concatenated, reduced and copied back. The default path divides by the
             number of ranks before the sum and this one after it. On two ranks a / 2 + b / 2 and (a + b) / 2 are the
             same floating-point value. On three, a / 3 + b / 3 + c / 3 and (a + b + c) / 3 round differently, and
             with more ranks the order of the sum can depend on where an element sits in the buffer. So this one
             is applied on one or two ranks only.
"""


def _fused_grad_norm(zero_optimizer):
    """DeepSpeedZeroOptimizer.get_grad_norm_direct (0.16.9) with the fp64 cast fused into the norm kernel.

    DeepSpeed computes the clipping norm per parameter as vector_norm(g.double()): every gradient of this rank's
    partition (2.25 B elements) is first materialized in fp64, 18 GB written and read back per step.
    vector_norm(g, dtype=float64) accumulates in fp64 without the copy and returns the same bits (33 shapes compared
    on B300: identical; 68 ms -> 2 ms). Everything else is DeepSpeed's code; the inf-norm branch is delegated."""
    import types
    from math import inf

    import torch
    from deepspeed import comm as dist
    from deepspeed.accelerator import get_accelerator
    from deepspeed.runtime.utils import is_model_parallel_parameter, mask_nan_or_inf_with_val_inplace
    from deepspeed.runtime.zero import stage_1_and_2 as zero

    original = type(zero_optimizer).get_grad_norm_direct

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        norm_type = float(norm_type)
        if norm_type == inf:
            return original(self, gradients, params, norm_type=norm_type)
        all_norms = []
        for g, p in zip(gradients, params):
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, zero.PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue
            if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                all_norms.append(
                    torch.linalg.vector_norm(g.data, ord=norm_type, dtype=torch.float64).to(
                        get_accelerator().current_device_name()
                    )
                )
        if len(all_norms) > 0:
            total_norm = torch.stack(all_norms).square().sum().float()
        else:
            total_norm = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        # Sum across all model parallel Device.
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=total_norm, op=dist.ReduceOp.SUM)

        total_norm = total_norm.pow(1.0 / norm_type)

        mask_nan_or_inf_with_val_inplace(total_norm, device=self.device)

        return total_norm

    zero_optimizer.get_grad_norm_direct = types.MethodType(get_grad_norm_direct, zero_optimizer)


def _zero_optimizer(engine):
    optimizer = getattr(engine, "optimizer", None)
    return optimizer if type(optimizer).__name__ == "DeepSpeedZeroOptimizer" else None


def fused_grad_norm(engine):
    optimizer = _zero_optimizer(engine)
    if optimizer is None:
        return False
    _fused_grad_norm(optimizer)
    return True


def plain_allreduce(engine):
    optimizer = _zero_optimizer(engine)
    if optimizer is None or not hasattr(optimizer, "reduce_scatter"):
        return False
    from deepspeed import comm as dist

    if dist.get_world_size(group=getattr(optimizer, "dp_process_group", None)) > 2:
        return False  # bitwise identical on one or two ranks only (see above)
    optimizer.reduce_scatter = False
    return True


def apply(engine):
    """Apply both to the engine the trainer steps (the DeepSpeed engine Accelerate returns); anything else is left
    alone. Returns the names of what was applied."""
    if getattr(engine, "_starvla_deepspeed_patch", None) is not None:
        return engine._starvla_deepspeed_patch
    applied = []
    for name, function in (("grad norm without fp64 copies", fused_grad_norm), ("in-place all-reduce", plain_allreduce)):
        try:
            if function(engine):
                applied.append(name)
        except Exception as error:  # must never stop a run that works without it
            print(f"[starVLA] DeepSpeed {name}: left off ({type(error).__name__}: {error})", flush=True)
    if applied:
        print(f"[starVLA] DeepSpeed optimizer step: {', '.join(applied)}", flush=True)
    try:
        engine._starvla_deepspeed_patch = applied
    except AttributeError:
        pass
    return applied
