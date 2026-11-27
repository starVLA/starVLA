"""worker_prep through whole frameworks (CPU, a few minutes): each framework takes one training step on a batch from a
DataLoader with workers, and must (1) get every backbone input it asks for from what the workers built and (2) arrive
at the same loss and gradients as without workers. The backbone is a small random model behind the real processor,
tokenizer and config of a model directory, so the prompts and the image pipeline are the real ones.

    STARVLA_TEST_VLMS=<dir of Qwen3-VL-4B-Instruct>[,<dir of Qwen2.5-VL-3B-Instruct>] python tests/test_speedups_frameworks.py

With STARVLA_TEST_PALIGEMMA_TOKENIZER=<paligemma_tokenizer.model>, also worker_examples on PI0 / PI05.
"""

import dataclasses
import hashlib
import importlib
import os
import subprocess
import sys
import tempfile

TRAIN_CONFIGS = "starVLA/config/training"
LIBERO = "examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml"
# framework -> (where its framework settings come from, backbone inputs built per training step)
FRAMEWORKS = {
    "QwenOFT": (LIBERO, 1),
    "QwenGR00T": (LIBERO, 1),
    "QwenPI": (LIBERO, 1),
    "QwenPI_v3": (LIBERO, 1),
    "QwenAdapter": (f"{TRAIN_CONFIGS}/starvla_train_adapter.yaml", 1),
    "QwenDiscreteDiffusion": (f"{TRAIN_CONFIGS}/starvla_train_discrete_diffusion.yaml", 1),
    "LangForce": ("QwenGR00T head", 2),  # one prompt with the action query before the instruction, one with it after
    "QwenDual": ("QwenGR00T head", 1),  # passes resized copies of the images
    "CosmosGR00T": (LIBERO, 1),  # the Cosmos-Reason2 interface (Qwen3-VL architecture)
}
HIDDEN = 256


def _small_backbones():
    """from_pretrained of the Qwen-VL classes returns a small random fp32 model with the directory's config otherwise
    unchanged (vocabulary, special token ids, patch size, rotary sections)."""
    import torch
    import transformers
    from transformers import AutoConfig

    def small(real_class):
        def from_pretrained(model_id, **kwargs):
            config = AutoConfig.from_pretrained(model_id)
            text, vision = config.text_config, config.vision_config
            text.hidden_size, text.intermediate_size, text.num_attention_heads, text.num_key_value_heads = (
                HIDDEN,
                512,
                2,
                2,
            )
            vision.hidden_size, vision.intermediate_size, vision.num_heads, vision.out_hidden_size = 64, 128, 4, HIDDEN
            vision.depth = 3
            if hasattr(vision, "deepstack_visual_indexes"):
                vision.deepstack_visual_indexes = [0, 1, 2]
            else:
                vision.fullatt_block_indexes = [1]
            torch.manual_seed(0)
            return real_class._from_config(config, attn_implementation="sdpa").float()

        return staticmethod(from_pretrained)

    for name in ("Qwen3VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"):
        setattr(getattr(transformers, name), "from_pretrained", small(getattr(transformers, name)))


def _config(framework, source, model_dir):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(LIBERO)
    if source == "QwenGR00T head":  # a framework without a training config of its own
        from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig

        module = importlib.import_module(f"starVLA.model.framework.VLM4A.{framework}")
        own = next(c for n, c in vars(module).items() if n.endswith("DefaultConfig") and dataclasses.is_dataclass(c))
        cfg.framework = OmegaConf.merge(
            {"action_model": QwenGR00TDefaultConfig().action_model}, dataclasses.asdict(own()), cfg.framework
        )
        cfg.datasets.vla_data.CoT_prompt = "{instruction}"
    elif source != LIBERO:
        cfg.framework = OmegaConf.load(source).framework
        cfg.framework.action_model.hidden_dim = HIDDEN  # QwenAdapter's head is sized in its config
    cfg.framework.name = framework
    cfg.framework.qwenvl.update(base_vlm=model_dir, attn_implementation="sdpa", vl_hidden_dim=HIDDEN)
    return cfg


class _Samples:
    """Four samples the way the LeRobot datasets return them."""

    def __init__(self, action_steps, action_dim, state_dim):
        self.shapes = (action_steps, action_dim), (1, state_dim)

    def __len__(self):
        return 4

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(index)
        return {
            "image": [Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)],
            "lang": "put the bowl on the plate" + " and then close the drawer" * (index % 3),
            "action": rng.uniform(-1, 1, self.shapes[0]).astype(np.float32),
            "state": rng.uniform(-1, 1, self.shapes[1]).astype(np.float32),
        }


def _step(model, batch):
    import torch

    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    loss = model(batch)["action_loss"]
    loss.backward()
    digest = lambda t: hashlib.sha256(t.detach().float().numpy().tobytes()).hexdigest()
    return float(loss), {name: digest(p.grad) for name, p in model.named_parameters() if p.grad is not None}


def _check(model_dir):
    from torch.utils.data import DataLoader

    _small_backbones()
    from starVLA.dataloader.lerobot_datasets import collate_fn
    from starVLA.model.framework import base_framework
    from starVLA.model.modules import vlm
    from starVLA.dataloader import worker_prep
    from starVLA.model.framework.base_framework import build_framework

    with tempfile.TemporaryDirectory() as tmp:
        cosmos = os.path.join(tmp, "Cosmos-Reason2-like")  # the name selects the interface
        os.symlink(os.path.abspath(model_dir), cosmos)
        for framework, (source, calls) in FRAMEWORKS.items():
            if framework == "CosmosGR00T" and "Qwen3-VL" not in model_dir:
                continue
            vlm.BACKBONE_SPEEDUPS = base_framework.FRAMEWORK_SPEEDUPS = ()  # the reference step runs the original code
            cfg = _config(framework, source, cosmos if framework == "CosmosGR00T" else model_dir)
            model = build_framework(cfg).train()
            head = model.config.framework.action_model
            samples = _Samples(
                int(head.get("action_horizon", None) or head.get("num_actions_chunk", 16)),
                int(head.get("action_dim", 7)),
                int(head.get("state_dim", 7)),
            )
            reference = _step(model, [samples[i] for i in range(len(samples))])
            applied, _ = vlm.run_speedups(
                model.qwen_vl_interface, ["starVLA.dataloader.worker_prep:prepare_inputs_in_workers"]
            )
            assert applied == ["prepare_inputs_in_workers"], (framework, applied)
            before = dict(worker_prep.input_stats)
            loader = DataLoader(
                samples, batch_size=len(samples), num_workers=1, collate_fn=worker_prep.inputs_collate(cfg, collate_fn)
            )
            candidate = _step(model, next(iter(loader)))
            used = worker_prep.input_stats["used"] - before["used"]
            assert used == calls and worker_prep.input_stats["other prompt"] == before["other prompt"], (
                framework,
                worker_prep.input_stats,
            )
            assert candidate == reference, framework
            print(
                f"ok {framework} on {os.path.basename(model_dir)}: {used} backbone input(s) per step from the workers, "
                f"loss and {len(reference[1])} gradients identical",
                flush=True,
            )
            del model, loader


def _check_openpi(tokenizer_model):
    """PI0 and PI05 (small Gemma towers, fp64) on samples with two of the three camera views. worker_examples: a batch
    from a DataLoader with workers gives the same loss, gradients and predicted actions bit for bit."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from torch.utils.data import DataLoader

    from starVLA.dataloader.lerobot_datasets import collate_fn
    from starVLA.model.framework import base_framework
    from starVLA.dataloader import worker_prep as openpi
    from starVLA.model.modules import vlm
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.trainer_utils.config_tracker import wrap_config

    rng = np.random.default_rng(0)
    samples = [
        {
            "image": [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(2)],
            "lang": "pick up the bowl" + " and put it on the plate" * index,
            "action": rng.uniform(-1, 1, (8, 7)).astype(np.float32),
            "state": rng.uniform(-1, 1, (1, 8)).astype(np.float32),
        }
        for index in range(2)
    ]
    small = {"width": 128, "depth": 3, "mlp_dim": 256, "num_heads": 2, "num_kv_heads": 1, "head_dim": 64}
    everything = base_framework.FRAMEWORK_SPEEDUPS
    for framework in ("PI0", "PI05"):
        cfg = OmegaConf.load(f"examples/simBenchmarks/LIBERO/train_files/openpi/{framework.lower()}_libero_8gpu.yaml")
        cfg.framework.update(
            precision="float32", paligemma_config=small, action_expert_config=dict(small, width=64, mlp_dim=128)
        )
        cfg.framework.tokenizer.model_path = tokenizer_model

        def run(*paths):
            vlm.BACKBONE_SPEEDUPS = ()
            base_framework.FRAMEWORK_SPEEDUPS = everything if paths else ()
            torch.manual_seed(0)
            model = build_framework(wrap_config(cfg)).double().train()  # wrapped as the trainer wraps it
            batch = samples
            if paths:
                batch = next(
                    iter(
                        DataLoader(
                            samples,
                            batch_size=len(samples),
                            num_workers=1,
                            collate_fn=openpi.examples_collate(cfg, collate_fn),
                        )
                    )
                )
            torch.manual_seed(1)
            loss = model(batch)["action_loss"]
            loss.backward()
            grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            torch.manual_seed(2)
            actions = model.eval().predict_action([{k: v for k, v in s.items() if k != "action"} for s in samples])
            return float(loss), grads, torch.as_tensor(actions["normalized_actions"])

        loss, grads, actions = run()  # the original code
        used = openpi.example_stats["used"]
        worker_loss, worker_grads, worker_actions = run("worker_examples")
        assert openpi.example_stats["used"] == used + 1, openpi.example_stats
        assert worker_loss == loss and torch.equal(worker_grads, grads) and torch.equal(worker_actions, actions)
        print(
            f"ok {framework} worker_examples: samples prepared in a worker, loss and {grads.numel():,} gradient "
            f"values identical",
            flush=True,
        )


def test_openpi():
    tokenizer_model = os.environ.get("STARVLA_TEST_PALIGEMMA_TOKENIZER", "")
    if not tokenizer_model:
        print("skipped PI0 / PI05: STARVLA_TEST_PALIGEMMA_TOKENIZER is not set")
        return
    result = subprocess.run([sys.executable, __file__, "--openpi", tokenizer_model], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    print("\n".join(line for line in result.stdout.splitlines() if line.startswith("ok ")))


def test_frameworks_train_on_worker_inputs():
    model_dirs = [d for d in os.environ.get("STARVLA_TEST_VLMS", "").split(",") if d]
    if not model_dirs:
        print("skipped: STARVLA_TEST_VLMS is not set")
    for model_dir in model_dirs:
        result = subprocess.run([sys.executable, __file__, model_dir], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        print("\n".join(line for line in result.stdout.splitlines() if line.startswith("ok ")))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if sys.argv[1:2] == ["--openpi"]:
        _check_openpi(sys.argv[2])
    elif len(sys.argv) > 1:
        _check(sys.argv[1])
    else:
        test_frameworks_train_on_worker_inputs()
        test_openpi()
