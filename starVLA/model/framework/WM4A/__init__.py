"""WM4A — World-Model-for-Action frameworks.

Frameworks that use a World Model (e.g. Cosmos-Reason2) as the perception
backbone live here.  World models may share the same Qwen3-VL architecture
under the hood but are trained on video prediction / physical reasoning,
providing richer temporal representations for action prediction.

The interface contract is identical to VLM4A frameworks:
  - forward(examples) -> {"action_loss": Tensor}
  - predict_action(examples) -> {"normalized_actions": np.ndarray}
"""
