
## 🌟 How does starVLA make model development Lego-like again?
👇 StarVLA achieves “Lego-like” development via the following designs:

<a id="model"></a>
<details close>
<summary><b>1. Model: Modular & Extensible Framework</b></summary>

StarVLA emphasizes modular model design, following top‑down decomposition and a principle of high cohesion & low coupling. We define the following conventions:

1. `starVLA.model.framework.yourframework.py` is the only external API of the model; it should correspond to (be isomorphic with) the framework figure in your paper.  
2. Each `yourframework.py` or `module.py` can run standalone (e.g., `python yourframework.py` to demo forward + inference).  

</details>


<a id="data"></a>
<details close>
<summary><b>2. DataLoader: Model-Agnostic Data Processing</b></summary>

Best practice references: GR00T / LeRobot action data schemas; multimodal data can reuse LLaVA JSON style. Conventions:

1. Dataloader returns raw data: `PIL.Image`, `str`, normalized actions, state, etc. Must return a single dict
2. Any model‑specific preprocessing should not be processing in dataloader,  but only lives inside `yourframework.forward()`
Dataloader saves any data-processing contexts (normalization stats, transforms, etc.) to the output path.
3. Each `dataset.py` should been run standalone and print/validate one legal sample dict. e.g., `python lerobot_datasets.py`.

</details>


 
<a id="config"></a>
<details close>
<summary><b>3. Config System: Global & Extensible Unified Configuration</b></summary>

StarVLA uses a single global configuration object; all parameter accesses should follow absolute (fully qualified) keys.
The configuration is read from `config_yaml` and converted into an `OmegaConf DictConfig`, which permits redundancy, flexible grouping, and easy addition of new parameters.

Conventions:
1. Use `OmegaConf.load(args.config_yaml)` as the single configuration entry; standalone debugging also uses `args.config_yaml`.
2. Parameters may be intentionally redundant; you can freely add or override them via the CLI. Example:
`--framework.name QwenOFT` to overwite and  `--framework.action_model.new_arg ${action_type}` for adding new arg.
3. Config snapshot: save the unified config in the output directory so experiments can be restarted quickly.

</details>


<a id="trainer"></a>
<details close>
<summary><b>4. Trainer: Lightweight & Strategy-Oriented</b></summary>

StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.

Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  

</details>

<a id="inference"></a>
<details close>
<summary><b>5. Inference: Unified WebSocket Abstraction</b></summary>

StarVLA uses a unified WebSocket layer to decouple complex training and evaluation environments, providing an environment-agnostic inference interface (`deployment/model_server`) and simulator-specific adapters (e.g., `model2simpler_interface.py`).

Conventions:
1. `policy_server.py` exposes only the core inference call: `framework.predict_action()`  
2. Disallow ad‑hoc test‑time and simulator‑specific  on‑the‑fly parameter injection (e.g., extra un‑normalization flags, stats, execution heuristics) to preserve a stable, reproducible evaluation pipeline.
3. Provide per‑environment policy clients (e.g., `examples/SimplerEnv/model2simpler_interface.py`) that handle connection, request packing, retries, and action post‑processing for vairous benchmarks.

</details>




---
