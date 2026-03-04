# NeuroVLA Historical State Support (StarVLA Integration)

NeuroVLA relies on **historical robot states** (a fixed-length state history window) for both training and evaluation.  
To enable this in StarVLA/LIBERO, you must:

1. **Modify the dataset state indexing logic** (so the dataloader can fetch historical states).
2. **Modify the evaluation script** (so the simulator returns historical states during inference/evaluation).

To make this straightforward, we provide **two ready-to-use example files** that you can directly replace in your repo.

---

## Quick Start

### 1) Replace Required Files

Please replace the following files with the provided examples:

- **Data config (state history indices)**
  - Target path:  
    `starVLA/starVLA/dataloader/gr00t_lerobot/data_config.py`

- **LIBERO evaluation script (sim env returns historical states)**
  - Target path:  
    `starVLA/examples/LIBERO/eval_files/eval_libero.py`

> After replacement, your pipeline will support reading **historical states** required by NeuroVLA.

---

## Configuration: State History Length

### Dataset / Training Side

In `data_config.py`, the historical state window is controlled by:

```python
state_indices = list(range(-16, 0))

	•	-16 means “take the previous 16 steps”
	•	0 means “up to the current step (exclusive)”
	•	Total history length = 16

✅ To change the history length, adjust the range accordingly. For example:
	•	8-step history:

state_indices = list(range(-8, 0))


	•	32-step history:

state_indices = list(range(-32, 0))



⸻

Evaluation / Inference: Returning Historical States

During inference, the evaluation script must return a matching length of historical states.

In eval_libero.py, update the following:

n = 16  # adjust n as needed

And ensure the per-step state vector is formed consistently:

state = np.concatenate(
    (
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    )
)

What this state contains
	•	robot0_eef_pos : end-effector position
	•	_quat2axisangle(robot0_eef_quat) : end-effector orientation (converted)
	•	robot0_gripper_qpos : gripper joint position

Important: If you change the history length in data_config.py, you should also update n in evaluation to keep them consistent.

⸻

Notes & Best Practices
	•	Keep training and evaluation history lengths aligned (e.g., both 16).
	•	If you change the state definition (e.g., add joint angles), update both:
	•	dataset processing (data_config.py / dataloader)
	•	evaluation state extraction (eval_libero.py)
	•	When tuning history length, common choices are 8 / 16 / 32 depending on task temporal complexity.

⸻

File Paths Summary

Purpose	Path
Dataset state history config	starVLA/starVLA/dataloader/gr00t_lerobot/data_config.py
LIBERO eval returns historical states	starVLA/examples/LIBERO/eval_files/eval_libero.py


⸻

Troubleshooting

Shape mismatch / runtime errors

If you see errors related to state tensor shapes, verify:
	•	len(state_indices) in data_config.py
	•	n in eval_libero.py
	•	The concatenated state dimension is unchanged across training and evaluation

⸻

Contact

For questions, issues, or integration help, please contact:
guoweiyu96@gmail.com

⸻


