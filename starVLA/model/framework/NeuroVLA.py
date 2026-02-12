
from __future__ import annotations
from typing import Union, List, Dict, Optional, Tuple, Sequence

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.projector.QFormer import get_layerwise_qformer
from starVLA.model.modules.action_model.spike_action_model_multitimestep import (
    get_action_model,
    get_gruedit_model
)





@FRAMEWORK_REGISTRY.register("NeuroVLA")
class NeuroVLA(baseframework):
    """
    NeuroVLA: Vision-Language-Action model for robotic manipulation.

    This model combines a vision-language model (Qwen-VL) with action prediction
    to generate robot actions from visual observations and language instructions.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config

        # Vision-language model for processing images and instructions
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Q-Former for extracting action-relevant features from VLM hidden states
        self.layer_qformer = get_layerwise_qformer(config=self.config)

        # Action prediction model (input_dim=768, hidden_dim=1536, action_dim=7)
        self.action_model = get_action_model(input_dim=768, hidden_dim=768*2, action_dim=7)

        # Edit model for refining actions based on robot states
        self.edit_model = get_gruedit_model(input_dim=768, hidden_dim=256, robot_state_dim=8)

        self.L1_loss = nn.L1Loss()
        self.norm_stats = norm_stats




    def forward(
        self,
        examples: List[dict] = None,
        repeated_diffusion_steps: int = 4,
        **kwargs,
    ) -> Tuple:
        """
        Run a forward pass through the VLM, returning loss for training.

        Args:
            examples: List of training examples, each containing:
                - "image": Input images
                - "lang": Language instructions
                - "action": Ground truth actions [B, T, 7]
                - "state": Robot states [B, T, 8]
                - "solution" (optional): Chain-of-thought solutions

        Returns:
            Dictionary containing action_loss
        """
        inference_num = 0

        # Extract data from examples
        images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        states = [example["state"] for example in examples]

        if "solution" in examples[0]:
            solutions = [example["solution"] for example in examples]
        else:
            solutions = None

        # Build inputs for vision-language model
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions, solutions=solutions
        )

        # Forward pass through VLM to get hidden states
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        vlm_cot_loss = qwenvl_outputs.loss

        if vlm_cot_loss is None or torch.isnan(vlm_cot_loss):
            vlm_cot_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)

        # Action prediction with iterative refinement
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action-relevant features from VLM hidden states
            start_layer = self.config.framework.layer_qformer.qformer_start_layer if self.config else -6
            end_layer = self.config.framework.layer_qformer.qformer_end_layer if self.config else -1
            action_latent_feature = self.layer_qformer(qwenvl_outputs.hidden_states[start_layer:end_layer])

            states = torch.tensor(np.array(states), dtype=torch.float32, device=action_latent_feature.device)
            all_predicted_actions = []
            inference_num = 0

            # Iterative action prediction (can be configured from 2 to 10 iterations)
            while inference_num < 2:
                # Edit action features based on current robot states
                edit_action_feature = self.edit_model(action_latent_feature, states)

                # Predict action chunk
                predicted_actions = self.action_model.predict_action(edit_action_feature)
                all_predicted_actions.append(predicted_actions)

                # Update states for next iteration
                predicted_states = torch.zeros_like(states)
                predicted_states[:, :predicted_actions.shape[1], :7] = predicted_actions
                predicted_states[:, :, 7] = states[:, :, 7]  # Keep gripper state
                states = predicted_states.clone()
                inference_num += 1

            # Compute action loss
            action_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=predicted_actions.device)
            predicted_action_tensor = torch.cat(all_predicted_actions, dim=1)
            action_loss = self.L1_loss(predicted_action_tensor, action_tensor)

        return {"action_loss": action_loss}


    def predict_action(
        self,
        batch_images: Union[Image, List[Image]],
        instructions: List[str],
        states: Optional[List[Sequence[float]]] = None,
        solutions: Union[Dict, List[Dict]] = None,
        unnorm_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        **kwargs: str
    ) -> np.ndarray:
        """
        Predict action from images and instructions.

        Args:
            batch_images: Input images (PIL Image or list of PIL Images)
            instructions: Task instructions (list of strings)
            states: Robot states history [B, T, 8], where last dim is [x,y,z,roll,pitch,yaw,gripper,pad]
            solutions: Optional solution dict for chain-of-thought
            unnorm_key: Key for unnormalization (if using norm_stats)
            cfg_scale: Classifier-free guidance scale (>1.0 enables CFG)
            use_ddim: Whether to use DDIM sampling
            num_ddim_steps: Number of DDIM steps

        Returns:
            Dictionary containing "normalized_actions" [B, T, 7]
        """
        predict_num = 0

        # Resize images to model input size
        batch_images = resize_images(batch_images, target_size=(224, 224))

        # Build VLM inputs
        inferface_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        qwen_inputs = inferface_inputs

        all_predicted_actions = []

        # Generate cognition features through VLM
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs.input_ids,
                attention_mask=qwen_inputs.attention_mask,
                pixel_values=qwen_inputs.pixel_values,
                image_grid_thw=qwen_inputs.image_grid_thw,
                labels=qwen_inputs.input_ids.clone(),
                output_hidden_states=True,
                return_dict=True,
            )

        # Action prediction with iterative refinement
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action features from VLM hidden states
            start_layer = self.config.framework.layer_qformer.qformer_start_layer if self.config else -6
            end_layer = self.config.framework.layer_qformer.qformer_end_layer if self.config else -1

            action_latent_feature = self.layer_qformer(qwenvl_outputs.hidden_states[start_layer:end_layer])

            using_cfg = cfg_scale > 1.0
            B = action_latent_feature.shape[0]

            # Convert states to tensor
            states = torch.tensor(
                np.array(states, dtype=np.float32),
                dtype=torch.float32,
                device=action_latent_feature.device
            )

            # Iterative action prediction (default: 2 iterations)
            while predict_num < 2:
                # Edit action features based on current states
                edit_action_feature = self.edit_model(action_latent_feature, states)

                # Predict action chunk
                samples = self.action_model.predict_action(edit_action_feature)
                all_predicted_actions.append(samples)

                # Update states for next iteration
                predicted_states = torch.zeros_like(states)
                predicted_states[:, :samples.shape[1], :7] = samples
                predicted_states[:, :, 7] = states[:, :, 7]  # Keep gripper state
                states = predicted_states.clone()
                predict_num += 1

        # Concatenate all predicted action chunks
        predicted_action_tensor = torch.cat(all_predicted_actions, dim=1)
        normalized_actions = predicted_action_tensor.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


def build_model_framework(config: dict = {}) -> NeuroVLA:
    """Build NeuroVLA model from config."""
    model = NeuroVLA(config=config)
    return model


if __name__ == "__main__":
    """
    Example usage for testing the model.

    This demonstrates how to:
    1. Load a pretrained model
    2. Prepare input data
    3. Run inference to predict actions
    """
    import pickle
    from omegaconf import OmegaConf

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Option 1: Load from pretrained checkpoint
    # model = NeuroVLA.from_pretrained("path/to/checkpoint.pt").to(device)
    model = NeuroVLA.from_pretrained("/workspace/nature_submit/NeuroVLA/playground/Checkpoints/1104_neurovla_gru_xiaonao_goal_dualimage_spike_multistep_ac8_768*2_yibu/checkpoints/steps_10000_pytorch_model.pt").to(device)
    # Option 2: Build from config
    # config = OmegaConf.load("path/to/config.yaml")
    # model = NeuroVLA(config).to(device)

    # Prepare sample data
    # Each sample should contain:
    # - "image": List of PIL Images
    # - "lang": Language instruction (string)
    # - "state": Robot state history [T, 8]
    # - "action": Ground truth actions [T, 7] (for training only)

    # Example data structure:
    # samples = [
    #     {
    #         "image": [],  # List of PIL Images
    #         "lang": "pick up the red block",
    #         "state": np.zeros((16, 8)),  # [T, 8] state history
    #         "action": np.zeros((8, 7)),  # [T, 7] action sequence
    #     }
    # ]
    import pickle
    from omegaconf import OmegaConf
    with open("/workspace/samples_states.pkl", "rb") as f:
        samples = pickle.load(f)
    device = torch.device("cuda:0")

    # Extract data for inference
    images = [sample["image"] for sample in samples]
    instructions = [sample["lang"] for sample in samples]
    states = [sample["state"] for sample in samples]

    # Run inference
    with torch.inference_mode():
        result = model.predict_action(
            batch_images=images,
            instructions=instructions,
            states=states,
        )
        normalized_actions = result["normalized_actions"]
        print(f"Predicted actions shape: {normalized_actions.shape}")

    print("Test example ready. Uncomment the code above to run inference.")

