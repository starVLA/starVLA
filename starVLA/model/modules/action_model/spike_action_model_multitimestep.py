"""
Spiking Neural Network (SNN) based action prediction models.

This module implements action prediction using spiking neural networks with:
- Leaky Integrate-and-Fire (LIF) neurons
- Temporal processing for sequential data
- FiLM-based state conditioning
"""

import math

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate


def _reset_snn_states(module: nn.Module):
    """Reset all SNN neuron states in the module."""
    for m in module.modules():
        if isinstance(m, snn.Leaky):
            if hasattr(m, "reset_state") and callable(m.reset_state):
                m.reset_state()
            else:
                for attr in ("mem", "spk", "syn", "state"):
                    if hasattr(m, attr):
                        v = getattr(m, attr)
                        if torch.is_tensor(v):
                            setattr(m, attr, v.detach())
                        else:
                            setattr(m, attr, None)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for timestep embeddings.

    Produces sine and cosine embeddings for a batch of timesteps.
    Adapted from: https://github.com/real-stanford/diffusion_policy
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x: Timesteps tensor of shape (batch_size,)

        Returns:
            Positional embeddings of shape (batch_size, dim)
        """
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)
        emb = torch.exp(exponent)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLPResNetBlock(nn.Module):
    """Single MLP ResNet block with spiking neurons."""

    def __init__(self, dim):
        super().__init__()
        beta_hidden = torch.rand(dim)
        spike_grad = surrogate.fast_sigmoid()

        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            snn.Leaky(beta=beta_hidden, init_hidden=True, spike_grad=spike_grad),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)

        Returns:
            Spike output of shape (batch_size, hidden_dim)
        """
        spike = self.ffn(x)
        return spike


class MLPResNet(nn.Module):
    """
    MLP with spiking neurons for temporal action prediction.

    Processes sequential data through Leaky Integrate-and-Fire neurons,
    using membrane potential for continuous action regression.
    """

    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Initialize SNN parameters
        beta_in = torch.rand(hidden_dim)
        thr_in = torch.rand(hidden_dim)
        spike_grad = surrogate.fast_sigmoid()
        beta_out = torch.rand(1)

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Input LIF layer with learnable parameters
        self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad, init_hidden=True)

        # Residual blocks
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output LIF layer (no reset, accumulates membrane potential)
        self.li_out = snn.Leaky(
            beta=beta_out,
            threshold=1.0,
            learn_beta=True,
            spike_grad=spike_grad,
            reset_mechanism="none",
            init_hidden=True,
        )

        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def reset_state(self):
        """Reset all SNN neuron states."""
        _reset_snn_states(self)

    def forward(self, x):
        """
        Process sequential input through spiking neurons.

        Args:
            x: Input tensor of shape [Batch, Seq, Dim]

        Returns:
            Output tensor of shape [Batch, Seq, output_dim]
        """
        # Reset SNN states at the start of each forward pass
        self.reset_state()

        # Transpose to [Seq, Batch, Dim] for temporal processing
        x = x.transpose(0, 1)

        outputs = []

        # Process each timestep
        for step in range(x.shape[0]):
            x_step = x[step]

            # Encoder
            out = self.layer_norm1(x_step)
            out = self.fc1(out)
            out = self.lif_in(out)

            # ResNet blocks
            for block in self.mlp_resnet_blocks:
                out = block(out)

            out = self.layer_norm2(out)
            out = self.fc2(out)

            # Output layer: use membrane potential for regression
            _ = self.li_out(out)
            current_mem = self.li_out.mem
            final_out = self.fc3(current_mem)

            outputs.append(final_out)

        # Stack and transpose back to [Batch, Seq, Out]
        return torch.stack(outputs).transpose(0, 1)


class L1RegressionActionHead(nn.Module):
    """SNN-based action head for continuous action prediction via L1 regression."""

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(num_blocks=2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=action_dim)

    def predict_action(self, actions_hidden_states):
        """
        Predict continuous actions from hidden states.

        Args:
            actions_hidden_states: Hidden states of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Predicted actions of shape (batch_size, seq_len, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, actions_hidden_states.shape[1], -1)
        action = self.model(rearranged_actions_hidden_states)
        return action


def get_action_model(input_dim=768, hidden_dim=768, action_dim=7):
    """
    Create an action prediction model.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        action_dim: Output action dimension

    Returns:
        L1RegressionActionHead instance
    """
    action_head = L1RegressionActionHead(input_dim=input_dim, hidden_dim=hidden_dim, action_dim=action_dim)
    return action_head


class FiLMedActionStateModulator(nn.Module):
    """
    FiLM-style modulation for action features conditioned on robot states.

    Applies Feature-wise Linear Modulation (FiLM) to action hidden states
    based on historical robot states.
    """

    def __init__(
        self,
        action_hidden_dim: int,
        robot_state_dim: int,
        projector_hidden_dim: int = 512,
    ):
        super().__init__()
        self.gamma_projector = nn.Sequential(
            nn.LayerNorm(robot_state_dim),
            nn.Linear(robot_state_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, action_hidden_dim),
        )
        self.beta_projector = nn.Sequential(
            nn.LayerNorm(robot_state_dim),
            nn.Linear(robot_state_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, action_hidden_dim),
        )

    def forward(self, actions_hidden_states: torch.Tensor, robot_states: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation to action features.

        Args:
            actions_hidden_states: Action features (batch_size, seq_len, action_hidden_dim)
            robot_states: Robot state history (batch_size, history_len, robot_state_dim)

        Returns:
            Modulated action features of same shape as input
        """
        batch_size, _, action_hidden_dim = actions_hidden_states.shape

        # Aggregate historical robot states via mean pooling
        pooled_robot_state = robot_states.mean(dim=1)

        # Generate FiLM parameters
        gamma = self.gamma_projector(pooled_robot_state)
        beta = self.beta_projector(pooled_robot_state)

        gamma = gamma.view(batch_size, 1, action_hidden_dim)
        beta = beta.view(batch_size, 1, action_hidden_dim)

        # Apply FiLM: y = x * (1 + gamma) + beta
        modulated_actions = actions_hidden_states * (1 + gamma) + beta
        return modulated_actions


def get_edit_model(input_dim=768, hidden_dim=768, robot_state_dim=8):
    """
    Create a FiLM-based action state modulator.

    Args:
        input_dim: Action hidden dimension
        hidden_dim: Projector hidden dimension
        robot_state_dim: Robot state dimension

    Returns:
        FiLMedActionStateModulator instance
    """
    edit_head = FiLMedActionStateModulator(
        action_hidden_dim=input_dim, robot_state_dim=robot_state_dim, projector_hidden_dim=hidden_dim
    )
    return edit_head


class GRU_GatedFiLModulator(nn.Module):
    """
    GRU-based gated FiLM modulator for action refinement.

    Uses a GRU encoder to capture temporal patterns in robot state history,
    then applies gated fusion with VLM features to generate FiLM parameters.
    """

    def __init__(
        self,
        action_hidden_dim: int,
        robot_state_dim: int,
        projector_hidden_dim: int = 512,
        gru_hidden_dim: int = 64,
    ):
        super().__init__()

        # GRU encoder for robot state history
        self.robot_state_encoder = nn.GRU(
            input_size=robot_state_dim,
            hidden_size=gru_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )

        # Pre-projectors for VLM and robot state features
        self.action_pre_projector = nn.Sequential(
            nn.LayerNorm(action_hidden_dim),
            nn.Linear(action_hidden_dim, projector_hidden_dim),
            nn.ReLU(),
        )

        self.robot_state_pre_projector = nn.Sequential(
            nn.LayerNorm(gru_hidden_dim),
            nn.Linear(gru_hidden_dim, projector_hidden_dim),
            nn.ReLU(),
        )

        # Gating mechanism
        self.gate_projector = nn.Sequential(nn.Linear(projector_hidden_dim, projector_hidden_dim), nn.Sigmoid())

        # FiLM parameter generators
        fused_input_dim = projector_hidden_dim + projector_hidden_dim

        self.gamma_projector = nn.Sequential(
            nn.LayerNorm(fused_input_dim),
            nn.Linear(fused_input_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, action_hidden_dim),
        )

        self.beta_projector = nn.Sequential(
            nn.LayerNorm(fused_input_dim),
            nn.Linear(fused_input_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, action_hidden_dim),
        )

    def forward(self, actions_hidden_states: torch.Tensor, robot_states: torch.Tensor) -> torch.Tensor:
        """
        Apply GRU-gated FiLM modulation.

        Args:
            actions_hidden_states: VLM features (batch_size, seq_len, action_hidden_dim)
            robot_states: Robot state history (batch_size, history_len, robot_state_dim)

        Returns:
            Modulated action features of same shape as input
        """
        batch_size, _, action_hidden_dim = actions_hidden_states.shape
        original_dtype = actions_hidden_states.dtype

        # Encode robot state history with GRU
        robot_states_device = actions_hidden_states.device
        gru_dtype = next(self.robot_state_encoder.parameters()).dtype

        robot_states_cast = robot_states.to(dtype=gru_dtype, device=robot_states_device)
        if not robot_states_cast.is_contiguous():
            robot_states_cast = robot_states_cast.contiguous()

        # Run GRU without autocast to avoid dtype mismatch
        with torch.autocast("cuda", enabled=False):
            output, _ = self.robot_state_encoder(robot_states_cast)

        output = output.to(original_dtype)
        pooled_robot_state = output[:, -1, :]  # Take last timestep

        # Aggregate VLM features
        pooled_action_state = actions_hidden_states.mean(dim=1)

        # Project and apply gating
        action_proj = self.action_pre_projector(pooled_action_state)
        robot_state_proj = self.robot_state_pre_projector(pooled_robot_state)

        gate = self.gate_projector(robot_state_proj)
        gated_action_proj = action_proj * gate

        # Fuse features
        fused_vector = torch.cat([gated_action_proj, robot_state_proj], dim=-1)

        # Generate FiLM parameters
        gamma = self.gamma_projector(fused_vector)
        beta = self.beta_projector(fused_vector)

        gamma = gamma.view(batch_size, 1, action_hidden_dim)
        beta = beta.view(batch_size, 1, action_hidden_dim)

        # Apply FiLM modulation
        modulated_actions = actions_hidden_states * (1 + gamma) + beta
        return modulated_actions


def get_gruedit_model(input_dim=768, hidden_dim=768, robot_state_dim=8):
    """
    Create a GRU-based gated FiLM modulator.

    Args:
        input_dim: Action hidden dimension
        hidden_dim: Projector hidden dimension
        robot_state_dim: Robot state dimension

    Returns:
        GRU_GatedFiLModulator instance
    """
    edit_head = GRU_GatedFiLModulator(
        action_hidden_dim=input_dim, robot_state_dim=robot_state_dim, projector_hidden_dim=hidden_dim
    )
    return edit_head


if __name__ == "__main__":
    """
    Test script for action prediction models.

    Demonstrates:
    1. Creating action model and edit model
    2. Processing sequential hidden states
    3. Applying state-conditioned modulation
    4. Predicting actions
    """
    import torch

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Test parameters
    batch_size = 2
    sequence_length = 16
    hidden_dim = 768
    robot_state_dim = 8
    action_dim = 7

    print("=" * 50)
    print("Testing Action Prediction Models")
    print("=" * 50)

    # Create test input: hidden states from VLM
    test_hidden_states = torch.randn(batch_size, sequence_length, hidden_dim).to(device)
    print(f"\nInput hidden states shape: {test_hidden_states.shape}")

    # Create test robot states: [x, y, z, roll, pitch, yaw, gripper, pad]
    test_robot_states = torch.randn(batch_size, sequence_length, robot_state_dim).to(device)
    print(f"Robot states shape: {test_robot_states.shape}")

    # Test 1: Action Model
    print("\n" + "-" * 50)
    print("Test 1: Action Prediction Model")
    print("-" * 50)
    action_model = get_action_model(input_dim=hidden_dim, hidden_dim=hidden_dim * 2, action_dim=action_dim).to(device)

    with torch.no_grad():
        predicted_actions = action_model.predict_action(test_hidden_states)
        print(f"Predicted actions shape: {predicted_actions.shape}")
        print(f"Expected shape: [{batch_size}, {sequence_length}, {action_dim}]")

    # Test 2: FiLM Edit Model
    print("\n" + "-" * 50)
    print("Test 2: FiLM Edit Model")
    print("-" * 50)
    edit_model = get_edit_model(input_dim=hidden_dim, hidden_dim=256, robot_state_dim=robot_state_dim).to(device)

    with torch.no_grad():
        modulated_features = edit_model(test_hidden_states, test_robot_states)
        print(f"Modulated features shape: {modulated_features.shape}")
        print(f"Same as input: {modulated_features.shape == test_hidden_states.shape}")

    # Test 3: GRU-Gated FiLM Edit Model
    print("\n" + "-" * 50)
    print("Test 3: GRU-Gated FiLM Edit Model")
    print("-" * 50)
    gru_edit_model = get_gruedit_model(input_dim=hidden_dim, hidden_dim=256, robot_state_dim=robot_state_dim).to(device)

    with torch.no_grad():
        gru_modulated_features = gru_edit_model(test_hidden_states, test_robot_states)
        print(f"GRU modulated features shape: {gru_modulated_features.shape}")
        print(f"Same as input: {gru_modulated_features.shape == test_hidden_states.shape}")

    # Test 4: Full Pipeline
    print("\n" + "-" * 50)
    print("Test 4: Full Pipeline (Edit + Action)")
    print("-" * 50)
    with torch.no_grad():
        # Apply state-conditioned modulation
        edited_features = gru_edit_model(test_hidden_states, test_robot_states)
        # Predict actions from edited features
        final_actions = action_model.predict_action(edited_features)
        print(f"Final predicted actions shape: {final_actions.shape}")

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)
