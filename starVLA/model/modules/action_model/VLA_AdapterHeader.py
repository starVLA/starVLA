"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        self.config = full_config

        input_dim = full_config.framework.qwenvl.vl_hidden_dim
        hidden_dim = full_config.framework.action_model.hidden_dim
        action_dim = full_config.framework.action_model.action_dim
        
        self.action_query_num = full_config.framework.action_model.get("action_query_num", 64)
        use_pro_version = full_config.framework.action_model.use_pro_version

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.num_actions_chunk = self.config.framework.action_model.num_actions_chunk
        self.model = MLPResNet(
            num_blocks=24, 
            input_dim=input_dim*action_dim, 
            hidden_dim=hidden_dim, 
            output_dim=action_dim,
            use_pro_version=use_pro_version
            )

    def predict_action(
            self, 
            actions_hidden_states, 
            proprio=None, 
            proprio_projector=None,
            phase="Inference"
            ):
        """
        Args:
            actions_hidden_states: [B, Layers, Total_Len, D]
            
            根据 Qwen_Adapter 的逻辑，Total_Len = (Vision_Len + Action_Query_Num)。
            Language Tokens 已经在 Adapter 阶段被过滤掉了，所以这里不需要额外处理 Language。
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        # 1. Proprioception Processing
        if proprio is not None and proprio_projector is not None:
            proprio = proprio.reshape(batch_size, -1).to(dtype=actions_hidden_states.dtype)
            proprio_features = proprio_projector(proprio)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
        else:
            proprio_features = None
        
        # Action Query Tokens (h_a)
        action_query_states = actions_hidden_states[:, :, -self.action_query_num:, :]
        
        task_hidden_states = actions_hidden_states[:, :, :-self.action_query_num, :]

        # 3. Action Chunk Queries Init
        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * self.num_actions_chunk, self.hidden_dim),
            device=device, dtype=actions_hidden_states.dtype
        ).detach()  

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, self.num_actions_chunk, -1
        ) 

        # Denoising Noise
        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations)

        # 4. MLP Forward
        action = self.model(
            rearranged_actions_hidden_states,
            h_a=action_query_states,  # [B, Layers, query_num, D]
            p=proprio_features,       # [B, 1, D]
            h_t=task_hidden_states    # [B, Layers, vis_len, D]
            )

        return action
    

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
            use_pro_version=False
            ):
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
                
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, h_a=None, h_t=None, p=None):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)
        x = self.fc1(x) 
        x = self.relu(x)
        
        for i, block in enumerate(self.mlp_resnet_blocks):
            idx = i + 1
            
            cur_h_t = None
            if h_t is not None and h_t.shape[1] > idx:
                cur_h_t = h_t[:, idx, :] 

            cur_h_a = None
            if h_a is not None and h_a.shape[1] > idx:
                cur_h_a = h_a[:, idx, :]
            
            x = block(x, h_t=cur_h_t, h_a=cur_h_a, p=p)
            
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x   


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class MLPResNetBlock(nn.Module):
    """
    Standard MLP ResNet Block.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.num_heads = 8
        self.head_dim = dim // self.num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))

    def forward(self, x, h_t=None, h_a=None, p=None):
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        conditions = []
        if h_a is not None:
            if h_a.dim() == 2: h_a = h_a.unsqueeze(1)
            conditions.append(h_a)
        if p is not None:
            if p.dim() == 2: p = p.unsqueeze(1)
            conditions.append(p)
            
        h_cond = torch.cat(conditions, dim=1) if len(conditions) > 0 else None

        if h_t is not None:
            if h_t.dim() == 2: h_t = h_t.unsqueeze(1)
        
        B, T, C = x.shape
        K_cond = h_cond.size(1) if h_cond is not None else 0
        K_task = h_t.size(1) if h_t is not None else 0

        # Self Attention Projection
        q_1 = self.q_proj(x)
        k_tokens = self.k_proj(x)
        v_tokens = self.v_proj(x)

        # Reshape Self
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_list = []
        # Score: Self
        attn_scores_list.append(torch.matmul(q_1, k_tokens.transpose(-2, -1)))

        # Process Task (Vision)
        v_task_reshaped = None
        if h_t is not None:
            k_task = self.k_proj(h_t)
            v_task = self.v_proj(h_t)
            k_task = k_task.view(B, K_task, self.num_heads, self.head_dim).transpose(1, 2)
            v_task_reshaped = v_task.view(B, K_task, self.num_heads, self.head_dim).transpose(1, 2)
            
            attn_scores_list.append(torch.matmul(q_1, k_task.transpose(-2, -1)))

        # Process Adapter (Action/Proprio)
        v_cond_reshaped = None
        if h_cond is not None:
            k_cond = self.k_proj(h_cond)
            v_cond = self.v_proj(h_cond)
            k_cond = k_cond.view(B, K_cond, self.num_heads, self.head_dim).transpose(1, 2)
            v_cond_reshaped = v_cond.view(B, K_cond, self.num_heads, self.head_dim).transpose(1, 2)

            attn_scores_list.append(torch.matmul(q_1, k_cond.transpose(-2, -1)) * ratio_g)

        # Softmax
        attn_scores = torch.cat(attn_scores_list, dim=-1)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Combine Values
        v_combined_list = [v_tokens]
        if v_task_reshaped is not None: v_combined_list.append(v_task_reshaped)
        if v_cond_reshaped is not None: v_combined_list.append(v_cond_reshaped)
        
        v_combined = torch.cat(v_combined_list, dim=2)

        # Output Projection
        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x)
        return x


class MLPResNetBlock_Pro(nn.Module):
    """
    MLP ResNet Block Pro with RoPE and dimension checks.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            )

        self.q_proj = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x, h_a=None, h_t=None, p=None):
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        # 1. Prepare Conditions
        cond_list = []
        if h_a is not None:
            if h_a.dim() == 2: h_a = h_a.unsqueeze(1)
            cond_list.append(h_a)
        if p is not None:
            if p.dim() == 2: p = p.unsqueeze(1)
            cond_list.append(p)
        h_adapter = torch.cat(cond_list, dim=1) if cond_list else None

        if h_t is not None:
            if h_t.dim() == 2: h_t = h_t.unsqueeze(1)

        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_adapter is not None else 0
        K_t = h_t.size(1) if h_t is not None else 0

        def to_heads(t, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Self Attention
        q_1 = self.q_proj(x)
        k_self = self.k_self(x)
        v_self = self.v_self(x)
        
        q_1 = to_heads(q_1, T)
        k_self = to_heads(k_self, T)
        v_self = to_heads(v_self, T)

        # RoPE: Self
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_self = apply_rope(q_1, k_self, cos_main, sin_main)

        attn_scores = [torch.matmul(q_1, k_self.transpose(-2, -1))]
        v_list = [v_self]

        # Adapter Attention (Action/Proprio) - With RoPE
        if h_adapter is not None:
            k_adp = self.k_adapter(h_adapter)
            v_adp = self.v_adapter(h_adapter)
            k_adp, v_adp = to_heads(k_adp, K_a), to_heads(v_adp, K_a)
            
            cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
            _, k_adp = apply_rope(k_adp, k_adp, cos_a, sin_a)
            
            attn_scores.append(torch.matmul(q_1, k_adp.transpose(-2, -1)))
            v_list.append(v_adp)

        # Task Attention (Vision) - With RoPE & Gating
        if h_t is not None:
            k_tsk = self.k_task(h_t)
            v_tsk = self.v_task(h_t)
            k_tsk, v_tsk = to_heads(k_tsk, K_t), to_heads(v_tsk, K_t)

            cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
            _, k_tsk = apply_rope(k_tsk, k_tsk, cos_t, sin_t)

            attn_scores.append(torch.matmul(q_1, k_tsk.transpose(-2, -1)) * ratio_g)
            v_list.append(v_tsk)

        # Merge & Output
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        v_combined = torch.cat(v_list, dim=2)
        output = torch.matmul(attn_weights, v_combined)
        
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x)
        return x


def get_action_model(config=None):
    return L1RegressionActionHead(
        full_config=config
    )