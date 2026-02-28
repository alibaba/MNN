import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from .model_mapper import ModelMapper
from .custom_op import FusedAttention, MoE, FusedLinearAttention

class Embedding(torch.nn.Module):
    def __init__(self, embed, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embed = embed
        self.embed_scale = 1.0
        if config.model_type == 'gemma' or config.model_type == 'gemma2':
            self.embed_scale = self.hidden_size**0.5
        if hasattr(embed, 'embed_scale'):
            self.embed_scale = embed.embed_scale

    def forward(self, input_ids):
        inputs_embeds = self.embed(input_ids).view(-1, 1, self.hidden_size)
        return inputs_embeds

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Attention(torch.nn.Module):
    def __init__(self, attn, layer_id, config, rotary, mapper):
        super().__init__()
        self.export_fused_attn = False
        if config is None: return
        self.config = config
        self.kv_cache = True
        self.layer_id = layer_id
        self.rotary = rotary
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        if isinstance(config.num_attention_heads, list):
            self.num_heads = config.num_attention_heads[layer_id]
            self.num_key_value_heads = config.num_key_value_heads[layer_id]
        else:
            self.head_dim = config.head_dim
            self.num_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.fused_attn = FusedAttention(self.num_heads * self.head_dim, self.kv_cache, f'/layers.{layer_id}/self_attn/FusedAttention')

        ModelMapper.do_map(self, attn, mapper['attention'])

        if hasattr(self, 'qkv_proj') and self.qkv_proj is not None:
            # split qkv linear to q, k, v
            split_sizes = [self.hidden_size] * 3
            if self.qkv_proj.weight.shape[0] != self.hidden_size * 3:
                # M/GQA
                split_sizes = [
                    self.num_heads * self.head_dim,           # q_size
                    self.num_key_value_heads * self.head_dim, # k_size
                    self.num_key_value_heads * self.head_dim  # v_size
                ]

            self.q_proj = torch.nn.Linear(self.hidden_size, split_sizes[0])
            self.k_proj = torch.nn.Linear(self.hidden_size, split_sizes[1])
            self.v_proj = torch.nn.Linear(self.hidden_size, split_sizes[2])
            if config.model_type == 'chatglm':
                # chatglm-6b
                qkv_weight = self.qkv_proj.weight.data.view(self.num_heads, 3, self.head_dim, self.hidden_size)
                self.q_proj.weight.data = qkv_weight[:, 0, :, :].reshape(self.hidden_size, self.hidden_size)
                self.k_proj.weight.data = qkv_weight[:, 1, :, :].reshape(self.hidden_size, self.hidden_size)
                self.v_proj.weight.data = qkv_weight[:, 2, :, :].reshape(self.hidden_size, self.hidden_size)
                qkv_bias = self.qkv_proj.bias.data.view(self.num_heads, 3, self.head_dim)
                self.q_proj.bias.data = qkv_bias[:, 0, :].reshape(self.hidden_size)
                self.k_proj.bias.data = qkv_bias[:, 1, :].reshape(self.hidden_size)
                self.v_proj.bias.data = qkv_bias[:, 2, :].reshape(self.hidden_size)
            else:
                # other
                qw, kw, vw = torch.split(self.qkv_proj.weight, split_sizes)
                self.q_proj.weight.data = qw
                self.k_proj.weight.data = kw
                self.v_proj.weight.data = vw
                if self.qkv_proj.bias is not None:
                    qb, kb, vb = torch.split(self.qkv_proj.bias, split_sizes)
                    self.q_proj.bias.data = qb
                    self.k_proj.bias.data = kb
                    self.v_proj.bias.data = vb
                else:
                    data_type = self.q_proj.weight.dtype
                    self.q_proj.bias.data = torch.zeros(split_sizes[0], dtype=data_type)
                    self.k_proj.bias.data = torch.zeros(split_sizes[1], dtype=data_type)
                    self.v_proj.bias.data = torch.zeros(split_sizes[2], dtype=data_type)
            self.q_proj.weight.requires_grad = False
            self.k_proj.weight.requires_grad = False
            self.v_proj.weight.requires_grad = False
            self.q_proj.bias.requires_grad = False
            self.k_proj.bias.requires_grad = False
            self.v_proj.bias.requires_grad = False

        self.past_key_value = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        if self.q_proj.out_features == 2 * self.num_heads * self.head_dim:
            reshaped = query_states.view(bsz, q_len, self.num_heads, self.head_dim * 2)
            query_states, gate = torch.split(reshaped, self.head_dim, dim=-1)
            gate = gate.reshape(bsz, q_len, -1)
        else:
            gate = None

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        # openelm model has qk_norm
        if hasattr(self, 'q_norm') and self.q_norm is not None and \
           hasattr(self, 'k_norm') and self.k_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        kv_seq_len = key_states.shape[1]
        if self.past_key_value is not None:
            kv_seq_len += self.past_key_value[0].shape[1]
        # rope
        if self.rotary is not None:
            cos, sin = rotary_pos_emb[0], rotary_pos_emb[1]
            query_states = self.rotary.apply_rotary_pos(query_states, cos, sin)
            key_states = self.rotary.apply_rotary_pos(key_states, cos, sin)

        # MobileLLM model llama4_text has qk_norm after rotary
        if hasattr(self, 'qk_norm') and self.qk_norm is not None :
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        if self.export_fused_attn and torch.onnx.is_in_onnx_export():
            attn_output = self.fused_attn(query_states, key_states, value_states, attention_mask)
            if gate is not None:
                attn_output = attn_output * torch.sigmoid(gate)
            attn_output = self.o_proj(attn_output)
            return attn_output

        # kv cache
        if self.past_key_value is not None:
            past_key, past_value = self.past_key_value[0], self.past_key_value[1]
            key_states = torch.cat((past_key, key_states), dim=1)
            value_states = torch.cat((past_value, value_states), dim=1)

        self.past_key_value = torch.stack((key_states, value_states))
        query_states = query_states.transpose(1, 2)
        key_states = key_states.permute([0, 2, 3, 1])
        value_states = value_states.transpose(1, 2)
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        #------- attention ----------
        # query_states @ key_states
        attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim)
        # attention_mask
        if attention_mask.dtype in (torch.bool, torch.int32):
            # chatglm
            attn_weights.masked_fill_(attention_mask, -10000.0)
        else:
            attn_weights = attn_weights + attention_mask

        if hasattr(self, 'sinks'):
            sinks = self.sinks.reshape(1, -1, 1, 1).to(torch.float32).expand(query_states.shape[0], -1, query_states.shape[-2], -1)
            combined_logits = torch.cat([attn_weights, sinks], dim=-1)
            combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
            probs = F.softmax(combined_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = probs[..., :-1]  # we drop the sink here
        else:
            # upcast softmax to fp32
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # attn_weights @ value_states
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_gated_delta_rule(
    query,       # [B, L, H, K]   query vectors
    key,         # [B, L, H, K]   key vectors
    value,       # [B, L, H, V]   value vectors
    g,           # [B, L, H]      log-space decay (negative values)
    beta,        # [B, L, H]      learning rate for delta update
    initial_state=None,           # [B, H, K, V]  initial recurrent state
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """
    Non-chunk implementation of Gated Delta Rule (Linear Attention).
    Mathematically equivalent to torch_chunk_gated_delta_rule.

    Maintains a key-value memory (state S) of shape [K, V] per head,
    updated at each timestep using the Delta Learning Rule with gating.

    Per-step formula (for each head independently):
    ─────────────────────────────────────────────────
        S_t  = S_{t-1} * exp(g_t)              # 1. decay old memory
        v_pred = S_t^T @ k_t                   # 2. predict value for current key
        delta  = beta_t * (v_t - v_pred)       # 3. prediction error * learning rate
        S_t    = S_t + k_t @ delta^T           # 4. update memory (outer product)
        o_t    = S_t^T @ (q_t / sqrt(d_k))     # 5. query the memory
    ─────────────────────────────────────────────────

    Shapes:
        S:     [B, H, K, V]   recurrent state (key-value memory)
        k_t:   [B, H, K]      key at timestep t
        v_t:   [B, H, V]      value at timestep t
        q_t:   [B, H, K]      query at timestep t
        g_t:   [B, H]         log-decay scalar per head
        beta_t:[B, H]         learning rate scalar per head
        o_t:   [B, H, V]      output at timestep t
    """
    initial_dtype = query.dtype

    # Optional: L2 normalize Q, K before computation
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # [B, L, H, D] -> [B, H, L, D], all in float32
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    B, H, L, K = key.shape
    V = value.shape[-1]

    # Scale query: q = q / sqrt(d_k)
    query = query * (K ** -0.5)

    # Initialize recurrent state S: [B, H, K, V]
    if initial_state is None:
        S = torch.zeros(B, H, K, V, device=query.device, dtype=torch.float32)
    else:
        S = initial_state.to(torch.float32)

    outputs = []

    for t in range(L):
        q_t = query[:, :, t]       # [B, H, K]
        k_t = key[:, :, t]         # [B, H, K]
        v_t = value[:, :, t]       # [B, H, V]
        g_t = g[:, :, t]           # [B, H]
        beta_t = beta[:, :, t]     # [B, H]

        # ── Step 1: Decay ──
        # S = S * exp(g_t),  g_t < 0 so this shrinks old memory
        S = S * g_t[:, :, None, None].exp()

        # ── Step 2: Read (predict value for current key) ──
        # v_pred = S^T @ k_t : [B, H, K, V]^T @ [B, H, K, 1] -> [B, H, V]
        v_pred = (S.transpose(-1, -2) @ k_t.unsqueeze(-1)).squeeze(-1)

        # ── Step 3: Delta (prediction error * learning rate) ──
        # delta = beta_t * (v_t - v_pred) : [B, H, V]
        delta = beta_t[:, :, None] * (v_t - v_pred)

        # ── Step 4: Write (update memory with outer product) ──
        # S += k_t @ delta^T : [B, H, K, 1] @ [B, H, 1, V] -> [B, H, K, V]
        S = S + k_t.unsqueeze(-1) @ delta.unsqueeze(-2)

        # ── Step 5: Query (read output from updated memory) ──
        # o_t = S^T @ q_t : [B, H, K, V]^T @ [B, H, K, 1] -> [B, H, V]
        o_t = (S.transpose(-1, -2) @ q_t.unsqueeze(-1)).squeeze(-1)

        outputs.append(o_t)

    # Stack: list of [B, H, V] -> [B, H, L, V]
    core_attn_out = torch.stack(outputs, dim=2)

    if not output_final_state:
        S = None

    # [B, H, L, V] -> [B, L, H, V], restore original dtype
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, S


class LinearAttention(torch.nn.Module):
    def __init__(self, attn, layer_id, config, rotary, mapper):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_state_size = self.conv_kernel_size - 1
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads

        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim

        ModelMapper.do_map(self, attn, mapper['linear_attention'])

        original_norm = self.norm
        self.norm = RMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.norm.weight.data = original_norm.weight.data

        self.fused_attn = FusedLinearAttention(
            name=f'/layers.{layer_id}/self_attn/FusedLinearAttention',
            attn_type="gated_delta_rule",
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            use_qk_l2norm=True
        )
        self.conv_state = None
        self.rnn_state = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Linear Projections
        # mixed_qkv: [B, L, 2*key_dim + value_dim]
        mixed_qkv = self.in_proj_qkv(hidden_states)
        # Transpose for Conv1d: [B, Dim, L]
        mixed_qkv = mixed_qkv.transpose(1, 2)

        # Gate, Beta, Z projections
        z = self.in_proj_z(hidden_states) # [B, L, value_dim]
        b = self.in_proj_b(hidden_states) # [B, L, num_v_heads]
        a = self.in_proj_a(hidden_states) # [B, L, num_v_heads]

        # 2. Pre-compute gates
        beta = torch.sigmoid(b)
        gate = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if torch.onnx.is_in_onnx_export():
            attn_out = self.fused_attn(mixed_qkv, gate, beta, self.conv1d.weight.data.detach())
            attn_out = attn_out.reshape(-1, self.head_v_dim)
            z = z.reshape(-1, self.head_v_dim)
            attn_out = self.norm(attn_out, z)
            attn_out = attn_out.view(batch_size, seq_len, -1)
            output = self.out_proj(attn_out)
            return output

        # === Normal path: full computation for testing ===
        # 3. State Management (Conv State & Recurrent State)
        if self.conv_state is not None:
            conv_state = self.conv_state
            conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)
            mixed_qkv = F.silu(F.conv1d(conv_input, self.conv1d.weight, self.conv1d.bias, padding=0, groups=self.conv_dim))
            new_conv_state = conv_input[:, :, -self.conv_state_size:]
        else:
            new_conv_state = F.pad(mixed_qkv, (self.conv_state_size - mixed_qkv.shape[-1], 0))
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        # 4. Split Q, K, V
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1
        )
        query = query.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # 5. GQA Expansion
        if self.num_v_heads > self.num_k_heads:
            factor = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(factor, dim=2)
            key = key.repeat_interleave(factor, dim=2)

        # 6. Gated Delta Rule
        if self.rnn_state is None:
            attn_out, last_recurrent_state = torch_gated_delta_rule(
                query, key, value,
                g=gate, beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            recurrent_state = self.rnn_state
            attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query, key, value,
                g=gate, beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        # 7. Post-process
        attn_out = attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        attn_out = self.norm(attn_out, z)
        attn_out = attn_out.view(batch_size, seq_len, -1)
        output = self.out_proj(attn_out)

        # Update internal state
        self.conv_state = new_conv_state
        self.rnn_state = last_recurrent_state

        return output


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _compute_yarn_parameters(rotary_dim, base_theta, scaling_config, max_position_embeddings):
    """
    计算 YaRN (Yet another RoPE extensioN method) 的参数。
    此函数等价于 Hugging Face Transformers 中的 YaRN 实现。

    Args:
        rotary_dim (int): RoPE 的维度。
        base_theta (float): RoPE 的基础 theta 值。
        scaling_config (dict): 包含 YaRN 特定配置的字典。
        max_position_embeddings (int): 模型的最大位置编码。

    Returns:
        tuple[torch.Tensor, float]:
            - inv_freq (torch.Tensor): 计算好的、用于 RoPE 的逆频率 (即 theta)。
            - attention_scaling (float): 应用于 Query 向量的缩放因子。
    """
    def get_mscale(scale, m_scale):
        if scale <= 1:
            return 1.0
        return 0.1 * m_scale * math.log(scale) + 1.0

    def find_correction_dim(num_rotations, d, b, max_pos):
        return (d * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(b))

    def find_correction_range(low_rot, high_rot, d, b, max_pos):
        low = find_correction_dim(low_rot, d, b, max_pos)
        high = find_correction_dim(high_rot, d, b, max_pos)
        return max(0, math.floor(low)), min(d - 1, math.ceil(high))

    def linear_ramp_factor(mn, mx, d):
        if mn == mx:
            mx += 0.001
        linear_func = (torch.arange(d, dtype=torch.float32) - mn) / (mx - mn)
        return torch.clamp(linear_func, 0, 1)

    # 1. 提取 YaRN 参数
    factor = scaling_config['factor']
    beta_fast = scaling_config.get("beta_fast", 32)
    beta_slow = scaling_config.get("beta_slow", 1)
    original_max_pos = scaling_config.get("original_max_position_embeddings", max_position_embeddings)
    mscale = scaling_config.get("mscale", 1.0)

    # 2. 计算 attention_scaling (即 attention_factor)
    attention_scaling = get_mscale(factor, mscale)

    # 3. 计算 inv_freq (即 theta)
    dim = rotary_dim

    # 计算插值和外推的频率
    pos_freqs = base_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    # 找到需要修正的维度范围
    low, high = find_correction_range(beta_fast, beta_slow, dim, base_theta, original_max_pos)

    # 创建维度混合的 ramp (作用于 dim//2 的频率上)
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2)

    # 混合插值和外推频率，得到最终的 inv_freq
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )

    return inv_freq, attention_scaling

class Rotary(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if config is None: return
        self.rope_theta = config.rope_theta
        self.rope_ratio = config.rope_ratio
        if self.rope_ratio is not None:
            self.rope_theta *= self.rope_ratio
        self.rotary_dim = config.head_dim
        self.model_type = config.model_type
        if hasattr(config, 'rotary_dim'):
            self.rotary_dim = config.rotary_dim
        if self.model_type == 'chatglm':
            self.rotary_dim = config.head_dim // 2

        # Qwen3.5
        if hasattr(config, 'rope_parameters') and config.rope_parameters is not None:
            if 'rope_theta' in config.rope_parameters:
                self.rope_theta = config.rope_parameters['rope_theta']
            if 'partial_rotary_factor' in config.rope_parameters:
                self.partial_rotary_factor = config.rope_parameters['partial_rotary_factor']
                self.rotary_dim = int(self.rotary_dim * self.partial_rotary_factor)
            config.rope_scaling = config.rope_parameters

        self.mrope_section = None
        self.theta_sections = None
        self.attention_scaling = 1.0
        self.is_scaled = False
        self.mrope_interleaved = False

        def get_theta():
            return 1.0 / (self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        # default rope type's theta
        self.theta = get_theta()
        # other type
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            scaling_config = config.rope_scaling
            # get rope_type
            rope_type = 'default'
            if 'type' in config.rope_scaling:
                rope_type = config.rope_scaling['type']
            elif 'rope_type' in config.rope_scaling:
                rope_type = config.rope_scaling['rope_type']
            # gen theta for rope_type
            if rope_type == 'dynamic': # NTK
                if 'alpha' in config.rope_scaling: # NTKAlpha in Hunyuan
                    self.rope_theta *= (config.rope_scaling['alpha'] ** (self.rotary_dim / (self.rotary_dim - 2)))
                else: # NTKScaling
                    pass
                self.theta = get_theta()
            elif rope_type == 'yarn':
                self.is_scaled = True
                self.theta, self.attention_scaling = _compute_yarn_parameters(
                    rotary_dim=self.rotary_dim,
                    base_theta=self.rope_theta,
                    scaling_config=scaling_config,
                    max_position_embeddings=config.max_position_embeddings
                )
            elif rope_type == 'longrope': # longrope in MiniCPM
                self.is_scaled = True
                original_max_position_embeddings = config.rope_scaling['original_max_position_embeddings']
                scale = (config.max_position_embeddings / original_max_position_embeddings)
                self.attention_scaling = math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))
                # long_factor = config.rope_scaling['long_factor']
                short_factor = config.rope_scaling['short_factor']
                self.theta = get_theta() / torch.tensor(short_factor, dtype=torch.float32)

            # mrope for multimode
            if 'mrope_section' in scaling_config:
                self.mrope_interleaved = scaling_config.get('mrope_interleaved', False)
                self.mrope_section = scaling_config['mrope_section']
                self.theta = get_theta().unsqueeze(0)
                self.theta_sections = self.theta.split(self.mrope_section, dim=-1)
                def apply_interleaved_mrope(freqs, mrope_section):
                    # mrope apply func from qwen3-vl
                    freqs_t = freqs[0]  # just overwrite the first dimension T
                    for dim, offset in enumerate((1, 2), start=1):  # H, W
                        length = mrope_section[dim] * 3
                        idx = slice(offset, length, 3)
                        freqs_t[..., idx] = freqs[dim, ..., idx]
                    return freqs_t
                if self.mrope_interleaved:
                    half_rotary = self.rotary_dim // 2
                    freq_idx = torch.arange(0, 3 * half_rotary).reshape(3, 1, half_rotary)
                    self.mrope_reindex = apply_interleaved_mrope(freq_idx, self.mrope_section).flatten()

        self.is_mrope = self.theta_sections is not None or self.mrope_interleaved

    def forward(self, position_ids):
        if self.is_mrope:
            return self.mrope_forward(position_ids)
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * self.theta.to(position_ids.device)
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        if self.model_type == 'ernie4_5':
            rotary_pos_emb = torch.stack((rotary_pos_emb, rotary_pos_emb), dim=-1)
            rotary_pos_emb = rotary_pos_emb.reshape(*rotary_pos_emb.shape[:-2], -1)
        elif self.model_type != 'chatglm2':
            rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
        if self.is_scaled:
            rotary_pos_emb *= self.attention_scaling
        return rotary_pos_emb

    def mrope_forward(self, position_ids):
        position_ids = position_ids.float().unsqueeze(-1)
        if self.mrope_interleaved:
            idx_theta = position_ids * self.theta.to(position_ids.device)
            idx_theta = idx_theta.transpose(1, 0).reshape(-1, 3 * self.rotary_dim // 2)
            idx_theta = idx_theta[:, self.mrope_reindex]
        else:
            idx_theta = torch.concat([
                position_ids[0] * self.theta_sections[0],
                position_ids[1] * self.theta_sections[1],
                position_ids[2] * self.theta_sections[2]
            ], dim=-1)
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
        return rotary_pos_emb

    def apply_rotary_pos(self, x, cos, sin):
        if self.model_type == 'chatglm':
            return self.chatglm_rotary_pos(x, cos, sin)
        if self.model_type == 'chatglm2':
            return self.chatglm2_rotary_pos(x, cos, sin)
        if self.model_type in ['phi-msft', 'qwen3_5', 'qwen3_5_moe']:
            return self.phi_rotary_pos(x, cos, sin)
        if self.model_type == 'ernie4_5':
            return self.ernie_rotary_pos(x, cos, sin)
        return self.llama_rotary_pos(x, cos, sin)

    def llama_rotary_pos(self, x, cos, sin):
        x = (x * cos) + (rotate_half(x) * sin)
        return x

    def ernie_rotary_pos(self, x, cos, sin):
        rotate_half_x = torch.stack(
            [-x[:, :, :, 1::2], x[:, :, :, 0::2]], dim=-1
        ).reshape(x.shape)
        x = (x * cos) + (rotate_half_x * sin)
        return x

    def phi_rotary_pos(self, x, cos, sin):
        x, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        x = (x * cos) + (rotate_half(x) * sin)
        return torch.cat((x, x_pass), dim=-1)

    def chatglm2_rotary_pos(self, x, cos, sin):
        x, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        b, s, n, h = x.shape
        xshaped = x.view(b, s, n, h//2, 2)
        x = torch.concat(
            [
                xshaped[..., 0] * cos - xshaped[..., 1] * sin,
                xshaped[..., 1] * cos + xshaped[..., 0] * sin,
            ],
            -1,
        )
        return torch.cat((x, x_pass), dim=-1)

    def chatglm_rotary_pos(self, x, cos, sin):
        seq = x.shape[1]
        x1, x2 = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        cos1, sin1 = cos[:, :seq, ...], sin[:, :seq, ...]
        cos2, sin2 = cos[:, seq:, ...], sin[:, seq:, ...]
        x1 = (x1 * cos1) + (rotate_half(x1) * sin1)
        x2 = (x2 * cos2) + (rotate_half(x2) * sin2)
        return torch.cat((x1, x2), dim=-1)

class VisionRotary(Rotary):
    def __init__(self, config):
        super().__init__(config)

    # support [h_pos, w_pos]
    def forward(self, position_ids):
        # [2, patch_len, 1]
        position_ids = position_ids.float().unsqueeze(-1)
        idx_theta = position_ids * self.theta
        # [patch_len, rotary_dim]
        idx_theta = idx_theta.permute(1, 0, 2).reshape(-1, self.rotary_dim)
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
        return rotary_pos_emb

    def apply_rotary_pos(self, x, cos, sin):
        x = (x * cos) + (rotate_half(x) * sin)
        return x

class GptOssExpert(torch.nn.Module):
    def __init__(self, hidden_size, expert_dim):
        super().__init__()
        self.expert_dim = expert_dim
        self.gate_up_proj_linear = torch.nn.Linear(hidden_size, 2 * expert_dim)
        self.down_proj_linear = torch.nn.Linear(expert_dim, hidden_size)
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor, debug=False) -> torch.Tensor:
        gate_up = self.gate_up_proj_linear(hidden_states)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        # gate = gate.clamp(min=None, max=self.limit)
        limit_tensor = torch.tensor(self.limit, device=gate.device, dtype=gate.dtype)
        gate = torch.min(gate, limit_tensor)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        out = self.down_proj_linear(gated_output)
        return out

class Qwen3Expert(torch.nn.Module):
    def __init__(self, hidden_size, expert_dim, act_fn):
        super().__init__()
        self.expert_dim = expert_dim
        self.gate_up_proj_linear = torch.nn.Linear(hidden_size, 2 * expert_dim, bias=False)
        self.down_proj_linear = torch.nn.Linear(expert_dim, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, hidden_states: torch.Tensor, debug=False) -> torch.Tensor:
        gate_up = self.gate_up_proj_linear(hidden_states)
        # gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate, up = gate_up.chunk(2, dim=-1)
        out = self.down_proj_linear(up * self.act_fn(gate))
        return out

class Mlp(torch.nn.Module):
    def __init__(self, mlp, mapper, layer_id):
        super().__init__()
        self.layer_id = layer_id
        ModelMapper.do_map(self, mlp, mapper['mlp'])
        self.is_moe = hasattr(self, 'experts')
        self.export_moe = False
        self.custom_moe = MoE(self.num_experts, self.top_k, layer_id)
        if isinstance(self.experts, torch.nn.ModuleList):
            self.moe_type = 'qwen3_moe'
        else:
            self.moe_type = 'qwen3_5_moe'
            self.norm_topk_prob = True
            # refacte experts to qwen3_experts
            original_experts = self.experts
            hidden_size = original_experts.hidden_dim
            expert_dim = original_experts.intermediate_dim
            act_fn = original_experts.act_fn
            new_experts_list = torch.nn.ModuleList()
            for i in range(self.num_experts):
                expert_mlp = Qwen3Expert(hidden_size, expert_dim, act_fn)
                expert_mlp.gate_up_proj_linear.weight.data = original_experts.gate_up_proj.data[i]
                expert_mlp.down_proj_linear.weight.data = original_experts.down_proj.data[i]
                new_experts_list.append(expert_mlp)
            self.experts = new_experts_list

            if not isinstance(self.gate, torch.nn.Linear):
                gate = torch.nn.Linear(hidden_size, self.num_experts, bias=False)
                gate.weight.data = self.gate.weight.data
                self.gate = gate

        if hasattr(self, 'router'):
            self.moe_type = 'gpt_oss'
            hidden_dim = self.router.weight.shape[1]
            self.gate = torch.nn.Linear(hidden_dim, self.num_experts, bias=True)
            self.gate.weight.data = self.router.weight.data
            self.gate.bias.data = self.router.bias.data
            # refacte experts to qwen3_experts
            original_experts = self.experts
            expert_dim = original_experts.expert_dim
            new_experts_list = torch.nn.ModuleList()
            for i in range(self.num_experts):
                expert_mlp = GptOssExpert(hidden_dim, expert_dim)
                expert_mlp.gate_up_proj_linear.weight.data = original_experts.gate_up_proj.data[i].transpose(0, 1)
                expert_mlp.gate_up_proj_linear.bias.data = original_experts.gate_up_proj_bias.data[i]
                expert_mlp.down_proj_linear.weight.data = original_experts.down_proj.data[i].transpose(0, 1)
                expert_mlp.down_proj_linear.bias.data = original_experts.down_proj_bias.data[i]
                new_experts_list.append(expert_mlp)
            self.experts = new_experts_list
            del self.router

    def forward(self, hidden_states: torch.Tensor):
        if not self.is_moe:
            # general Mlp
            return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

        # MoE Mlp
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if hasattr(self, 'shared_expert'):
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * self.shared_expert(hidden_states)
            shared_expert_output = shared_expert_output.reshape(batch_size, sequence_length, hidden_dim)
        else:
            shared_expert_output = None

        if self.moe_type == 'gpt_oss':
            router_logits = self.gate(hidden_states)
            routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
            routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float).to(hidden_states.dtype)
        else:
            router_logits = self.gate(hidden_states)
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

        if self.export_moe:
            expert_output = self.custom_moe(hidden_states, routing_weights, selected_experts)
            if shared_expert_output is not None:
                expert_output = expert_output + shared_expert_output
            return expert_output

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        if False: # cpp impl
            seqlen, topk = selected_experts.shape
            if seqlen == 1:
                expert_idx = int(selected_experts[0, 0])
                scale = float(routing_weights[0, 0])
                output = self.experts[expert_idx](hidden_states) * scale
                for i in range(1, topk):
                    expert_idx = int(selected_experts[0, i])
                    scale = float(routing_weights[0, i])
                    output += self.experts[expert_idx](hidden_states) * scale
                return output

            hss = torch.split(hidden_states, 1)
            expertWorks = [[] for i in range(self.num_experts)]

            for i in range(seqlen):
                for j in range(topk):
                    expert_idx = int(selected_experts[i, j])
                    scale = float(routing_weights[i, j])
                    expertWorks[expert_idx].append((i, scale))

            for i in range(self.num_experts):
                if len(expertWorks[i]) == 0:
                    continue
                input_hs = []
                for token_id, scale in expertWorks[i]:
                    input_hs.append(hss[token_id])
                output_hs = self.experts[i](torch.concat(input_hs))
                output_hss = torch.split(output_hs, 1)
                for j in range(len(expertWorks[i])):
                    token_id, scale = expertWorks[i][j]
                    scale_hs = output_hss[j] * scale
                    final_hidden_states[token_id] += scale_hs.squeeze(0)
            return final_hidden_states

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        # Loop over all available experts in the model and perform the computation on each expert
        # for expert_idx in range(self.num_experts):
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            idx, top_x = torch.where(expert_mask[expert_idx])
            expert_layer = self.experts[expert_idx]

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        if shared_expert_output is not None:
            final_hidden_states = final_hidden_states + shared_expert_output

        return final_hidden_states

class Decoder(torch.nn.Module):
    def __init__(self, decoder, layer_id, config, rotary=None, mapper=None):
        super().__init__()
        if rotary is None:
            rotary = config.rotary
        if mapper is None:
            mapper = config.model_map
        ModelMapper.do_map(self, decoder, mapper['decoder'])
        if 'mlp' in mapper:
            self.mlp = Mlp(self.mlp, mapper, layer_id)

        self.layer_type = 'full_attention'
        if hasattr(self, 'self_attn') and self.self_attn is not None:
            self.self_attn = Attention(self.self_attn, layer_id, config, rotary, mapper)
        if hasattr(self, 'linear_attn') and self.linear_attn is not None:
            self.self_attn = LinearAttention(self.linear_attn, layer_id, config, rotary, mapper)
            self.layer_type = 'linear_attention'

        self.hidden_size = config.hidden_size
        if hasattr(config, 'num_hidden_layers'):
            # minicpm
            self.num_hidden_layers = config.num_hidden_layers
            # chatglm
            self.alpha = (2 * config.num_hidden_layers) ** 0.5 if config.model_type == 'chatglm' else 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        norm_hidden_states = hidden_states

        # Self Attention or Linear Attention
        if self.layer_type == 'full_attention':
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
            )
        elif self.layer_type == 'linear_attention':
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        # Fully Connected
        if not hasattr(self, 'post_attention_layernorm'):
            # phi
            feed_forward_hidden_states = self.mlp(norm_hidden_states)
            hidden_states = hidden_states + feed_forward_hidden_states + residual
        elif hasattr(self, 'alpha') and self.alpha != 1.0:
            # chatglm-6b
            hidden_states = norm_hidden_states * self.alpha + hidden_states
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_input * self.alpha + mlp_output
        elif hasattr(self, 'pre_feedforward_layernorm'):
            # gemma2
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states
        elif hasattr(self, 'scale_depth'):
            # minicpm
            hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        else:
            # general
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states

class Lm(torch.nn.Module):
    def __init__(self, lm_):
        super().__init__()
        self.lm = lm_

    def forward(self, hidden_states):
        m_logits = self.lm(hidden_states)
        return m_logits
