import math
import torch
from typing import Optional, Tuple

from .model_mapper import ModelMapper
from .custom_op import FusedAttention, MoE

class Embedding(torch.nn.Module):
    def __init__(self, embed, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embed = embed
        self.embed_scale = 1.0
        if config.model_type == 'gemma2':
            self.embed_scale = self.hidden_size**0.5
        if hasattr(embed, 'embed_scale'):
            self.embed_scale = embed.embed_scale

    def forward(self, input_ids):
        inputs_embeds = self.embed(input_ids).view(-1, 1, self.hidden_size)
        return inputs_embeds

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Attention(torch.nn.Module):
    def __init__(self, attn, layer_id, config):
        super().__init__()
        self.export_fused_attn = False
        if config is None: return
        self.config = config
        self.fused_attn = FusedAttention(config.hidden_size, f'/layers.{layer_id}/self_attn/FusedAttention')
        self.layer_id = layer_id
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
        self.rotary = config.rotary

        ModelMapper.do_map(self, attn, config.model_map['attention'])

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


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        if cross_attention_states is not None:
            hidden_states = cross_attention_states
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        # openelm model has qk_norm
        if hasattr(self, 'q_norm') and self.q_norm is not None and \
           hasattr(self, 'k_norm') and self.k_norm is not None :
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        kv_seq_len = key_states.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[1]
        # rope
        if self.rotary is not None:
            cos, sin = rotary_pos_emb[0], rotary_pos_emb[1]
            query_states = self.rotary.apply_rotary_pos(query_states, cos, sin)
            key_states = self.rotary.apply_rotary_pos(key_states, cos, sin)

        if self.export_fused_attn:
            attn_output = self.fused_attn(query_states, key_states, value_states, attention_mask)
            attn_output = self.o_proj(attn_output)
            return attn_output, past_key_value

        # kv cache
        if past_key_value is not None:
            past_key, past_value = past_key_value[0], past_key_value[1]
            key_states = torch.cat((past_key, key_states), dim=1)
            value_states = torch.cat((past_value, value_states), dim=1)

        past_key_value = torch.stack((key_states, value_states))
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
            probs = torch.nn.functional.softmax(combined_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = probs[..., :-1]  # we drop the sink here
        else:
            # upcast softmax to fp32
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights @ value_states
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_value

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
        self.rope_theta *= self.rope_ratio
        self.rotary_dim = config.head_dim
        self.model_type = config.model_type
        if hasattr(config, 'rotary_dim'):
            self.rotary_dim = config.rotary_dim
        if self.model_type == 'chatglm':
            self.rotary_dim = config.head_dim // 2
        self.mrope_section = None
        self.theta_sections = None
        self.attention_scaling = 1.0
        self.is_scaled = False

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
                if 'alpha' in config.rope_scaling: # NTKAlpha
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
            # mrope for multimode
            if 'mrope_section' in scaling_config:
                self.mrope_section = scaling_config['mrope_section']
                self.theta_sections = get_theta().unsqueeze(0).split(self.mrope_section, dim=-1)

    def forward(self, position_ids):
        if self.theta_sections is not None:
            return self.mrope_forward(position_ids)
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * self.theta
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
        if self.model_type == 'phi-msft':
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

class Mlp(torch.nn.Module):
    def __init__(self, mlp, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        ModelMapper.do_map(self, mlp, config.model_map['mlp'])
        self.is_moe = hasattr(self, 'experts')
        self.export_moe = False
        self.custom_moe = MoE(self.num_experts, self.top_k, layer_id)
        self.moe_type = 'qwen3_moe'
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
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        if self.moe_type == 'gpt_oss':
            routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
            routing_weights = torch.nn.functional.softmax(routing_weights, dim=-1, dtype=torch.float).to(hidden_states.dtype)
        else:
            routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

        if self.export_moe:
            return self.custom_moe(hidden_states, routing_weights, selected_experts)

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
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states

class Decoder(torch.nn.Module):
    def __init__(self, decoder, layer_id, config):
        super().__init__()
        self.cross_decoder = False
        ModelMapper.do_map(self, decoder, config.model_map['decoder'])
        if 'mlp' in config.model_map:
            self.mlp = Mlp(self.mlp, config, layer_id)
        # mllama has cross_attn
        if hasattr(self, 'cross_attn') and self.cross_attn is not None:
            self.cross_decoder = True
            self.self_attn = Attention(self.cross_attn, layer_id, config)
        else:
            self.self_attn = Attention(self.self_attn, layer_id, config)
        self.hidden_size = config.hidden_size
        # chatglm
        self.alpha = (2 * config.num_hidden_layers) ** 0.5 if config.model_type == 'chatglm' else 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        norm_hidden_states = hidden_states
        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cross_attention_states=cross_attention_states,
        )
        # Fully Connected
        if not hasattr(self, 'post_attention_layernorm'):
            # phi
            feed_forward_hidden_states = self.mlp(norm_hidden_states)
            hidden_states = hidden_states + feed_forward_hidden_states + residual
        elif self.alpha != 1.0:
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
        elif cross_attention_mask is not None and self.cross_decoder:
            hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = cross_attention_mask * hidden_states
            hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states
        else:
            # general
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states, present_key_value

class Lm(torch.nn.Module):
    def __init__(self, lm_):
        super().__init__()
        self.lm = lm_

    def forward(self, hidden_states):
        m_logits = self.lm(hidden_states)
        return m_logits