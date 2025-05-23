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
        self.theta = 1.0 / (self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            self.mrope_section = config.rope_scaling['mrope_section']
            self.theta_sections = self.theta.unsqueeze(0).split(self.mrope_section, dim=-1)
        else:
            self.mrope_section = None
            self.theta_sections = None

    def forward(self, position_ids):
        if self.theta_sections is not None:
            return self.mrope_forward(position_ids)
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * self.theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        if self.model_type != 'chatglm2':
            rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
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
        return self.llama_rotary_pos(x, cos, sin)

    def llama_rotary_pos(self, x, cos, sin):
        x = (x * cos) + (rotate_half(x) * sin)
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

class Mlp(torch.nn.Module):
    def __init__(self, mlp, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        ModelMapper.do_map(self, mlp, config.model_map['mlp'])
        self.is_moe = hasattr(self, 'experts')
        self.export_moe = False
        self.custom_moe = MoE(self.num_experts, self.top_k, layer_id)

    def forward(self, hidden_states: torch.Tensor):
        if not self.is_moe:
            # general Mlp
            return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

        # MoE Mlp
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

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
            # print(routing_weights, selected_experts)
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
    def __init__(self, lm_, final_layernorm_, config):
        super().__init__()
        self.final_layernorm = final_layernorm_
        self.lm = lm_
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states, logits_index: int = -1):
        hidden_states = hidden_states[:, logits_index:, :]
        hidden_states = self.final_layernorm(hidden_states)
        m_logits = self.lm(hidden_states)
        return m_logits