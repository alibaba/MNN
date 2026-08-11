import torch
import torch.nn as nn
from typing import Optional, Tuple

from .transformers import Attention
from utils.custom_op import FakeLinear
from utils.spinner import spinner_run
from .torch_utils import onnx_export

class Mtp(torch.nn.Module):
    def __init__(self, mtp, base):
        super().__init__()
        self.model_type = base.config.model_type
        self.mtp = mtp
        self.embed_ = base.embed
        self.lm_ = base.lm
        self.rotary = base.rotary

        self.config = base.config
        if not hasattr(base.config, 'head_dim'):
            self.config.head_dim = base.head_dim
        self.hidden_size = self.config.hidden_size
        self.num_attention_heads = self.config.num_attention_heads
        self.past_kv_shape = [self.config.num_hidden_layers, 2, 1, 0, self.config.num_key_value_heads, self.config.head_dim]
        self.load()
        self.unloaded_ops = {}


    @staticmethod
    def get_mtp(model_type):
        mtps = {
            'mimo': MimoMtp,
            'poi_qwen2_mtp' : PoiQwenMtp,
        }
        if model_type in mtps:
            return mtps[model_type]
        return None

    @spinner_run(f'export onnx model to ')
    def export(self, onnx_path):
        onnx_model = f'{onnx_path}/mtp.onnx'

        # unload linear weight to save export memory
        self.unload_param()

        self.seq_len = 3
        input_ids = torch.arange(3, dtype=torch.long)
        attention_mask =  (1 - torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]))) * torch.finfo(torch.float32).min
        position_ids = torch.arange(self.seq_len, dtype=torch.int).unsqueeze(0)
        hidden_states = torch.ones([self.seq_len, 1, self.hidden_size], dtype=torch.float)

        # For export onnx, don't need image or audio's embedding
        input_embed = self.embed_(input_ids)
        past_key_values = torch.zeros(self.past_kv_shape[1:])
        logits_index = torch.tensor([-1], dtype=torch.int32)
        # export to onnx
        with torch.no_grad():
            onnx_export(
                self, (input_embed, hidden_states, attention_mask, position_ids, past_key_values, logits_index),
                onnx_model,
                input_names=[
                    'input_embed', 'hidden_states',
                    'attention_mask', 'position_ids',
                    'past_key_values', 'logits_index'
                ],
                output_names=['logits', 'presents'],
                dynamic_axes={
                    "input_embed" : { 0: "seq_len" },
                    "hidden_states" : { 0: "seq_len" },
                    "attention_mask" : { 2: "seq_len", 3: "seq_len" },
                    "position_ids" : { 1: "seq_len" },
                    "past_key_values" : { 2: "history_len" }
                })
        return onnx_model

    def load(self):
        raise NotImplementedError

    def forward(self, images):
        raise NotImplementedError


class MimoMtp(Mtp):
    def __init__(self, mtp, base):
        super().__init__(mtp, base)

    def load(self):
        self.mtp.eval()
        self.token_layernorm = getattr(self.mtp[0], 'token_layernorm')
        self.hidden_layernorm = getattr(self.mtp[0], 'hidden_layernorm')
        self.input_proj = getattr(self.mtp[0], 'input_proj')
        self.input_layernorm = getattr(self.mtp[0], 'input_layernorm')
        self.self_attn = getattr(self.mtp[0], 'self_attn')
        self.post_attention_layernorm = getattr(self.mtp[0], 'post_attention_layernorm')
        self.mlp = getattr(self.mtp[0], 'mlp')
        self.final_layernorm = getattr(self.mtp[0], 'final_layernorm')
        self.self_attn = Attention(self.self_attn, 0, self.config, self.rotary, self.config.model_map)

    def unload_param(self):
        def build_faker(real, name):
            faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
            self.unloaded_ops[name] = real
            return faker
        # replace linear with fakelinear to save export memory and time
        with torch.no_grad():
            # different kv cache shape in different layers
            if isinstance(self.num_attention_heads, list):
                self.self_attn.export_fused_attn = True
            for name, child in self.self_attn.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(self.self_attn, name, build_faker(child, f'/mtp_layers.0/self_attn/{name}/Linear'))
            for name, child in self.mlp.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(self.mlp, name, build_faker(child, f'/mtp_layers.0/mlp/{name}/Linear'))
            self.input_proj = build_faker(self.input_proj, f'/mtp/input_proj/Linear')

    def forward(self,
                input_embeds: torch.Tensor,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: Optional[Tuple[torch.Tensor]] = None,
                logits_index: int = -1
                ):
        input_embeds = input_embeds.view(1, -1, self.hidden_size)
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        hidden_states = hidden_states[:, 0 : input_embeds.size(1), :]

        input_embeds = self.token_layernorm(input_embeds)
        previous_hidden_states = self.hidden_layernorm(hidden_states)
        hidden_states = self.input_proj(torch.cat([previous_hidden_states, input_embeds], dim=-1))
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        rotary_pos_emb = self.rotary(position_ids)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            past_key_value=past_key_values,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states[:, logits_index:, :]
        hidden_states = self.final_layernorm(hidden_states)

        logits = self.lm_(hidden_states)
        return logits, present_key_value

class PoiQwenMtp(Mtp):
    def __init__(self, mtp, base):
        self.num_mtp_layers = 2
        super().__init__(mtp, base)

    def load(self):
        self.mtp[0].eval()
        self.mtp[1].eval()
        self.decode_layers = nn.ModuleList([])
        self.hidden_norm = nn.ModuleList([])
        self.last_norm = nn.ModuleList([])

        with torch.no_grad():
            for i in range(self.num_mtp_layers):
                self.decode_layers.append(getattr(self.mtp[i], 'layers'))
                self.hidden_norm.append(getattr(self.mtp[i], 'RMSorm_MTP_1'))
                self.last_norm.append(getattr(self.mtp[i], 'norm'))

        self.input_layernorm = nn.ModuleList([])
        self.post_attention_layernorm = nn.ModuleList([])
        self.mlp = nn.ModuleList([])
        self.self_attn = nn.ModuleList([])

        with torch.no_grad():
            for i in range(self.num_mtp_layers):
                self.input_layernorm.append(getattr(self.decode_layers[i], 'input_layernorm'))
                self.ori_attn = getattr(self.decode_layers[i], 'self_attn')
                self.post_attention_layernorm.append(getattr(self.decode_layers[i], 'post_attention_layernorm'))
                self.mlp.append(getattr(self.decode_layers[i], 'mlp'))
                self.self_attn.append(Attention(self.ori_attn, i, self.config))

    def unload_param(self):
        def build_faker(real, name):
            faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
            self.unloaded_ops[name] = real
            return faker
        # replace linear with fakelinear to save export memory and time
        with torch.no_grad():
            for i in range(self.num_mtp_layers):
                # different kv cache shape in different layers
                if isinstance(self.num_attention_heads, list):
                    self.self_attn[i].export_fused_attn = True
                for name, child in self.self_attn[i].named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.self_attn[i], name, build_faker(child, f'/mtp_layers.{i}/self_attn/{name}/Linear'))
                for name, child in self.mlp[i].named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.mlp[i], name, build_faker(child, f'/mtp_layers.{i}/mlp/{name}/Linear'))

    def forward(self,
                input_embeds: torch.Tensor,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: Optional[Tuple[torch.Tensor]] = None,
                logits_index: int = -1
                ):
        present_key_value = []
        # [1, -1, self.hidden_size]
        mtp_hidden_states = []

        rotary_pos_emb = self.rotary(position_ids)
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        hidden_states = hidden_states[:, 0 : input_embeds.size(0), :]

        for i in range(self.num_mtp_layers):
            # first norm
            hidden_states = self.hidden_norm[i](hidden_states)

            # Decoder Layer
            residual = hidden_states
            hidden_states = self.input_layernorm[i](hidden_states)

            # Self Attention
            hidden_states, kv = self.self_attn[i](
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
            )
            present_key_value.append(kv)

            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm[i](hidden_states)
            hidden_states = self.mlp[i](hidden_states)
            hidden_states = residual + hidden_states

            # last norm
            hidden_states = self.last_norm[i](hidden_states)

            mtp_hidden_states.append(hidden_states)
            hidden_states = mtp_hidden_states[i]

        for i in range(self.num_mtp_layers):
            mtp_hidden_states[i] = mtp_hidden_states[i][:, logits_index:, :]

        mtp_logits = self.lm_(mtp_hidden_states[0])
        for i in range(self.num_mtp_layers-1):
            logits = self.lm_(mtp_hidden_states[i+1])
            mtp_logits = torch.cat([mtp_logits, logits], dim=0)
        return mtp_logits, present_key_value