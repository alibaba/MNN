import math
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional, List, Tuple
from .transformers import Attention
from .transformers import RMSNorm
from .transformers import Rotary
from .transformers import Embedding
from utils.custom_op import FakeLinear
from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN
from utils.spinner import spinner_run
from .torch_utils import onnx_export


class Eagle(torch.nn.Module):
    def __init__(self, eagle_path, base):
        super().__init__()
        # load eagle config.json
        config_file_path = eagle_path + "/config.json"
        self.eagle_config = PretrainedConfig.from_json_file(config_file_path)

        self.model_type = base.config.model_type
        self.eagle_path = eagle_path

        self.config = base.config
        if hasattr(self.eagle_config, "head_dim"):
            self.config.head_dim = self.eagle_config.head_dim

        self.rope_theta = 10000
        self.rope_ratio = 1.0
        self.head_dim = self.config.head_dim
        self.hidden_size = self.config.hidden_size
        if self.eagle_config.hidden_size != self.hidden_size:
            raise RuntimeError(f'eagle_config hidden_size not equal: {self.eagle_config.hidden_size}, {self.hidden_size}!')
        # self.past_kv_shape = base.past_kv_shape
        self.num_attention_heads = self.config.num_attention_heads
        self.past_kv_shape = [self.config.num_hidden_layers, 2, 1, 0, self.config.num_key_value_heads, self.config.head_dim]

        self.head_dim = self.config.head_dim
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        # self.config.head_dim = self.head_dim
        self.config.rotary = Rotary(self)
        # eagle config params
        self.padding_idx = self.eagle_config.pad_token_id
        self.vocab_size = self.eagle_config.vocab_size
        self.draft_vocab_size = self.eagle_config.draft_vocab_size
        # embed_tokens api
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)
        if not hasattr(self.eagle_config, "target_hidden_size"):
            self.embed_tokens.weight = base.embed.embed.weight

        # fc api
        if hasattr(self.eagle_config, "target_hidden_size"):
            self.fc = nn.Linear(self.eagle_config.target_hidden_size * 3, self.hidden_size, bias=False)
        else:
            self.fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)

        self.midlayer = nn.Module()
        # midlayer.hidden_norm
        self.midlayer.hidden_norm = RMSNorm(self.hidden_size, eps=self.eagle_config.rms_norm_eps)
        # midlayer.input_layernorm
        self.midlayer.input_layernorm = RMSNorm(self.hidden_size, eps=self.eagle_config.rms_norm_eps)
        # midlayer.self_attn
        self.midlayer.self_attn = Attention(None, 0, self.config, base.rotary, self.config.model_map)
        self.midlayer.self_attn.q_proj = nn.Linear(self.hidden_size * 2, self.num_attention_heads * self.head_dim, bias=False)
        self.midlayer.self_attn.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.midlayer.self_attn.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.midlayer.self_attn.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)
        # midlayer.post_attention_layernorm
        self.midlayer.post_attention_layernorm = RMSNorm(self.hidden_size, eps=self.eagle_config.rms_norm_eps)
        # midlayer.mlp
        self.midlayer.mlp = nn.Module()
        self.intermediate_size = self.eagle_config.intermediate_size
        self.midlayer.mlp.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.midlayer.mlp.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.midlayer.mlp.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.midlayer.mlp.act_fn = ACT2FN[self.eagle_config.hidden_act]

        # norm api
        self.norm = RMSNorm(self.hidden_size, eps=self.eagle_config.rms_norm_eps)
        # lm_head api
        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size,bias=False)
        # logsoftmax api
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        # d2t
        d2t = torch.zeros((self.draft_vocab_size), dtype=torch.int64)
        self.register_buffer("d2t", d2t)

        self.load()
        self.unloaded_ops = {}

    def unload_param(self):
        def build_faker(real, name):
            faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
            self.unloaded_ops[name] = real
            return faker
        # replace linear with fakelinear to save export memory and time
        with torch.no_grad():
            # different kv cache shape in different layers
            if isinstance(self.num_attention_heads, list):
                self.midlayer.self_attn.export_fused_attn = True
            for name, child in self.midlayer.self_attn.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(self.midlayer.self_attn, name, build_faker(child, f'/eagle_layers.0/self_attn/{name}/Linear'))
            for name, child in self.midlayer.mlp.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(self.midlayer.mlp, name, build_faker(child, f'/eagle_layers.0/mlp/{name}/Linear'))
            self.lm_head = build_faker(self.lm_head, f'/eagle/lm_head/Linear')
            self.fc = build_faker(self.fc, f'/eagle/fc/Linear')

    @staticmethod
    def get_eagle(model_type):
        eagles = {
            'llama': LlamaEagle,
            'qwen3': LlamaEagle,
        }
        if model_type in eagles:
            return eagles[model_type]
        return LlamaEagle

    @spinner_run(f'export onnx model to ')
    def export(self, onnx_path):
        # save d2t to file
        import MNN.expr as expr
        torch_d2t = self.d2t.detach().to(torch.int32).contiguous().cpu()
        mnn_d2t = expr.const(torch_d2t.data_ptr(), torch_d2t.shape, expr.data_format.NHWC, expr.dtype.int)
        mnn_d2t.name = 'd2t'
        expr.save([mnn_d2t], f'{onnx_path}/../eagle_d2t.mnn')

        eagle_model = f'{onnx_path}/eagle.onnx'
        eagle_fc_model = f'{onnx_path}/eagle_fc.onnx'
        # unload linear weight to save export memory
        # self.unload_param()

        self.seq_len = 3
        input_ids = torch.arange(3, dtype=torch.long)
        attention_mask =  (1 - torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]))) * torch.finfo(torch.float32).min
        position_ids = torch.arange(self.seq_len, dtype=torch.int).unsqueeze(0)
        hidden_states = torch.ones([1, self.seq_len, self.hidden_size], dtype=torch.float)
        logits_index = torch.tensor([-1], dtype=torch.int32)

        fc_hidden = torch.ones([1, self.seq_len, self.hidden_size * 3], dtype=torch.float)

        # For export onnx, don't need image or audio's embedding
        input_embed = self.embed_tokens(input_ids)
        past_key_values = torch.zeros(self.past_kv_shape[1:-1] + [self.head_dim])
        # export to onnx
        with torch.no_grad():
            onnx_export(self.fc, (fc_hidden),
                eagle_fc_model,
                input_names=['fc_hidden'],
                output_names=['hidden_states'],
                dynamic_axes={ "fc_hidden" : { 1: "seq_len" } })
            onnx_export(
                self, (input_embed, hidden_states, attention_mask, position_ids, past_key_values, logits_index),
                eagle_model,
                input_names=[
                    'input_embed', 'hidden_states',
                    'attention_mask', 'position_ids',
                    'past_key_values', 'logits_index'
                ],
                output_names=['logits', 'out_hidden_states', 'presents'],
                dynamic_axes={
                    "input_embed" : { 0: "seq_len" },
                    "hidden_states" : { 1: "seq_len" },
                    "attention_mask" : { 2: "seq_len", 3: "seq_len" },
                    "position_ids" : { 1: "seq_len" },
                    "past_key_values" : { 2: "history_len" }
                    })
        return eagle_model, eagle_fc_model

    def load(self):
        raise NotImplementedError

    def forward(self, images):
        raise NotImplementedError

class LlamaEagle(Eagle):
    def __init__(self, eagle_path, base):
        super().__init__(eagle_path, base)

    def load(self):
        safetensors_path = os.path.join(self.eagle_path, "model.safetensors")
        bin_path = os.path.join(self.eagle_path, "pytorch_model.bin")
        ea_layer_state_dict = None
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            ea_layer_state_dict = load_file(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):
            ea_layer_state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"Eagle path '{self.eagle_path}' not found 'model.safetensors' or 'pytorch_model.bin'."
            )
        self.load_state_dict(ea_layer_state_dict, strict=False)

    def forward(self,
                input_embeds: torch.Tensor,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: Optional[Tuple[torch.Tensor]] = None,
                logits_index: int = -1
                ):
        # hidden_states = self.fc(hidden_states)
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        input_embeds = input_embeds.view(1, -1, self.hidden_size)

        residual = hidden_states

        input_embeds = self.midlayer.input_layernorm(input_embeds)
        previous_hidden_states = self.midlayer.hidden_norm(hidden_states)
        hidden_states = torch.cat([input_embeds, previous_hidden_states], dim=-1)

        rotary_pos_emb = self.config.rotary(position_ids)

        # Self Attention
        hidden_states, present_key_value = self.midlayer.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            past_key_value=past_key_values,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.midlayer.post_attention_layernorm(hidden_states)
        hidden_states = self.midlayer.mlp.down_proj(self.midlayer.mlp.act_fn(self.midlayer.mlp.gate_proj(hidden_states)) * self.midlayer.mlp.up_proj(hidden_states))
        hidden_states = residual + hidden_states

        hidden_states = hidden_states[:, logits_index:, :]
        last_hidden = self.norm(hidden_states)

        logits = self.lm_head(last_hidden)
        logits = self.logsoftmax(logits)
        return logits, hidden_states, present_key_value
