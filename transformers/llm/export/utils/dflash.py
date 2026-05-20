import os
import json
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .transformers import Attention, RMSNorm, Rotary, Embedding
from utils.custom_op import FakeLinear
from utils.spinner import spinner_run
from .torch_utils import onnx_export
from transformers.activations import ACT2FN


class DFlashAttention(torch.nn.Module):
    """DFlash non-causal attention: Q from noise, K/V from cat(context, noise)"""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, hidden_states, context_hidden, q_cos, q_sin, k_cos, k_sin, attention_mask):
        """
        hidden_states: [1, block_size, hidden_size] (noise)
        context_hidden: [1, context_len, hidden_size]
        q_cos/q_sin: [1, 1, block_size, head_dim] - RoPE for Q
        k_cos/k_sin: [1, 1, context_len + block_size, head_dim] - RoPE for K
        attention_mask: [1, 1, block_size, context_len + block_size]
        """
        bsz = 1
        q_len = hidden_states.shape[1]
        ctx_len = context_hidden.shape[1]
        total_len = ctx_len + q_len

        # Q from noise only
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_attention_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [1, num_heads, q_len, head_dim]

        # K/V from cat(context, noise)
        kv_input = torch.cat([context_hidden, hidden_states], dim=1)  # [1, total_len, hidden_size]
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)
        k = k.view(bsz, total_len, self.num_key_value_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)  # [1, num_kv_heads, total_len, head_dim]
        v = v.view(bsz, total_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (pre-computed, no dynamic slicing needed)
        q = self._apply_rope(q, q_cos, q_sin)
        k = self._apply_rope(k, k_cos, k_sin)

        # GQA repeat
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)

    @staticmethod
    def _apply_rope(x, cos, sin):
        """Apply rotary position embedding."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


class DFlashDecoderLayer(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.mlp.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, context_hidden, q_cos, q_sin, k_cos, k_sin, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, context_hidden, q_cos, q_sin, k_cos, k_sin, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp.down_proj(
            self.mlp.act_fn(self.mlp.gate_proj(hidden_states)) * self.mlp.up_proj(hidden_states)
        )
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashFc(torch.nn.Module):
    """Feature compression module: fc + hidden_norm"""
    def __init__(self, fc, hidden_norm):
        super().__init__()
        self.fc = fc
        self.hidden_norm = hidden_norm

    def forward(self, target_hidden):
        return self.hidden_norm(self.fc(target_hidden))


class DFlash(torch.nn.Module):
    """DFlash Draft Model for export."""
    def __init__(self, dflash_path, base):
        super().__init__()
        from transformers.configuration_utils import PretrainedConfig

        # Load DFlash config
        config_path = os.path.join(dflash_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        self.dflash_config = config_dict
        self.model_type = base.config.model_type

        # Base model config
        self.hidden_size = base.config.hidden_size
        self.head_dim = base.config.head_dim
        self.num_attention_heads = base.config.num_attention_heads
        self.num_key_value_heads = base.config.num_key_value_heads
        self.rms_norm_eps = getattr(base.config, 'rms_norm_eps', 1e-6)

        # DFlash-specific config
        dflash_cfg = config_dict.get('dflash_config', {})
        self.block_size = config_dict.get('block_size', 16)
        self.mask_token_id = dflash_cfg.get('mask_token_id', 0)

        num_hidden_layers = config_dict.get('num_hidden_layers', 1)
        num_target_layers = config_dict.get('num_target_layers', 3)
        # Use origin_config (the original HF config) for attributes not in LlmConfig
        origin_cfg = getattr(base.config, 'origin_config', base.config)
        intermediate_size = config_dict.get('intermediate_size', getattr(origin_cfg, 'intermediate_size', 9728))
        hidden_act = config_dict.get('hidden_act', 'silu')

        # Build target layer ids
        target_layer_ids = dflash_cfg.get('target_layer_ids', None)
        if target_layer_ids is None:
            # Use build_target_layer_ids logic
            target_num_layers = getattr(base.config, 'num_hidden_layers', 32)
            if num_hidden_layers == 1:
                target_layer_ids = [target_num_layers // 2]
            else:
                start = 1
                end = target_num_layers - 3
                span = end - start
                target_layer_ids = [
                    int(round(start + (i * span) / (num_target_layers - 1)))
                    for i in range(num_target_layers)
                ]
        self.target_layer_ids = target_layer_ids

        # Build a simple config namespace for sub-modules
        class SimpleConfig:
            pass
        cfg = SimpleConfig()
        cfg.hidden_size = self.hidden_size
        cfg.head_dim = self.head_dim
        cfg.num_attention_heads = self.num_attention_heads
        cfg.num_key_value_heads = self.num_key_value_heads
        cfg.intermediate_size = intermediate_size
        cfg.hidden_act = hidden_act
        cfg.rms_norm_eps = self.rms_norm_eps

        # FC: Linear(num_target_layers * hidden_size, hidden_size)
        self.fc = nn.Linear(len(self.target_layer_ids) * self.hidden_size, self.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # Decoder layers
        self.layers = nn.ModuleList([
            DFlashDecoderLayer(cfg, i) for i in range(num_hidden_layers)
        ])

        # Final norm
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # Shared lm_head from base model (for inclusion in dflash.onnx output)
        self.lm_head = base.lm.lm

        # Shared embed_tokens from base model (for embedding block tokens)
        self.embed_tokens = base.embed.embed

        # Rotary embedding
        # Compatibility: transformers>=5.x moved rope_theta into rope_parameters dict
        self.rope_theta = getattr(base.config, 'rope_theta', None)
        if self.rope_theta is None or self.rope_theta == 10000.0:
            origin_cfg = getattr(base.config, 'origin_config', base.config)
            rp = getattr(origin_cfg, 'rope_parameters', None) or getattr(origin_cfg, 'rope_scaling', None)
            if isinstance(rp, dict) and 'rope_theta' in rp:
                self.rope_theta = rp['rope_theta']
        if self.rope_theta is None:
            self.rope_theta = 10000.0
        self.max_position_embeddings = getattr(base.config, 'max_position_embeddings', 32768)

        # Load weights
        self._load_weights(dflash_path)

        self.unloaded_ops = {}

    def _load_weights(self, dflash_path):
        """Load DFlash model weights from safetensors or bin file."""
        safetensors_path = os.path.join(dflash_path, "model.safetensors")
        bin_path = os.path.join(dflash_path, "pytorch_model.bin")
        state_dict = None
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"DFlash path '{dflash_path}' has no 'model.safetensors' or 'pytorch_model.bin'."
            )

        # Map weights to our structure
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            new_state_dict[new_key] = value

        # Filter to only load our parameters (exclude lm_head, embed_tokens, rotary)
        own_keys = set(k for k, _ in self.named_parameters())
        filtered = {}
        for key, value in new_state_dict.items():
            if key in own_keys:
                filtered[key] = value
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        # lm_head and embed_tokens are shared from base, so they'll be in missing - that's fine

    def unload_param(self):
        """Replace linear layers with FakeLinear for memory-efficient export."""
        def build_faker(real, name):
            faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
            self.unloaded_ops[name] = real
            return faker

        with torch.no_grad():
            for i in range(len(self.layers)):
                for name, child in self.layers[i].self_attn.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.layers[i].self_attn, name, build_faker(child, f'/dflash_layers.{i}/self_attn/{name}/Linear'))
                for name, child in self.layers[i].mlp.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.layers[i].mlp, name, build_faker(child, f'/dflash_layers.{i}/mlp/{name}/Linear'))
            self.fc = build_faker(self.fc, '/dflash/fc/Linear')
            self.lm_head = build_faker(self.lm_head, '/lm/lm_head/Linear')

    def forward(self, noise_embedding, context_hidden, attention_mask, q_position_ids, k_position_ids):
        """
        DFlash main forward pass.
        Args:
            noise_embedding: [1, block_size, hidden_size] - embedded block tokens
            context_hidden: [1, context_len, hidden_size] - output from fc module
            attention_mask: [1, 1, block_size, context_len + block_size] - all zeros (non-causal)
            q_position_ids: [1, block_size] - position ids for Q (block positions only)
            k_position_ids: [1, context_len + block_size] - position ids for K/V (all positions)
        Returns:
            logits: [1, block_size, vocab_size]
        """
        hidden_states = noise_embedding

        # Compute rotary embeddings separately for Q and K
        q_cos, q_sin = self._compute_rope(q_position_ids)  # [1, 1, block_size, head_dim]
        k_cos, k_sin = self._compute_rope(k_position_ids)  # [1, 1, total_len, head_dim]

        for layer in self.layers:
            hidden_states = layer(hidden_states, context_hidden, q_cos, q_sin, k_cos, k_sin, attention_mask)

        hidden_states = self.norm(hidden_states)
        # Apply lm_head to get logits
        logits = self.lm_head(hidden_states)
        return logits

    def _compute_rope(self, position_ids):
        """Compute rotary position embeddings (cos, sin) for given positions."""
        # position_ids: [1, seq_len]
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=position_ids.device) / self.head_dim))
        # [seq_len] x [head_dim/2] -> [seq_len, head_dim/2]
        freqs = position_ids.float().squeeze(0).unsqueeze(-1) * inv_freq.unsqueeze(0)
        # [seq_len, head_dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
        sin = emb.sin().unsqueeze(0).unsqueeze(1)
        return cos, sin

    @spinner_run(f'export onnx model to ')
    def export(self, onnx_path):
        dflash_model = f'{onnx_path}/dflash.onnx'
        dflash_fc_model = f'{onnx_path}/dflash_fc.onnx'

        block_size = self.block_size
        context_len = 3  # dummy context length for export

        # Export dflash_fc.onnx
        fc_module = DFlashFc(self.fc, self.hidden_norm)
        fc_hidden = torch.ones([1, context_len, len(self.target_layer_ids) * self.hidden_size], dtype=torch.float)
        with torch.no_grad():
            onnx_export(
                fc_module, (fc_hidden,),
                dflash_fc_model,
                input_names=['target_hidden'],
                output_names=['context_hidden'],
                dynamic_axes={"target_hidden": {1: "seq_len"}}
            )

        # Unload params for main model export
        self.unload_param()

        # Export dflash.onnx (main model)
        noise_embedding = torch.ones([1, block_size, self.hidden_size], dtype=torch.float)
        context_hidden = torch.ones([1, context_len, self.hidden_size], dtype=torch.float)
        attention_mask = torch.zeros([1, 1, block_size, context_len + block_size], dtype=torch.float)
        q_position_ids = torch.arange(context_len, context_len + block_size, dtype=torch.int).unsqueeze(0)
        k_position_ids = torch.arange(context_len + block_size, dtype=torch.int).unsqueeze(0)

        with torch.no_grad():
            onnx_export(
                self, (noise_embedding, context_hidden, attention_mask, q_position_ids, k_position_ids),
                dflash_model,
                input_names=['noise_embedding', 'context_hidden', 'attention_mask', 'q_position_ids', 'k_position_ids'],
                output_names=['logits'],
                dynamic_axes={
                    "noise_embedding": {1: "block_size"},
                    "context_hidden": {1: "context_len"},
                    "attention_mask": {2: "block_size", 3: "total_len"},
                    "q_position_ids": {1: "block_size"},
                    "k_position_ids": {1: "total_len"},
                }
            )

        return dflash_model, dflash_fc_model