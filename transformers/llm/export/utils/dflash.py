import os
import json
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .transformers import Attention, RMSNorm, Rotary, Embedding
from utils.custom_op import FakeLinear, FusedAttention
from utils.spinner import spinner_run
from .torch_utils import onnx_export
from transformers.activations import ACT2FN


def dflash_rope(position_ids, head_dim, rope_theta):
    """RoPE (cos, sin) [1, 1, seq_len, head_dim]; shared by the draft graph and DFlashKVMat."""
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                                  device=position_ids.device) / head_dim))
    freqs = position_ids.float().squeeze(0).unsqueeze(-1) * inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0).unsqueeze(1), emb.sin().unsqueeze(0).unsqueeze(1)


class DFlashAttention(torch.nn.Module):
    """DFlash non-causal attention: Q from noise, K/V = committed cache + noise."""
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

        self.fused_attn = FusedAttention(
            self.num_attention_heads * self.head_dim,
            kv_cache=0,
            name=f'/dflash_layers.{layer_idx}/self_attn/FusedAttention',
            layer_index=-1,
            kv_shared_layer_index=-1)

    def forward(self, hidden_states, kv_k, kv_v, cos, sin, attention_mask):
        """kv_k/kv_v are the committed K/V (already k_norm'ed and RoPE'd by dflash_kvmat); k/v_proj run only on the block."""
        bsz = 1
        q_len = hidden_states.shape[1]

        # Projections + q/k norm in [B, seq, heads, head_dim] layout
        q = self.q_norm(self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim))
        k_new = self.k_norm(self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim))
        v_new = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # RoPE, then one fused Attention op (K/V un-repeated; the op does GQA internally)
        q = self._apply_rope(q, cos, sin)
        k_new = self._apply_rope(k_new, cos, sin)
        k = torch.cat([kv_k, k_new], dim=1)
        v = torch.cat([kv_v, v_new], dim=1)
        attn_output = self.fused_attn(q, k, v, attention_mask)  # [1, q_len, num_heads*head_dim]

        # No-op reshape: FusedAttentionOp.symbolic annotates the output as rank-4 while the runtime tensor is rank-3
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)

    @staticmethod
    def _apply_rope(x, cos, sin):
        """RoPE for [B, seq, heads, dim] layout: transpose cos/sin [1,1,seq,dim]->[1,seq,1,dim]."""
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
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

    def forward(self, hidden_states, kv_k, kv_v, cos, sin, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, kv_k, kv_v, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp.down_proj(
            self.mlp.act_fn(self.mlp.gate_proj(hidden_states)) * self.mlp.up_proj(hidden_states)
        )
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashKVMat(torch.nn.Module):
    """Projects newly committed context rows to per-layer K/V, output as (kv_k_0, kv_v_0, ..., kv_k_L-1, kv_v_L-1).

    new_context: [1, rows, hidden] -- target hidden states after dflash_fc.
    position_ids: [1, rows] draft-local positions, torch.arange(start, start + rows):
    the engine numbers the draft context 0-based and contiguous (see appendDraftKv in
    dflash.cpp), which is RoPE-equivalent to absolute positions and never leaves holes.
    """
    def __init__(self, layers, head_dim, rope_theta):
        super().__init__()
        self.layers = layers
        self.head_dim = head_dim
        self.rope_theta = rope_theta

    def forward(self, new_context, position_ids):
        cos, sin = dflash_rope(position_ids, self.head_dim, self.rope_theta)
        outs = []
        for layer in self.layers:
            attn = layer.self_attn
            k = attn.k_norm(attn.k_proj(new_context).view(1, -1, attn.num_key_value_heads, attn.head_dim))
            k = DFlashAttention._apply_rope(k, cos, sin)
            v = attn.v_proj(new_context).view(1, -1, attn.num_key_value_heads, attn.head_dim)
            outs += [k, v]
        return tuple(outs)


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

        # Only hidden_size is shared with the target; the head geometry is the draft's own.
        self.hidden_size = base.config.hidden_size
        self.head_dim = config_dict.get('head_dim', base.config.head_dim)
        self.num_attention_heads = config_dict.get('num_attention_heads', base.config.num_attention_heads)
        self.num_key_value_heads = config_dict.get('num_key_value_heads', base.config.num_key_value_heads)
        self.rms_norm_eps = config_dict.get('rms_norm_eps', getattr(base.config, 'rms_norm_eps', 1e-6))

        # DFlash-specific config
        dflash_cfg = config_dict.get('dflash_config', {})
        self.block_size = config_dict.get('block_size', 8)
        self.mask_token_id = dflash_cfg.get('mask_token_id', 0)
        # Not derivable from the graph, so the runtime reads it from config.json.
        self.shift_label = bool(dflash_cfg.get('shift_label', False))

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

        # The draft's own rope config wins: a target composite config can hide rope_parameters in text_config.
        self.rope_theta = config_dict.get('rope_theta', None)
        if self.rope_theta is None:
            rp = config_dict.get('rope_parameters')
            if isinstance(rp, dict):
                self.rope_theta = rp.get('rope_theta')
        # Compatibility: transformers>=5.x moved rope_theta into rope_parameters dict
        if self.rope_theta is None:
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

    def _build_faker(self, real, name):
        """Replace a Linear with a FakeLinear and record the real op for MNN rebuild."""
        faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
        self.unloaded_ops[name] = real
        return faker

    def unload_param(self):
        """Replace linear layers with FakeLinear for memory-efficient export.
        lm_head is not part of the main graph; it is handled in export() when needed."""
        with torch.no_grad():
            for i in range(len(self.layers)):
                for name, child in self.layers[i].self_attn.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.layers[i].self_attn, name, self._build_faker(child, f'/dflash_layers.{i}/self_attn/{name}/Linear'))
                for name, child in self.layers[i].mlp.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.layers[i].mlp, name, self._build_faker(child, f'/dflash_layers.{i}/mlp/{name}/Linear'))
            self.fc = self._build_faker(self.fc, '/dflash/fc/Linear')

    def forward(self, noise_embedding, *args):
        """args order matches the ONNX input_names: (kv_k_0, kv_v_0, ..., kv_k_L-1, kv_v_L-1, attention_mask, position_ids)."""
        n = len(self.layers)
        kv_args = args[:2 * n]
        attention_mask, position_ids = args[2 * n], args[2 * n + 1]
        hidden_states = noise_embedding

        cos, sin = dflash_rope(position_ids, self.head_dim, self.rope_theta)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, kv_args[2 * i], kv_args[2 * i + 1],
                                  cos, sin, attention_mask)

        hidden_states = self.norm(hidden_states)
        # lm_head is applied on the engine side (shared from target or separate file)
        return hidden_states

    @spinner_run(f'export onnx model to ')
    def export(self, onnx_path):
        """Export the DFlash draft model to ONNX."""
        dflash_model = f'{onnx_path}/dflash.onnx'
        dflash_fc_model = f'{onnx_path}/dflash_fc.onnx'
        dflash_kvmat_model = f'{onnx_path}/dflash_kvmat.onnx'

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

        kv_names = []
        for i in range(len(self.layers)):
            kv_names += [f'kv_k_{i}', f'kv_v_{i}']

        # Reuses dflash.onnx's k_proj/k_norm/v_proj FakeLinears, so both files carry those weights.
        kvmat_module = DFlashKVMat(self.layers, self.head_dim, self.rope_theta)
        mat_context = torch.ones([1, context_len, self.hidden_size], dtype=torch.float)
        mat_pos = torch.arange(context_len, dtype=torch.int).unsqueeze(0)
        with torch.no_grad():
            onnx_export(
                kvmat_module, (mat_context, mat_pos),
                dflash_kvmat_model,
                input_names=['new_context', 'position_ids'],
                output_names=kv_names,
                dynamic_axes={"new_context": {1: "seq_len"}, "position_ids": {1: "seq_len"}}
            )

        # Export dflash.onnx (main model, outputs hidden_states; lm_head not baked in)
        noise_embedding = torch.ones([1, block_size, self.hidden_size], dtype=torch.float)
        kv_dummy = [torch.ones([1, context_len, self.num_key_value_heads, self.head_dim], dtype=torch.float)
                    for _ in kv_names]
        attention_mask = torch.zeros([1, 1, block_size, context_len + block_size], dtype=torch.float)
        position_ids = torch.arange(context_len, context_len + block_size, dtype=torch.int).unsqueeze(0)

        with torch.no_grad():
            onnx_export(
                self, (noise_embedding, *kv_dummy, attention_mask, position_ids),
                dflash_model,
                input_names=['noise_embedding'] + kv_names + ['attention_mask', 'position_ids'],
                output_names=['hidden_states'],
                dynamic_axes={
                    "noise_embedding": {1: "block_size"},
                    "attention_mask": {2: "block_size", 3: "total_len"},
                    "position_ids": {1: "block_size"},
                    **{n: {1: "kv_len"} for n in kv_names},
                }
            )

        return dflash_model, dflash_fc_model, dflash_kvmat_model
