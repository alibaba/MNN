"""DSpark draft export: DFlash backbone + Markov logit bias + shared lm_head.

Everything here reuses the DFlash export path; DSpark only adds the Markov
head and folds the target's lm_head into the draft graph, because the bias has
to land on the logits before argmax.
"""

import json
import os

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from .dflash import DFlashAttention, DFlashFc, DFlashKVMat, dflash_rope
from utils.custom_op import FakeLinear, FusedAttention
from utils.spinner import spinner_run
from .torch_utils import onnx_export
from .transformers import RMSNorm

_FC_SCALE = 8.0


class DSparkMarkovW1(torch.nn.Module):
    """Maps the previous slot's token to its rank-space coefficient.

    DSpark conditions each slot on the token at the previous slot, and at inference that token
    is the draft's own prediction, so the bias forms a chain the engine must walk. Shipping w1
    as a graph keeps that gather on-device instead of copying a 148 MB table to the host.
    """

    def __init__(self, w1):
        super().__init__()
        self.w1 = w1

    def forward(self, prev_ids):
        # w1 is [vocab, rank]; the gather returns that token's rank vector as [1, rank].
        return torch.index_select(self.w1, 0, prev_ids.view(-1)).reshape(1, -1)


class DSparkMarkovW2(torch.nn.Module):
    """The rank-space coefficient times w2, i.e. one slot's vocab-wide bias.

    Kept as its own graph so the exporter can quantize it: at inference this is the only
    matrix the Markov chain multiplies, and a weight-only quantized 1x1 conv is what makes
    the seven chain steps cheap.
    """

    def __init__(self, w2):
        super().__init__()
        self.w2 = w2          # Linear [vocab, rank] from the checkpoint

    def forward(self, coef):
        # coef [1, rank] -> [1, vocab]
        return self.w2(coef)


class DSparkLogits(torch.nn.Module):
    """Draft + lm_head, plus the Markov bias of one slot when the engine supplies it.

    The bias is W2 @ W1[prev], a full [vocab] vector per row, so it must be added in the
    graph; the engine walks the chain and passes the rank-space embedding per slot.
    """

    def __init__(self, dspark, block_size):
        super().__init__()
        self.dspark = dspark
        self.block_size = block_size

    def input_names(self):
        n_ctx = len(self.dspark.target_layer_ids)
        n_kv = len(self.dspark.layers)
        names = ['noise_embedding'] + [f'hidden_{i}' for i in range(n_ctx)]
        for i in range(n_kv):
            names += [f'kv_k_{i}', f'kv_v_{i}']
        return names + ['attention_mask', 'position_ids', 'block_ids']

    def forward(self, noise_embedding, *args):
        d = self.dspark
        n_ctx = len(d.target_layer_ids)
        n_kv = len(d.layers)
        # args = (hidden_0..N-1, kv_k_0, kv_v_0, ..., attention_mask, position_ids, block_ids)
        lead = n_ctx + 2 * n_kv
        block_ids = args[lead + 2]
        hidden_states = d(noise_embedding, *args[:lead + 2])
        # Rows are [1, block, hidden]; lm_head wants [seq, hidden] and returns [seq, vocab].
        # Both reshapes name the block axis explicitly: a -1 placeholder there is resolved by
        # the converter into the vocab axis, which silently transposes the block logits.
        block = hidden_states.shape[1]
        hidden = hidden_states.shape[-1]
        # Flat rows -> [rows, vocab]; lm_head's FakeLinear keeps the leading axes.
        logits = d.lm_head(hidden_states.reshape(block, hidden))
        logits = logits.reshape(1, block, -1)
        # Markov bias is applied by the engine, one slot at a time: slot k depends on slot k-1's
        # sampled token, which is only known after the previous `rowArgmax`. See dspark_markov.
        return logits


class DSparkDecoderLayer(torch.nn.Module):
    """DSpark layer: block hidden states attend to the whole context plus their own block."""

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
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.fused_attn = FusedAttention(
            self.num_attention_heads * self.head_dim,
            kv_cache=0,
            name=f'/dspark_layers.{layer_idx}/self_attn/FusedAttention',
            layer_index=-1,
            kv_shared_layer_index=-1)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.mlp.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, kv_k, kv_v, cos, sin, attention_mask):
        """kv_k/kv_v are the committed context K/V; the block's own K/V are projected here."""
        bsz = 1
        q_len = hidden_states.shape[1]

        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        q = self.q_norm(self.q_proj(normed).view(bsz, q_len, self.num_attention_heads, self.head_dim))
        k_noise = self.k_norm(self.k_proj(normed).view(bsz, q_len, self.num_key_value_heads, self.head_dim))
        v_noise = self.v_proj(normed).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        # Context rows keep the positions they were committed at, so only the block gets rotated.
        q = DFlashAttention._apply_rope(q, cos[:, :, -q_len:], sin[:, :, -q_len:])
        k_noise = DFlashAttention._apply_rope(k_noise, cos[:, :, -q_len:], sin[:, :, -q_len:])
        k = torch.cat([kv_k, k_noise], dim=1)
        v = torch.cat([kv_v, v_noise], dim=1)

        # FusedAttention already emits [b, seq, hidden]; a reshape here would be an identity
        # op, and a dynamic one at that, which blocks the converter's attention-output C4 fuse
        # (the same fuse the target model relies on) and leaves the op on plain NCHW.
        attn_output = self.fused_attn(q, k, v, attention_mask)
        attn_output = self.o_proj(attn_output)

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp.down_proj(
            self.mlp.act_fn(self.mlp.gate_proj(hidden_states)) * self.mlp.up_proj(hidden_states)
        )
        return residual + hidden_states


class DSparkKVMat(torch.nn.Module):
    """Projects committed context rows to per-layer K/V, output as (kv_k_0, kv_v_0, ...).

    Mirrors DFlashKVMat but reads the projections off the layer: DSpark layers hold
    them directly instead of under self_attn.
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
            k = layer.k_norm(layer.k_proj(new_context).view(
                1, -1, layer.num_key_value_heads, self.head_dim))
            k = DFlashAttention._apply_rope(k, cos, sin)
            v = layer.v_proj(new_context).view(1, -1, layer.num_key_value_heads, self.head_dim)
            outs += [k, v]
        return tuple(outs)


class DSpark(torch.nn.Module):
    """DSpark draft model: one block forward yields hidden states and the Markov bias."""

    def __init__(self, dspark_path, base):
        super().__init__()
        with open(os.path.join(dspark_path, 'config.json'), 'r') as f:
            config_dict = json.load(f)
        self.dspark_config = config_dict

        # DSpark checkpoints keep the draft geometry in the top-level fields.
        self.hidden_size = base.config.hidden_size
        self.head_dim = config_dict.get('head_dim', base.config.head_dim)
        self.num_attention_heads = config_dict.get('num_attention_heads', base.config.num_attention_heads)
        self.num_key_value_heads = config_dict.get('num_key_value_heads', base.config.num_key_value_heads)
        self.rms_norm_eps = config_dict.get('rms_norm_eps', 1e-6)
        self.block_size = config_dict.get('block_size', 8)
        self.mask_token_id = config_dict.get('mask_token_id', 0)
        # DSpark always predicts from the anchor slot onward; DFlash-style label shifting is the
        # same thing expressed as a flag, so map it to the one field the engine reads.
        # DSpark's default (sample_from_anchor) means block slot k predicts the token at
        # k+1, i.e. the draft row k feeds verification slot k+1 -- DFlash calls that shift_label.
        self.sample_from_anchor = config_dict.get('sample_from_anchor', True)
        self.shift_label = bool(self.sample_from_anchor)
        self.markov_rank = config_dict.get('markov_rank', 0)
        if self.markov_rank and config_dict.get('markov_head_type', 'vanilla') != 'vanilla':
            raise NotImplementedError('only the vanilla DSpark Markov head is supported')

        target_layer_ids = (config_dict.get('target_layer_ids')
                            or config_dict.get('aux_hidden_state_layer_ids')
                            or config_dict.get('dflash_config', {}).get('target_layer_ids'))
        if not target_layer_ids:
            raise ValueError(f'{dspark_path}/config.json carries no target layer ids')
        self.target_layer_ids = list(target_layer_ids)
        self.num_hidden_layers = config_dict.get('num_hidden_layers', 1)

        class SimpleConfig:
            pass
        cfg = SimpleConfig()
        cfg.hidden_size = self.hidden_size
        cfg.head_dim = self.head_dim
        cfg.num_attention_heads = self.num_attention_heads
        cfg.num_key_value_heads = self.num_key_value_heads
        cfg.intermediate_size = config_dict.get('intermediate_size')
        cfg.hidden_act = config_dict.get('hidden_act', 'silu')
        cfg.rms_norm_eps = self.rms_norm_eps

        # Context compression: [target layers] -> draft hidden.
        self.fc = nn.Linear(len(self.target_layer_ids) * self.hidden_size, self.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.layers = nn.ModuleList([DSparkDecoderLayer(cfg, i) for i in range(self.num_hidden_layers)])
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        if self.markov_rank:
            # W1 indexes the previous token (verifier vocab), W2 projects the rank back to logits.
            self.markov_w1 = nn.Embedding(config_dict['vocab_size'], self.markov_rank)
            self.markov_w2 = nn.Linear(self.markov_rank, config_dict['vocab_size'], bias=False)
        self.lm_head = base.lm.lm
        self.embed_tokens = base.embed.embed
        self.rope_theta = self._read_rope_theta(config_dict)

        self._load_weights(dspark_path)
        self.markov_proj = self.markov_w2 if self.markov_rank else None
        # Kept aside because unload_param replaces the projections with placeholders.
        self.markov_tables = None
        if self.markov_rank:
            self.markov_tables = (self.markov_w1.weight.detach().clone(),
                                  self.markov_w2.weight.detach().clone())
        # Export runs in fp32: the draft checkpoints ship bf16, which the ONNX graph then keeps.
        self.float()
        with torch.no_grad():
            self.fc.weight.div_(_FC_SCALE)
        self.hidden_norm.variance_epsilon /= _FC_SCALE * _FC_SCALE
        self.unloaded_ops = {}

    @staticmethod
    def _read_rope_theta(config_dict):
        theta = config_dict.get('rope_theta')
        if theta is None:
            theta = (config_dict.get('rope_parameters') or {}).get('rope_theta')
        return 1000000.0 if theta is None else float(theta)

    def _load_weights(self, dspark_path):
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(dspark_path, 'model.safetensors'), device='cpu')
        own_keys = set(dict(self.named_parameters()).keys())
        # The checkpoint names the Markov factors markov_head.markov_w*; the export holds them
        # as embeddings so the engine can gather any row, so the keys need rewriting.
        mapped = {}
        for key, value in state_dict.items():
            # markov_head.markov_w1.weight -> markov_w1.weight (embedding table),
            # markov_head.markov_w2.weight -> markov_w2.weight (Linear [vocab, rank], same layout).
            if key.startswith('markov_head.'):
                key = key[len('markov_head.'):]
            # The attention projections live directly on the layer here, not under self_attn.
            key = key.replace('.self_attn.', '.')
            if key in own_keys:
                mapped[key] = value
        missing, _ = self.load_state_dict(mapped, strict=False)
        shared = ('lm_head.', 'embed_tokens.')
        strict_missing = [m for m in missing if not m.startswith(shared)]
        if strict_missing:
            raise RuntimeError(f'DSpark checkpoint misses {strict_missing}')

    def _build_faker(self, real, name):
        faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
        self.unloaded_ops[name] = real
        return faker

    def unload_param(self):
        with torch.no_grad():
            for i in range(len(self.layers)):
                layer = self.layers[i]
                for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                    child = getattr(layer, name)
                    if isinstance(child, torch.nn.Linear):
                        setattr(layer, name,
                                self._build_faker(child, f'/dspark_layers.{i}/self_attn/{name}/Linear'))
                for name, child in layer.mlp.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(layer.mlp, name,
                                self._build_faker(child, f'/dspark_layers.{i}/mlp/{name}/Linear'))
            self.fc = self._build_faker(self.fc, '/dspark/fc/Linear')
            # lm_head is shared with the target, so detach before faking it: the converter only
            # refills ops it unloaded, and a plain Linear would be serialized as all-zero weight.
            lm = torch.nn.Linear(self.lm_head.in_features, self.lm_head.out_features,
                                 bias=self.lm_head.bias is not None)
            with torch.no_grad():
                lm.weight.copy_(self.lm_head.weight)
                if lm.bias is not None:
                    lm.bias.copy_(self.lm_head.bias)
            self.lm_head = self._build_faker(lm, '/lm_head/Linear')
            if self.markov_rank and isinstance(self.markov_w2, torch.nn.Linear):
                self.markov_w2 = self._build_faker(self.markov_w2, '/markov_w2/Linear')
                self.markov_proj = self.markov_w2

    def forward(self, noise_embedding, *args):
        """args: (hidden_0..hidden_{N-1}, kv_k_0, kv_v_0, ..., attention_mask, position_ids)."""
        n_ctx = len(self.target_layer_ids)
        n_kv = len(self.layers)
        lead = n_ctx + 2 * n_kv
        hidden_args = args[:n_ctx]
        kv_args = args[n_ctx:lead]
        attention_mask, position_ids = args[lead], args[lead + 1]

        # The target hidden states reach the draft only through the per-layer context K/V
        # (dspark_kvmat); the block itself starts from the noise embedding.
        # cos/sin span the whole key length so the block rows rotate at their own positions;
        # the context K/V arrive already rotated by dspark_kvmat. position_ids is the full
        # [context | block] range so its values stay a graph input (a traced arange is constant).
        cos, sin = dflash_rope(position_ids, self.head_dim, self.rope_theta)
        hidden_states = noise_embedding
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, kv_args[2 * i], kv_args[2 * i + 1], cos, sin, attention_mask)
        return self.norm(hidden_states)

    @spinner_run(f'export onnx model to ')
    def export(self, onnx_path):
        """Export the DSpark draft; dspark.onnx emits unbiased block logits.

        The Markov bias is a separate module because it is a chain: slot k is biased by the
        token sampled at slot k-1, so the engine has to interleave argmax with bias.
        """
        dspark_model = f'{onnx_path}/dspark.onnx'
        dspark_fc_model = f'{onnx_path}/dspark_fc.onnx'
        dspark_kvmat_model = f'{onnx_path}/dspark_kvmat.onnx'
        dspark_markov_w1_model = f'{onnx_path}/dspark_markov_w1.onnx'
        dspark_markov_w2_model = f'{onnx_path}/dspark_markov_w2.onnx'

        block_size = self.block_size
        context_len = 3  # dummy context length for export
        n_kv = len(self.layers)
        n_ctx = len(self.target_layer_ids)

        # Export dspark_fc.onnx
        fc_module = DFlashFc(self.fc, self.hidden_norm)
        fc_hidden = torch.ones([1, context_len, len(self.target_layer_ids) * self.hidden_size], dtype=torch.float)
        with torch.no_grad():
            onnx_export(
                fc_module, (fc_hidden,),
                dspark_fc_model,
                input_names=['target_hidden'],
                output_names=['context_hidden'],
                dynamic_axes={"target_hidden": {1: "seq_len"}}
            )

        self.unload_param()

        kv_names = []
        for i in range(n_kv):
            kv_names += [f'kv_k_{i}', f'kv_v_{i}']

        kvmat_module = DSparkKVMat(self.layers, self.head_dim, self.rope_theta)
        mat_context = torch.ones([1, context_len, self.hidden_size], dtype=torch.float)
        mat_pos = torch.arange(context_len, dtype=torch.int).unsqueeze(0)
        with torch.no_grad():
            onnx_export(
                kvmat_module, (mat_context, mat_pos),
                dspark_kvmat_model,
                input_names=['new_context', 'position_ids'],
                output_names=kv_names,
                dynamic_axes={"new_context": {1: "seq_len"}, "position_ids": {1: "seq_len"}}
            )

        # dspark.onnx: draft hidden states -> shared lm_head -> markov bias -> biased logits.
        # The bias has to land on the logits before argmax, so it lives in the draft graph.
        noise_embedding = torch.ones([1, block_size, self.hidden_size], dtype=torch.float)
        kv_dummy = [torch.ones([1, context_len, self.num_key_value_heads, self.head_dim], dtype=torch.float)
                    for _ in kv_names]
        # Dense additive plane, 0 everywhere: every block row sees the whole key axis.
        attention_mask = torch.zeros([1, 1, block_size, context_len + block_size], dtype=torch.float)
        # [context | block] positions: the graph slices the block rows off the tail.
        position_ids = torch.arange(0, context_len + block_size, dtype=torch.int).unsqueeze(0)
        block_ids = torch.full((1, block_size), self.mask_token_id, dtype=torch.int)
        block_ids[0, 0] = 100
        hidden_dummy = [torch.ones([1, block_size, self.hidden_size], dtype=torch.float) for _ in range(n_ctx)]

        wrapper = DSparkLogits(self, block_size)
        with torch.no_grad():
            onnx_export(
                wrapper, (noise_embedding, *hidden_dummy, *kv_dummy, attention_mask, position_ids, block_ids),
                dspark_model,
                input_names=wrapper.input_names(),
                output_names=['block_logits'],
                dynamic_axes={
                    "noise_embedding": {1: "block_size"},
                    # Only the key axis varies; the query axis is the fixed block size, and
                    # marking it dynamic stops MNN from classifying the mask as dense.
                    "attention_mask": {3: "total_len"},
                    "position_ids": {1: "total_len"},
                    "block_ids": {1: "block_size"},
                    **{n: {1: "kv_len"} for n in kv_names},
                    **{f'hidden_{i}': {1: "block_size"} for i in range(n_ctx)},
                }
            )

        # Markov head, split in two so each half stays on-device. w1 maps the previous token
        # to its rank vector; w2 turns that into the vocab-wide bias. Keeping w2 as its own
        # graph is what lets the exporter quantize it -- a weight-only 1x1 conv measures ~9x
        # faster than the engine sweeping the table from the host.
        markov_w1_onnx = None
        markov_w2_onnx = None
        if self.markov_rank:
            w1_weight, w2_weight = self.markov_tables
            w2_linear = torch.nn.Linear(self.markov_rank, w2_weight.shape[0], bias=False)
            with torch.no_grad():
                w2_linear.weight.copy_(w2_weight)
            m1 = DSparkMarkovW1(w1_weight).float()
            m2 = DSparkMarkovW2(w2_linear).float()
            prev_ids = torch.zeros(1, dtype=torch.int32)
            coef = torch.zeros(1, self.markov_rank)
            with torch.no_grad():
                onnx_export(m1, (prev_ids,), dspark_markov_w1_model,
                            input_names=['prev_ids'], output_names=['coef'])
                onnx_export(m2, (coef,), dspark_markov_w2_model,
                            input_names=['coef'], output_names=['bias'])

        return dspark_model, dspark_fc_model, dspark_kvmat_model, dspark_markov_w1_model, dspark_markov_w2_model
