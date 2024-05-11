# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright (c) 2022, Tri Dao, trid@cs.stanford.edu.
# Licensed under the BSD 3-Clause License.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_phi import PhiConfig

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
    from flash_attn.modules.mha import FlashCrossAttention, FlashSelfAttention
    from flash_attn.ops.fused_dense import FusedDense
except:
    pad_input, unpad_input = None, None
    FlashRotaryEmbedding = None
    FlashSelfAttention, FlashCrossAttention = None, None
    FusedDense = None


@dataclass
class InferenceParams:
    """Inference parameters passed to model to efficiently calculate
    and store context during inference.

    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/generation.py.

    Args:
        max_seqlen: Maximum sequence length.
        max_batch_size: Maximum batch size.
        seqlen_offset: Sequence length offset.
        batch_size_offset: Batch size offset.
        key_value_memory_dict: Key value memory dictionary.
        lengths_per_sample: Lengths per sample.

    """

    max_seqlen: int = field(metadata={"help": "Maximum sequence length."})

    max_batch_size: int = field(metadata={"help": "Maximum batch size."})

    seqlen_offset: int = field(default=0, metadata={"help": "Sequence length offset."})

    batch_size_offset: int = field(default=0, metadata={"help": "Batch size offset."})

    key_value_memory_dict: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Key value memory dictionary."}
    )

    lengths_per_sample: torch.Tensor = field(default=None, metadata={"help": "Lengths per sample."})


class Embedding(nn.Module):
    """Token embedding with dropout."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.wte(input_ids)
        hidden_states = self.drop(hidden_states)

        return hidden_states


def _apply_rotary_emb(
    x: torch.FloatTensor,
    cos: torch.FloatTensor,
    sin: torch.FloatTensor,
) -> torch.FloatTensor:
    _, seqlen, _, _ = x.shape
    _, rotary_dim = cos.shape
    rotary_dim *= 2

    x_rot = x[:, :, :, :rotary_dim]
    x_pass = x[:, :, :, rotary_dim:]

    x1, x2 = x_rot.chunk(2, dim=-1)
    c, s = rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d")
    x1, x2, c, s = [t.to(dtype=torch.float32) for t in [x1, x2, c, s]]

    x_rot = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).to(x.dtype)

    return torch.cat([x_rot, x_pass], axis=-1)


def _apply_rotary_emb_kv(
    kv: torch.FloatTensor,
    cos: torch.FloatTensor,
    sin: torch.FloatTensor,
    cos_k: Optional[torch.FloatTensor] = None,
    sin_k: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    _, seqlen, _, _, _ = kv.shape
    _, rotary_dim = cos.shape
    rotary_dim *= 2

    k_rot = kv[:, :, 0, :, :rotary_dim]
    k_pass = kv[:, :, 0, :, rotary_dim:]

    k1, k2 = k_rot.chunk(2, dim=-1)
    c, s = rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d")
    k1, k2, c, s = [t.to(dtype=torch.float32) for t in [k1, k2, c, s]]

    k_rot = torch.cat([k1 * c - k2 * s, k1 * s + k2 * c], axis=-1).to(kv.dtype)

    return torch.cat(
        [
            torch.cat([k_rot, k_pass], axis=-1).unsqueeze(2),
            kv[:, :, 1:2, :, :],
        ],
        axis=2,
    )


def _apply_rotary_emb_qkv(
    qkv: torch.FloatTensor,
    cos: torch.FloatTensor,
    sin: torch.FloatTensor,
    cos_k: Optional[torch.FloatTensor] = None,
    sin_k: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    _, seqlen, _, _, _ = qkv.shape
    _, rotary_dim = cos.shape
    rotary_dim *= 2

    q_rot = qkv[:, :, 0, :, :rotary_dim]
    q_pass = qkv[:, :, 0, :, rotary_dim:]

    k_rot = qkv[:, :, 1, :, :rotary_dim]
    k_pass = qkv[:, :, 1, :, rotary_dim:]

    q1, q2 = q_rot.chunk(2, dim=-1)
    k1, k2 = k_rot.chunk(2, dim=-1)
    c, s = rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d")
    q1, q2, k1, k2, c, s = [t.to(dtype=torch.float32) for t in [q1, q2, k1, k2, c, s]]

    q_rot = torch.cat([q1 * c - q2 * s, q1 * s + q2 * c], axis=-1).to(qkv.dtype)
    k_rot = torch.cat([k1 * c - k2 * s, k1 * s + k2 * c], axis=-1).to(qkv.dtype)

    return torch.cat(
        [
            torch.cat([q_rot, q_pass], axis=-1).unsqueeze(2),
            torch.cat([k_rot, k_pass], axis=-1).unsqueeze(2),
            qkv[:, :, 2:3, :, :],
        ],
        axis=2,
    )


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE).

    Reference:
        RoFormer: Enhanced Transformer with Rotary Position Embedding.
        https://arxiv.org/pdf/2104.09864.pdf.

    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        scale_base: Optional[float] = None,
        pos_idx_in_fp32: bool = True,
        max_position_embeddings: int = 2048,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if scale_base is not None:
            raise NotImplementedError

        self.dim = dim
        self.base = float(base)
        self.scale_base = scale_base
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.max_position_embeddings = max_position_embeddings
        self.device = device

        # Generate and save the inverse frequency buffer (non-trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Generate and save the scale buffer (non-trainable)
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        # Initialize cached attributes since ONNX can't rely on dynamic initialization
        self._update_cos_sin_cache(max_position_embeddings, device=device, dtype=torch.float32)

    def _compute_inv_freq(self, device: Optional[str] = None) -> torch.FloatTensor:
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(
        self,
        seqlen: int,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self._seq_len_cached = seqlen

        # fp32 is preferred since the output of `torch.arange` can be quite large
        # and bf16 would lose a lot of precision
        if self.pos_idx_in_fp32:
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            if self.inv_freq.dtype != torch.float32:
                inv_freq = self._compute_inv_freq(device=device)
            else:
                inv_freq = self.inv_freq
        else:
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            inv_freq = self.inv_freq

        # `torch.outer` is preferred since `torch.einsum` converts from fp32 to fp16 if used with AMP
        freqs = torch.outer(t, inv_freq)
        if self.scale is None:
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)
        else:
            power = (
                torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
            ) / self.scale_base
            scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")

            # Force the scale multiplication to happen in fp32
            self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._seq_len_cached < qkv.shape[1] + seqlen_offset
            or self._cos_cached.device != qkv.device
            or self._cos_cached.dtype != qkv.dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._update_cos_sin_cache(qkv.shape[1] + seqlen_offset, device=qkv.device, dtype=qkv.dtype)

        if kv is None:
            return _apply_rotary_emb_qkv(
                qkv,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
            )
        else:
            q = _apply_rotary_emb(
                qkv,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
            )
            kv = _apply_rotary_emb_kv(
                kv,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
            )

            return q, kv


class MLP(nn.Module):
    """Multi-Layer Perceptron.

    Reference:
        Attention Is All You Need.
        https://arxiv.org/pdf/1706.03762.pdf.

    """

    def __init__(
        self,
        config: PretrainedConfig,
        n_inner: Optional[int] = None,
        act_fn: Optional[str] = None,
    ) -> None:
        super().__init__()

        act_fn = config.activation_function if act_fn is None else act_fn

        n_inner = getattr(config, "n_inner", None) if n_inner is None else n_inner
        n_inner = n_inner if n_inner is not None else 4 * config.n_embd

        self.fc1 = nn.Linear(config.n_embd, n_inner)
        self.fc2 = nn.Linear(n_inner, config.n_embd)
        self.act = ACT2FN[act_fn]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SelfAttention(nn.Module):
    """Self-attention layer (compatible with PyTorch).

    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py.

    """

    def __init__(
        self,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    @torch.autocast("cpu", enabled=False)
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        qkv: torch.FloatTensor,
        causal: bool = None,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)

        q = q.to(torch.float32)
        k = k.to(torch.float32)

        causal = self.causal if causal is None else causal
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

        # Autocast is manually disabled to avoid `torch.einsum` performing the operation
        # using float16, which might lead to overflow
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)

            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

        if causal:
            causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
            scores = scores + causal_mask.to(dtype=scores.dtype)

        attention = torch.softmax(scores, dim=-1).to(v.dtype)
        attention = self.drop(attention)

        output = torch.einsum("bhts,bshd->bthd", attention, v)

        return output


class CrossAttention(nn.Module):
    """Cross-attention layer (compatible with PyTorch).

    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py.

    """

    def __init__(
        self,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    @torch.autocast("cpu", enabled=False)
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        q: torch.FloatTensor,
        kv: torch.FloatTensor,
        causal: bool = None,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = kv.shape[1]

        if kv.shape[3] != q.shape[2]:
            kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
        k, v = kv.unbind(dim=2)

        q = q.to(torch.float32)
        k = k.to(torch.float32)

        causal = self.causal if causal is None else causal
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

        # Autocast is manually disabled to avoid `torch.einsum` performing the operation
        # using float16, which might lead to overflow
        # scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        q = q.permute(0, 2, 1, 3)
        k_ = (k * softmax_scale).permute(0, 2, 3, 1)
        scores = torch.matmul(q, k_)

        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen_k),
                -10000.0,
                dtype=scores.dtype,
                device=scores.device,
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)

            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, -10000.0)
        elif causal:
            rows = rearrange(torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1")
            cols = torch.arange(seqlen_k, device=k.device, dtype=torch.long)
            causal_mask = cols > rows + seqlen_k - seqlen_q

            scores = scores.masked_fill(causal_mask, -10000.0)

        attention = torch.softmax(scores, dim=-1).to(v.dtype)
        attention = self.drop(attention)

        # output = torch.einsum("bhts,bshd->bthd", attention, v)
        v = v.permute(0, 2, 1, 3)
        output = torch.matmul(attention, v).permute(0, 2, 1, 3)
        return output


def _find_mha_dims(
    config: PretrainedConfig,
    n_head: Optional[int] = None,
    n_head_kv: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> Tuple[int, int]:
    if n_head is None and head_dim is None:
        head_dim = config.n_embd // config.n_head
        n_head = config.n_head
    elif n_head is None or head_dim is None:
        raise ValueError("`n_head` and `head_dim` must be both specified or `None`.")

    if n_head_kv is None:
        n_head_kv = getattr(config, "n_head_kv", None) or n_head

    return n_head, n_head_kv, head_dim


def _update_kv_cache(kv: torch.FloatTensor, inference_params: InferenceParams, layer_idx: int) -> torch.FloatTensor:
    num_heads, head_dim = kv.shape[-2:]

    if layer_idx not in inference_params.key_value_memory_dict:
        inference_params.key_value_memory_dict[layer_idx] = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_seqlen,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )

    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]

    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]

    # When the current sequence length is equal to or larger than the maximum sequence length,
    # we need to concatenate the current `kv` with the cached `kv` to expand its length
    if sequence_end >= inference_params.max_seqlen:
        inference_params.key_value_memory_dict[layer_idx] = torch.concatenate((inference_params.key_value_memory_dict[layer_idx], kv), dim=1)

    inference_params.key_value_memory_dict[layer_idx][batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    kv = inference_params.key_value_memory_dict[layer_idx][batch_start:batch_end, :sequence_end, ...]
        
    return kv


class MHA(nn.Module):
    """Multi-head attention layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        rotary_dim: Optional[int] = None,
        rotary_base: float = 10000.0,
        rotary_scale_base: Optional[float] = None,
        n_head: Optional[int] = None,
        n_head_kv: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = True,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        layer_idx: Optional[int] = None,
        return_residual: bool = False,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()

        # Rotary embedding
        self.rotary_dim = rotary_dim if rotary_dim is not None else getattr(config, "rotary_dim", 0)
        if self.rotary_dim > 0:
            rotary_cls = FlashRotaryEmbedding if config.flash_rotary else RotaryEmbedding
            if rotary_cls is None:
                rotary_cls = RotaryEmbedding

            rotary_kwargs = {}
            if rotary_cls is RotaryEmbedding:
                rotary_kwargs["max_position_embeddings"] = config.n_positions

            self.rotary_emb = rotary_cls(
                self.rotary_dim,
                base=rotary_base,
                scale_base=rotary_scale_base,
                device=device,
                **rotary_kwargs,
            )

        # MLP
        self.n_head, self.n_head_kv, self.head_dim = _find_mha_dims(
            config, n_head=n_head, n_head_kv=n_head_kv, head_dim=head_dim
        )
        op_size = self.head_dim * (self.n_head + 2 * self.n_head_kv)
        hidden_size = config.n_embd

        linear_cls = FusedDense if config.fused_dense else nn.Linear
        if linear_cls is None:
            linear_cls = nn.Linear

        self.Wqkv = linear_cls(hidden_size, op_size, bias=bias, device=device, dtype=dtype)
        self.out_proj = linear_cls(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Attention
        attn_cls = FlashSelfAttention if config.flash_attn else SelfAttention
        if attn_cls is None:
            attn_cls = SelfAttention

        cross_attn_cls = FlashCrossAttention if config.flash_attn else CrossAttention
        if cross_attn_cls is None:
            cross_attn_cls = CrossAttention

        self.inner_attn = attn_cls(
            causal=causal,
            softmax_scale=softmax_scale,
            attention_dropout=config.attn_pdrop,
        )
        self.inner_cross_attn = cross_attn_cls(
            causal=causal,
            softmax_scale=softmax_scale,
            attention_dropout=config.attn_pdrop,
        )

        self.flash_attn = config.flash_attn and attn_cls is FlashSelfAttention
        self.layer_idx = layer_idx
        self.return_residual = return_residual
        self.checkpointing = checkpointing

    def _forward_self_attn(
        self, x: torch.FloatTensor, key_padding_mask: Optional[torch.BoolTensor]
    ) -> torch.FloatTensor:
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)

        if self.rotary_dim > 0:
            qkv = self.rotary_emb(qkv)

        if self.flash_attn:
            batch_size, seqlen = qkv.shape[0], qkv.shape[1]

            cu_seqlens, max_seqlen = None, None
            if key_padding_mask is not None:
                # If `key_padding_mask` is supplied, we need to unpad the input and retrieve
                # the `cu_seqlens` and `max_seqlen` to be used by `flash-attn`
                qkv, indices, cu_seqlens, max_seqlen = unpad_input(qkv, key_padding_mask)

            if self.checkpointing:
                attn_output = torch.utils.checkpoint.checkpoint(
                    self.inner_attn, qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
                )
            else:
                attn_output = self.inner_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).to(qkv.device)

            # If `key_padding_mask` is supplied, we need to pad the output back to the original shape
            return pad_input(attn_output, indices, batch_size, seqlen) if key_padding_mask is not None else attn_output

        if self.checkpointing:
            return torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, key_padding_mask=key_padding_mask)

        return self.inner_attn(qkv, key_padding_mask=key_padding_mask)

    def _forward_cross_attn(
        self,
        x: torch.FloatTensor,
        past_key_values: Optional[Union[torch.Tensor, InferenceParams]],
        key_padding_mask: Optional[torch.BoolTensor],
        rotary_pos_emb: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        qkv = self.Wqkv(x)

        q = qkv[..., : self.n_head * self.head_dim]
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)

        kv = qkv[..., self.n_head * self.head_dim :]
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)

        if rotary_pos_emb is None:
            seqlen_offset = past_key_values.seqlen_offset if past_key_values is not None else 0
            causal = None if seqlen_offset == 0 else False
            if self.rotary_dim > 0:
                q, kv = self.rotary_emb(q, kv=kv, seqlen_offset=seqlen_offset)
        else:
            causal = False
            cos_pos, sin_pos = rotary_pos_emb
            q = _apply_rotary_emb(q, cos_pos, sin_pos)
            kv = _apply_rotary_emb_kv(kv, cos_pos, sin_pos)

        if past_key_values is not None:
            if type(past_key_values) is InferenceParams:
                kv = _update_kv_cache(kv, past_key_values, self.layer_idx)
            else:
                # kv.shape is [1, past + 1, 2, 32, 80]
                #print('kv.shape', kv.shape)
                #print('past_key_values.shape', past_key_values.shape)
                kv = torch.cat((past_key_values, kv), dim=1)

        if self.flash_attn:
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            seqlen_k = kv.shape[1]

            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = (
                None,
                None,
                None,
                None,
            )
            if key_padding_mask is not None:
                kv, _, cu_seqlens_k, max_seqlen_k = unpad_input(kv, key_padding_mask)

                if seqlen_q == 1:
                    key_padding_mask = torch.ones(batch_size, 1, device=q.device)
                elif seqlen_q != seqlen_k:
                    key_padding_mask = key_padding_mask[:, -seqlen_q:]

                q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, key_padding_mask)

            if self.checkpointing:
                attn_output = torch.utils.checkpoint.checkpoint(
                    self.inner_cross_attn,
                    q,
                    kv,
                    causal=causal,
                    cu_seqlens=cu_seqlens_q,
                    max_seqlen=max_seqlen_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_k=max_seqlen_k,
                )
            else:
                attn_output = self.inner_cross_attn(
                    q,
                    kv,
                    causal=causal,
                    cu_seqlens=cu_seqlens_q,
                    max_seqlen=max_seqlen_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_k=max_seqlen_k,
                )

            return (
                pad_input(attn_output, indices_q, batch_size, max_seqlen_q)
                if key_padding_mask is not None
                else attn_output
            )

        if self.checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self.inner_cross_attn,
                q,
                kv,
                key_padding_mask=key_padding_mask,
                causal=causal,
            )
        output = self.inner_cross_attn(q, kv, key_padding_mask=key_padding_mask, causal=causal, causal_mask=causal_mask)
        return output, kv

    def forward(
        self,
        x: torch.FloatTensor,
        past_key_values: Optional[InferenceParams] = None,
        attention_mask: Optional[Union[torch.LongTensor, torch.BoolTensor]] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        else:
            attention_mask = None

        attention_mask = None
        kv = None
        # MHA
        if self.n_head == self.n_head_kv:
            if past_key_values is None and False:
                # If `past_key_values` are not supplied, we run self-attention
                attn_output = self._forward_self_attn(x, attention_mask)
            else:
                # If `past_key_values` are supplied, it means that we might have cached values and
                # could take advantage of cross-attention
                attn_output, kv = self._forward_cross_attn(x, past_key_values, attention_mask, rotary_pos_emb, causal_mask)
        # MQA / GQA
        else:
            # Regardless of `past_key_values` being supplied or not, it always use cross-attention
            # because `q` and `kv` lengths might be different
            attn_output = self._forward_cross_attn(x, past_key_values, attention_mask)

        output = rearrange(attn_output, "... h d -> ... (h d)")
        output = self.out_proj(output)

        # return output if not self.return_residual else (output, x)
        return output, kv


class ParallelBlock(nn.Module):
    """Parallel block.

    This block applies parallel mixer and MLP layers to the input (used in GPT-J and CodeGen).

    """

    def __init__(
        self,
        config: PretrainedConfig,
        block_idx: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.block_idx = block_idx

        self.mixer = MHA(config, layer_idx=block_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.ln(hidden_states)

        attn_outputs, kv = self.mixer(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            causal_mask=causal_mask,
        )

        if isinstance(attn_outputs, tuple):
            attn_outputs = attn_outputs[0]

        # attn_outputs = self.resid_dropout(attn_outputs)
        # feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        return hidden_states, kv


class CausalLMHead(nn.Module):
    """Causal Language Modeling head.

    Reference:
        Improving Language Understanding by Generative Pre-Training.
        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.

    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.ln(hidden_states)
        logits = self.linear(hidden_states).to(torch.float32)
        return logits


class CausalLMLoss(nn.Module):
    """Causal Language Modeling loss.

    Reference:
        Improving Language Understanding by Generative Pre-Training.
        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.

    """

    def __init__(self, shift_labels: bool = True) -> None:
        super().__init__()

        self.shift_labels = shift_labels
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        if self.shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss


class PhiPreTrainedModel(PreTrainedModel):
    """Phi pre-trained model."""

    config_class = PhiConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["ParallelBlock"]

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear,)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[Union[torch.LongTensor, torch.BoolTensor]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if past_key_values is None or not (isinstance(past_key_values, InferenceParams)):
            past_key_values = InferenceParams(
                max_seqlen=self.config.n_positions,
                max_batch_size=input_ids.shape[0],
                seqlen_offset=0,
                batch_size_offset=0,
                key_value_memory_dict={},
                lengths_per_sample=None,
            )
        else:
            # Assume that `past_key_values` has cached all tokens up to the last token in `input_ids`
            past_key_values.seqlen_offset = input_ids.shape[1] - 1
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }


class PhiModel(PhiPreTrainedModel):
    """Phi model."""

    _keys_to_ignore_on_load_missing = [""]
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.mlp.(fc_in|fc_out)\.(weight|bias)"]

    def __init__(self, config: PhiConfig) -> None:
        super().__init__(config)

        self.embd = Embedding(config)
        self.h = nn.ModuleList([ParallelBlock(config, block_idx=i) for i in range(config.n_layer)])
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embd.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embd.wte = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        hidden_states = self.embd(input_ids)

        for layer in self.h:
            hidden_states, _ = layer(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )

        return hidden_states


class PhiForCausalLM(PhiPreTrainedModel):
    """Phi for Causal Language Modeling."""

    _keys_to_ignore_on_load_missing = [""]
    _keys_to_ignore_on_load_unexpected = [r"transformer\.h\.\d+\.mlp.(fc_in|fc_out)\.(weight|bias)"]

    def __init__(self, config: PhiConfig) -> None:
        super().__init__(config)

        self.transformer = PhiModel(config)
        self.lm_head = CausalLMHead(config)
        self.loss = CausalLMLoss()

        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head.linear

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head.linear = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss(lm_logits, labels)

        return CausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=past_key_values)
