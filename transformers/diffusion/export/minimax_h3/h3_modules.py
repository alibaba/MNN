# SPDX-License-Identifier: Apache-2.0
"""Export-friendly PyTorch re-implementation of the MiniMax-H3 DiT.

The reference `MiniMaxH3Transformer3DModel` is written for eager execution; three of its choices do not
survive an ONNX export that MNN can fold into its fused ops:

* `nn.RMSNorm` traces to a `SimplifiedLayerNormalization` node the MNN converter does not know. The explicit
  `Mul(x, Rsqrt(Add(ReduceMean(Square(x)), eps)))` decomposition followed by `Mul(gamma)` is what
  `FuseLayerNormRMS` / `FuseLayerNormRMSGamm` match.
* `dispatch_attention_fn` traces to whatever backend torch picked. `FuseAttention` matches an explicit
  `MatMul -> Div -> Add(mask) -> Softmax -> MatMul -> Transpose -> Reshape` chain, and the additive mask is
  mandatory: without it MNN's fused attention falls back to a lower-triangular mask, which would silently make
  H3 causal.
* `index_select` of the AdaLN table per row materializes six `(seq_len, hidden_size)` tensors per block. The
  packed sequence is grouped by `(timestep, modality)`, so the same modulation is applied segment-wise with a
  broadcast instead, which is what makes the block affordable on a phone.

The AdaLN projection is not exported at all. It consumes only the timestep embedding, so the whole 13B branch
is folded offline into a small per-(step, layer) modulation table -- see `h3_adaln.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# The LLM exporter's custom ops. `FusedAttention` traces to a zero-filled output and an
# `LlmExporter::FusedAttention` node, so the score tensor is never materialized: export memory stops depending
# on sequence length, which is what makes a 21763-row layout exportable at all.
sys.path.insert(0, (Path(__file__).resolve().parents[3] / "llm" / "export").as_posix())
from utils.custom_op import FusedAttention  # noqa: E402


def rms_norm(hidden_states, weight, eps):
    """RMSNorm in the decomposition `FuseLayerNormRMS` + `FuseLayerNormRMSGamm` fold into one MNN LayerNorm.

    `weight` may be None for the query/key norms of the VAE decoder, which have no learnable scale.
    """
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    normalized = hidden_states * torch.rsqrt(variance + eps)
    return normalized if weight is None else normalized * weight


def apply_rotary_emb(hidden_states, cos, sin):
    """Rotate the leading `2 * rotary_half` channels of every head, pass the rest through."""
    rotary_dim = cos.shape[-1]
    rotary = hidden_states[..., :rotary_dim]
    passthrough = hidden_states[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1)


def merge_runs(segments):
    """Collapse adjacent runs that address the same table row."""
    merged = []
    for length, row in segments:
        if merged and merged[-1][1] == row:
            merged[-1] = (merged[-1][0] + length, row)
        else:
            merged.append((int(length), int(row)))
    return merged


class SegmentModulation(nn.Module):
    """Apply per-row AdaLN parameters as a broadcast over each contiguous run of rows sharing a table row.

    `segments` is the run-length encoding of the packed sequence's `(timestep, modality)` tags, so a
    modulation is a broadcast over each run rather than a gather of two `(seq_len, hidden_size)` tensors.
    """

    def __init__(self, segments):
        super().__init__()
        segments = merge_runs(segments)
        self.lengths = [length for length, _ in segments]
        self.rows = [row for _, row in segments]

    def _map(self, hidden_states, apply):
        if len(self.rows) == 1:
            return apply(hidden_states, self.rows[0])
        pieces = []
        offset = 0
        for length, row in zip(self.lengths, self.rows):
            pieces.append(apply(hidden_states[:, offset : offset + length], row))
            offset += length
        return torch.cat(pieces, dim=1)

    def forward(self, hidden_states, shift, scale):
        return self._map(
            hidden_states, lambda piece, row: piece * (1.0 + scale[row : row + 1]) + shift[row : row + 1]
        )

    def gate(self, hidden_states, gate):
        return self._map(hidden_states, lambda piece, row: piece * gate[row : row + 1])


class H3Attention(nn.Module):
    """Full non-causal self-attention over one packed document.

    The attention itself is emitted as the LLM exporter's `FusedAttention` custom op, which becomes MNN's
    `OpType_Attention` with `kv_cache = false`. Two reasons over writing the matmul chain out and relying on
    the converter's `FuseAttention` pattern: tracing never allocates the `heads x seq x seq` score tensor, so
    export memory is independent of sequence length; and the op is emitted directly rather than pattern-matched,
    which is what silently failed for the SDPA form and for the VAE decoder's graph.
    """

    # `FusedAttention` traces to a zero-filled output, so it is only valid while exporting. Every numerical
    # gate -- the fixture parity, the self-test -- runs the real computation with this off.
    export_fused_attn = False

    def __init__(self, hidden_size, heads, head_dim, qk_norm_eps, name="attn"):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.qk_norm_eps = qk_norm_eps
        inner_dim = heads * head_dim
        self.to_q = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, inner_dim, bias=False)
        self.norm_q = nn.Parameter(torch.ones(head_dim))
        self.norm_k = nn.Parameter(torch.ones(head_dim))
        self.to_out = nn.Linear(inner_dim, hidden_size, bias=False)
        # H3 attends over one document with no cache, so `kv_cache` is false.
        self.fused_attn = FusedAttention(inner_dim, False, name, head_dim=head_dim)

    def forward(self, hidden_states, cos, sin, mask):
        batch, sequence_length, _ = hidden_states.shape
        shape = (batch, sequence_length, self.heads, self.head_dim)
        query = self.to_q(hidden_states).view(shape)
        key = self.to_k(hidden_states).view(shape)
        value = self.to_v(hidden_states).view(shape)

        query = rms_norm(query, self.norm_q, self.qk_norm_eps)
        key = rms_norm(key, self.norm_k, self.qk_norm_eps)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        # MNN's Attention takes B-S-H-D and returns (batch, seq, heads * head_dim). It derives the
        # 1/sqrt(head_dim) scale itself, and needs the materialized mask to stay non-causal.
        if self.export_fused_attn:
            attention = self.fused_attn(query, key, value, mask)
            # `FusedAttentionOp.symbolic` derives its output type from the rank-4 query and so annotates a
            # rank-4 result, while its forward returns rank 3. Reshaping to the real rank pins the type, without
            # which ONNX shape inference carries the wrong rank into whatever consumes the attention output.
            attention = attention.reshape(batch, sequence_length, self.heads * self.head_dim)
            return self.to_out(attention)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / float(np.sqrt(self.head_dim))
        attention = torch.matmul((scores + mask).softmax(dim=-1), value)
        attention = attention.transpose(1, 2).reshape(batch, sequence_length, self.heads * self.head_dim)
        return self.to_out(attention)


class H3FeedForward(nn.Module):
    """SwiGLU: the first half of the projection is the value, the second half the gate."""

    def __init__(self, hidden_size, ffn_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_size, ffn_dim * 2, bias=False)
        self.out = nn.Linear(ffn_dim, hidden_size, bias=False)

    def forward(self, hidden_states):
        value, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return self.out(value * torch.nn.functional.silu(gate))


class H3Block(nn.Module):
    def __init__(self, config, segments):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.norm_eps = config["norm_eps"]
        self.norm1 = nn.Parameter(torch.ones(hidden_size))
        self.norm2 = nn.Parameter(torch.ones(hidden_size))
        self.attn = H3Attention(
            hidden_size, config["num_attention_heads"], config["attention_head_dim"], config["qk_norm_eps"]
        )
        self.ff = H3FeedForward(hidden_size, config["ffn_dim"])
        self.modulation = SegmentModulation(segments)

    def forward(self, hidden_states, modulation, cos, sin, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

        normed = rms_norm(hidden_states, self.norm1, self.norm_eps)
        normed = self.modulation(normed, shift_msa, scale_msa)
        hidden_states = hidden_states + self.modulation.gate(self.attn(normed, cos, sin, mask), gate_msa)

        normed = rms_norm(hidden_states, self.norm2, self.norm_eps)
        normed = self.modulation(normed, shift_mlp, scale_mlp)
        return hidden_states + self.modulation.gate(self.ff(normed), gate_mlp)


class H3BlockGroup(nn.Module):
    """One sequential slice of the block stack, exported as a single graph.

    Splitting the 50 blocks keeps the ONNX export, the MNN weight file and -- on device -- the QNN context
    binary of each partition bounded, and lets the runtime hold the whole stack resident without ever building
    one 20B graph.
    """

    def __init__(self, config, segments, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([H3Block(config, segments) for _ in range(num_layers)])

    def forward(self, hidden_states, cos, sin, mask, *modulation):
        for index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, modulation[index * 6 : (index + 1) * 6], cos, sin, mask)
        return hidden_states


class H3TokenRefinerBlock(nn.Module):
    """Plain pre-norm block over the text stream: no AdaLN, no rotary embedding."""

    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.norm_eps = config["norm_eps"]
        self.norm1 = nn.Parameter(torch.ones(hidden_size))
        self.norm2 = nn.Parameter(torch.ones(hidden_size))
        self.attn = H3Attention(
            hidden_size, config["num_attention_heads"], config["attention_head_dim"], config["qk_norm_eps"]
        )
        self.ff = H3FeedForward(hidden_size, config["ffn_dim"])

    def forward(self, hidden_states, cos, sin, mask):
        hidden_states = hidden_states + self.attn(rms_norm(hidden_states, self.norm1, self.norm_eps), cos, sin, mask)
        return hidden_states + self.ff(rms_norm(hidden_states, self.norm2, self.norm_eps))


class H3Embed(nn.Module):
    """The three input projections, the text refiner, and the packing of the rows into the packed sequence.

    The packed layout is `[text | keyframe conditions | audio | target video]`, so the scatter the reference
    writes as three `index_copy` calls is a concatenation of contiguous slices here.
    """

    def __init__(self, config, num_condition_video_rows):
        super().__init__()
        patch = config["patch_size"]
        video_patch_dim = config["in_channels"] * patch[0] * patch[1] * patch[2]
        hidden_size = config["hidden_size"]
        self.num_condition_video_rows = int(num_condition_video_rows)
        self.proj_in = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(config["audio_in_channels"], hidden_size, bias=True)
        self.context_embedder = nn.Linear(config["text_dim"], hidden_size, bias=True)
        self.refiner_blocks = nn.ModuleList(
            [H3TokenRefinerBlock(config) for _ in range(config["num_refiner_layers"])]
        )
        self.final_norm = nn.Parameter(torch.ones(hidden_size))
        self.final_norm_eps = config["final_norm_eps"]
        # The refiner has no rotary embedding; a zero cos/sin pair keeps one attention implementation.
        rotary_half = 3 * config["rope_freq_dim"]
        self.register_buffer("refiner_cos", torch.ones(1, 1, 1, 2 * rotary_half), persistent=False)
        self.register_buffer("refiner_sin", torch.zeros(1, 1, 1, 2 * rotary_half), persistent=False)

    def forward(self, video_rows, audio_rows, text_rows, text_mask):
        text = self.context_embedder(text_rows)
        for block in self.refiner_blocks:
            text = block(text, self.refiner_cos, self.refiner_sin, text_mask)
        text = rms_norm(text, self.final_norm, self.final_norm_eps)

        video = self.proj_in(video_rows)
        audio = self.audio_proj_in(audio_rows)
        condition = self.num_condition_video_rows
        if condition:
            return torch.cat([text, video[:, :condition], audio, video[:, condition:]], dim=1)
        return torch.cat([text, audio, video], dim=1)


class H3Head(nn.Module):
    """Final norm and the two output heads.

    The reference runs both heads over every row and selects afterwards; slicing first is identical and skips
    two projections over the rows of the other modality.
    """

    def __init__(self, config, segments, num_text_tokens, num_audio_rows, num_condition_video_rows):
        super().__init__()
        patch = config["patch_size"]
        video_patch_dim = config["in_channels"] * patch[0] * patch[1] * patch[2]
        hidden_size = config["hidden_size"]
        self.final_norm_eps = config["final_norm_eps"]
        self.norm_out = nn.Parameter(torch.ones(hidden_size))
        self.proj_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(hidden_size, config["audio_in_channels"], bias=True)
        self.modulation = SegmentModulation(segments)
        self.num_text_tokens = int(num_text_tokens)
        self.num_audio_rows = int(num_audio_rows)
        self.num_condition_video_rows = int(num_condition_video_rows)

    def forward(self, hidden_states, shift, scale):
        normed = rms_norm(hidden_states, self.norm_out, self.final_norm_eps)
        normed = self.modulation(normed, shift, scale)

        condition_end = self.num_text_tokens + self.num_condition_video_rows
        audio_end = condition_end + self.num_audio_rows
        audio = normed[:, condition_end:audio_end]
        video = (
            torch.cat([normed[:, self.num_text_tokens : condition_end], normed[:, audio_end:]], dim=1)
            if self.num_condition_video_rows
            else normed[:, audio_end:]
        )
        return self.proj_out(video), self.audio_proj_out(audio)

