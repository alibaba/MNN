# SPDX-License-Identifier: Apache-2.0
"""Export the MiniMax-H3 text conditioner -- Qwen3-VL-32B's text stack -- to ONNX, then MNN.

MiniMax-H3 conditions on `hidden_states[50]` of its Qwen3-VL conditioner: the output of the 50th decoder layer,
*before* the model's final norm. Only those 50 layers matter, and for a text-only request neither the vision
tower nor the language-model head is ever touched, so this exports:

* `h3_text_embed.bin` -- the token embedding table, gathered on the host rather than run as a graph.
* `h3_text_layers_{g}.onnx` -- one partition of the 50 decoder layers.

Two things are baked rather than recomputed. The rotary tables are captured from the reference's own rotary
module, so Qwen3-VL's interleaved mrope is reproduced exactly instead of being reimplemented; and the causal
mask is a constant of the sequence length. Prompts shorter than `max_tokens` need no padding mask: attention is
causal, so a real token never attends to a padding slot, and the padding rows are discarded afterwards.

    python h3_encoder_export.py --text_encoder /path/to/MiniMax-H3/text_encoder \\
        --output /path/to/onnx --max_tokens 256 --layers_per_group 5 --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
from h3_fixture import compare, format_report  # noqa: E402
from h3_modules import rms_norm  # noqa: E402

OPSET = 17
# The hidden state MiniMax-H3 conditions on, i.e. how many decoder layers are needed.
H3_TEXT_ENCODER_LAYER = 50


def apply_rope(hidden_states, cos, sin):
    """Qwen's rotate-half convention over the whole head dimension."""
    first, second = hidden_states.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return hidden_states * cos + rotated * sin


class QwenAttention(nn.Module):
    """Grouped-query attention with per-head query/key norms, causal, no bias."""

    def __init__(self, config):
        super().__init__()
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.eps = config["rms_norm_eps"]
        hidden = config["hidden_size"]
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=False)
        self.q_norm = nn.Parameter(torch.ones(self.head_dim))
        self.k_norm = nn.Parameter(torch.ones(self.head_dim))

    def forward(self, hidden_states, cos, sin, mask):
        batch, tokens, _ = hidden_states.shape
        query = rms_norm(self.q_proj(hidden_states).view(batch, tokens, self.heads, self.head_dim), self.q_norm,
                         self.eps)
        key = rms_norm(self.k_proj(hidden_states).view(batch, tokens, self.kv_heads, self.head_dim), self.k_norm,
                       self.eps)
        value = self.v_proj(hidden_states).view(batch, tokens, self.kv_heads, self.head_dim)
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # Grouped queries: repeat each key/value head across its group.
        repeat = self.heads // self.kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

        scores = torch.matmul(query, key.transpose(-1, -2)) / float(np.sqrt(self.head_dim))
        attention = torch.matmul((scores + mask).softmax(dim=-1), value)
        attention = attention.transpose(1, 2).reshape(batch, tokens, self.heads * self.head_dim)
        return self.o_proj(attention)


class QwenLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config["rms_norm_eps"]
        hidden = config["hidden_size"]
        intermediate = config["intermediate_size"]
        self.input_norm = nn.Parameter(torch.ones(hidden))
        self.attn = QwenAttention(config)
        self.post_attn_norm = nn.Parameter(torch.ones(hidden))
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden_states, cos, sin, mask):
        normed = rms_norm(hidden_states, self.input_norm, self.eps)
        hidden_states = hidden_states + self.attn(normed, cos, sin, mask)
        normed = rms_norm(hidden_states, self.post_attn_norm, self.eps)
        gated = nn.functional.silu(self.gate_proj(normed)) * self.up_proj(normed)
        return hidden_states + self.down_proj(gated)


class QwenLayerGroup(nn.Module):
    """One sequential slice of the 50 decoder layers, exported as a single graph."""

    def __init__(self, config, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([QwenLayer(config) for _ in range(num_layers)])

    def forward(self, hidden_states, cos, sin, mask):
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, mask)
        return hidden_states


def layer_mapping(start, count):
    mapping = {}
    for offset in range(count):
        source = f"model.language_model.layers.{start + offset}"
        target = f"layers.{offset}"
        mapping[f"{source}.input_layernorm.weight"] = f"{target}.input_norm"
        mapping[f"{source}.post_attention_layernorm.weight"] = f"{target}.post_attn_norm"
        for part in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mapping[f"{source}.self_attn.{part}.weight"] = f"{target}.attn.{part}.weight"
        mapping[f"{source}.self_attn.q_norm.weight"] = f"{target}.attn.q_norm"
        mapping[f"{source}.self_attn.k_norm.weight"] = f"{target}.attn.k_norm"
        for part in ("gate_proj", "up_proj", "down_proj"):
            mapping[f"{source}.mlp.{part}.weight"] = f"{target}.{part}.weight"
    return mapping


def read_tensors(model_path, keys, dtype=None):
    from safetensors import safe_open

    directory = Path(model_path)
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    per_shard = {}
    for key in keys:
        if key not in weight_map:
            raise KeyError(f"{key} is not in the text encoder index")
        per_shard.setdefault(weight_map[key], []).append(key)
    tensors = {}
    for shard, shard_keys in sorted(per_shard.items()):
        with safe_open((directory / shard).as_posix(), framework="pt") as handle:
            for key in shard_keys:
                tensor = handle.get_tensor(key)
                tensors[key] = tensor if dtype is None else tensor.to(dtype)
    return tensors


def text_config(model_path):
    config = json.loads((Path(model_path) / "config.json").read_text())["text_config"]
    config.setdefault("head_dim", config["hidden_size"] // config["num_attention_heads"])
    return config


def capture_rope(model_path, config, max_tokens, device="cpu"):
    """Take the rotary tables from the reference's own rotary module.

    Qwen3-VL uses interleaved mrope over three position axes. For a text-only request all three carry the
    sequence position, but rather than argue that this collapses to plain rotary, the reference module is called
    and its output is what gets baked.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    from transformers import AutoConfig

    full = AutoConfig.from_pretrained(model_path)
    rotary = Qwen3VLTextRotaryEmbedding(full.text_config).to(device).eval()
    hidden = torch.zeros(1, max_tokens, config["hidden_size"], device=device)
    # (3, batch, seq): the temporal, height and width axes all carry the text position.
    position_ids = torch.arange(max_tokens, device=device).view(1, 1, -1).expand(3, 1, -1)
    with torch.no_grad():
        cos, sin = rotary(hidden, position_ids)
    return cos, sin


def main():
    parser = argparse.ArgumentParser(description="Export the MiniMax-H3 Qwen3-VL text conditioner.")
    parser.add_argument("--text_encoder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_tokens", type=int, default=256, help="Fixed sequence length of the exported graph.")
    parser.add_argument("--layers_per_group", type=int, default=5)
    parser.add_argument("--num_layers", type=int, default=H3_TEXT_ENCODER_LAYER)
    parser.add_argument("--only_group", type=int)
    parser.add_argument("--parts", default="embed,layers")
    parser.add_argument("--verify", action="store_true", help="Check the first partition against the reference.")
    parser.add_argument("--verify_tokens", type=int, default=16)
    args = parser.parse_args()

    config = text_config(args.text_encoder)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    parts = set(args.parts.split(","))
    hidden_size = config["hidden_size"]

    cos, sin = capture_rope(args.text_encoder, config, args.max_tokens)
    # (1, seq, dim) -> (1, seq, 1, dim) so it broadcasts over heads.
    cos_t = cos[:, :, None, :].float()
    sin_t = sin[:, :, None, :].float()
    # Causal: a token attends to itself and everything before it. Padding slots need no separate mask.
    mask = torch.full((1, 1, args.max_tokens, args.max_tokens), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    print(
        f"conditioner: {args.num_layers} of {config['num_hidden_layers']} layers, {args.max_tokens} tokens, "
        f"hidden {hidden_size}, rotary dim {cos_t.shape[-1]}"
    )

    group_bounds = [
        (start, min(args.layers_per_group, args.num_layers - start))
        for start in range(0, args.num_layers, args.layers_per_group)
    ]

    if "embed" in parts:
        key = "model.language_model.embed_tokens.weight"
        table = read_tensors(args.text_encoder, [key])[key]
        # bfloat16 keeps the table at the checkpoint's own precision and halves what the host has to hold.
        raw = table.to(torch.bfloat16).view(torch.uint16).numpy()
        (output / "h3_text_embed.bin").write_bytes(np.ascontiguousarray(raw).tobytes())
        print(f"  wrote h3_text_embed.bin: {tuple(table.shape)} bfloat16, {raw.nbytes / 1e9:.2f} GB")
        del table, raw

    if args.verify:
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

        start, count = group_bounds[0]
        mapping = layer_mapping(start, count)
        tensors = read_tensors(args.text_encoder, list(mapping), torch.float32)
        group = QwenLayerGroup(config, count)
        group.load_state_dict({name: tensors[key] for key, name in mapping.items()}, strict=True)
        group = group.eval()

        # Only the layers under test are materialized; the rest of the 32B conditioner is never read.
        print(f"  verifying layers {start}..{start + count - 1} against the reference")
        reference_config = AutoConfig.from_pretrained(args.text_encoder).text_config
        reference_layers = []
        for offset in range(count):
            with torch.device("meta"):
                layer = Qwen3VLTextDecoderLayer(reference_config, start + offset)
            state = {}
            for key, name in mapping.items():
                if name.startswith(f"layers.{offset}."):
                    state[key.split(f"layers.{start + offset}.", 1)[1]] = tensors[key]
            layer.load_state_dict(state, assign=True, strict=True)
            reference_layers.append(layer.eval())

        tokens = args.verify_tokens
        generator = torch.Generator().manual_seed(0)
        hidden = torch.randn(1, tokens, hidden_size, generator=generator)
        reference_cos = cos[:, :tokens].float()
        reference_sin = sin[:, :tokens].float()
        reference_mask = torch.triu(torch.full((1, 1, tokens, tokens), float("-inf")), diagonal=1)
        with torch.no_grad():
            theirs = hidden
            for layer in reference_layers:
                theirs = layer(
                    theirs, position_embeddings=(reference_cos, reference_sin), attention_mask=reference_mask
                )
                theirs = theirs[0] if isinstance(theirs, tuple) else theirs
            mine = group(hidden, reference_cos[:, :, None, :], reference_sin[:, :, None, :], reference_mask)
        rows = [compare(f"layers[{start}..{start + count - 1}]", mine.numpy(), theirs.numpy())]
        print(format_report(rows))
        if not rows[0]["ok"]:
            raise SystemExit("the export-friendly conditioner layers do not match the reference")
        del reference_layers, group, tensors

    if "layers" in parts:
        for index, (start, count) in enumerate(group_bounds):
            if args.only_group is not None and index != args.only_group:
                continue
            name = f"h3_text_layers_{index}"
            print(f"  {name}: layers {start}..{start + count - 1}")
            group = QwenLayerGroup(config, count)
            mapping = layer_mapping(start, count)
            tensors = read_tensors(args.text_encoder, list(mapping), torch.float32)
            group.load_state_dict({key_name: tensors[key] for key, key_name in mapping.items()}, strict=True)
            directory = output / name
            directory.mkdir(parents=True, exist_ok=True)
            torch.onnx.export(
                group.eval(),
                (torch.zeros(1, args.max_tokens, hidden_size), cos_t, sin_t, mask),
                (directory / f"{name}.onnx").as_posix(),
                input_names=["hidden", "rope_cos", "rope_sin", "mask"],
                output_names=["hidden_out"],
                opset_version=OPSET,
                do_constant_folding=True,
                dynamo=False,
            )
            size = sum(item.stat().st_size for item in directory.iterdir() if item.is_file())
            print(f"    wrote {name}/ ({size / 1e9:.2f} GB)")
            del group, tensors

    rope = np.concatenate(
        [
            np.ascontiguousarray(cos_t.numpy()).reshape(-1),
            np.ascontiguousarray(sin_t.numpy()).reshape(-1),
        ]
    )
    (output / "h3_text_rope.bin").write_bytes(rope.astype(np.float32).tobytes())
    manifest = {
        "hidden_size": hidden_size,
        "vocab_size": config["vocab_size"],
        "max_tokens": args.max_tokens,
        "num_layers": args.num_layers,
        "layers_per_group": args.layers_per_group,
        "rotary_dim": int(cos_t.shape[-1]),
        "encoder_layer": H3_TEXT_ENCODER_LAYER,
        "groups": [{"start": start, "num_layers": count} for start, count in group_bounds],
    }
    (output / "h3_text_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote h3_text_manifest.json and h3_text_rope.bin ({rope.nbytes / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
