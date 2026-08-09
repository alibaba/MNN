# SPDX-License-Identifier: Apache-2.0
"""Export the MiniMax-H3 video VAE decoder to ONNX, plus the decode plan the runtime executes.

The decoder is a non-causal ViT over latent voxels: one token per voxel, 36 blocks of 2048 channels, plus four
register tokens and a zero token that are dropped again before the patch projection. The graph exported here is
**one spatial tile of one temporal clip** -- a fixed shape, which is what the runtime needs anyway and what an
NPU backend requires.

Everything outside that tile is index arithmetic: the reference decodes in `tokens_chunk_size` chunks that
overlap because `token_drop` removed each encoded chunk's tail, cross-fades the overlap, tiles space with its
own overlaps and blends those too. Those schedules depend only on the latent shape, so they are computed here
and written to `h3_vae_plan.json` for the runtime to replay.

    python h3_vae_export.py --vae /path/to/MiniMax-H3/vae --output /path/to/onnx \\
        --num_latent_frames 17 --latent_height 16 --latent_width 28 --verify
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
from h3_fixture import compare, format_report  # noqa: E402
from h3_modules import apply_rotary_emb, rms_norm  # noqa: E402

OPSET = 17


class VaeAttention(nn.Module):
    """Full self-attention over the tile's tokens, in the shape MNN's fused Attention expects.

    The query/key norms have no learnable scale here, and the reference runs them -- and both block norms -- in
    float32 whatever the compute dtype is.
    """

    def __init__(self, dim, heads, head_dim, eps):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.eps = eps
        inner_dim = heads * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=True)

    def forward(self, hidden_states, cos, sin, mask):
        batch, tokens, _ = hidden_states.shape
        shape = (batch, tokens, self.heads, self.head_dim)
        query = rms_norm(self.to_q(hidden_states).view(shape), None, self.eps)
        key = rms_norm(self.to_k(hidden_states).view(shape), None, self.eps)
        value = self.to_v(hidden_states).view(shape)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / float(np.sqrt(self.head_dim))
        attention = torch.matmul((scores + mask).softmax(dim=-1), value)
        attention = attention.transpose(1, 2).reshape(batch, tokens, self.heads * self.head_dim)
        return self.to_out(attention)


class VaeBlock(nn.Module):
    """Pre-norm block with a learned per-channel scale on each branch."""

    def __init__(self, dim, heads, head_dim, ffn_dim, eps):
        super().__init__()
        self.eps = eps
        self.norm1 = nn.Parameter(torch.ones(dim))
        self.attn = VaeAttention(dim, heads, head_dim, eps)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.Parameter(torch.ones(dim))
        self.ff_proj = nn.Linear(dim, ffn_dim * 2, bias=True)
        self.ff_out = nn.Linear(ffn_dim, dim, bias=True)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states, cos, sin, mask):
        normed = rms_norm(hidden_states, self.norm1, self.eps)
        hidden_states = hidden_states + self.attn(normed, cos, sin, mask) * self.scale1
        normed = rms_norm(hidden_states, self.norm2, self.eps)
        value, gate = self.ff_proj(normed).chunk(2, dim=-1)
        return hidden_states + self.ff_out(value * nn.functional.silu(gate)) * self.scale2


class H3VaeTileDecoder(nn.Module):
    """`post_quant_conv` plus the ViT decoder over one fixed tile, ending in the unpatchified pixel block."""

    def __init__(self, config, latent_frames, latent_height, latent_width):
        super().__init__()
        self.latent_channels = config["latent_channels"]
        self.latent_frames = latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.patch = config["spatial_downsample_factors_product"]
        self.patch_t = config["temporal_compression_ratio"]
        self.out_channels = config["out_channels"]
        self.num_register_tokens = config["decoder_num_register_tokens"]
        eps = config["decoder_norm_eps"]
        heads = config["decoder_num_attention_heads"]
        head_dim = config["decoder_attention_head_dim"]
        dim = heads * head_dim

        # 1x1x1 over the latent, i.e. a per-voxel linear map, so it folds into the token projection's input.
        self.post_quant = nn.Linear(self.latent_channels, self.latent_channels, bias=True)
        self.proj_in = nn.Linear(self.latent_channels, dim, bias=True)
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, dim))
        self.blocks = nn.ModuleList(
            [
                VaeBlock(dim, heads, head_dim, dim * config["decoder_ffn_mult"], eps)
                for _ in range(config["decoder_num_layers"])
            ]
        )
        self.norm_out_weight = nn.Parameter(torch.ones(dim))
        self.norm_out_bias = nn.Parameter(torch.zeros(dim))
        self.norm_out_eps = eps
        self.proj_out = nn.Linear(dim, self.out_channels * self.patch_t * self.patch * self.patch, bias=True)

    def forward(self, latent, cos, sin, mask):
        # (1, C, F, H, W) -> one token per voxel.
        tokens = latent.permute(0, 2, 3, 4, 1).reshape(1, -1, self.latent_channels)
        hidden_states = self.proj_in(self.post_quant(tokens))
        num_patches = hidden_states.shape[1]
        # The register tokens and a zero token ride along at position 0 and are dropped before the projection.
        suffix = torch.cat([self.register_tokens, torch.zeros_like(hidden_states[:, :1])], dim=1)
        hidden_states = torch.cat([hidden_states, suffix], dim=1)

        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin, mask)

        # This one is a LayerNorm, not RMSNorm: it subtracts the mean.
        mean = hidden_states.mean(-1, keepdim=True)
        centered = hidden_states - mean
        variance = centered.pow(2).mean(-1, keepdim=True)
        hidden_states = centered * torch.rsqrt(variance + self.norm_out_eps)
        hidden_states = hidden_states * self.norm_out_weight + self.norm_out_bias

        hidden_states = self.proj_out(hidden_states)[:, :num_patches]
        hidden_states = hidden_states.view(
            1,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            self.out_channels,
            self.patch_t,
            self.patch,
            self.patch,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return hidden_states.reshape(
            1,
            self.out_channels,
            self.latent_frames * self.patch_t,
            self.latent_height * self.patch,
            self.latent_width * self.patch,
        )


def tile_rope(latent_frames, latent_height, latent_width, head_dim, rope_dim_ratio, theta, num_register_tokens):
    """The decoder's rotary tables: coordinates normalized to `[-1, 1)` per axis and scaled by `2 * pi`.

    The register tokens and the zero token all sit at position 0, so their angles are zero.
    """
    dim = int(head_dim * rope_dim_ratio)
    inv_freq = 1.0 / theta ** np.arange(0, 1, 2 * 3 / dim, dtype=np.float32)
    grids = [
        2.0 * (np.arange(0.5, size, dtype=np.float32) / size) - 1.0
        for size in (latent_frames, latent_height, latent_width)
    ]
    position_ids = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)
    position_ids = np.concatenate([position_ids, np.zeros((num_register_tokens + 1, 3), dtype=np.float32)])
    angles = 2.0 * math.pi * position_ids[:, :, None] * inv_freq[None, None, :]
    angles = angles.reshape(position_ids.shape[0], -1)
    angles = np.concatenate([angles, angles], axis=-1)
    return np.cos(angles), np.sin(angles)


def split_tiles(length, tile_size, min_overlap, ratio):
    """Lay `tile_size` tiles over `length`, distributing slack over the overlaps in whole `ratio` steps."""
    if tile_size >= length:
        return [0], [length], []
    num_tiles = math.ceil(length / tile_size)
    while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
        num_tiles += 1
    overlaps = [min_overlap] * (num_tiles - 1)
    remaining = tile_size * num_tiles - sum(overlaps) - length
    for index in range(remaining // ratio):
        overlaps[index % (num_tiles - 1)] += ratio
    starts = [0]
    for index in range(num_tiles - 1):
        starts.append(starts[-1] + tile_size - overlaps[index])
    return starts, [tile_size] * num_tiles, overlaps


def decode_plan(config, num_latent_frames, latent_height, latent_width, tile_size, min_overlap):
    """Everything the runtime needs to replay the reference's chunking and tiling."""
    ratio = config["spatial_downsample_factors_product"]
    temporal_ratio = config["temporal_compression_ratio"]
    clip_length = config["clip_length"]
    token_drop = config["token_drop"]

    frame_pre_padding = (-clip_length) % temporal_ratio
    tokens_chunk_size = math.ceil(clip_length / temporal_ratio)
    token_overlap = (-token_drop) % tokens_chunk_size
    frame_overlap = max(token_overlap * temporal_ratio - frame_pre_padding, 0)
    chunk_num_frames = tokens_chunk_size * temporal_ratio

    num_tokens = num_latent_frames + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)

    pad_frames = 0
    if pad_tokens > 0:
        intra_tail = clip_length % temporal_ratio
        before_pad = num_latent_frames
        pad_frames = sum(
            intra_tail if intra_tail and (before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
            for k in range(pad_tokens)
        )

    y_starts, _, y_overlaps = split_tiles(latent_height * ratio, tile_size, min_overlap, ratio)
    x_starts, _, x_overlaps = split_tiles(latent_width * ratio, tile_size, min_overlap, ratio)

    return {
        "latent_channels": config["latent_channels"],
        "latents_mean": config["latents_mean"],
        "latents_std": config["latents_std"],
        "video_patch_size": [1, 2, 2],
        "spatial_ratio": ratio,
        "temporal_ratio": temporal_ratio,
        "num_latent_frames": num_latent_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "tokens_chunk_size": tokens_chunk_size,
        "token_overlap": token_overlap,
        "token_drop": token_drop,
        "frame_pre_padding": frame_pre_padding,
        "frame_overlap": frame_overlap,
        "chunk_num_frames": chunk_num_frames,
        "num_chunks": num_chunks,
        "pad_tokens": pad_tokens,
        "pad_frames": pad_frames,
        "tile_latent_frames": tokens_chunk_size + token_overlap,
        "tile_latent_height": min(tile_size, latent_height * ratio) // ratio,
        "tile_latent_width": min(tile_size, latent_width * ratio) // ratio,
        "y_starts": y_starts,
        "y_overlaps": y_overlaps,
        "x_starts": x_starts,
        "x_overlaps": x_overlaps,
    }


def load_decoder_state_dict(vae_path, config, dtype=torch.float32):
    """Map the checkpoint onto `H3VaeTileDecoder`, reading only the decoder side."""
    from safetensors import safe_open

    directory = Path(vae_path)
    index = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    mapping = {
        "post_quant_conv.weight": "post_quant.weight",
        "post_quant_conv.bias": "post_quant.bias",
        "decoder.proj_in.weight": "proj_in.weight",
        "decoder.proj_in.bias": "proj_in.bias",
        "decoder.register_tokens": "register_tokens",
        "decoder.norm_out.weight": "norm_out_weight",
        "decoder.norm_out.bias": "norm_out_bias",
        "decoder.proj_out.weight": "proj_out.weight",
        "decoder.proj_out.bias": "proj_out.bias",
    }
    for index_ in range(config["decoder_num_layers"]):
        source = f"decoder.transformer_blocks.{index_}"
        target = f"blocks.{index_}"
        mapping[f"{source}.norm1.weight"] = f"{target}.norm1"
        mapping[f"{source}.norm2.weight"] = f"{target}.norm2"
        mapping[f"{source}.scale1"] = f"{target}.scale1"
        mapping[f"{source}.scale2"] = f"{target}.scale2"
        for part in ("to_q", "to_k", "to_v"):
            mapping[f"{source}.attn.{part}.weight"] = f"{target}.attn.{part}.weight"
            mapping[f"{source}.attn.{part}.bias"] = f"{target}.attn.{part}.bias"
        mapping[f"{source}.attn.to_out.0.weight"] = f"{target}.attn.to_out.weight"
        mapping[f"{source}.attn.to_out.0.bias"] = f"{target}.attn.to_out.bias"
        mapping[f"{source}.ff.net.0.proj.weight"] = f"{target}.ff_proj.weight"
        mapping[f"{source}.ff.net.0.proj.bias"] = f"{target}.ff_proj.bias"
        mapping[f"{source}.ff.net.2.weight"] = f"{target}.ff_out.weight"
        mapping[f"{source}.ff.net.2.bias"] = f"{target}.ff_out.bias"

    per_shard = {}
    for key in mapping:
        if key not in weight_map:
            raise KeyError(f"{key} is not in the VAE checkpoint index")
        per_shard.setdefault(weight_map[key], []).append(key)

    state_dict = {}
    for shard, keys in sorted(per_shard.items()):
        with safe_open((directory / shard).as_posix(), framework="pt") as handle:
            for key in keys:
                tensor = handle.get_tensor(key)
                # post_quant_conv is a 1x1x1 Conv3d; as a per-voxel linear map it is the same weight.
                if key.startswith("post_quant_conv.weight"):
                    tensor = tensor.reshape(tensor.shape[0], tensor.shape[1])
                state_dict[mapping[key]] = tensor.to(dtype)
    return state_dict


def vae_config(vae_path):
    config = json.loads((Path(vae_path) / "config.json").read_text())
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    config["spatial_downsample_factors_product"] = int(np.prod(config["spatial_downsample_factors"]))
    config["temporal_compression_ratio"] = int(np.prod(config["temporal_downsample_factors"]))
    return config


def main():
    parser = argparse.ArgumentParser(description="Export the MiniMax-H3 video VAE decoder.")
    parser.add_argument("--vae", required=True, help="MiniMax-H3 vae directory.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_latent_frames", type=int, required=True)
    parser.add_argument("--latent_height", type=int, required=True)
    parser.add_argument("--latent_width", type=int, required=True)
    parser.add_argument("--tile", type=int, default=256, help="Spatial tile size in pixels.")
    parser.add_argument("--tile_overlap", type=int, default=64)
    parser.add_argument("--verify", action="store_true", help="Check one tile against the reference decoder.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip_onnx", action="store_true")
    args = parser.parse_args()

    config = vae_config(args.vae)
    plan = decode_plan(
        config, args.num_latent_frames, args.latent_height, args.latent_width, args.tile, args.tile_overlap
    )
    print(
        f"plan: {plan['num_chunks']} temporal chunk(s) x "
        f"{len(plan['y_starts']) * len(plan['x_starts'])} spatial tile(s), tile latent "
        f"{plan['tile_latent_frames']}x{plan['tile_latent_height']}x{plan['tile_latent_width']} -> "
        f"{plan['tile_latent_frames'] * plan['temporal_ratio']}x"
        f"{plan['tile_latent_height'] * plan['spatial_ratio']}x{plan['tile_latent_width'] * plan['spatial_ratio']}"
    )

    device = torch.device(args.device)
    decoder = H3VaeTileDecoder(
        config, plan["tile_latent_frames"], plan["tile_latent_height"], plan["tile_latent_width"]
    )
    decoder.load_state_dict(load_decoder_state_dict(args.vae, config), strict=True)
    decoder = decoder.to(device).eval()
    parameters = sum(p.numel() for p in decoder.parameters())
    print(f"decoder: {parameters / 1e9:.2f}B parameters")

    cos, sin = tile_rope(
        plan["tile_latent_frames"],
        plan["tile_latent_height"],
        plan["tile_latent_width"],
        config["decoder_attention_head_dim"],
        config["decoder_rope_dim_ratio"],
        config["decoder_rope_theta"],
        config["decoder_num_register_tokens"],
    )
    num_tokens = cos.shape[0]
    cos_t = torch.from_numpy(cos)[None, :, None, :].to(device)
    sin_t = torch.from_numpy(sin)[None, :, None, :].to(device)
    mask_t = torch.zeros(1, 1, num_tokens, num_tokens, device=device)
    latent_shape = (
        1,
        config["latent_channels"],
        plan["tile_latent_frames"],
        plan["tile_latent_height"],
        plan["tile_latent_width"],
    )
    print(f"tile graph: {num_tokens} tokens, latent {latent_shape}")

    if args.verify:
        from diffusers import AutoencoderKLMiniMaxH3

        reference = AutoencoderKLMiniMaxH3.from_pretrained(args.vae, torch_dtype=torch.float32).to(device).eval()
        generator = torch.Generator().manual_seed(0)
        latent = torch.randn(latent_shape, generator=generator).to(device)
        with torch.no_grad():
            mine = decoder(latent, cos_t, sin_t, mask_t)
            theirs = reference.decoder(reference.post_quant_conv(latent))
        rows = [compare("tile decode vs reference", mine.cpu().numpy(), theirs.cpu().numpy())]
        print(format_report(rows))
        if not rows[0]["ok"]:
            raise SystemExit("the export-friendly tile decoder does not match the reference")
        del reference

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "h3_vae_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rope = np.concatenate([np.ascontiguousarray(cos).reshape(-1), np.ascontiguousarray(sin).reshape(-1)])
    (output / "h3_vae_rope.bin").write_bytes(rope.astype(np.float32).tobytes())
    print(f"wrote h3_vae_plan.json and h3_vae_rope.bin ({rope.nbytes / 1e6:.1f} MB)")

    if not args.skip_onnx:
        directory = output / "h3_vae_decoder"
        directory.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            decoder,
            (torch.zeros(latent_shape), cos_t.cpu(), sin_t.cpu(), mask_t.cpu()),
            (directory / "h3_vae_decoder.onnx").as_posix(),
            input_names=["latent", "rope_cos", "rope_sin", "mask"],
            output_names=["pixels"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
        size = sum(item.stat().st_size for item in directory.iterdir() if item.is_file())
        print(f"wrote h3_vae_decoder/ ({size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
