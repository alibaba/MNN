# SPDX-License-Identifier: Apache-2.0
"""Dump golden MiniMax-H3 fixtures from the reference diffusers implementation.

The block forward is written out step by step here rather than read off diffusers' module, so every
intermediate the MNN port has to reproduce -- the modulated norms, q/k/v, the per-head q/k norms, the rotated
q/k, the attention output, the gated residuals and the SwiGLU feed-forward -- lands in the fixture. The
decomposition is then checked against the real `MiniMaxH3TransformerBlock`, so the trace is a verified spec
rather than a second implementation.

    # single real block, the 448x256 / 56-frame layout, plus a truncated full-model trace
    python h3_reference.py --model_path /path/to/MiniMax-H3 --output /path/to/fixtures/h3_block0

    # tiny random-weight model, small enough to commit as a test asset
    python h3_reference.py --tiny --output ../../../../test/resource/minimax_h3
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
import h3_layout  # noqa: E402
from h3_fixture import FixtureWriter, compare, format_report  # noqa: E402

from diffusers.models.transformers.transformer_minimax_h3 import (  # noqa: E402
    MiniMaxH3Transformer3DModel,
    _apply_rotary_emb,
)


TINY_CONFIG = {
    "num_attention_heads": 3,
    "attention_head_dim": 32,
    "hidden_size": 64,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 96,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 48,
    "freq_dim": 32,
    "time_embed_hidden_dim": 64,
    "time_embed_dim": 40,
    "rope_freq_dim": 4,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
}


def rms_norm(hidden_states, weight, eps):
    """`nn.RMSNorm`: the statistic is float32, the scale is applied at the input dtype."""
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normalized = hidden_states.to(torch.float32) * torch.rsqrt(variance + eps)
    return (normalized.to(hidden_states.dtype) * weight) if weight is not None else normalized.to(hidden_states.dtype)


class BlockTrace:
    """One `MiniMaxH3TransformerBlock` forward, decomposed into the tensors the MNN port is compared against."""

    def __init__(self, block, hidden_size, heads, head_dim, norm_eps, qk_norm_eps):
        self.block = block
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.qk_norm_eps = qk_norm_eps

    def modulation(self, temb):
        """The six AdaLN parameters of this block, as the runtime precomputes them per (timestep, modality)."""
        linear = self.block.adaln_proj.linear
        projected = linear(nn.functional.silu(temb).to(linear.weight.dtype))
        return projected.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)

    def __call__(self, hidden_states, temb, adaln_indices, cos, sin, tensors, prefix):
        attn = self.block.attn
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(temb)
        # The table the runtime precomputes offline and feeds the block graph, one row per (timestep, modality).
        tensors[f"{prefix}.adaln"] = torch.stack(
            [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        )

        residual = hidden_states
        normed = rms_norm(hidden_states, self.block.norm1.weight, self.norm_eps)
        tensors[f"{prefix}.norm1"] = normed
        normed = normed * (1.0 + scale_msa.index_select(0, adaln_indices)) + shift_msa.index_select(0, adaln_indices)
        tensors[f"{prefix}.norm1_modulated"] = normed

        query = attn.to_q(normed).unflatten(-1, (self.heads, -1))
        key = attn.to_k(normed).unflatten(-1, (self.heads, -1))
        value = attn.to_v(normed).unflatten(-1, (self.heads, -1))
        tensors[f"{prefix}.query"] = query
        tensors[f"{prefix}.key"] = key
        tensors[f"{prefix}.value"] = value

        query = rms_norm(query, attn.norm_q.weight, self.qk_norm_eps)
        key = rms_norm(key, attn.norm_k.weight, self.qk_norm_eps)
        tensors[f"{prefix}.query_normed"] = query
        tensors[f"{prefix}.key_normed"] = key

        query = _apply_rotary_emb(query, cos, sin)
        key = _apply_rotary_emb(key, cos, sin)
        tensors[f"{prefix}.query_rope"] = query
        tensors[f"{prefix}.key_rope"] = key

        # Full non-causal self-attention over one document. Computed in float32 so the fixture is a clean target
        # rather than a record of one backend's accumulation order.
        scale = 1.0 / np.sqrt(self.head_dim)
        query_t = query.transpose(1, 2).to(torch.float32)
        key_t = key.transpose(1, 2).to(torch.float32)
        value_t = value.transpose(1, 2).to(torch.float32)
        scores = torch.matmul(query_t, key_t.transpose(-1, -2)) * scale
        probabilities = scores.softmax(dim=-1)
        attention = torch.matmul(probabilities, value_t).transpose(1, 2).to(query.dtype)
        tensors[f"{prefix}.attn_probs_row0"] = probabilities[:, :, 0]
        tensors[f"{prefix}.attn_out_heads"] = attention

        attention = attn.to_out[0](attention.flatten(2, 3).type_as(query))
        tensors[f"{prefix}.attn_proj"] = attention
        hidden_states = residual + gate_msa.index_select(0, adaln_indices) * attention
        tensors[f"{prefix}.after_attn"] = hidden_states

        residual = hidden_states
        normed = rms_norm(hidden_states, self.block.norm2.weight, self.norm_eps)
        normed = normed * (1.0 + scale_mlp.index_select(0, adaln_indices)) + shift_mlp.index_select(0, adaln_indices)
        tensors[f"{prefix}.norm2_modulated"] = normed

        # SwiGLU: the first half of the projection is the value, the second half the gate.
        projected = self.block.ff.net[0].proj(normed)
        value_half, gate_half = projected.chunk(2, dim=-1)
        activated = value_half * nn.functional.silu(gate_half)
        tensors[f"{prefix}.ff_gated"] = activated
        feed_forward = self.block.ff.net[2](activated)
        tensors[f"{prefix}.ff_out"] = feed_forward

        hidden_states = residual + gate_mlp.index_select(0, adaln_indices) * feed_forward
        tensors[f"{prefix}.output"] = hidden_states
        return hidden_states


def load_transformer(model_path, num_layers, dtype, device):
    """Load the checkpoint with the block stack truncated to `num_layers`, keeping peak host memory bounded."""
    from safetensors import safe_open

    directory = Path(model_path)
    config = json.loads((directory / "config.json").read_text())
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    if num_layers is not None:
        config["num_layers"] = num_layers

    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel.from_config(config)

    index = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())
    wanted = set(name for name, _ in model.named_parameters()) | set(
        name for name, _ in model.named_buffers() if name != "rope.inv_freq"
    )
    per_shard = {}
    for name, shard in index["weight_map"].items():
        if name in wanted:
            per_shard.setdefault(shard, []).append(name)

    keep_fp32 = tuple(MiniMaxH3Transformer3DModel._keep_in_fp32_modules)
    state_dict = {}
    for shard, names in sorted(per_shard.items()):
        with safe_open((directory / shard).as_posix(), framework="pt") as handle:
            for name in names:
                target = torch.float32 if any(pattern in name for pattern in keep_fp32) else dtype
                state_dict[name] = handle.get_tensor(name).to(target)

    missing = wanted - set(state_dict)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} tensor(s), first: {sorted(missing)[:3]}")
    model.load_state_dict(state_dict, assign=True)
    del state_dict
    # `rope.inv_freq` is computed, not loaded, so it is still a meta tensor; rebuild it for real.
    model.rope = type(model.rope)(rope_freq_dim=config["rope_freq_dim"], rope_theta=config["rope_theta"])
    return model.to(device).eval(), config


def build_tiny(dtype, device, seed=0):
    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel.from_config(dict(TINY_CONFIG))
    for name, parameter in model.named_parameters():
        # Real H3 norms sit near 1 and the AdaLN projection near 0; mirror that so the tiny model is not degenerate.
        if "norm" in name and parameter.ndim == 1 and "linear" not in name:
            nn.init.normal_(parameter, mean=1.0, std=0.02)
        elif "adaln_proj" in name:
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        elif parameter.ndim >= 2:
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(parameter)
    keep_fp32 = tuple(MiniMaxH3Transformer3DModel._keep_in_fp32_modules)
    for name, module in model.named_modules():
        if not any(pattern in name for pattern in keep_fp32) and name:
            for parameter_name, parameter in list(module.named_parameters(recurse=False)):
                module._parameters[parameter_name] = nn.Parameter(parameter.data.to(dtype), requires_grad=False)
    return model.to(device).eval(), dict(TINY_CONFIG)


def main():
    parser = argparse.ArgumentParser(description="Dump golden MiniMax-H3 reference fixtures.")
    parser.add_argument("--model_path", help="MiniMax-H3 transformer directory, e.g. <repo>/transformer.")
    parser.add_argument("--tiny", action="store_true", help="Use a tiny random-weight model instead.")
    parser.add_argument("--output", required=True, help="Fixture directory to write.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num_frames", type=int, default=56)
    parser.add_argument("--num_text_tokens", type=int, default=37)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--step", type=int, default=0, help="Which schedule step the fixture is taken at.")
    parser.add_argument("--layers", type=int, default=1, help="Blocks to keep; the fixture traces all of them.")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keyframe", action="store_true", help="Anchor a first-frame keyframe (fl2va).")
    args = parser.parse_args()

    if not args.tiny and not args.model_path:
        parser.error("pass either --model_path or --tiny")

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    if args.tiny:
        model, config = build_tiny(dtype, device, args.seed)
    else:
        model, config = load_transformer(args.model_path, args.layers, dtype, device)

    text_token_tags = np.full(args.num_text_tokens, h3_layout.TEXT_TAG, dtype=np.int64)
    layout = h3_layout.H3Layout(
        args.height,
        args.width,
        args.num_frames,
        text_token_tags,
        patch_size=tuple(config["patch_size"]),
        keyframe_anchors=("first",) if args.keyframe else (),
    )
    schedule = h3_layout.H3Schedule(args.num_inference_steps, layout)
    unique_timesteps, timestep_indices = schedule.row_timestep_plan[args.step]

    generator = torch.Generator().manual_seed(args.seed)
    video_patch_dim = config["in_channels"] * int(np.prod(config["patch_size"]))
    num_video_rows = int(layout.video_indices.shape[0])
    num_audio_rows = int(layout.audio_indices.shape[0])

    video_rows = torch.randn(1, num_video_rows, video_patch_dim, generator=generator, dtype=torch.float32)
    audio_rows = torch.randn(1, num_audio_rows, config["audio_in_channels"], generator=generator, dtype=torch.float32)
    text_rows = torch.randn(1, args.num_text_tokens, config["text_dim"], generator=generator, dtype=torch.float32)

    forward_kwargs = {
        "hidden_states": video_rows.to(device),
        "audio_hidden_states": audio_rows.to(device),
        "encoder_hidden_states": text_rows.to(device, dtype),
        "timestep": torch.from_numpy(unique_timesteps).to(device),
        "timestep_indices": torch.from_numpy(timestep_indices).to(device),
        "token_tags": torch.from_numpy(layout.token_tags).to(device),
        "position_ids": torch.from_numpy(layout.position_ids).to(device),
        "video_indices": torch.from_numpy(layout.video_indices).to(device),
        "audio_indices": torch.from_numpy(layout.audio_indices).to(device),
        "text_indices": torch.from_numpy(layout.text_indices).to(device),
    }

    tensors = {}
    with torch.no_grad():
        cos, sin = model.rope(forward_kwargs["position_ids"])
        tensors["rope_cos"] = cos
        tensors["rope_sin"] = sin

        temb = model.time_embedder(model.time_proj(forward_kwargs["timestep"]).to(torch.float32))
        tensors["temb"] = temb

        adaln_indices = forward_kwargs["timestep_indices"] * h3_layout.MODALITY_NUM + forward_kwargs["token_tags"]
        tensors["adaln_indices"] = adaln_indices

        video_embeds = model.proj_in(video_rows.to(device))
        audio_embeds = model.audio_proj_in(audio_rows.to(device))
        text_embeds = model.context_embedder(forward_kwargs["encoder_hidden_states"])
        tensors["video_embeds"] = video_embeds
        tensors["audio_embeds"] = audio_embeds
        tensors["text_embeds_pre_refiner"] = text_embeds

        refined = model.token_refiner(text_embeds)
        tensors["text_embeds"] = refined

        packed = refined.new_zeros((1, layout.sequence_length, config["hidden_size"]))
        packed = packed.index_copy(1, forward_kwargs["text_indices"], refined)
        packed = packed.index_copy(1, forward_kwargs["video_indices"], video_embeds.to(refined.dtype))
        packed = packed.index_copy(1, forward_kwargs["audio_indices"], audio_embeds.to(refined.dtype))
        tensors["packed_input"] = packed

        hidden_states = packed
        for index, block in enumerate(model.transformer_blocks):
            trace = BlockTrace(
                block,
                config["hidden_size"],
                config["num_attention_heads"],
                config["attention_head_dim"],
                config["norm_eps"],
                config["qk_norm_eps"],
            )
            reference = block(hidden_states, temb, adaln_indices, (cos, sin))
            hidden_states = trace(hidden_states, temb, adaln_indices, cos, sin, tensors, f"block{index}")
            # The trace only reorders the reference's own ops, but it runs attention in float32 while diffusers
            # dispatches a bfloat16 kernel. H3 activations carry outliers hundreds of times their RMS, so the
            # agreement is measured by direction and relative energy rather than by an absolute bound.
            row = compare(
                f"block{index} trace vs diffusers",
                hidden_states.float().numpy(),
                reference.float().numpy(),
                cos_threshold=0.9999,
            )
            relative_rms = row["rms"] / row["ref_rms"] if row["ref_rms"] > 0 else 0.0
            print(f"block {index}: cosine={row['cosine']:.8f} rel_rms={relative_rms:.3e}")
            if not row["ok"] or relative_rms > 5e-3:
                raise RuntimeError(f"block {index} trace diverges from diffusers: {row}")
            hidden_states = reference

        normed_out = model.norm_out(hidden_states, temb, forward_kwargs["timestep_indices"])
        tensors["norm_out"] = normed_out
        head_linear = model.norm_out.linear
        head_shift, head_scale = head_linear(
            nn.functional.silu(temb).to(head_linear.weight.dtype)
        ).chunk(2, dim=-1)
        # One row per distinct timestep -- the final norm carries no modality tag.
        tensors["norm_out_adaln"] = torch.stack([head_shift, head_scale])
        normed_out = normed_out.to(model.proj_out.weight.dtype)
        tensors["video_velocity"] = model.proj_out(normed_out).index_select(1, forward_kwargs["video_indices"])
        tensors["audio_velocity"] = model.audio_proj_out(normed_out).index_select(1, forward_kwargs["audio_indices"])

        full = model(**forward_kwargs, return_dict=False)

    rows = [
        compare("video_velocity vs model()", tensors["video_velocity"].float().numpy(), full[0].float().cpu().numpy()),
        compare("audio_velocity vs model()", tensors["audio_velocity"].float().numpy(), full[1].float().cpu().numpy()),
    ]
    print(format_report(rows))
    if not all(row["ok"] for row in rows):
        raise RuntimeError("the traced forward does not reproduce the reference model output")

    writer = FixtureWriter(
        args.output,
        metadata={
            "source": "tiny" if args.tiny else str(args.model_path),
            "config": config,
            "dtype": args.dtype,
            "seed": args.seed,
            "step": args.step,
            "num_inference_steps": args.num_inference_steps,
            "height": layout.height,
            "width": layout.width,
            "num_frames": layout.num_frames,
            "num_latent_frames": layout.num_latent_frames,
            "latent_height": layout.latent_height,
            "latent_width": layout.latent_width,
            "num_audio_latents": layout.num_audio_latents,
            "num_text_tokens": layout.num_text_tokens,
            "sequence_length": layout.sequence_length,
            "num_condition_video_rows": layout.num_condition_video_rows,
            "keyframe_anchors": list(layout.keyframe_anchors),
            "video_sigmas": schedule.video_sigmas.tolist(),
            "audio_sigmas": schedule.audio_sigmas.tolist(),
            "timesteps": schedule.timesteps.tolist(),
            "audio_timesteps": schedule.audio_timesteps.tolist(),
        },
    )
    writer.add("input_video_rows", video_rows[0])
    writer.add("input_audio_rows", audio_rows[0])
    writer.add("input_text_rows", text_rows[0])
    writer.add("timestep", torch.from_numpy(unique_timesteps))
    writer.add("timestep_indices", torch.from_numpy(timestep_indices))
    writer.add("token_tags", torch.from_numpy(layout.token_tags))
    writer.add("position_ids", torch.from_numpy(layout.position_ids))
    writer.add("video_indices", torch.from_numpy(layout.video_indices))
    writer.add("audio_indices", torch.from_numpy(layout.audio_indices))
    writer.add("text_indices", torch.from_numpy(layout.text_indices))
    for name, tensor in tensors.items():
        writer.add(name, tensor[0] if tensor.ndim == 3 and tensor.shape[0] == 1 else tensor)
    manifest = writer.close()

    total = sum(np.prod(entry["shape"]) * (2 if "16" in entry["storage_dtype"] else 8 if "64" in entry["storage_dtype"] else 4) for entry in manifest["tensors"].values())
    print(f"\nwrote {len(manifest['tensors'])} tensors ({total / 1e6:.1f} MB) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
