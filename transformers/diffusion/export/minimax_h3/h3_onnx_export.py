# SPDX-License-Identifier: Apache-2.0
"""Export the MiniMax-H3 DiT to ONNX, partitioned so no single graph holds the whole 20B stack.

The DiT is emitted as three kinds of graph:

* `h3_embed` -- the three input projections, the text refiner and the packing into the packed sequence.
* `h3_blocks_{g}` -- one sequential slice of the block stack. Partitioning keeps the ONNX proto, the MNN
  weight file and, on device, the QNN context binary of each slice bounded, and lets the runtime hold the
  whole stack resident without ever finalizing one 20B graph.
* `h3_head` -- the final norm and the two output heads.

The AdaLN branch is not exported; `h3_adaln.py` folds it into a per-(step, layer) table that the block graphs
take as inputs. Shapes are static: the layout of one request is baked in, which is what QNN needs anyway and
what lets the AdaLN modulation be applied per contiguous run rather than gathered per row.

    python h3_onnx_export.py --model_path /path/to/MiniMax-H3/transformer \
        --output /path/to/onnx --num_text_tokens 37 --layers_per_group 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import h3_layout  # noqa: E402
from h3_adaln import compact_adaln, head_rows, head_segments  # noqa: E402
from h3_modules import H3Attention, H3BlockGroup, H3Embed, H3Head, merge_runs  # noqa: E402
from h3_weights import checkpoint_config, load_group_state_dict  # noqa: E402

OPSET = 17


def export(module, args, root, name, input_names, output_names):
    # Tensors above the 2 GB protobuf limit spill into sibling files named after the producing node, so each
    # model needs its own directory or two exports would overwrite each other's weights.
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.onnx"
    torch.onnx.export(
        module,
        args,
        path.as_posix(),
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    size = sum(item.stat().st_size for item in directory.iterdir() if item.is_file())
    print(f"  wrote {name}/ ({size / 1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Export the MiniMax-H3 DiT to ONNX.")
    parser.add_argument("--model_path", required=True, help="MiniMax-H3 transformer directory.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num_frames", type=int, default=56)
    parser.add_argument("--num_text_tokens", type=int, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--keyframe", action="store_true")
    parser.add_argument("--layers_per_group", type=int, default=5)
    parser.add_argument(
        "--num_layers", type=int, help="Override the checkpoint's layer count, to export a partial stack."
    )
    parser.add_argument("--parts", default="embed,blocks,head", help="Comma-separated subset to export.")
    parser.add_argument(
        "--only_group", type=int, help="Export just this block partition, so one process holds one partition."
    )
    args = parser.parse_args()

    # Emit the fused attention op while tracing: the score tensor is never allocated, so export memory stops
    # scaling with the square of the sequence length. `h3_rebuild.py` turns it into MNN's Attention op.
    H3Attention.export_fused_attn = True

    config = checkpoint_config(args.model_path)
    num_layers = args.num_layers or config["num_layers"]
    parts = set(args.parts.split(","))
    output = Path(args.output)

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
    _, timestep_indices = schedule.row_timestep_plan[0]
    segments, row_order = compact_adaln(layout.token_tags, timestep_indices)
    final_norm_rows = head_rows(row_order)
    final_norm_segments = merge_runs(head_segments(segments, row_order))

    sequence_length = layout.sequence_length
    hidden_size = config["hidden_size"]
    num_video_rows = int(layout.video_indices.shape[0])
    num_audio_rows = int(layout.audio_indices.shape[0])
    rotary_dim = 6 * config["rope_freq_dim"]
    video_patch_dim = config["in_channels"] * int(np.prod(config["patch_size"]))

    print(
        f"layout: {sequence_length} rows = {layout.num_text_tokens} text + "
        f"{layout.num_condition_video_rows} condition + {num_audio_rows} audio + "
        f"{num_video_rows - layout.num_condition_video_rows} video"
    )
    print(f"AdaLN runs {segments} over {len(row_order)} table rows; head runs {final_norm_segments}")

    cos = torch.ones(1, sequence_length, 1, rotary_dim)
    sin = torch.zeros(1, sequence_length, 1, rotary_dim)
    mask = torch.zeros(1, 1, sequence_length, sequence_length)

    if "embed" in parts:
        print("embed:")
        embed = H3Embed(config, layout.num_condition_video_rows)
        embed.load_state_dict(load_group_state_dict(args.model_path, "embed", config, dtype=torch.float32))
        export(
            embed.eval(),
            (
                torch.zeros(1, num_video_rows, video_patch_dim),
                torch.zeros(1, num_audio_rows, config["audio_in_channels"]),
                torch.zeros(1, layout.num_text_tokens, config["text_dim"]),
                torch.zeros(1, 1, layout.num_text_tokens, layout.num_text_tokens),
            ),
            output,
            "h3_embed",
            ["video_rows", "audio_rows", "text_rows", "text_mask"],
            ["packed"],
        )
        del embed

    group_bounds = [
        (start, min(args.layers_per_group, num_layers - start))
        for start in range(0, num_layers, args.layers_per_group)
    ]
    if "blocks" in parts:
        for index, (start, count) in enumerate(group_bounds):
            if args.only_group is not None and index != args.only_group:
                continue
            print(f"blocks[{index}]: layers {start}..{start + count - 1}")
            group = H3BlockGroup(config, segments, count)
            group.load_state_dict(
                load_group_state_dict(
                    args.model_path, "block", config, layer=start, num_layers=count, dtype=torch.float32
                )
            )
            modulation = tuple(
                torch.zeros(len(row_order), hidden_size) for _ in range(6 * count)
            )
            names = [
                f"adaln_l{layer}_{part}"
                for layer in range(count)
                for part in ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
            ]
            export(
                group.eval(),
                (torch.zeros(1, sequence_length, hidden_size), cos, sin, mask) + modulation,
                output,
                f"h3_blocks_{index}",
                ["hidden", "rope_cos", "rope_sin", "mask"] + names,
                ["hidden_out"],
            )
            del group

    if "head" in parts:
        print("head:")
        head = H3Head(
            config, final_norm_segments, layout.num_text_tokens, num_audio_rows, layout.num_condition_video_rows
        )
        head.load_state_dict(load_group_state_dict(args.model_path, "head", config, dtype=torch.float32))
        export(
            head.eval(),
            (
                torch.zeros(1, sequence_length, hidden_size),
                torch.zeros(len(final_norm_rows), hidden_size),
                torch.zeros(len(final_norm_rows), hidden_size),
            ),
            output,
            "h3_head",
            ["hidden", "norm_out_shift", "norm_out_scale"],
            ["video_velocity", "audio_velocity"],
        )
        del head

    manifest = {
        "config": config,
        "num_layers": num_layers,
        "layers_per_group": args.layers_per_group,
        "groups": [{"start": start, "num_layers": count} for start, count in group_bounds],
        "height": layout.height,
        "width": layout.width,
        "num_frames": layout.num_frames,
        "num_latent_frames": layout.num_latent_frames,
        "latent_height": layout.latent_height,
        "latent_width": layout.latent_width,
        "num_audio_latents": layout.num_audio_latents,
        "num_text_tokens": layout.num_text_tokens,
        "sequence_length": sequence_length,
        "num_video_rows": num_video_rows,
        "num_audio_rows": num_audio_rows,
        "num_condition_video_rows": layout.num_condition_video_rows,
        "rotary_dim": rotary_dim,
        "adaln_segments": [list(segment) for segment in segments],
        "adaln_row_order": row_order,
        "head_segments": [list(segment) for segment in final_norm_segments],
        "num_inference_steps": schedule.num_steps,
        "timesteps": schedule.timesteps.tolist(),
        "audio_timesteps": schedule.audio_timesteps.tolist(),
        "video_sigmas": schedule.video_sigmas.tolist(),
        "audio_sigmas": schedule.audio_sigmas.tolist(),
    }
    (output / "h3_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output / 'h3_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
