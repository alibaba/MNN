# SPDX-License-Identifier: Apache-2.0
"""Fold the MiniMax-H3 AdaLN branch into a small per-(step, layer) table.

Every AdaLN projection in H3 reads only the timestep embedding, so for a fixed sampling schedule the whole
branch is a constant. That branch is 13B of the checkpoint's 33B parameters -- 26 GB in bfloat16, more than a
phone can hold or stream -- while the table it collapses to is a few tens of MB:

    num_steps * num_layers * 6 * num_table_rows * hidden_size + the final norm's two rows

The table rows are compacted to the `(timestep, modality)` pairs the packed sequence actually addresses,
usually three: the video rows, the text rows and the audio rows. The compaction has to agree with the row
indices baked into the exported block graphs, so `compact_adaln` is the single definition of both.
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
from h3_weights import checkpoint_config, read_tensors, time_embedder_state_dict  # noqa: E402


def compact_adaln(token_tags, timestep_indices):
    """Run-length encode the packed sequence over a compacted AdaLN table.

    Returns `(segments, row_order)`: `segments` is `(length, compact_row)` per run and `row_order` lists the
    original `timestep_index * 3 + modality` rows in compacted order. The exported graph bakes `segments`, and
    the table this module writes holds its rows in `row_order`.
    """
    indices = np.asarray(timestep_indices, dtype=np.int64) * h3_layout.MODALITY_NUM + np.asarray(
        token_tags, dtype=np.int64
    )
    row_order = sorted(set(int(value) for value in indices))
    compact = {row: position for position, row in enumerate(row_order)}
    boundaries = np.flatnonzero(np.diff(indices)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [indices.shape[0]]])
    segments = [(int(end - start), compact[int(indices[start])]) for start, end in zip(starts, ends)]
    if len({row for _, row in segments}) != len(segments):
        raise ValueError(
            "the packed sequence is not grouped by AdaLN table row, so the segment-wise modulation the export "
            f"relies on would need a row permutation first; runs = {[row for _, row in segments]}"
        )
    return segments, row_order


def head_rows(row_order):
    """The final norm's table rows: it is indexed by timestep alone, so the modality tag drops out."""
    return sorted(set(row // h3_layout.MODALITY_NUM for row in row_order))


def head_segments(segments, row_order):
    """`segments` re-addressed against the final norm's compacted table."""
    rows = head_rows(row_order)
    return [(length, rows.index(row_order[row] // h3_layout.MODALITY_NUM)) for length, row in segments]


def timestep_embedding(timesteps, freq_dim, state_dict):
    """The reference timestep MLP: sinusoidal features, then a float32 two-layer MLP with SiLU."""
    from diffusers.models.embeddings import Timesteps

    features = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)(
        torch.as_tensor(timesteps, dtype=torch.float32)
    )
    hidden = nn.functional.linear(
        features.to(torch.float32), state_dict["time_embedder.linear_1.weight"], state_dict["time_embedder.linear_1.bias"]
    )
    hidden = nn.functional.silu(hidden)
    return nn.functional.linear(
        hidden, state_dict["time_embedder.linear_2.weight"], state_dict["time_embedder.linear_2.bias"]
    )


def project_modulation(temb, weight, bias, hidden_size, parts):
    """`linear(silu(temb))` reshaped the way the reference chunks it, at the projection's own precision.

    The reference activates `temb` at the timestep MLP's float32 precision and only then casts to the
    bfloat16 projection, so a rounding applied before the activation would bias every block identically at
    every step. This keeps that order.
    """
    activated = nn.functional.silu(temb).to(weight.dtype)
    projected = nn.functional.linear(activated, weight, bias)
    if parts == 6:
        # (num_timesteps, 18 * hidden) -> (num_timesteps * 3, 6 * hidden) -> 6 x (rows, hidden)
        return torch.stack(projected.view(-1, 6 * hidden_size).chunk(6, dim=-1))
    return torch.stack(projected.chunk(parts, dim=-1))


class AdalnTableWriter:
    """Writes `h3_adaln.bin` plus its index: `[step][layer][6][rows][hidden]` then the final norm's rows."""

    def __init__(self, path, dtype=np.float16):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("wb")
        self.dtype = dtype
        self.offset = 0
        self.entries = []

    def write(self, name, array):
        array = np.ascontiguousarray(np.asarray(array, dtype=self.dtype))
        self.handle.write(array.tobytes())
        self.entries.append({"name": name, "offset": self.offset, "shape": list(array.shape)})
        self.offset += array.nbytes

    def close(self):
        self.handle.close()
        return self.entries


def main():
    parser = argparse.ArgumentParser(description="Precompute the MiniMax-H3 AdaLN modulation table.")
    parser.add_argument("--model_path", required=True, help="MiniMax-H3 transformer directory.")
    parser.add_argument("--output", required=True, help="Directory to write h3_adaln.bin and h3_adaln.json into.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num_frames", type=int, default=56)
    parser.add_argument("--num_text_tokens", type=int, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--keyframe", action="store_true")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float16", "float32"),
        help=(
            "Storage precision of the table. float16 costs about 2.8e-3 relative RMS on a block output -- the "
            "same order as bfloat16 itself -- and the reference notes this rounding biases every block "
            "identically at every step, so it accumulates over the trajectory. The table is tens of MB either "
            "way, so float32 is the default."
        ),
    )
    args = parser.parse_args()

    config = checkpoint_config(args.model_path)
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]

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

    # One graph serves every step, so the compacted run layout has to be the same at every step.
    plans = []
    reference_segments = None
    for index in range(schedule.num_steps):
        _, timestep_indices = schedule.row_timestep_plan[index]
        segments, row_order = compact_adaln(layout.token_tags, timestep_indices)
        if reference_segments is None:
            reference_segments = segments
        elif segments != reference_segments:
            raise RuntimeError(
                f"step {index} compacts to runs {segments}, step 0 to {reference_segments}; one exported graph "
                "cannot serve both"
            )
        plans.append((row_order, timestep_indices))
    print(f"AdaLN runs {reference_segments}, {len(plans[0][0])} table rows, {schedule.num_steps} steps")

    shared = time_embedder_state_dict(args.model_path)
    writer = AdalnTableWriter(Path(args.output) / "h3_adaln.bin", getattr(np, args.dtype))

    tembs = []
    for index in range(schedule.num_steps):
        unique_timesteps, _ = schedule.row_timestep_plan[index]
        tembs.append(timestep_embedding(unique_timesteps, config["freq_dim"], shared))

    for layer in range(num_layers):
        keys = [
            f"transformer_blocks.{layer}.adaln_proj.linear.weight",
            f"transformer_blocks.{layer}.adaln_proj.linear.bias",
        ]
        tensors = read_tensors(args.model_path, keys)
        weight, bias = tensors[keys[0]], tensors[keys[1]]
        for index, (row_order, _) in enumerate(plans):
            table = project_modulation(tembs[index], weight, bias, hidden_size, 6)
            writer.write(f"step{index}.layer{layer}", table[:, row_order].float().numpy())
        del tensors, weight, bias
        if (layer + 1) % 10 == 0 or layer + 1 == num_layers:
            print(f"  folded {layer + 1}/{num_layers} layers, {writer.offset / 1e6:.1f} MB")

    head_weight = shared["norm_out.linear.weight"]
    head_bias = shared["norm_out.linear.bias"]
    for index, (row_order, timestep_indices) in enumerate(plans):
        table = project_modulation(tembs[index], head_weight, head_bias, hidden_size, 2)
        writer.write(f"step{index}.head", table[:, head_rows(row_order)].float().numpy())

    entries = writer.close()

    # The rotary tables come from an fp64 position grid that depends on numpy's linspace rounding, so they are
    # baked here rather than reproduced by the runtime. The attention mask is all zeros, so the runtime builds
    # that one itself.
    cos, sin = layout.rope_cos_sin(config["rope_freq_dim"], config["rope_theta"])
    rope_writer = AdalnTableWriter(Path(args.output) / "h3_rope.bin", np.float32)
    rope_writer.write("rope_cos", cos)
    rope_writer.write("rope_sin", sin)
    rope_entries = rope_writer.close()

    final_norm_segments = head_segments(reference_segments, plans[0][0])
    index_path = Path(args.output) / "h3_adaln.json"
    index_path.write_text(
        json.dumps(
            {
                "dtype": args.dtype,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_steps": schedule.num_steps,
                "segments": [list(segment) for segment in reference_segments],
                "head_segments": [list(segment) for segment in final_norm_segments],
                "timesteps": schedule.timesteps.tolist(),
                "audio_timesteps": schedule.audio_timesteps.tolist(),
                "video_sigmas": schedule.video_sigmas.tolist(),
                "audio_sigmas": schedule.audio_sigmas.tolist(),
                "entries": entries,
                "rope_entries": rope_entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {writer.offset / 1e6:.1f} MB to {index_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
