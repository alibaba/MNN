# SPDX-License-Identifier: Apache-2.0
"""Check the export-friendly H3 modules reproduce the golden reference fixture.

This is the gate the ONNX export sits behind: `h3_modules` rewrites RMSNorm, attention and the AdaLN
modulation into forms the MNN converter can fold, and this harness proves the rewrite is numerically the same
computation as `h3_reference` traced from diffusers.

    python h3_module_parity.py --fixture /path/to/fixtures/block0_bf16 \
        --model_path /path/to/MiniMax-H3/transformer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import h3_layout  # noqa: E402
from h3_fixture import FixtureReader, compare, format_report  # noqa: E402
from h3_adaln import compact_adaln, head_segments  # noqa: E402
from h3_modules import H3Block, H3Embed, H3Head  # noqa: E402
from h3_weights import load_group_state_dict  # noqa: E402


def build_mask(sequence_length, dtype):
    """The all-zero additive mask that keeps MNN's fused attention non-causal."""
    return torch.zeros(1, 1, sequence_length, sequence_length, dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description="Compare h3_modules against a golden reference fixture.")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model_path", required=True, help="MiniMax-H3 transformer directory.")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    fixture = FixtureReader(args.fixture)
    config = fixture.metadata["config"]
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    token_tags = fixture.get("token_tags", np.int64)
    timestep_indices = fixture.get("timestep_indices", np.int64)
    segments, row_order = compact_adaln(token_tags, timestep_indices)
    print(f"packed sequence: {token_tags.shape[0]} rows, AdaLN runs {segments}, table rows {row_order}")

    def tensor(name, target=dtype):
        return torch.from_numpy(fixture.get(name)).to(device=device, dtype=target)

    rows = []

    # 1. The embedding stage: three projections, the text refiner and the packing.
    embed = H3Embed(config, fixture.metadata["num_condition_video_rows"])
    embed.load_state_dict(load_group_state_dict(args.model_path, "embed", config), strict=True)
    embed = embed.to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        packed = embed(
            tensor("input_video_rows")[None],
            tensor("input_audio_rows")[None],
            tensor("input_text_rows")[None],
            build_mask(fixture.metadata["num_text_tokens"], dtype).to(device),
        )
    rows.append(compare("embed -> packed_input", packed[0].float().cpu().numpy(), fixture.get("packed_input")))

    # 2. One block, driven by the fixture's own packed input so an embed mismatch does not cascade.
    num_layers = sum(1 for name in fixture.names() if name.endswith(".adaln") and name.startswith("block"))
    hidden_states = tensor("packed_input")[None]
    mask = build_mask(token_tags.shape[0], dtype).to(device)
    cos = tensor("rope_cos")[None, :, None, :]
    sin = tensor("rope_sin")[None, :, None, :]
    for index in range(num_layers):
        block = H3Block(config, segments)
        block.load_state_dict(load_group_state_dict(args.model_path, "block", config, layer=index), strict=True)
        block = block.to(device=device, dtype=dtype).eval()
        modulation = tensor(f"block{index}.adaln")[:, row_order].unbind(0)
        with torch.no_grad():
            hidden_states = block(hidden_states, modulation, cos, sin, mask)
        rows.append(
            compare(
                f"block{index} -> output",
                hidden_states[0].float().cpu().numpy(),
                fixture.get(f"block{index}.output"),
            )
        )
        hidden_states = tensor(f"block{index}.output")[None]

    # 3. The head, again from the fixture's own input.
    head = H3Head(
        config,
        head_segments(segments, row_order),
        fixture.metadata["num_text_tokens"],
        int(fixture.get("audio_indices", np.int64).shape[0]),
        fixture.metadata["num_condition_video_rows"],
    )
    head.load_state_dict(load_group_state_dict(args.model_path, "head", config), strict=True)
    head = head.to(device=device, dtype=dtype).eval()
    head_shift, head_scale = tensor("norm_out_adaln")[:, sorted(set(r // h3_layout.MODALITY_NUM for r in row_order))].unbind(0)
    with torch.no_grad():
        video_velocity, audio_velocity = head(tensor(f"block{num_layers - 1}.output")[None], head_shift, head_scale)
    rows.append(
        compare("head -> video_velocity", video_velocity[0].float().cpu().numpy(), fixture.get("video_velocity"))
    )
    rows.append(
        compare("head -> audio_velocity", audio_velocity[0].float().cpu().numpy(), fixture.get("audio_velocity"))
    )

    print()
    print(format_report(rows))
    failed = [row["name"] for row in rows if not row["ok"]]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nthe export-friendly modules reproduce the reference fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
