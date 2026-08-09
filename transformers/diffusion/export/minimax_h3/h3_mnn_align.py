# SPDX-License-Identifier: Apache-2.0
"""Compare the converted MiniMax-H3 MNN graphs against the golden reference fixture.

Runs the `h3_embed`, `h3_blocks_*` and `h3_head` modules through pymnn on a chosen backend and reports
cosine similarity and relative error against the tensors `h3_reference.py` dumped from diffusers. Each stage
is driven from the fixture's own input so one stage's error does not cascade into the next, and the block
stack is additionally run end to end so accumulated drift is visible.

    python h3_mnn_align.py --mnn /path/to/mnn_1layer --fixture /path/to/fixtures/block0_bf16_s0 \
        --adaln /path/to/t2va_448x256_f56 --backend cpu --precision high
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
from h3_fixture import FixtureReader, compare, format_report  # noqa: E402

BACKENDS = {"cpu": "CPU", "cuda": "CUDA", "opencl": "OPENCL", "vulkan": "VULKAN", "metal": "METAL"}
PRECISIONS = {"normal": "Normal", "high": "High", "low": "Low", "low_bf16": "Low_BF16"}
MEMORIES = {"normal": "Normal", "high": "High", "low": "Low"}


class AdalnTable:
    """Reader for the table `h3_adaln.py` folded the AdaLN branch into."""

    def __init__(self, root):
        root = Path(root)
        self.index = json.loads((root / "h3_adaln.json").read_text())
        self.dtype = np.dtype(self.index["dtype"])
        self.blob = np.memmap(root / "h3_adaln.bin", dtype=self.dtype, mode="r")
        self.entries = {entry["name"]: entry for entry in self.index["entries"]}

    def get(self, name):
        entry = self.entries[name]
        count = int(np.prod(entry["shape"]))
        start = entry["offset"] // self.dtype.itemsize
        return np.asarray(self.blob[start : start + count], dtype=np.float32).reshape(entry["shape"])

    def layer(self, step, layer):
        return self.get(f"step{step}.layer{layer}")

    def head(self, step):
        return self.get(f"step{step}.head")


def load_module(path, backend, precision, memory="normal"):
    import MNN.expr as F
    import MNN.nn as nn

    config = {
        "backend": getattr(F.Backend, BACKENDS[backend]),
        "precision": getattr(F.PrecisionMode, PRECISIONS[precision]),
        "memory": getattr(F.MemoryMode, MEMORIES[memory]),
        "numThread": 8 if backend == "cpu" else 1,
    }
    return nn.load_module_from_file(path.as_posix(), [], [], **config)


def run(module, inputs):
    import MNN.expr as F

    variables = [F.const(array.flatten().tolist(), list(array.shape), F.NCHW, F.float) for array in inputs]
    outputs = module.forward(variables)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return [np.array(output.read()).reshape(output.shape) for output in outputs]


def zero_mask(module, name, rows):
    """`h3_rebuild.py --drop_attention_mask` collapses the mask input, so take the size from the module."""
    info = module.get_info()
    for input_name, variable in zip(info["inputNames"], info["inputs"]):
        if input_name == name and variable.shape:
            rows = variable.shape[-1]
            break
    return np.zeros((1, 1, rows, rows), dtype=np.float32)


def dump_io(directory, names, inputs, output_names, outputs):
    """Write `input.mnn` / `output.mnn` so `ModuleBasic.out` can replay this case on any backend.

    The installed pymnn is a self-contained build, so a backend it was not compiled with -- CUDA here -- has
    to be exercised through the C++ tools against the same tensors. ModuleBasic takes the graph's input and
    output names from the variable names in these two files, so both have to carry the graph's own names.
    """
    import MNN.expr as F

    directory.mkdir(parents=True, exist_ok=True)

    def named(name, array):
        variable = F.const(array.flatten().tolist(), list(array.shape), F.NCHW, F.float)
        variable.name = name
        return variable

    F.save([named(name, array) for name, array in zip(names, inputs)], (directory / "input.mnn").as_posix())
    F.save(
        [named(name, array) for name, array in zip(output_names, outputs)],
        (directory / "output.mnn").as_posix(),
    )
    print(f"  dumped {len(inputs)} inputs / {len(outputs)} outputs to {directory}")


def main():
    parser = argparse.ArgumentParser(description="Align the MiniMax-H3 MNN graphs against the golden fixture.")
    parser.add_argument("--mnn", required=True, help="Directory holding the converted .mnn modules.")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--adaln", help="AdaLN table directory; defaults to the fixture's own tables.")
    parser.add_argument("--step", type=int, default=0, help="Schedule step the AdaLN table is read at.")
    parser.add_argument("--backend", default="cpu", choices=sorted(BACKENDS))
    parser.add_argument("--precision", default="high", choices=sorted(PRECISIONS))
    parser.add_argument("--cos_threshold", type=float, default=0.999)
    parser.add_argument("--parts", default="embed,blocks,head")
    parser.add_argument("--dump_io", help="Also write ModuleBasic replay tensors under this directory.")
    args = parser.parse_args()

    fixture = FixtureReader(args.fixture)
    root = Path(args.mnn)
    manifest = json.loads((root / "h3_manifest.json").read_text())
    table = AdalnTable(args.adaln) if args.adaln else None
    parts = set(args.parts.split(","))
    rows = []

    def modulation(layer):
        if table is not None:
            return table.layer(args.step, layer)
        return fixture.get(f"block{layer}.adaln")

    sequence_length = manifest["sequence_length"]
    rotary_dim = manifest["rotary_dim"]
    cos = fixture.get("rope_cos").reshape(1, sequence_length, 1, rotary_dim)
    sin = fixture.get("rope_sin").reshape(1, sequence_length, 1, rotary_dim)
    mask = None

    if "embed" in parts:
        module = load_module(root / "h3_embed.mnn", args.backend, args.precision)
        inputs = [
            fixture.get("input_video_rows")[None],
            fixture.get("input_audio_rows")[None],
            fixture.get("input_text_rows")[None],
            zero_mask(module, "text_mask", manifest["num_text_tokens"]),
        ]
        packed = run(module, inputs)[0]
        rows.append(compare("embed -> packed", packed[0], fixture.get("packed_input"), args.cos_threshold))
        if args.dump_io:
            dump_io(
                Path(args.dump_io) / "h3_embed",
                ["video_rows", "audio_rows", "text_rows", "text_mask"],
                inputs,
                ["packed"],
                [fixture.get("packed_input")[None]],
            )
        del module

    num_fixture_layers = sum(
        1 for name in fixture.names() if name.startswith("block") and name.endswith(".adaln")
    )
    if "blocks" in parts:
        chained = None
        for group in manifest["groups"]:
            start, count = group["start"], group["num_layers"]
            if start >= num_fixture_layers:
                break
            path = root / f"h3_blocks_{start // manifest['layers_per_group']}.mnn"
            module = load_module(path, args.backend, args.precision)
            last = min(start + count, num_fixture_layers) - 1
            if mask is None:
                mask = zero_mask(module, "mask", sequence_length)

            modulations = []
            for layer in range(count):
                modulations.extend(modulation(start + layer))

            # Per-partition, driven by the reference input so one partition's error does not cascade.
            reference_input = fixture.get("packed_input" if start == 0 else f"block{start - 1}.output")[None]
            inputs = [reference_input, cos, sin, mask] + modulations
            hidden = run(module, inputs)[0]
            rows.append(
                compare(
                    f"blocks[{start}..{last}] -> hidden",
                    hidden[0],
                    fixture.get(f"block{last}.output"),
                    args.cos_threshold,
                )
            )

            # And chained, so drift accumulated across the whole stack is visible.
            chained = run(
                module, [chained if chained is not None else reference_input, cos, sin, mask] + modulations
            )[0]
            if last == num_fixture_layers - 1:
                rows.append(
                    compare(
                        f"blocks[0..{last}] chained -> hidden",
                        chained[0],
                        fixture.get(f"block{last}.output"),
                        args.cos_threshold,
                    )
                )
            if args.dump_io:
                names = ["hidden", "rope_cos", "rope_sin", "mask"] + [
                    f"adaln_l{layer}_{part}"
                    for layer in range(count)
                    for part in ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
                ]
                dump_io(
                    Path(args.dump_io) / path.stem,
                    names,
                    inputs,
                    ["hidden_out"],
                    [fixture.get(f"block{last}.output")[None]],
                )
            del module

    if "head" in parts:
        module = load_module(root / "h3_head.mnn", args.backend, args.precision)
        head_table = table.head(args.step) if table is not None else fixture.get("norm_out_adaln")
        inputs = [fixture.get(f"block{num_fixture_layers - 1}.output")[None], head_table[0], head_table[1]]
        video, audio = run(module, inputs)
        rows.append(compare("head -> video_velocity", video[0], fixture.get("video_velocity"), args.cos_threshold))
        rows.append(compare("head -> audio_velocity", audio[0], fixture.get("audio_velocity"), args.cos_threshold))
        if args.dump_io:
            dump_io(
                Path(args.dump_io) / "h3_head",
                ["hidden", "norm_out_shift", "norm_out_scale"],
                inputs,
                ["video_velocity", "audio_velocity"],
                [fixture.get("video_velocity")[None], fixture.get("audio_velocity")[None]],
            )
        del module

    print(f"backend={args.backend} precision={args.precision} step={args.step}")
    print(format_report(rows))
    failed = [row["name"] for row in rows if not row["ok"]]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nMNN reproduces the reference fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
