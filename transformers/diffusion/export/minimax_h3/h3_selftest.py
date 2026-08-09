# SPDX-License-Identifier: Apache-2.0
"""End-to-end self-test of the MiniMax-H3 tooling on a tiny random checkpoint.

Writes a small checkpoint in the released layout, then runs the real pipeline over it -- golden fixture,
AdaLN fold, ONNX export, MNNConvert, MNN alignment -- and asserts every stage agrees. Nothing here needs the
33B weights, so it is the gate that can run anywhere MNNConvert and diffusers are available.

    python h3_selftest.py                       # uses a temporary directory
    python h3_selftest.py --workdir /tmp/h3st   # keeps the artifacts for inspection
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import h3_layout  # noqa: E402
from h3_build_mnn import find_mnnconvert  # noqa: E402
from h3_reference import TINY_CONFIG, build_tiny  # noqa: E402
from h3_weights import FP32_PATTERNS  # noqa: E402

# Small enough to convert and run in seconds, large enough that all three AdaLN runs are non-empty.
CASE = {"height": 64, "width": 96, "num_frames": 22, "num_text_tokens": 5, "num_inference_steps": 4}


def write_tiny_checkpoint(directory, seed=0):
    """Save a random tiny model under the reference's own key names and mixed-precision layout."""
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model, config = build_tiny(torch.bfloat16, torch.device("cpu"), seed)

    state_dict = {}
    for name, parameter in model.state_dict().items():
        target = torch.float32 if any(pattern in name for pattern in FP32_PATTERNS) else torch.bfloat16
        state_dict[name] = parameter.detach().to(target).contiguous()

    shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(state_dict, (directory / shard).as_posix())
    total = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
    (directory / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": {name: shard for name in state_dict}}, indent=2)
    )
    stored = dict(config)
    stored["_class_name"] = "MiniMaxH3Transformer3DModel"
    (directory / "config.json").write_text(json.dumps(stored, indent=2))
    print(f"tiny checkpoint: {len(state_dict)} tensors, {total / 1e6:.1f} MB")
    return config


def run(command, cwd=None):
    print(f"  $ {Path(command[1]).name if len(command) > 1 else command[0]} ...")
    completed = subprocess.run(
        [str(item) for item in command], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    text = completed.stdout.decode(errors="replace")
    if completed.returncode != 0:
        print(text[-4000:])
        raise SystemExit(f"FAILED: {' '.join(str(i) for i in command)}")
    return text


def find_pymnn_python(explicit=None):
    """An interpreter whose `import MNN` works.

    Only the alignment step needs pymnn, and a machine can carry more than one install -- a broken one shadows
    a working one easily enough that probing beats guessing.
    """
    candidates = [explicit] if explicit else [sys.executable, shutil.which("python3"), shutil.which("python")]
    for candidate in candidates:
        if not candidate:
            continue
        probe = subprocess.run(
            [candidate, "-c", "import MNN.expr, MNN.nn"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if probe.returncode == 0:
            return candidate
    raise SystemExit(
        "No interpreter with a working pymnn was found. Build it from pymnn/pip_package, or pass "
        "--pymnn_python explicitly."
    )


def main():
    parser = argparse.ArgumentParser(description="Self-test the MiniMax-H3 tooling on a tiny checkpoint.")
    parser.add_argument("--workdir", help="Keep artifacts here instead of a temporary directory.")
    parser.add_argument("--mnnconvert")
    parser.add_argument("--pymnn_python", help="Interpreter to run the pymnn alignment step with.")
    parser.add_argument("--cos_threshold", type=float, default=0.999)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    temporary = None
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.mkdtemp(prefix="h3_selftest_")
        workdir = Path(temporary)
    print(f"workdir: {workdir}")

    try:
        checkpoint = workdir / "checkpoint"
        write_tiny_checkpoint(checkpoint)

        layout_args = [
            "--model_path", checkpoint.as_posix(),
            "--height", str(CASE["height"]),
            "--width", str(CASE["width"]),
            "--num_frames", str(CASE["num_frames"]),
            "--num_text_tokens", str(CASE["num_text_tokens"]),
            "--num_inference_steps", str(CASE["num_inference_steps"]),
        ]

        print("1. golden fixture from diffusers")
        fixture = workdir / "fixture"
        run([sys.executable, (here / "h3_reference.py").as_posix(), "--output", fixture.as_posix(),
             "--layers", str(TINY_CONFIG["num_layers"]), "--dtype", "float32", "--step", "1"] + layout_args)

        print("2. export-friendly modules vs the fixture")
        text = run([sys.executable, (here / "h3_module_parity.py").as_posix(), "--fixture", fixture.as_posix(),
                    "--model_path", checkpoint.as_posix(), "--dtype", "float32"])
        if "reproduce the reference fixture" not in text:
            print(text[-3000:])
            raise SystemExit("FAILED: the export-friendly modules do not match the fixture")

        print("3. export, fold AdaLN and convert")
        resources = workdir / "mnn"
        run([sys.executable, (here / "h3_build_mnn.py").as_posix(), "--output", resources.as_posix(),
             "--layers_per_group", "1", "--mnnconvert", find_mnnconvert(args.mnnconvert)] + layout_args)

        print("4. MNN vs the fixture")
        pymnn_python = find_pymnn_python(args.pymnn_python)
        text = run([pymnn_python, (here / "h3_mnn_align.py").as_posix(), "--mnn", resources.as_posix(),
                    "--fixture", fixture.as_posix(), "--adaln", resources.as_posix(), "--step", "1",
                    "--cos_threshold", str(args.cos_threshold)])
        report = "\n".join(line for line in text.splitlines() if "cosine" in line or " ok" in line or "FAIL" in line)
        print(report)
        if "MNN reproduces the reference fixture" not in text:
            print(text[-3000:])
            raise SystemExit("FAILED: MNN does not match the fixture")

        print("\nself-test passed")
        return 0
    finally:
        if temporary and not args.workdir:
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
