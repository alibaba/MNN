# SPDX-License-Identifier: Apache-2.0
"""Build the MiniMax-H3 MNN resources: export each partition, convert it, and drop the ONNX again.

The 20B block stack does not fit anywhere as one artifact, so this driver walks the partitions one at a time
and keeps only the converted `.mnn` plus its external `.weight` sidecar. Peak disk is one partition's ONNX
rather than the whole stack's, which is the difference between ~8 GB and ~80 GB of scratch.

    python h3_build_mnn.py --model_path /path/to/MiniMax-H3/transformer \
        --output /path/to/h3_mnn --num_text_tokens 37 --layers_per_group 5 --quant_bit 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def find_mnnconvert(explicit=None):
    if explicit:
        return explicit
    repo_root = Path(__file__).resolve().parents[4]
    for candidate in ("build_h3/MNNConvert", "build/MNNConvert"):
        path = repo_root / candidate
        if path.exists():
            return path.as_posix()
    found = shutil.which("MNNConvert") or shutil.which("mnnconvert")
    return found or "MNNConvert"


def run(command, log):
    log.write(" ".join(str(item) for item in command) + "\n")
    log.flush()
    completed = subprocess.run([str(item) for item in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write(completed.stdout.decode(errors="replace"))
    log.flush()
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with {completed.returncode}: {' '.join(str(i) for i in command)}")
    return completed.stdout.decode(errors="replace")


def convert(mnnconvert, onnx_dir, name, destination, quant, log, transformer_fuse=True):
    command = [
        mnnconvert,
        "-f",
        "ONNX",
        "--modelFile",
        (onnx_dir / name / f"{name}.onnx").as_posix(),
        "--MNNModel",
        (destination / f"{name}.mnn").as_posix(),
        "--bizCode",
        "MNN",
        "--saveExternalData",
        # h3_modules emits LlmExporter::FusedAttention; without this the converter rejects the unknown node.
        "--allowCustomOp",
    ]
    if transformer_fuse:
        command.append("--transformerFuse")
    if quant["bits"]:
        command += [
            "--weightQuantBits",
            str(quant["bits"]),
            "--weightQuantBlock",
            str(quant["block"]),
        ]
        if quant["asymmetric"]:
            command.append("--weightQuantAsymmetric")
        if quant["hqq"]:
            command.append("--hqq")
    output = run(command, log)
    if "Converted Success" not in output:
        raise RuntimeError(f"{name} did not convert cleanly; see the log")
    # Turn the traced custom ops into real MNN ops.
    run([sys.executable, (Path(__file__).resolve().parent / "h3_rebuild.py").as_posix(),
         "--mnn", (destination / f"{name}.mnn").as_posix(), "--mnnconvert", str(mnnconvert)], log)
    fused = output.count("Fuse Attention")
    size = sum(item.stat().st_size for item in destination.glob(f"{name}.mnn*"))
    print(f"  converted {name}: {size / 1e9:.2f} GB, {fused} pattern-fused attention op(s)")
    return {"name": name, "bytes": size, "fused_attention": fused}


def main():
    parser = argparse.ArgumentParser(description="Export and convert the MiniMax-H3 DiT to MNN.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scratch", help="Where partition ONNX lands; defaults to <output>/onnx_scratch.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num_frames", type=int, default=56)
    parser.add_argument("--num_text_tokens", type=int, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--keyframe", action="store_true")
    parser.add_argument("--layers_per_group", type=int, default=5)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument(
        "--quant_bit", type=int, default=0, help="Block-stack weight bits; 0 keeps float, 4 and 8 are the useful ones."
    )
    parser.add_argument(
        "--embed_quant_bit",
        type=int,
        default=0,
        help=(
            "Embed-stage weight bits. Defaults to float: the text refiner runs once per generation rather than "
            "once per step, and W4 there costs about 1e-1 relative RMS."
        ),
    )
    parser.add_argument(
        "--head_quant_bit", type=int, default=0, help="Head weight bits. Defaults to float; the heads are under 1M."
    )
    parser.add_argument("--quant_block", type=int, default=64)
    parser.add_argument("--symmetric", action="store_true", help="Symmetric weight quant; asymmetric is the default.")
    parser.add_argument("--hqq", action="store_true")
    parser.add_argument("--mnnconvert")
    parser.add_argument("--keep_onnx", action="store_true")
    parser.add_argument("--skip_adaln", action="store_true")
    parser.add_argument(
        "--skip_transformer",
        action="store_true",
        help="Reuse the transformer modules already in --output and build only the VAE / conditioner.",
    )
    parser.add_argument("--vae", help="MiniMax-H3 vae directory; also builds the video decoder when given.")
    parser.add_argument(
        "--text_encoder", help="MiniMax-H3 text_encoder directory; also builds the conditioner when given."
    )
    parser.add_argument("--max_text_tokens", type=int, default=256, help="Fixed sequence length of the conditioner.")
    parser.add_argument("--text_layers_per_group", type=int, default=5)
    parser.add_argument("--text_quant_bit", type=int, default=4, help="Conditioner weight bits.")
    parser.add_argument(
        "--vae_quant_bit",
        type=int,
        default=8,
        help="VAE decoder weight bits. W4 costs 1.7e-1 relative RMS on a tile, W8 1.0e-2, and the decoder runs "
        "once per generation, so 8 is the default.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch) if args.scratch else output / "onnx_scratch"
    mnnconvert = find_mnnconvert(args.mnnconvert)
    python = sys.executable

    def quant_for(bits):
        return {
            "bits": bits,
            "block": args.quant_block,
            "asymmetric": not args.symmetric,
            "hqq": args.hqq,
        }

    quant = quant_for(args.quant_bit)

    layout_args = [
        "--model_path", args.model_path,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_frames", str(args.num_frames),
        "--num_text_tokens", str(args.num_text_tokens),
        "--num_inference_steps", str(args.num_inference_steps),
    ] + (["--keyframe"] if args.keyframe else [])

    log_path = output / "h3_build.log"
    print(f"MNNConvert: {mnnconvert}")
    print(f"log: {log_path}")
    with log_path.open("w") as log:
        if not args.skip_adaln:
            print("folding the AdaLN branch")
            run([python, (here / "h3_adaln.py").as_posix(), "--output", output.as_posix()] + layout_args, log)

        # One export pass writes the manifest and tells us how many partitions there are.
        export_args = [
            python, (here / "h3_onnx_export.py").as_posix(),
            "--output", scratch.as_posix(),
            "--layers_per_group", str(args.layers_per_group),
        ] + layout_args + (["--num_layers", str(args.num_layers)] if args.num_layers else [])

        if args.skip_transformer:
            manifest = json.loads((output / "h3_manifest.json").read_text())
            converted = list(manifest.get("modules", []))
        else:
            print("exporting embed + head")
            run(export_args + ["--parts", "embed,head"], log)
            manifest = json.loads((scratch / "h3_manifest.json").read_text())
            converted = [
                convert(mnnconvert, scratch, "h3_embed", output, quant_for(args.embed_quant_bit), log),
                convert(mnnconvert, scratch, "h3_head", output, quant_for(args.head_quant_bit), log),
            ]
            if not args.keep_onnx:
                shutil.rmtree(scratch / "h3_embed", ignore_errors=True)
                shutil.rmtree(scratch / "h3_head", ignore_errors=True)

            for index, group in enumerate(manifest["groups"]):
                name = f"h3_blocks_{index}"
                print(f"exporting {name} (layers {group['start']}..{group['start'] + group['num_layers'] - 1})")
                # Re-running the export per partition keeps only one partition's weights in host memory.
                run(export_args + ["--parts", "blocks", "--only_group", str(index)], log)
                converted.append(convert(mnnconvert, scratch, name, output, quant, log))
                if not args.keep_onnx:
                    shutil.rmtree(scratch / name, ignore_errors=True)

    if args.vae:
        with log_path.open("a") as log:
            print("exporting the video VAE decoder")
            run(
                [
                    python, (here / "h3_vae_export.py").as_posix(),
                    "--vae", args.vae,
                    "--output", scratch.as_posix(),
                    "--num_latent_frames", str(manifest["num_latent_frames"]),
                    "--latent_height", str(manifest["latent_height"]),
                    "--latent_width", str(manifest["latent_width"]),
                ],
                log,
            )
            for name in ("h3_vae_plan.json", "h3_vae_rope.bin"):
                shutil.copy(scratch / name, output / name)
            # FuseFmhaV2 aborts on this graph, so the decoder is converted without --transformerFuse and its
            # attention stays MatMul/Softmax/MatMul.
            converted.append(
                convert(
                    mnnconvert, scratch, "h3_vae_decoder", output, quant_for(args.vae_quant_bit), log,
                    transformer_fuse=False,
                )
            )
            if not args.keep_onnx:
                shutil.rmtree(scratch / "h3_vae_decoder", ignore_errors=True)

    if args.text_encoder:
        with log_path.open("a") as log:
            encoder_args = [
                python, (here / "h3_encoder_export.py").as_posix(),
                "--text_encoder", args.text_encoder,
                "--output", scratch.as_posix(),
                "--max_tokens", str(args.max_text_tokens),
                "--layers_per_group", str(args.text_layers_per_group),
            ]
            print("exporting the conditioner's embedding table")
            run(encoder_args + ["--parts", "embed"], log)
            text_manifest = json.loads((scratch / "h3_text_manifest.json").read_text())
            for name in ("h3_text_manifest.json", "h3_text_rope.bin", "h3_text_embed.bin"):
                shutil.move((scratch / name).as_posix(), (output / name).as_posix())
            for index, group in enumerate(text_manifest["groups"]):
                name = f"h3_text_layers_{index}"
                print(f"exporting {name} (layers {group['start']}..{group['start'] + group['num_layers'] - 1})")
                run(encoder_args + ["--parts", "layers", "--only_group", str(index)], log)
                converted.append(
                    convert(mnnconvert, scratch, name, output, quant_for(args.text_quant_bit), log)
                )
                if not args.keep_onnx:
                    shutil.rmtree(scratch / name, ignore_errors=True)

    manifest["quant"] = {
        "blocks": quant,
        "embed": quant_for(args.embed_quant_bit),
        "head": quant_for(args.head_quant_bit),
        "vae": quant_for(args.vae_quant_bit) if args.vae else None,
        "text_encoder": quant_for(args.text_quant_bit) if args.text_encoder else None,
    }
    manifest["modules"] = converted
    (output / "h3_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.keep_onnx:
        shutil.rmtree(scratch, ignore_errors=True)
    total = sum(item["bytes"] for item in converted)
    print(f"\n{len(converted)} modules, {total / 1e9:.2f} GB in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
