# SPDX-License-Identifier: Apache-2.0
"""Decode MiniMax-H3 video latents into an MP4.

Takes the latent rows the MNN engine writes -- `video_latent_rows.bin` plus `h3_latents.json` -- unpacks them,
denormalizes with the VAE's per-channel statistics, decodes through the reference video VAE and reverts the
ImageNet normalization the VAE decodes into. Tiling is the VAE's own: it blends overlapping spatial tiles and
chunks time, and the released frames are the blended-tile ones, so it is left enabled.

The decoder here is the reference implementation. It is the target an MNN port is measured against, and it is
what makes the DiT's output inspectable before that port exists.

    python h3_vae_decode.py --vae /path/to/MiniMax-H3/vae --latents /path/to/latents --output out.mp4
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

# The VAE decodes into ImageNet-normalized RGB over a [0, 1] base range.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def load_latents(directory, override_rows=None):
    directory = Path(directory)
    meta = json.loads((directory / "h3_latents.json").read_text())
    rows = np.fromfile(override_rows or (directory / "video_latent_rows.bin"), dtype=np.float32)
    expected = meta["video_rows"] * meta["video_patch_dim"]
    if rows.size != expected:
        raise ValueError(f"expected {expected} floats of video rows, got {rows.size}")
    rows = rows.reshape(meta["video_rows"], meta["video_patch_dim"])
    # The loop never writes the conditioning rows, so they are dropped rather than decoded.
    generated = rows[meta.get("num_condition_video_rows", 0) :]
    latents = h3_layout.unpatchify_video_latents(
        generated, meta["num_latent_frames"], meta["latent_height"], meta["latent_width"]
    )
    return latents, meta


def main():
    parser = argparse.ArgumentParser(description="Decode MiniMax-H3 video latents into an MP4.")
    parser.add_argument("--vae", required=True, help="MiniMax-H3 vae directory.")
    parser.add_argument("--latents", required=True, help="Directory holding h3_latents.json.")
    parser.add_argument("--rows", help="Override the video_latent_rows.bin path.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=h3_layout.FPS)
    parser.add_argument("--tile", type=int, default=256, help="Spatial tile size in pixels.")
    parser.add_argument("--no_tiling", action="store_true")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help=(
            "Decode under the reference's float16 autocast. Off by default: it makes the ViT decoder's "
            "24-channel input projection a k=24 half GEMM that cuBLAS rejects with CUBLAS_STATUS_INVALID_VALUE."
        ),
    )
    args = parser.parse_args()

    from diffusers import AutoencoderKLMiniMaxH3

    latents, meta = load_latents(args.latents, args.rows)
    print(
        f"latents {latents.shape} -> {meta['num_frames']} frames at {meta['width']}x{meta['height']}, "
        f"{meta['num_latent_frames']} latent frames"
    )

    device = torch.device(args.device)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(args.vae, torch_dtype=torch.float32).to(device).eval()
    if args.no_tiling:
        vae.disable_tiling()
    else:
        vae.enable_tiling(tile_sample_min_height=args.tile, tile_sample_min_width=args.tile)

    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    sample = torch.from_numpy(latents).to(device)[None] * std + mean

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"
    ):
        video = vae.decode(sample, return_dict=False)[0]
    pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)

    frames = (video[0].permute(1, 2, 3, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    print(f"decoded {frames.shape[0]} frames of {frames.shape[2]}x{frames.shape[1]}")

    import imageio

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"wrote {args.output} ({Path(args.output).stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
