# SPDX-License-Identifier: Apache-2.0
"""Check `h3_layout` reproduces the diffusers MiniMax-H3 layout, schedule and patchify bit-exactly.

Needs a diffusers build that ships `MiniMaxH3Scheduler` and the `minimax_h3` modular blocks:

    python h3_layout_parity.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import h3_layout  # noqa: E402

from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: E402
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)
from diffusers.schedulers import MiniMaxH3Scheduler  # noqa: E402


CASES = [
    # (height, width, num_frames, num_text_tokens, keyframe_anchors)
    (256, 448, 56, 37, ()),
    (256, 448, 56, 37, ("first",)),
    (256, 448, 124, 512, ("first", "last")),
    (768, 1344, 124, 301, ("first",)),
    (256, 256, 39, 8, ()),
]


def check(name, mine, theirs, atol=0.0):
    mine = np.asarray(mine)
    theirs = theirs.cpu().numpy() if isinstance(theirs, torch.Tensor) else np.asarray(theirs)
    if mine.shape != theirs.shape:
        print(f"  FAIL {name}: shape {mine.shape} vs {theirs.shape}")
        return False
    delta = np.abs(mine.astype(np.float64) - theirs.astype(np.float64)).max() if mine.size else 0.0
    ok = bool(np.array_equal(mine, theirs)) if atol == 0.0 else bool(delta <= atol)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f" max|d|={delta:.3e}"))
    return ok


def main():
    failures = 0
    for height, width, num_frames, num_text_tokens, anchors in CASES:
        print(f"case {height}x{width} frames={num_frames} text={num_text_tokens} anchors={anchors}")
        rng = np.random.default_rng(0)
        text_token_tags = rng.integers(0, 2, size=num_text_tokens, dtype=np.int64)
        # Text rows are tagged 1 except the rows of a keyframe vision block, which are tagged 0.
        layout = h3_layout.H3Layout(height, width, num_frames, text_token_tags, keyframe_anchors=anchors)

        reference = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.from_numpy(text_token_tags),
            layout.num_latent_frames,
            layout.latent_height,
            layout.latent_width,
            layout.num_audio_latents,
            (1, 2, 2),
            h3_layout.AUDIO_CHANNELS,
            h3_layout.AUDIO_TAG,
            h3_layout.VIDEO_TAG,
            anchors,
        )
        ref_pos, ref_tags, ref_video, ref_audio, ref_text, ref_cond_video, ref_cond_audio = reference

        failures += not check("position_ids", layout.position_ids, ref_pos)
        failures += not check("token_tags", layout.token_tags, ref_tags)
        failures += not check("video_indices", layout.video_indices, ref_video)
        failures += not check("audio_indices", layout.audio_indices, ref_audio)
        failures += not check("text_indices", layout.text_indices, ref_text)
        failures += not check("num_condition_video_rows", layout.num_condition_video_rows, ref_cond_video)
        failures += not check("num_condition_audio_rows", layout.num_condition_audio_rows, ref_cond_audio)

        for steps in (4, 5, 9, 33):
            video_scheduler = MiniMaxH3Scheduler(shift=h3_layout.VIDEO_FLOW_SHIFT)
            audio_scheduler = MiniMaxH3Scheduler(shift=h3_layout.AUDIO_FLOW_SHIFT)
            video_scheduler.set_timesteps(steps)
            audio_scheduler.set_timesteps(steps)
            # The standalone grid can land one float32 ulp off because torch.linspace rounds differently.
            failures += not check(
                f"sigma_schedule[{steps}]", h3_layout.sigma_schedule(steps, h3_layout.VIDEO_FLOW_SHIFT),
                video_scheduler.sigmas, atol=1e-7,
            )
            # Everything downstream is driven from the reference grids, the way the exporter bakes them.
            schedule = h3_layout.H3Schedule(
                steps,
                layout,
                video_sigmas=video_scheduler.sigmas.numpy(),
                audio_sigmas=audio_scheduler.sigmas.numpy(),
            )
            failures += not check(f"timesteps[{steps}]", schedule.timesteps, video_scheduler.timesteps)
            failures += not check(f"audio_timesteps[{steps}]", schedule.audio_timesteps, audio_scheduler.timesteps)

            for index in range(schedule.num_steps):
                ref_unique, ref_inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                    torch.from_numpy(layout.video_indices),
                    torch.from_numpy(layout.audio_indices),
                    layout.num_condition_video_rows,
                    layout.num_condition_audio_rows,
                    layout.num_text_tokens,
                    float(video_scheduler.timesteps[index]),
                    float(audio_scheduler.timesteps[index]),
                    max(float(video_scheduler.timesteps[index]), h3_layout.KEYFRAME_NOISE_AUG),
                    1.0,
                )
                unique, inverse = schedule.row_timestep_plan[index]
                failures += not check(f"plan[{steps}][{index}].timestep", unique, ref_unique)
                failures += not check(f"plan[{steps}][{index}].indices", inverse, ref_inverse)

            # One Euler step against the reference scheduler.
            rng = np.random.default_rng(steps)
            sample = rng.standard_normal((7, 96)).astype(np.float32)
            velocity = rng.standard_normal((7, 96)).astype(np.float32)
            video_scheduler.set_timesteps(steps)
            ref_next = video_scheduler.step(
                torch.from_numpy(velocity), video_scheduler.timesteps[0], torch.from_numpy(sample), return_dict=False
            )[0]
            failures += not check(f"euler[{steps}]", schedule.step(sample, velocity, 0), ref_next)

        rng = np.random.default_rng(1)
        latents = rng.standard_normal(
            (24, layout.num_latent_frames, layout.latent_height, layout.latent_width)
        ).astype(np.float32)
        rows = h3_layout.patchify_video_latents(latents)
        ref_rows = patchify_video_latents(torch.from_numpy(latents)[None], (1, 2, 2))
        failures += not check("patchify", rows, ref_rows)
        failures += not check(
            "unpatchify roundtrip",
            h3_layout.unpatchify_video_latents(
                rows, layout.num_latent_frames, layout.latent_height, layout.latent_width
            ),
            latents,
        )

        cos, sin = layout.rope_cos_sin()
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

        ref_cos, ref_sin = MiniMaxH3RotaryPosEmbed()(torch.from_numpy(layout.position_ids))
        # The angles are bit-exact; numpy and torch round their vectorized cos/sin differently by one float32 ulp.
        failures += not check("rope cos", cos, ref_cos, atol=1e-7)
        failures += not check("rope sin", sin, ref_sin, atol=1e-7)

    print()
    if failures:
        print(f"FAILED: {failures} mismatch(es)")
        return 1
    print("all layout / schedule / rope / patchify checks match diffusers exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
