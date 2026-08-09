# SPDX-License-Identifier: Apache-2.0
"""Packed-sequence layout of MiniMax-H3, independent of diffusers.

MiniMax-H3 runs one stack of blocks over a single packed 1-D sequence holding the text condition, the
keyframe conditioning rows, the target audio rows and the target video rows, in that order. Everything the
transformer needs to describe that sequence -- the fp64 `(t, h, w)` rotary grid, the per-row modality tag, the
per-row timestep index and the three row-index tensors -- is derived here so the export, the alignment harness
and the MNN runtime all read one definition.

Kept numerically identical to `diffusers.modular_pipelines.minimax_h3`: the spatial grids reproduce
`numpy.linspace(endpoint=False)` in float64, and the `"last"` keyframe anchor keeps numpy's pairwise summation
because the reference computes it that way.
"""

from __future__ import annotations

import numpy as np

# Per-row modality tags. They index the checkpoint's AdaLN table, so the values are a contract.
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
MODALITY_NUM = 3

FPS = 24
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
KEYFRAME_NOISE_AUG = 0.999

VAE_SPATIAL_RATIO = 16
VAE_FRAMES_PER_CHUNK = 17
VAE_LATENTS_PER_CHUNK = 5

ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
ROPE_SPATIAL_SCALE = 32

VIDEO_FLOW_SHIFT = 12.0
AUDIO_FLOW_SHIFT = 3.0


def align_num_frames(num_frames):
    """Snap a frame count up to the next `17 * n + 5` the video VAE can encode."""
    num_frames = max(int(num_frames), VAE_LATENTS_PER_CHUNK)
    while num_frames % VAE_FRAMES_PER_CHUNK != VAE_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames):
    if num_frames % VAE_FRAMES_PER_CHUNK != VAE_LATENTS_PER_CHUNK:
        raise ValueError(f"num_frames must be of the form 17 * n + 5, got {num_frames}")
    return (num_frames - VAE_LATENTS_PER_CHUNK) // VAE_FRAMES_PER_CHUNK * VAE_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames):
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def _spatial_position_grid(dim, patch, sqrt_area):
    """One aspect-normalized spatial rotary axis, `dim // patch` coordinates scaled up by 32."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # np.linspace(endpoint=False) is `start + arange(num) * (stop - start) / num`, which torch.linspace is not.
    return np.linspace(left, left + ratio, dim // patch, endpoint=False).astype(np.float64) * ROPE_SPATIAL_SCALE


def _temporal_position_grid(num_latent_frames, origin):
    """Rotary time of every latent frame. Spacing is non-uniform: `5/3 * (1, 4, 4, 4, 4)`."""
    spans = np.array(
        [
            ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[index % len(ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=np.float64,
    )
    return origin + np.concatenate([np.zeros(1, dtype=np.float64), spans[:-1].cumsum()])


def _frame_position_grid(latent_height, latent_width, patch_h, patch_w):
    """The `(h, w)` rotary coordinates of one latent frame, and the width axis they were built from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = np.meshgrid(height_grid, width_grid, indexing="ij")
    return np.stack([grid.reshape(-1) for grid in grids], axis=-1), width_grid


def _last_anchor_time(num_latent_frames, origin):
    spans = np.ones(num_latent_frames, dtype=np.float64) * ROPE_FRAME_RESCALE
    for offset in range(len(ROPE_FRAMES_PER_LATENT)):
        spans[offset :: len(ROPE_FRAMES_PER_LATENT)] *= ROPE_FRAMES_PER_LATENT[offset]
    return origin + float(spans.sum()) - ROPE_FRAME_RESCALE


class H3Layout:
    """Resolved geometry and packed layout of one `t2va` / `fl2va` request.

    Attributes hold plain numpy arrays: `position_ids` is `(seq_len, 3)` float64, `token_tags` is `(seq_len,)`
    int64 and the three index tensors are int64 positions into the packed sequence.
    """

    def __init__(
        self,
        height,
        width,
        num_frames,
        text_token_tags,
        patch_size=(1, 2, 2),
        keyframe_anchors=(),
    ):
        self.height = int(height)
        self.width = int(width)
        self.num_frames = align_num_frames(num_frames)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.keyframe_anchors = tuple(keyframe_anchors)

        canvas_multiple = VAE_SPATIAL_RATIO * self.patch_size[2]
        if self.height % canvas_multiple or self.width % canvas_multiple:
            raise ValueError(
                f"height and width must be multiples of {canvas_multiple}, got {self.height}x{self.width}"
            )

        self.num_latent_frames = video_latent_num_frames(self.num_frames)
        self.latent_height = self.height // VAE_SPATIAL_RATIO
        self.latent_width = self.width // VAE_SPATIAL_RATIO
        self.num_audio_latents = audio_latent_num_frames(self.num_frames)

        self.text_token_tags = np.asarray(text_token_tags, dtype=np.int64)
        self._build()


    @property
    def rows_per_frame(self):
        _, patch_h, patch_w = self.patch_size
        return (self.latent_height // patch_h) * (self.latent_width // patch_w)

    def _build(self):
        _, patch_h, patch_w = self.patch_size
        rows_per_frame = self.rows_per_frame
        num_text_tokens = int(self.text_token_tags.shape[0])
        num_condition_rows = len(self.keyframe_anchors) * rows_per_frame
        num_audio_rows = self.num_audio_latents * AUDIO_CHANNELS
        num_video_rows = self.num_latent_frames * rows_per_frame
        sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

        condition_start = num_text_tokens
        audio_start = condition_start + num_condition_rows
        video_start = audio_start + num_audio_rows

        position_ids = np.zeros((sequence_length, 3), dtype=np.float64)
        position_ids[:num_text_tokens, 0] = np.arange(num_text_tokens, dtype=np.float64)

        frame_grid, width_grid = _frame_position_grid(self.latent_height, self.latent_width, patch_h, patch_w)

        for index, anchor in enumerate(self.keyframe_anchors):
            if anchor == "first":
                anchor_time = float(num_text_tokens)
            elif anchor == "last":
                anchor_time = _last_anchor_time(self.num_latent_frames, float(num_text_tokens))
            else:
                raise ValueError(f"a keyframe anchor must be 'first' or 'last', got {anchor!r}")
            rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
            position_ids[rows, 0] = anchor_time
            position_ids[rows, 1:] = frame_grid

        # Audio rows are channel-major, share the video's rotary clock, carry no height coordinate and are pinned
        # to the two extremes of the width grid.
        audio_time = float(num_text_tokens) + np.arange(self.num_audio_latents, dtype=np.float64)
        position_ids[audio_start:video_start, 0] = np.tile(audio_time, AUDIO_CHANNELS)
        position_ids[audio_start:video_start, 2] = np.concatenate(
            [
                np.full(self.num_audio_latents, float(width_grid[0]), dtype=np.float64),
                np.full(num_audio_rows - self.num_audio_latents, float(width_grid[-1]), dtype=np.float64),
            ]
        )

        video_position_ids = np.empty((self.num_latent_frames, rows_per_frame, 3), dtype=np.float64)
        video_position_ids[:, :, 0] = _temporal_position_grid(self.num_latent_frames, float(num_text_tokens))[:, None]
        video_position_ids[:, :, 1:] = frame_grid[None]
        position_ids[video_start:] = video_position_ids.reshape(-1, 3)

        video_indices = np.concatenate(
            [np.arange(condition_start, audio_start), np.arange(video_start, sequence_length)]
        ).astype(np.int64)
        audio_indices = np.arange(audio_start, video_start, dtype=np.int64)
        text_indices = np.arange(num_text_tokens, dtype=np.int64)

        token_tags = np.empty(sequence_length, dtype=np.int64)
        token_tags[text_indices] = self.text_token_tags
        token_tags[audio_indices] = AUDIO_TAG
        token_tags[video_indices] = VIDEO_TAG

        self.sequence_length = sequence_length
        self.num_text_tokens = num_text_tokens
        self.position_ids = position_ids
        self.token_tags = token_tags
        self.video_indices = video_indices
        self.audio_indices = audio_indices
        self.text_indices = text_indices
        self.num_condition_video_rows = num_condition_rows
        self.num_condition_audio_rows = 0

    def rope_cos_sin(self, rope_freq_dim=16, rope_theta=10000.0):
        """The `(seq_len, 6 * rope_freq_dim)` rotary tables the transformer rotates its leading channels with."""
        inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, 2 * rope_freq_dim, 2, dtype=np.float32) / (2 * rope_freq_dim))
        )
        freqs = self.position_ids.astype(np.float32)[:, :, None] * inv_freq.reshape(1, 1, -1)
        freqs = np.concatenate([freqs[:, 0], freqs[:, 1], freqs[:, 2]], axis=-1)
        freqs = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(freqs), np.sin(freqs)


def sigma_schedule(num_inference_steps, shift):
    """`linspace(1, 0, n)` pushed through the exponential shift, consecutive duplicates collapsed.

    `torch.linspace` rounds its float32 grid differently from numpy at some step counts, so this can land one
    float32 ulp away from the reference schedule. The grid is a handful of scalars, so the exporter bakes the
    reference values into the model config and the runtime reads them rather than recomputing them.
    """
    if num_inference_steps < 2:
        raise ValueError(f"num_inference_steps must be >= 2, got {num_inference_steps}")
    base = np.linspace(1.0, 0.0, int(num_inference_steps), dtype=np.float32)
    sigmas = (shift * base / (1 + (shift - 1) * base)).astype(np.float32)
    keep = np.concatenate([[True], sigmas[1:] != sigmas[:-1]])
    return sigmas[keep]


class H3Schedule:
    """The two rectified-flow schedules of one request plus the per-step row-to-timestep plan."""

    def __init__(
        self,
        num_inference_steps,
        layout,
        video_shift=VIDEO_FLOW_SHIFT,
        audio_shift=AUDIO_FLOW_SHIFT,
        video_sigmas=None,
        audio_sigmas=None,
    ):
        self.video_sigmas = (
            sigma_schedule(num_inference_steps, video_shift)
            if video_sigmas is None
            else np.asarray(video_sigmas, dtype=np.float32)
        )
        self.audio_sigmas = (
            sigma_schedule(num_inference_steps, audio_shift)
            if audio_sigmas is None
            else np.asarray(audio_sigmas, dtype=np.float32)
        )
        self.timesteps = (1.0 - self.video_sigmas[:-1]).astype(np.float32)
        self.audio_timesteps = (1.0 - self.audio_sigmas[:-1]).astype(np.float32)
        self.num_steps = int(self.timesteps.shape[0])
        self.layout = layout
        self.row_timestep_plan = [self._plan(index) for index in range(self.num_steps)]

    def _plan(self, index):
        layout = self.layout
        row_timesteps = np.full(layout.sequence_length, self.timesteps[index], dtype=np.float32)
        row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = max(
            float(self.timesteps[index]), KEYFRAME_NOISE_AUG
        )
        row_timesteps[layout.audio_indices[layout.num_condition_audio_rows :]] = self.audio_timesteps[index]
        row_timesteps[layout.audio_indices[: layout.num_condition_audio_rows]] = 1.0
        unique, inverse = np.unique(row_timesteps, return_inverse=True)
        return unique.astype(np.float32), inverse.astype(np.int64).reshape(-1)

    def adaln_rows(self, index):
        """The `(timestep_index, modality)` table rows the packed sequence actually addresses at one step."""
        unique, inverse = self.row_timestep_plan[index]
        rows = inverse * MODALITY_NUM + self.layout.token_tags
        return np.unique(rows), unique

    def step(self, sample, velocity, index, audio=False):
        """One Euler step. The velocity is data-ward, so `x0 = x_t + (1 - t) * v`."""
        sigmas = self.audio_sigmas if audio else self.video_sigmas
        timesteps = self.audio_timesteps if audio else self.timesteps
        denoised = sample + (1.0 - timesteps[index]) * velocity
        ratio = sigmas[index + 1] / sigmas[index]
        return (ratio * sample + (1.0 - ratio) * denoised).astype(np.float32)


def patchify_video_latents(latents, patch_size=(1, 2, 2)):
    """`(C, F, H, W)` video latents to `(num_rows, C * prod(patch))` rows, frame-major then row-major."""
    patch_t, patch_h, patch_w = patch_size
    channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"latents of shape {latents.shape} are not divisible by the patch {patch_size}")
    latents = latents.reshape(
        channels, num_frames // patch_t, patch_t, height // patch_h, patch_h, width // patch_w, patch_w
    )
    latents = latents.transpose(1, 3, 5, 0, 2, 4, 6)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w)


def unpatchify_video_latents(rows, num_latent_frames, latent_height, latent_width, channels=24, patch_size=(1, 2, 2)):
    """Inverse of `patchify_video_latents`, back to `(C, F, H, W)`."""
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.transpose(3, 0, 4, 1, 5, 2, 6)
    return rows.reshape(channels, num_latent_frames, latent_height, latent_width)
