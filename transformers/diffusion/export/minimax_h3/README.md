# MiniMax-H3 on MNN

MiniMax-H3 generates video with native stereo audio. This directory holds the export, conversion and
validation tooling; the runtime lives in `transformers/diffusion/engine/` as `MinimaxH3Diffusion`
(`DiffusionModelType::MINIMAX_H3`).

The first release covers **T2VA / FL2VA at a fixed low-resolution layout with precomputed conditioning**.
Ref2VA, the audio VAE, the on-device H3 encoder and the QNN path are structured for but not part of it -- see
[Not yet done](#not-yet-done).

## What H3 is, and what that costs

H3 runs one stack of 50 blocks over a **single packed 1-D sequence** holding, in order, the text condition,
the keyframe conditioning rows, the target audio rows and the target video rows. Attention over that sequence
is full and non-causal; there is no cross-attention. The checkpoint is guidance-distilled, so there is no
unconditional pass and no CFG -- one forward per step, batch 1.

| | |
|---|---|
| layers / hidden / heads / head dim | 50 / 5376 / 56 / 128 (`heads * head_dim = 7168 > hidden`) |
| FFN | SwiGLU, inner 14336 |
| video / audio latent channels | 24 / 32 |
| patch | `(1, 2, 2)`, so a video row is 96 channels |
| text dim | 5120 (Qwen3-VL-32B hidden states) |
| VAE | f16 spatial, t4 temporal, 24 latent channels, ViT decoder |
| frames | `17n + 5`, 24 fps |
| modality tags | 0 video, 1 text, 2 audio |

Two facts about the checkpoint drive the whole design.

**The AdaLN branch is 13B of the 33B and is a constant.** Every `adaln_proj` reads only the timestep
embedding, and the timesteps of a fixed schedule are known offline. `h3_adaln.py` folds all 50 projections
into a per-`(step, layer)` table:

```
26 GB of bfloat16 AdaLN weights  ->  78 MB float32 table
```

That is a 340x reduction and it is what makes the model fit on a phone at all. The table is float32 on
purpose: float16 storage costs ~2.8e-3 relative RMS on a block output -- the same order as bfloat16 itself --
and the reference notes this rounding biases every block identically at every step, so it accumulates
coherently along the trajectory.

**The packed sequence is grouped by `(timestep, modality)`.** For T2VA the runs are exactly
`[text | audio | video]`, so the per-row AdaLN modulation is a broadcast over three contiguous runs instead of
a gather of six `(seq_len, 5376)` tensors per block. At the reference layout that is 137 MB of transient
allocation per block that never happens. `compact_adaln` checks the assumption and refuses layouts that would
need a row permutation first.

## Layout of the first target

`448x256`, 56 frames, no keyframe, 37 text tokens:

```
sequence = 37 text + 0 condition + 186 audio + 1904 video = 2127 rows
video   = 17 latent frames x (16/2 x 28/2) = 17 x 112 rows
audio   = round(56/24 * 40) = 93 latents x 2 channels, channel-major
AdaLN runs = [(37, 1), (186, 2), (1904, 0)]
```

`num_inference_steps=5` drives **4** model evaluations: the schedule is
`sigma = linspace(1, 0, n)` pushed through `s*sigma / (1 + (s-1)*sigma)` with `s = 12` for video and `s = 3`
for audio, and the terminal zero is part of the grid. The two modalities step down their own schedules in the
same forward.

Attention is only ~7% of the arithmetic at this size; the FFN is ~56%. That flips completely at 768p/8s,
where the sequence is ~59k rows and attention alone is ~98 TFLOP per layer -- which is why the low-resolution
layout is the first target and 2K is out of scope.

## Pipeline

```
                        h3_reference.py            (golden fixtures from diffusers)
                               |
HuggingFace checkpoint  --> h3_onnx_export.py --> MNNConvert --> h3_*.mnn + .mnn.weight
                               |                                        |
                          h3_adaln.py  --> h3_adaln.bin, h3_rope.bin    |
                                                       \               /
                                                    minimax_h3_demo (MinimaxH3Diffusion)
                                                               |
                                                    video_latent_rows.bin
                                                               |
                                                      h3_vae_decode.py --> .mp4
```

The DiT is exported as **three kinds of graph**, not one:

* `h3_embed` -- the three input projections, the 2-block text refiner, and the packing.
* `h3_blocks_{g}` -- one sequential slice of the block stack (5 layers by default).
* `h3_head` -- the final norm and the two output heads.

Partitioning is not cosmetic. It keeps each ONNX proto, each MNN weight file and -- on NPU backends -- each
context binary bounded, so nothing ever has to build or finalize a single 20B graph. The runtime holds all
partitions resident and walks them in order.

Shapes are static. QNN requires that anyway, and it is what lets the AdaLN modulation be a per-run broadcast.
A request therefore has to match the layout the resources were exported for.

## Text conditioner

MiniMax-H3 conditions on `hidden_states[50]` of Qwen3-VL-32B: the output of the 50th decoder layer, **before**
the model's final norm. That last detail is load-bearing -- a stack truncated to exactly 50 layers hands back
`norm(after layer 50)`, which is not what the released weights were trained against. For a text-only request
neither the vision tower nor the language-model head is ever touched, and the prompt is tokenized with no
special tokens and no chat template.

So the export covers the 50 layers and nothing else: 24.4B parameters in ten partitions, plus a 778M token
embedding table that is **gathered on the host** rather than run as a graph -- a prompt touches a few hundred KB
of a 1.5 GB table.

Two things are baked rather than reimplemented. Qwen3-VL's *interleaved* mrope is captured from the reference
rotary module, and the causal mask is a constant of the sequence length. Because attention is causal, a prompt
shorter than the exported `max_tokens` needs no padding mask at all: a real token never attends to a padding
slot, and the padding rows are dropped from the output.

| stage | cosine | relative RMS |
|---|---|---|
| embedding gather vs reference | 1.000000 | 0 (bit-exact) |
| captured mrope vs reference rotary | — | 0 (bit-exact) |
| export stack vs truncated reference model | 1.000000 | 2.4e-7 |
| MNN W4 conditioner vs export stack (50 layers) | 0.999865 | 2.0e-2 |
| MNN tokenizer vs HuggingFace | — | identical ids |

W4 is the default: 15.3 GB, and the conditioner runs once per generation and is released before the transformer
loads. The 2.0e-2 is the price of W4 across 50 layers; W8 would be 27.5 GB, which no longer fits a 24 GB device,
so a mixed-precision conditioner is the knob if conditioning fidelity turns out to matter.

`h3_encoder_verify.py` is the gate. It loads **one layer more than it reads**, for the post-norm reason above --
getting that wrong is a silent 0.70-cosine failure rather than an error.

## Video VAE decoder

The decoder is a non-causal ViT over latent voxels -- one token per voxel, 36 blocks of 2048 channels, 2.42B
parameters -- plus four register tokens and a zero token that ride along at position 0 and are dropped before
the patch projection. The exported graph is **one spatial tile of one temporal clip**, a fixed shape.

Everything around that tile is index arithmetic that depends only on the latent shape, so the exporter computes
it into `h3_vae_plan.json` and the runtime replays it: temporal chunks that overlap because the encoder's
`token_drop` removed each chunk's tail, cross-faded over `frame_overlap` pixel frames; spatial tiles with their
own overlaps, blended the same way; and the trailing pixel frames padded latent frames produced.

For `448x256 / 56 frames` that resolves to 3 temporal chunks x 2 spatial tiles, tile latent `7x16x16` ->
`28x256x256` pixels, 1797 tokens per graph call, and exactly 56 frames out.

| stage | cosine | relative RMS |
|---|---|---|
| export-friendly tile decoder vs reference | 1.000000 | 6.1e-7 |
| MNN W8 tile vs reference | 0.999951 | 1.0e-2 |
| MNN W4 tile vs reference | 0.985537 | 1.7e-1 |
| **C++ end-to-end video luma vs reference** | **0.999979** | **6.5e-3** |

W8 is the default: the decoder runs once per generation, W4 costs 17x the error, and W8 is 2.58 GB.

Two converter notes:

* `--transformerFuse` aborts on this graph. `GetFmhaV2NumHeads` in
  `tools/converter/source/optimizer/merge/FuseFmhaV2.cpp` indexed `inputs().at(1)` on Reshapes whose shape
  operand was folded into the op, and computed a negative index guarded only by `MNN_ASSERT`, which compiles
  away in release builds. MNN builds with `-fno-exceptions`, so an out-of-range `.at()` aborts the converter
  rather than reporting a non-match. Those two are fixed here, but the pass still aborts somewhere else on this
  graph, so the decoder is converted **without** `--transformerFuse` and its attention stays
  MatMul/Softmax/MatMul -- a transient `32 x 1797^2` score tensor per block.
* The reference decodes under a float16 autocast. That is off by default in `h3_vae_decode.py` because it makes
  the decoder's 24-channel input projection a `k=24` half GEMM that cuBLAS rejects.

## Build

```bash
cd transformers/diffusion/export/minimax_h3

python h3_build_mnn.py \
    --model_path /path/to/MiniMax-H3/transformer \
    --output /path/to/h3_mnn \
    --num_text_tokens 37 \
    --layers_per_group 5 \
    --quant_bit 4 \
    --vae /path/to/MiniMax-H3/vae \
    --text_encoder /path/to/MiniMax-H3/text_encoder --max_text_tokens 256
```

The conditioner also needs `tokenizer.mtok`, which MNN's own tokenizer exporter produces from the H3 processor:

```python
from utils.tokenizer import LlmTokenizer      # transformers/llm/export
LlmTokenizer(processor_dir, "qwen3_vl").export(resource_dir, model_path=processor_dir, model_type="qwen3_vl")
```

`--skip_transformer` reuses the transformer modules already in `--output`, for iterating on one stage.

`h3_build_mnn.py` walks the partitions one at a time and deletes each ONNX after converting it, so peak
scratch is one partition (~8 GB) rather than the whole stack (~80 GB).

The C++ side needs `MNN_BUILD_DIFFUSION=ON`, which in turn needs `MNN_BUILD_OPENCV=ON` and
`MNN_IMGCODECS=ON`, plus `MNN_BUILD_LLM=ON` and `MNN_SUPPORT_TRANSFORMER_FUSE=ON`:

```bash
cmake .. -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_DIFFUSION=ON -DMNN_BUILD_LLM=ON \
         -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_LOW_MEMORY=ON \
         -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_CUDA=ON
```

`MNN_SUPPORT_TRANSFORMER_FUSE` is required: without it `OpType_Attention` is not registered and the graph
falls back to raw MatMul/Softmax, which materializes the full `56 x 2127 x 2127` score tensor.

## Run

Prompt to video, entirely in C++ and MNN. Each stage is released before the next loads, which is both what a
24 GB device needs and what the on-device pipeline is meant to look like:

```bash
# <resource_dir> <prompt> <output.y4m> [backend] [seed] [resident_groups] [fps] [reuse_latents]
./minimax_h3_video_demo /path/to/h3_mnn "A red fox leaps over a mossy log ..." out.y4m cuda 0 1
ffmpeg -i out.y4m -c:v libx264 -pix_fmt yuv420p out.mp4      # optional, Y4M is already playable
```

Measured on one RTX 3090, `448x256 / 56 frames`, 4 steps, W4 conditioner and transformer, W8 VAE:

```
conditioner   37 tokens -> 5120 channels
transformer   4 steps, 78 - 89 s/step
VAE           3 chunks x 2 tiles -> 56 frames
peak device   15959 MiB
```

Passing a `.bin` path instead of a prompt takes it as precomputed conditioning and skips the conditioner, which
is how to run the transformer before the conditioner's resources are built. `reuse_latents=1` skips denoising
too and only re-decodes.

`resident_groups 1` is required on CUDA at float32; see [Backend findings](#backend-findings-from-the-full-stack-runs).
Passing `reuse_latents=1` skips denoising and decodes the latents already written, which is how to iterate on
the decoder without re-running the transformer.

The transformer alone, writing latents rather than pixels:

```bash
./minimax_h3_demo /path/to/h3_mnn prompt_embeds.bin /path/to/out cuda 0 0 1 high
python h3_vae_decode.py --vae /path/to/MiniMax-H3/vae --latents /path/to/out --output out.mp4
```

`prompt_embeds.bin` is `num_text_tokens * 5120` little-endian float32 -- the H3 encoder's hidden states. The
encoder is not in the device pipeline yet, so conditioning arrives as a tensor.

## Numerical validation

Every level is checked against a golden trace dumped from the reference diffusers implementation.
`h3_reference.py` writes out each intermediate a port has to reproduce and **verifies its own decomposition
against the real `MiniMaxH3TransformerBlock`**, so the trace is a verified spec rather than a second
implementation.

| gate | command | result |
|---|---|---|
| layout, schedule, rope, patchify | `h3_layout_parity.py` | 605 checks bit-exact vs diffusers |
| AdaLN fold and timestep MLP | inline in `h3_adaln.py` | bit-exact |
| export-friendly modules | `h3_module_parity.py` | cosine >= 0.99994 |
| MNN graphs, any backend | `h3_mnn_align.py` | see below |

Measured on the `448x256 / 56` layout, block 0, against an **fp32** golden reference:

| configuration | cosine | relative RMS |
|---|---|---|
| MNN CPU fp32, fp32 AdaLN table | 1.000000 | 8.0e-7 |
| MNN CPU fp32, fp16 AdaLN table | 1.000000 | 2.8e-3 |
| MNN CUDA fp32 | 0.999985 | 5.5e-3 (*) |
| MNN CUDA fp16 | 0.999984 | 5.8e-3 (*) |

(*) against the **bfloat16** golden reference, whose own quantization noise is the dominant term -- the fp32
comparison above shows the graph itself is exact.

Weight quantization of the block stack, 4 layers chained, vs the fp32 golden:

| configuration | cosine | relative RMS | bytes / param |
|---|---|---|---|
| float | 1.000000 | 8.0e-7 | 4 |
| W8 asym, block 64 | 0.999996 | 2.7e-3 | 1.13 |
| W4 asym, block 64 | 0.999374 | 3.5e-2 | 0.63 |

`--hqq` produced numerically identical output to plain W4 here, i.e. it did not take effect on these shapes.

**The embed and head stages default to float weights.** W4 there costs ~1e-1 relative RMS, and it buys
nothing: the text refiner runs once per generation rather than once per step, and the two output heads are
under a million parameters combined.

H3 activations carry outliers **hundreds of times their RMS** (a block-0 output has RMS ~51 and absmax
~35000). Any activation-quantization scheme has to account for that; an absmax-based threshold is meaningless
on these tensors, which is why every gate here reports cosine similarity and relative RMS.

Two tolerances are documented rather than fixed, because they are irreducible:

* `numpy.cos/sin` and `torch.cos/sin` differ by one float32 ulp on identical input, so the rope tables match
  to 1e-7 rather than bit-exactly. The tables are baked at export time, so the runtime never recomputes them.
* `torch.linspace` rounds its float32 grid differently from numpy at some step counts, so the sigma grid can
  land one ulp off. The exporter bakes the reference schedule into the manifest.

### Full 50-layer end-to-end run

`448x256 / 56 frames`, 4 steps, W4 blocks, float embed, CPU (4 threads), fp32 activations:

```
load                          55.9 s   (12 modules, 14 GB)
step 1..4              584 - 686 s     per step
total                       2558.7 s
video latent rows       rms 1.0161, absmax 3.720, 0 NaN
audio latent rows       rms 0.4256,             0 NaN
decoded                 56 frames at 448x256, 24 fps, 0.61 MB H.264
```

The latent statistics are the useful signal here. The VAE denormalizes with `latents_std`, so a correct DiT
output should sit near unit variance -- and it does, at rms 1.016. The same layout with only 4 of the 50 layers
gives rms 25.3, i.e. the check discriminates. The decoded frames carry clear spatial structure rather than
noise.

This run predates the conditioner, so its conditioning was random. With the conditioner wired up the same
layout produces a prompt-faithful video -- for "A red fox leaps over a mossy log in a sunlit pine forest ...",
a pine forest lit from behind, a mossy log across the foreground and a fox moving along it over the 56 frames.

## Backend findings from the full-stack runs

Two properties of H3 constrain which backends can run it, and both were found by running the whole stack
rather than by inspection.

**`Memory_Low` silently produces NaN on CUDA.** The cutlass convolutions stage their whole *dequantized*
weight in the CUDA static pool while loading -- 616 MB for the widest feed-forward projection -- and under
`Memory_Low` that allocation fails. `ConvCutlassExecution::Resource` only logs `CUDA alloc failed` and returns
from its constructor, leaving the weights unfilled, so the symptom is an all-NaN forward with no error. The
engine now refuses to hand CUDA that configuration. This was the first of two independent NaN sources and is
worth knowing before blaming precision for anything.

**Float16 activations really do overflow.** With the staging fixed, float16 still returns all-NaN latents:
H3's residual stream reaches absmax ~3.5e4 against float16's 6.55e4 ceiling and grows across 50 blocks. A
single block passes at float16 (cosine 0.999984), so this only appears at full depth. `Precision_Low_BF16`
would have the range, but MNN CUDA's `ConvCutlassBf16Execution` fails to allocate for these shapes
(`code=2`, 30 staging failures), so **float32 is the only working CUDA precision today**.

**MNN CUDA expands quantized weights at load.** Measured on a 3090 at float16:

| resident layers | peak device memory |
|---|---|
| 10 | 9105 MiB |
| 25 | 21255 MiB |

That is ~810 MiB per layer against 241 MiB of actual W4 weight -- the int4 weights are materialized, and there
is no low-bit GEMM. At float32 it is ~1.6 GB per layer, so the 50-layer stack needs ~77 GB, and MNN CUDA has no
multi-GPU support. `setResidentGroups(n)` keeps only a trailing window of partitions resident:

| window | precision | peak | per step | result |
|---|---|---|---|---|
| 3 | fp16 | 13193 MiB | 11.5 s | NaN (overflow) |
| 3 | fp32 | 24065 MiB | -- | OOM |
| 2 | fp32 | 23121 MiB | -- | OOM |
| **1** | **fp32** | **15959 MiB** | **77.5 s** | **rms 0.991, cosine 0.9857 vs CPU** |

So one partition resident at float32 is the working CUDA configuration. The reloads come out of the page cache
rather than storage. This is a CUDA-specific workaround, **not** the intended device design -- re-reading
weights every step is exactly what the phone budget must avoid, and a backend with native low-bit kernels wants
every partition resident (the default, `0`).

Fixing either MNN-side issue -- a CUDA int4 GEMM, or a working `Precision_Low_BF16` path for these convolutions
-- removes the constraint.

## Memory budget

Snapdragon 8 Elite / 24 GB shared, at the `448x256 / 56` layout, W4 blocks and float embed:

| | |
|---|---|
| block stack, W4 asym block 64 | ~12.1 GB |
| embed (float) | ~3.2 GB |
| head (float) | ~3 MB |
| AdaLN table, float32, 4 steps | 78 MB |
| rope tables | 1.6 MB |
| attention mask, `2127^2` float32 | 18 MB |
| activation working set, one partition | ~1 GB |

What the design already avoids:

* 26 GB of AdaLN weights, folded offline.
* Six `(2127, 5376)` gathers per block, replaced by three broadcasts.
* One 20B graph or context binary, replaced by 10 partitions.
* A materialized `56 x 2127 x 2127` score tensor, via the fused `OpType_Attention`.

Still to prove on device: that raw and QNN-packed weights do not both stay resident, the context
finalization peak per partition, and HTP RPC / dma-buf accounting.

## Android and Hexagon

The target is a OnePlus 13: `PJZ110`, SM8750, Hexagon `hvxArch 79`, 8 MB VTCM, 6 DSP threads, Android 16,
23.6 GB RAM of which ~15.5 GB is available, 759 GB free storage.

### What works

`project/android/build_64` builds with `MNN_HEXAGON=ON` and produces arm64 `minimax_h3_demo` and
`minimax_h3_video_demo`. One block partition runs on the device CPU and is **numerically identical to x86**:

```
h3_blocks_0 (5 layers, W4), CPU fp32, 4 threads
  cosine 0.999426, rel_rms 3.392e-02 vs the fp32 golden -- the same digits as x86
  85.8 s, 10.9 GB   (MNN expands the int4 weights to fp32, as on CUDA)
```

Two toolchain notes. The Hexagon SDK records `HexagonSDK6x_MinimalNDK` and `HexagonSDK6x_CMake` as installed
while their directories are absent, so `build_cmake` fails looking for `tools/android-ndk-r25c` and
`tools/cmake-3.28.3-linux-x86_64`; pointing those at a real NDK and cmake gets past it. And `updateTest.sh`
assumes adb is local -- with the device on a separate host the flow is build, `scp`, then `adb push` from there.

### The blocker: Hexagon has no bidirectional attention

`forwardtype=10` fails with `code=2` (`NOT_SUPPORT`). The op is `Attention`:

```
[H3DBG] resize failed code=2 type=Attention
  in[0..2] dims=4 shape=1 2127 56 128     Q / K / V
  in[3]    dims=4 shape=1 1 2127 2127     mask
```

`HexagonAttention::onBuildCmd` has exactly two paths, and both are built for LLM decode:

| path | requires | H3 has |
|---|---|---|
| streaming flash-attention | 7 inputs / 3 outputs | 4 inputs / 1 output |
| KV-cache | `V` in NC4HW4 shaped `(kvLen, kvHeads * headDim)` while Q/K are 4-D plain, and mandatorily goes through `HexagonKVCacheManager` | Q/K/V all 4-D plain, no cache |

`--transformerFuseC4=1` does not bridge it either: `hidden-state C4 region matched only 0 / 5 attention blocks`.
And **Hexagon registers no MatMul op**, so leaving attention unfused is not an alternative -- there is no DSP
path for the score matmul at all.

So H3 on Hexagon needs a genuinely new path: 4-input/1-output attention over plain 4-D Q/K/V with an explicit
mask and no KV cache. The hexagon skill rules out routing it to CPU as a solution.

### Two more measured problems

**The exported graph is far too fragmented for a DSP.** One 5-layer partition is 1064 ops, of which only ~55
are arithmetic (30 Convolution, 20 LayerNorm, 5 Attention):

| op | count | on Hexagon |
|---|---|---|
| BinaryOp | 305 | yes |
| StridedSlice | 215 | no |
| Concat | 70 | no |
| Reshape | 65 | no |
| Unsqueeze / Squeeze | 75 | no |
| SliceTf | 60 | no |
| ConvertTensor | 50 | no |
| Shape / Rank | 85 | no |
| Convolution / LayerNorm / Attention / UnaryOp | 70 | yes |

`Shape` and `Rank` are dead -- every shape is static -- and most of the slicing comes from two places in this
exporter: the hand-written rotary (slice into rotary/pass, chunk into halves, concat back) and the segment-wise
AdaLN modulation. **MNN has a native `RoPE` op that Hexagon supports**, so emitting that instead of the
slice/concat form should remove most of the 215 `StridedSlice`.

**Hexagon wants float16 tensors, and H3 overflows float16.** Most Hexagon ops reject non-2-byte floats, and
H3's residual stream already reaches absmax ~3.5e4 against float16's 6.55e4 ceiling at block 0, growing over 50
blocks. On CUDA the answer was float32; on the DSP there is no float32 fallback. This is unresolved and is
independent of the attention blocker.

### Order of work

1. Add a non-causal, cache-free attention path to the Hexagon backend. This is the blocker.
2. Slim the export: drop the dead shape ops, and emit MNN's `RoPE` op instead of the slice/concat rotary.
3. Only then kernel-level work, and a real answer for the float16 range problem.

QNN (`forwardType=5`) is the untried alternative: its `QNNAttention` does support `kv_cache=false` with 4-D
Q/K/V and an optional mask, which is much closer to what H3 emits. It needs the full Qualcomm QNN SDK, which is
not installed here -- `prepare_qnn_deps.sh` provides arm64-v8a runtime libs only.

## Not yet done

The conditioner's attention does **not** fold into MNN's fused Attention op (`0 fused attention op(s)` at
conversion): the pattern matcher does not recognise the grouped-query `repeat_interleave`. At 256 tokens the
materialized score tensor is 17 MB, so it is not urgent, but it will be at longer prompts.

**On-device (QNN / Android) is unstarted and was not reachable in this environment** -- no Android NDK, no QNN
SDK (`prepare_qnn_deps.sh` fetches QNN 2.37) and no device attached. The entry points are:

* MNN's QNN backend does support the shape H3 needs: `QNNAttention` accepts `kv_cache=false` with 4-D Q/K/V
  and `seqLenQ == seqLenKV`, and the exported graphs already carry `kvcache: 0` and static shapes.
* QNN has **no RoPE op** (the Hexagon backend does, `HexagonExecutionFactory.cpp`). H3's rotary is exported as
  explicit slice/concat/mul/add, so it needs no op -- but it should be measured before being kept.
* Offline context binaries come from `tools/cpp/MNN2QNNModel`, which runs on the host per `socId` /
  `hexagonArch`. Per-partition compilation is the reason for the partitioning, and it can be validated without
  a device.
* The three-process lifecycle (`:h3_encoder`, `:h3_dit`, `:h3_decoder`) has no precedent in the repo -- none of
  the Android apps use `android:process` today.

Also outstanding:

* The video VAE runs through the reference PyTorch decoder. An MNN port is a separate piece of work;
  `h3_vae_decode.py` is its numerical target. The reference's float16 autocast is off by default because it
  makes the decoder's 24-channel input projection a `k=24` half GEMM that cuBLAS rejects.
* The audio VAE is not wired up; the engine writes the audio latent rows and stops.
* The conditioner is exported for a fixed `max_tokens` and the transformer for a fixed text-token count, so a
  prompt has to tokenize to exactly the length the transformer resources were built for.
* Ref2VA uses the `transformer_ref` partition of the checkpoint. The layout code takes the keyframe-anchor
  path already, but `MiniMaxH3Ref2VAPrepareLayoutStep` has its own packing that is not ported.
* The engine seeds its initial noise with `std::mt19937`, so seeds do not reproduce the torch reference.
  Reproducing a reference sample needs the latents fed in rather than drawn.
* Per-op mixed-precision quantization needs weight-level control that `MNNConvert --weightQuantBits` does not
  give; the route is the JSON plus external `.weight` patching that `transformers/llm/export/safetensors2mnn.py`
  uses.
* MNN CUDA has no multi-GPU support, so the block stack has to fit one device. W4 does; float and W8 do not.

## Licensing

MiniMax-H3 weights are covered by the license in the model repository and are not redistributed here. Nothing
in this directory contains weights, quantized artifacts, latents or video. The reference implementation this
tooling is validated against is `diffusers` (Apache-2.0).
