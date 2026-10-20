# ARM NX PoC

For a Chinese/English overview of the design and limitations, see [ARM_NX.md](../../ARM_NX.md).

This experimental ARM NX path keeps MNN's Vulkan buffer backend and emits a single TOSA
`MATMUL` in `SPV_ARM_graph` directly. It does not link ARM's Model Converter,
VGF Library, Workload Library, or Scenario Runner. The graph is compiled by the
Vulkan implementation. ARM's Graph/Tensor Emulation Layers let the same API run
on an ordinary GPU through compute shaders.

MNN uses ARM NX names for the feature, classes, shaders and test tools. The Vulkan
extension and API names remain Data Graph as specified by Vulkan. Enabling the path
or recording a graph dispatch does not prove execution on physical NX hardware.
FP32 MatMul remains available for integration tests; the quantized Linear path uses INT8 MatMul.

## Scope

- Build option `MNN_VULKAN_ARM_NX=ON`, **off by default**.
- Runtime opt-in `MNN_VULKAN_ARM_NX=1`, evaluated when MNN creates its device.
- MNN routing: 2-D, contiguous FP32 MatMul, high precision, no transpose or bias.
  Unsupported configurations and graph creation failures use the original shader.
- Quantized single-input 1x1 Conv / Linear also uses this path, for both prefill
  and decode. W4 and W8 weights retain their original per-channel/per-group
  scale and offset. Activations are dynamically quantized to signed INT8 per
  row/group. Batched `MATMUL` accumulates INT32; a shader restores FP32/FP16 output,
  adds bias and applies ReLU/ReLU6. INT2/INT3 and unsupported groups use existing shaders.
  Groups must divide K and have a size divisible by four.
- Pipelines and sessions are retained by each execution and recreated on shape
  changes. Each owns separate, linear tensors and working memory. Input/output
  bridging uses GPU buffer copies; this is not a zero-copy or performance-tuned path.
  Clones share prepared static weights on the same device. Graphs still own another
  weight allocation and copy weights per execution; memory and transfer costs need
  optimization before any native performance assessment.
- Grouped Linear retains an INT32 accumulator of shape `[groups, M, N]` for each
  execution, plus graph tensor storage and emulation working memory. The current
  PoC does not enforce an aggregate memory budget. A Qwen3-0.6B run with a 128-token
  prefill and 64 decode steps reached about 9.6 GiB physical footprint on this Mac;
  a 512-token attempt caused severe system memory pressure and was stopped.
  Keep emulation smoke prompts short. Increasing sequence length requires a total
  CPU/GPU memory monitor and an automatic stop threshold; process RSS alone
  under-reports these GPU allocations.
- Requires Vulkan 1.3+, a compute queue also supporting data graphs, synchronization2,
  tensors/non-packed tensors/shader tensor access, TOSA and data graph shader modules.
  This initial implementation also requires the maintenance5/deferred-host-operations
  extensions and MNN's shader-int8 support. Imported application-owned devices are
  conservatively left on the existing path, since their enabled features are unknown.

## Mac setup

Apple silicon only. The following combination was tested on macOS 26.2 / Apple M3 Pro:
MoltenVK 1.4.2, Vulkan Loader/Headers/Validation Layers 1.4.363.0,
ARM Emulation Layer 0.11.0, Python 3.12. Darwin emulation is experimental.

Install a recent LunarG Vulkan SDK, or use Homebrew as in this test:

```sh
brew install molten-vk vulkan-loader vulkan-tools vulkan-headers spirv-headers spirv-tools
python3 -m venv /absolute/path/to/arm-vulkan-venv
/absolute/path/to/arm-vulkan-venv/bin/pip install --only-binary=:all: ai-ml-emulation-layer-for-vulkan==0.11.0
```

From the MNN repository root:

```sh
cmake -S . -B build-arm-nx \
  -DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF -DMNN_VULKAN_ARM_NX=ON \
  -DMNN_USE_SYSTEM_LIB=ON -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_SEP_BUILD=OFF \
  -DMNN_METAL=OFF -DMNN_BUILD_TEST=ON -DMNN_BUILD_TOOLS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-arm-nx --target vulkan_arm_nx_test.out -j8
```

The focused test target uses internal backend APIs and is only generated for static
test builds. Shared library builds can enable the experimental backend independently.

In a separate shell, run the `emulation_layer` helper and apply the three environment
exports it prints (layer library path, manifest path and Graph-before-Tensor order):

```sh
/absolute/path/to/arm-vulkan-venv/bin/emulation_layer
# Apply the printed DYLD_LIBRARY_PATH, VK_LAYER_PATH, VK_INSTANCE_LAYERS exports.
vulkaninfo --summary
MNN_VULKAN_ARM_NX=1 ./build-arm-nx/vulkan_arm_nx_test.out --require-arm-nx
```

With the LunarG SDK, source its `setup-env.sh` first and select its MoltenVK ICD.
`vulkaninfo` must report MoltenVK and both ARM layers. No Android device is needed.

## Verification

The focused test checks:

- FP32 and INT8->INT32 graphs against a CPU calculation on four matrix shapes,
  including non-multiples of four and square matrices; five forwards per graph.
- An MNN model `Square -> MatMul -> Add`, using `Module::load/onForward`, in both
  per-op and batch recording modes, repeated forwards and changed shapes.
- Transposed and low-precision MatMul staying on the original shader path.
- Recorded graph counts, so a numerical pass after silent CPU/compute fallback
  cannot satisfy `--require-arm-nx`.
- With `--require-arm-nx`, W4/W8 Linear against an explicit CPU A8 reference,
  symmetric/asymmetric groups, bias, zero activations, unaligned output channels,
  shared-weight Module clones, prefill/decode shape changes, ReLU/ReLU6, FP32/FP16
  storage and both recording modes.

Recorded results on the Mac above: all 40 low-level forwards, 32 MatMul Module forwards
and 64 INT8 Linear forwards passed. The FP32 Module/integer-graph validation run had no Vulkan validation or
synchronization errors. Both explicit-disable and missing-extension fallback runs
passed. With the experiment compiled out, the default Image and Buffer Vulkan
libraries also built successfully using MNN's bundled headers.

Without either ARM layer, explicitly requesting the feature must still pass the
fallback checks:

```sh
env -u VK_INSTANCE_LAYERS MNN_VULKAN_ARM_NX=1 \
  ./build-arm-nx/vulkan_arm_nx_test.out
```

Dump and validate the emitted graph IR:

```sh
./build-arm-nx/vulkan_arm_nx_test.out --dump-spirv /tmp/mnn-matmul
spirv-val --target-env vulkan1.3 /tmp/mnn-matmul-fp32.spv
spirv-val --target-env vulkan1.3 /tmp/mnn-matmul-int8.spv
```

For validation, install Vulkan Validation Layers, add their manifest/library paths,
put `VK_LAYER_KHRONOS_validation` before the two ARM layers, and run:

```sh
VK_LAYER_VALIDATE_SYNC=1 MNN_VULKAN_ARM_NX=1 \
  ./build-arm-nx/vulkan_arm_nx_test.out --require-arm-nx --fp32-only
```

`--fp32-only` excludes the original low-precision shader fallback from this validation
run. The existing FP16 path reports `uniformAndStorageBuffer16BitAccess-06332` with
these recent Validation Layers; this experiment does not change that path. Its
numerical fallback tests are run separately.

The tests use the loaded Module API. A preliminary eager-expression test with batch
recording returned incorrect results after a shape change even with ARM NX
disabled; that independent cache/recording issue is outside this experiment.
The A8 Linear reference tests require the graph path: the original fallback uses
floating-point activations. A separate attempt with the existing W4 General shader
also found a mismatch at K=32, N=7, group=8 on this Mac, with ARM NX disabled;
this change does not repair that existing shader. Fallback checks above cover MatMul.

## Qwen3-0.6B end-to-end test

Enable the existing LLM support in the same static build:

```sh
cmake -S . -B build-arm-nx \
  -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DLLM_SUPPORT_HTTP_RESOURCE=OFF
cmake --build build-arm-nx --target vulkan_arm_nx_llm_test.out -j8
hf download taobao-mnn/Qwen3-0.6B-MNN \
  --revision 34dfccda1187ded6e07ea06426da576b0b793c6b \
  --local-dir build-arm-nx/models/Qwen3-0.6B-MNN
```

Create `vulkan.json` beside the downloaded `config.json`, keeping model paths
relative to that directory. Set `backend_type` to `vulkan`, `precision` to `high`,
`thread_num` to `1` (Vulkan tuning-none), `sampler_type` to `greedy`, and `tmp_path`
to a writable absolute directory. Use a distinct tmp_path for each concurrent model.
Apply the emulation environment exports described above, then run:

```sh
MNN_VULKAN_ARM_NX=1 MNN_VULKAN_ARM_NX_TRACE=1 \
  ./build-arm-nx/vulkan_arm_nx_llm_test.out \
  build-arm-nx/models/Qwen3-0.6B-MNN/vulkan.json \
  '中国的首都是哪里？' 24 --require-arm-nx
```

This runs the public LLM forward API without tuning, disables thinking, checks finite
logits and prints greedy tokens. The test answered **中国的首都是北京。**, then EOS.
All **197 Linear operators** (28 layers × 7 + lm_head) used ARM NX INT8 graphs in
both prefill and decode, with no Linear fallback. There are 394 graph recordings;
later decode steps replay cached commands, so zero *new recordings* is expected.
The callback count confirms all 197 Linear executions per step. This downloaded
model uses 64-element weight groups; W4 is expanded to signed INT8 for the graph.

For an independent numerical comparison, use the optional `logits.bin` output and
`--teacher-tokens file` (one token ID per line), feeding identical histories to both
runs. A 15-step Hangzhou prompt test compared the original Vulkan path with this
INT8 path: 15/15 argmax agreement, mean KL(original || graph) 0.003306,
mean absolute logit difference 0.12249. CPU vs original Vulkan agreed on 14/15 steps
and had mean KL 0.04414, so CPU/backend differences must be measured separately.
This is a small correctness smoke test, not a model-quality evaluation.

The focused suite and a prefill + two-decode LLM run also passed Vulkan Validation
Layers with synchronization validation enabled, using high precision for surrounding
Vulkan operators. Low-precision surrounding LLM operators are not validated here.

## Hardware boundary

Passing emulation establishes API integration and numerical behavior of the simulated
implementation. It does not establish native NX operator coverage, compilation,
performance, power, or model quality. Graph execution may use shader fallback on a
native driver. Check native compilation diagnostics and NX hardware counters when
hardware becomes available. INT8 support here does not establish W4A16 support.

References: [ARM Darwin guide](https://arm.github.io/ai-ml-sdk-for-vulkan/darwin.html),
[SPV_ARM_graph](https://github.khronos.org/SPIRV-Registry/extensions/ARM/SPV_ARM_graph.html),
[Khronos single-op example](https://docs.vulkan.org/samples/latest/samples/extensions/tensor_and_data_graph/simple_tensor_and_data_graph/README.html).
