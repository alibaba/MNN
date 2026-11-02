# Online Core ML programs

Build with `-DMNN_COREML=ON`, select `MNN_FORWARD_NN` (`backend_type: "npu"` for LLM).
This implementation requires iOS 18 / macOS 15 or newer and uses documented Core ML compilation,
prediction and `MLState` APIs. No private ANE loader is used.

## CV graphs (default memory mode)

The default path retains high-level MNN operators, builds their MIL operations during resize,
and compiles each consecutive Core ML subgraph once. Fixed weights become external MIL constants.
Compilation and loading use public Core ML APIs on the device; the caller still supplies a single MNN model.
Unsupported MNN operations and transfer commands delimit subgraphs. Prediction runs at the last execution
in each group and propagates errors through `onExecute`. Group outputs use temporary backing buffers so
MNN's intermediate-buffer reuse cannot overwrite live inputs. Shape changes rebuild the affected graph.

Concat, split, reshape, transpose, slicing, padding and integer-factor nearest resize use native MIL
operations. NHWC spatial operations get graph-local transposes. Explicit Raster commands retain their
native MIL lowering and flat binding workaround; no Core ML custom operator is registered.
Static 1x1 CV convolutions remain convolutions, avoiding large spatial MatMul partitions on the CPU.
Long FP16 reductions use bounded partial convolutions with contiguous constant weights and balanced sums.

`CPUAndNeuralEngine` is the default compute policy. It permits Core ML CPU execution and does not force
all nodes onto ANE. Inspect the final compiled model with Xcode/MLComputePlan and validate runtime use
with the Neural Engine instrument. MNN operator callbacks prove backend assignment, not hardware placement.
Callback execution lazily prepares individual static-weight programs, materializes each command's outputs,
and honors skipped commands. Ordinary execution continues to use the fused graph. Callback compilation and
per-command prediction costs must not be used as fused inference benchmarks.
`MNN_COREML_DISABLE_FUSION=1` selects the diagnostic dynamic-weight
path. `MNN_COREML_COMPUTE_UNITS=cpu|all` provides public-API placement controls for diagnosis.

CV regression commands (Apple Silicon, `MNN_BUILD_TEST=ON`):

```sh
./run_test.out backend/coreml/cv 5 2 1
./run_test.out backend/coreml/graph 5 2 1
MNN_COREML_TEST_MODEL=model.mnn MNN_COREML_TEST_INPUT=input.txt ./run_test.out backend/coreml/model 5 2 4
python3 test/backend/coreml/run_cache_stress.py --help
```

The operator gate uses relative L2 <= 1%, normalized maximum error <= 2%, and exact integer outputs.
The external Module model gate uses relative L2 <= 5%, normalized maximum error <= 5%, cosine >= .999,
and zero MNN CPU fallback; `CV_CLASSIFIER=1` additionally checks Top1 and the Top5 set.
The graph test separately covers branches, multiple outputs, dynamic resize and an intentional CPU partition;
that partition test is not counted as native MIL coverage. Model quality also requires application checks
such as class-aware detection/NMS comparison. LLM behavior is not covered by this CV validation.

## Experimental low-memory programs

`BackendConfig::Memory_Low` retains the geometry/per-operator program route. Weights are runtime inputs;
canonical names and serialized program structure form the runtime cache key. Layers with equal shapes
and semantics can share a loaded model while retaining separate bindings and state.

Immutable convolution weights are shared across shape-specific executions by their serialized contents and
external-file identity/revision, including geometry-generated fused-projection children. The cache holds
weak references, so unloading the executions releases their weights. Ordinary memory mode retains FP16
projection weights. Dense Q4/Q8 decoding writes FP16 in bounded blocks without a full FP32 temporary.
Adjacent Q4 nibbles are decoded together to permit compiler vectorization; FP16 conversion retains MNN's
existing rounding and finite-range handling. Linear projections flatten batch/spatial positions into matrix
rows, avoiding a separate single-row product for each token in the MIL graph.

`BackendConfig::Memory_Low` (`memory: "low"` for LLM) retains supported quantized projection weights and
decodes them before each prediction into one runtime-owned FP16 workspace. Its size follows the largest
executing operator's weight inputs, rather than the whole model. A runtime lock covers decode and synchronous
prediction, so concurrent executions cannot overwrite the workspace. This reduces memory at the cost of
repeated CPU decoding; unsupported quantization forms keep the ordinary FP16 path.

The runtime retains up to 64 program entries for reuse while it is alive. MIL programs load through
`MLModelAsset`. Each fused program's serialized weights use an in-memory buffer up to and including
10 MiB; larger blobs use a temporary file. The in-memory buffer is retained with the model, so multiple
active programs can together exceed this per-program threshold. Core ML may also use its own caches.

Per-command callback programs use the file compilation path to limit retained asset memory. Programs
containing `resample` or `topk` also use it for numerical compatibility.
Temporary files live in per-program directories under the app cache at `MNN/CoreMLPrograms`. Each
directory is removed when its last program/session is released. A later compilation removes directories
left by terminated processes, while owner locks protect programs still used by other processes. Older
content-addressed cache files are not removed automatically because they may still be used by an older process.

Set `MNN_COREML_VERBOSE=1` to print loads, cache hits, source compilations and preparation times.
`MNN_COREML_CACHE_DIR` optionally overrides the temporary program directory, for isolated tests.

When experimental KV is explicitly enabled, attention uses independent FP16 key/value states, fixed capacity 4096, and runtime placement/mask inputs.
Token position does not change the decode program. Q/K can be dense NCHW or NHWC; C4 V inputs are reshaped
into heads. Score computation, normalization and fused SiLU gating retain FP32 intermediates for accuracy.
The fixed-capacity implementation updates and reads full buffers; it is not an in-place or constant-work
KV update. Batch size 1, ordinary append, tail removal and reset are supported. Reserve/paged or file-backed
KV caches, shared-KV layer indices, padding with `KVMeta.add != input sequence length`, and contexts exceeding
4096 are not supported by this implementation. Other model/operator combinations need separate validation.

Low-memory mode delegates attention and KV storage to the native CPU backend. This avoids Core ML's
full-capacity state/transfer buffers for score computations that already select its FP32 CPU path.
LLM prefill can additionally use the existing `chunk` setting (for example 128) to limit projection
workspaces; callers using `Llm` directly should apply configuration through `set_config()` before loading.
Program sharing reduces preparation cost. It does not remove per-prediction weight decoding, Core ML's
internal weight/layout preparation, or CPU fallback. Measure these costs against the native quantized CPU
backend before selecting this path for a workload.

`CPUAndNeuralEngine` allows CPU execution, including for FP32 operations. A macOS 26.2 / M3 Pro trace of
the tested 28-layer model in ordinary memory mode confirmed ANE predictions for 11 of 61 distinct programs.
Decode projections other than the vocabulary head were planned for CPU, and attention's FP32 scores/softmax
remained on CPU. The implementation validates online compilation and program reuse; the tested low-memory
mode substantially reduced process memory but remained slower than the floating-point CPU baseline.
Larger models, different quantization formats, iPhone memory limits and energy need separate validation.
Public API usage does not guarantee App Store approval.

Historical LLM program-sharing test (separate from the CV gate):

Stateful Attention is disabled by default and falls back at backend selection. The opt-in
`MNN_COREML_EXPERIMENTAL_KV=1` enables the experimental fixed-capacity path; its 4096-token limit,
shared-layer cloning restrictions and full-capacity updates remain outside the CV support contract.
State buffers are reused directly only with the same `MLModel` instance; switching compatible models
creates new state and copies its contents through the public multiarray API.

```sh
MNN_COREML_EXPERIMENTAL_KV=1 MNN_COREML_VERBOSE=1 ./run_test.out backend/coreml/program_reuse 5 1
```

It covers dynamic matmul, distinct per-layer weights, multiple models sharing a runtime, shape changes,
independent GQA KV states, NCHW/NHWC/C4 transfers, dense/scalar masks, prefill/decode cloning and reset.
Run an exported LLM against a CPU floating-point baseline as well; low-memory CPU activation quantization
is a different numerical baseline from the FP16 projection path.

## Experimental fixed-weight FFN component

`backend/CoreMLFusedFFN` builds one online MIL program for three bias-free linear
projections and SiLU gating. It accepts dense FP16 `[tokens, hidden]` IO and a
Core ML weight blob containing gate `[intermediate, hidden]`, up of the same
shape, and down `[hidden, intermediate]` at the supplied metadata offsets.
`prepare` compiles/loads the model once; `run` reuses it without CPU weight
unpacking. FP32 gate intermediates are the default. The optional FP16 gate
variant requires separate numerical validation.

The executor's optional external-weight path creates a temporary `.mlpackage`.
The runtime cache key includes the specification and full weight contents, so
different same-shaped layers cannot share stale constants. The source blob must
remain immutable until `prepare` completes. Each active instance retains its loaded model; this component does
not enforce a whole-model residency or compilation-memory budget.

This is an experimental component, not automatic FFN fusion in the scheduler.
It is not automatically inserted by the CV graph grouping or the low-memory path. It excludes
RMSNorm/residuals, attention and KV state; a layer benchmark is not an end-to-end
LLM acceleration result. Validate compilation memory, numerical quality, device placement and full-model
latency before enabling this component in a scheduler.

## Model serialization

The backend serializes only `mlProgram` models. Its protobuf-c sources contain
`Model`, `FeatureTypes`, and `MIL`; legacy `NeuralNetwork` descriptors are not built.
Model field 500 is reserved and `mlProgram` keeps field 502 for wire compatibility.

The schemas are synchronized with
[apple/coremltools](https://github.com/apple/coremltools/tree/a207b1a3452eafe02f2d870f486287834af2c82d/mlmodel/format)
at `a207b1a3452eafe02f2d870f486287834af2c82d` (also identical in release 9.0).
`MIL.proto` and `FeatureTypes.proto` are unmodified upstream files. `Model.proto`
retains the upstream feature, metadata and function descriptions, with only the
`mlProgram` model type and its required imports. Updating these definitions does
not change the backend's specification version 9 or `CoreML8` opset; INT8 model
I/O requires specification version 10 and iOS/macOS 26 before it can be enabled.

To regenerate all three protobuf-c source/header pairs, use protobuf-c 1.3.3
(matching the bundled runtime):

```sh
cd source/backend/coreml/mlmodel
out=$(mktemp -d)
protoc --plugin=protoc-gen-c=/path/to/protobuf-c-1.3.3/protoc-gen-c \
    --proto_path=proto --c_out="$out" Model.proto FeatureTypes.proto MIL.proto
cp "$out/"*.pb-c.h include/
cp "$out/"*.pb-c.c src/
```
