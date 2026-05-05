# Testing & CI

This document covers `./test_ci.sh` and `test_stages.json` — the declarative
CI driver for this fork — and walks through adding a new operator test
end-to-end.

## TL;DR

```bash
# Local host (CPU only):
./test_ci.sh local

# Android device (full matrix):
./test_ci.sh android <serial>

# Run a subset by filter tag:
RUNS=cpu          ./test_ci.sh android <serial>   # CPU + lowmem + llm
RUNS=opencl-image ./test_ci.sh android <serial>   # only OpenCL IMAGE stages
RUNS=vulkan       ./test_ci.sh android <serial>
RUNS=lowmem       ./test_ci.sh android <serial>
```

The full matrix is described in `test_stages.json`. Every parameter (forward
type, precision, gpuMode, thread count, tag, memory mode, dynamic-quant
option, KleidiAI flag, per-stage skip list, smoke model list, benchmark
arguments) lives there. Editing the JSON is the supported way to add, drop,
or retune a stage — no shell edits needed for the common cases.

## Architecture

```
test_ci.sh                           ← bash driver (local + android modes)
├── test_stages.json                 ← every stage's parameters live here
└── test/                            ← C++ test framework
    ├── MNNTestSuite.{h,cpp}         ← test registry + MNN_TEST_SKIP support
    ├── main.cpp                     ← argv → BackendConfig + RuntimeHint
    └── op/                          ← per-operator tests, one file each
```

Flow on android mode:

1. **Build** for `arm64-v8a` (`build_64.sh` + a parallel `make`).
2. **Push** the artefacts in `ANDROID_BIN_LIST` to `/data/local/tmp/MNN`.
3. **Push the LLM model** under `models/<repo>/` and the smoke caffe sources.
4. **Convert** smoke caffe sources → `.mnn` on-device with the just-pushed
   `MNNConvert` (`tools/converter/libMNNConvertDeps.so` is pushed alongside).
5. **Run stages** declared in `test_stages.json` in order:
   `unit/* → lowmem/* → smokeA/* → smokeB/* → bench/* → llm/*`.
6. **Print summary**: `total / passed / failed / skipped` plus per-stage
   `PASS / FAIL / SKIP` lines.

## `test_stages.json` shape

```jsonc
{
  "android": {                  // stages dispatched by `./test_ci.sh android`
    "stages":          [ ... ], // run_test.out unit + lowmem stages
    "smoke_models":    [ ... ], // public-model paths
    "smoke_a_stages":  [ ... ], // forward-smoke per (model × backend)
    "smoke_b_stages":  [ ... ], // numeric CPU-vs-backend per (model × backend)
    "bench_stages":    [ ... ]  // benchmark.out invocations
  },
  "local": {                    // stages dispatched by `./test_ci.sh local`
    "stages":          [ ... ],
    "smoke_a_stages":  [ ... ]  // (no smoke_b in local — there's no GPU oracle)
  },
  "llm": {                      // LLM smoke test (both modes)
    "model_repo":   "...",
    "config_file":  "config.json",
    "prompt_file":  "prompt.txt",
    "stage":        { ... }
  },
  "_documentation": { ... }     // self-describing schema doc; safe to read
}
```

### Stage object — every field

| Field             | Meaning                                                                                                                                                              |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`            | Stage label. Also the per-stage log filename (`/`s flatten to `_`).                                                                                                  |
| `filter`          | Filter tag. Matched against `RUNS=…` or the implicit `all`.                                                                                                          |
| `comment`         | Free-form note about why the stage exists / what it covers.                                                                                                          |
| `binary`          | (smoke/bench only) `run_test` (default), `v2basic`, `backendtest`, or `benchmark`.                                                                                   |
| `prefix`          | First positional arg to `run_test.out` — test-name prefix or the literal `all`.                                                                                      |
| `type`            | Forward type. `0`=CPU, `3`=OpenCL, `7`=Vulkan.                                                                                                                       |
| `precision`       | `0`=Normal, `1`=High, `2`=Low. (Per `BackendConfig::PrecisionMode`.)                                                                                                 |
| `threadOrGpuMode` | CPU: thread count. GPU: bitmask of `MNN_GPU_TUNING_*` (1=NONE, 2=HEAVY, 4=WIDE) **OR**'d with `MNN_GPU_MEMORY_*` (64=BUFFER, 128=IMAGE). e.g. `129` = NONE\|IMAGE.   |
| `tag`             | Free-form report tag forwarded to `run_test.out` (printed in `TEST_NAME_UNIT<tag>` lines).                                                                           |
| `memory`          | `0`=Normal, `1`=High, `2`=Low. **Omit** when not setting (some stages rely on the default).                                                                          |
| `dynamicOption`   | `RuntimeHint::dynamicQuantOption` (0..7). Omit when not setting.                                                                                                     |
| `kleidiAi`        | `argv[8]`: `1` enables KleidiAI on ARM. Omit when not setting.                                                                                                       |
| `skip`            | Array of **exact** test names to skip. Passed to `MNNTestSuite::run()` via `MNN_TEST_SKIP` env. Use this for known device/driver bugs you don't want to globally lose coverage on. |
| `args`            | (smoke/bench only) Positional argv array. `{model}` and `{models_dir}` get per-iteration substitution.                                                               |

### Filter tags

| Tag              | Meaning                                                                  |
|------------------|--------------------------------------------------------------------------|
| `cpu`            | Plain CPU stages (also covers lowmem and llm in `RUNS=cpu`).             |
| `opencl-image`   | OpenCL with `MNN_GPU_MEMORY_IMAGE`.                                      |
| `opencl-buffer`  | OpenCL with `MNN_GPU_MEMORY_BUFFER`.                                     |
| `vulkan`         | Vulkan backend.                                                          |
| `lowmem`         | Low-memory configurations (`memory=2`).                                  |
| `smoke-opencl`   | Smoke A/B per model on OpenCL.                                           |
| `smoke-vulkan`   | Smoke A/B per model on Vulkan.                                           |
| `llm`            | LLM smoke test.                                                          |

`RUNS=opencl` is a shortcut for `opencl-image | opencl-buffer | smoke-opencl`.
`RUNS=gpu` covers everything OpenCL + Vulkan.

## Per-stage type — what each kind covers

### `unit/cpu/*` — host CPU correctness

The plain-CPU sweep across the registered C++ tests. Variants in
`test_stages.json`:

* **`unit/cpu/all`** — single-thread, `Precision_Normal`, broadest sanity check.
* **`unit/cpu/op-mt`** — 4-thread, op-only (`prefix: "op"`). Catches threadpool races.
* **`unit/cpu/op-fp16-conv`** — `Precision_Low`, only convolution tests. Exercises
  the FP16 ARM82 path without the rest of the suite.
* **`unit/cpu/op-fp16-col2im`**, **`unit/cpu/op-fp16-roi`** — narrow FP16 sweeps
  for ops that are particularly sensitive.

### `unit/opencl/*` and `unit/vulkan/*` — GPU correctness

GPU stages run with `TUNING_NONE` because we want correctness, not perf —
`TUNING_WIDE` adds many seconds of per-kernel auto-tuning that's wasted on a
single-shot run. Bench stages flip back to `TUNING_WIDE` since perf is the
point there.

* **`unit/opencl/op`** — OpenCL with `MEMORY_IMAGE` (gpuMode `129`).
* **`unit/opencl/op-buffer`** — OpenCL with `MEMORY_BUFFER` (gpuMode `65`).
  Critical for catching regressions in BUFFER-only creators (e.g. Attention)
  that the IMAGE path masks via CPU fallback.
* **`unit/vulkan/op`** — Vulkan with `TUNING_NONE` (gpuMode `1`). Vulkan
  ignores `MEMORY_*` bits; image-vs-buffer is selected at build time via
  the `MNN_VULKAN_IMAGE` CMake option.

Each carries a `skip` list for device-specific upstream bugs that aren't
ours to fix in this fork. Each entry is documented in
`test_stages.json::_documentation.skip_rationale`.

### `lowmem/*` — low-memory matrix

`memory=2` (`Memory_Low`) plus various `precision` × `dynamicOption` × thread
combinations. Mostly exercises the `op/lowMemory/*` and weight-i8/i4 quantised
conv tests.

### `smokeA/<backend>/<model>` — forward-smoke per backend × model

Loads each public smoke model with `MNNV2Basic.out`, does one forward pass.
Catches model-load and shape-inference regressions; doesn't validate
numerics.

### `smokeB/<backend>/<model>` — CPU-vs-backend numeric check

Runs `backendTest.out` with the CPU oracle vs the named backend; tolerance
defaults to `0.05`. Catches backend kernel regressions that produce
numerically wrong output.

### `bench/<backend>` — micro-benchmark per backend

`benchmark.out` over the same model set. Just a perf sanity check (no pass/fail
on numbers — the stage passes as long as the binary doesn't crash and
produces timing lines).

### `llm/<model_name>` — LLM smoke

Runs `llm_demo` against the provisioned `config.json` + `prompt.txt`.

## How to add a new operator test

There are two pieces:

1. **Write the C++ test** under `test/op/`.
2. **(Optional) wire a dedicated `test_ci.sh` stage** in `test_stages.json` so
   the new test runs in a specific config (precision / threads / backend /
   memory).

If you only do (1), the new test is automatically picked up by every existing
`unit/*` stage that matches its prefix (e.g. `unit/cpu/all` runs everything;
`unit/cpu/op-mt` runs anything starting with `op`; `unit/opencl/op` runs
anything starting with `op` on OpenCL; …). That's usually enough.

### Step 1 — Write the test

Each test is a class deriving from `MNNTestCase` registered with
`MNNTestSuiteRegister`. Minimal template:

```cpp
// test/op/MyOpTest.cpp
//
//  MyOpTest.cpp
//  MNNTests
//
//  Copyright © 2026, ...
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class MyOpTest : public MNNTestCase {
public:
    virtual ~MyOpTest() = default;
    virtual bool run(int precision) override {
        // 1) Build a small graph with the op under test.
        auto x = _Input({1, 4}, NCHW, halide_type_of<float>());
        x->writeMap<float>()[0] = 1.0f;
        x->writeMap<float>()[1] = 2.0f;
        x->writeMap<float>()[2] = 3.0f;
        x->writeMap<float>()[3] = 4.0f;
        auto y = _Multiply(x, _Scalar<float>(2.0f));   // <- op under test

        // 2) Read result and compare against expected.
        const std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f};
        auto got = y->readMap<float>();
        if (!checkVector<float>(got, expected.data(), 4, 0.0001f)) {
            MNN_ERROR("MyOpTest: numerical mismatch\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(MyOpTest, "op/myop");
```

Conventions:

* The **test name** (`"op/myop"`) is what `prefix` matches in
  `test_stages.json`. Group related tests under a common prefix
  (e.g. `op/binary/myop`, `op/convolution/myop`) so a JSON stage with
  `prefix: "op/binary"` automatically picks them up.
* Use `checkVector` for absolute tolerance, `checkVectorByRelativeError` for
  relative — both live in `test/TestUtils.h`.
* Read `MNNTestSuite::get()->pStaus` if you need the runtime config inside
  the test (it carries `precision`, `memory`, `forwardType`, `dynamicOption`).
* `precision` arg into `run(int)` is the `Precision_Normal=0 / High=1 / Low=2`
  selector (so `FP32Converter[precision]` quantises reference values to
  match the runtime).

CMake picks up new files in `test/op/` automatically (the converter test
target globs the directory). After adding the file:

```bash
cd build && make -j$(nproc) run_test.out
./run_test.out op/myop          # local CPU
```

### Step 2a — Run on default stages (no JSON change)

If your test prefix already matches an existing stage, you're done. Examples:

* Test named `op/myop` runs in `unit/cpu/all`, `unit/cpu/op-mt`,
  `unit/opencl/op`, `unit/opencl/op-buffer`, `unit/vulkan/op`.
* Test named `op/convolution/myop` additionally runs in
  `unit/cpu/op-fp16-conv`.

### Step 2b — Add a dedicated stage

If you need a specific precision / thread / memory / dynamicOption config that
no existing stage covers, add a stage entry to `test_stages.json`:

```jsonc
// in test_stages.json → android.stages
{
  "name":            "unit/cpu/myop-fp16-mt",
  "filter":          "cpu",
  "comment":         "MyOp at FP16 precision, 4-thread.",
  "prefix":          "op/myop",
  "type":            0,
  "precision":       2,
  "threadOrGpuMode": 4,
  "tag":             "fp16myop64",
  "memory":          0
}
```

That's it. `./test_ci.sh android <serial>` will:

1. Build the test binary (your new `MyOpTest.cpp` is picked up automatically).
2. Push `run_test.out` to the device.
3. Run the new stage between the existing `unit/*` stages.

### Step 2c — Skip a known-broken test on a specific stage

If your test exposes an upstream backend bug that you don't want blocking
the overall run, add the test name to that stage's `skip` array. The skip
list is passed via `MNN_TEST_SKIP` env to `MNNTestSuite::run()`, which
matches by **exact name** and prints `skip <name> (in MNN_TEST_SKIP)`
instead of running it.

```jsonc
{
  "name":   "unit/opencl/op-buffer",
  ...
  "skip": [
    "op/myop"        // <-- add here; document why in skip_rationale
  ]
}
```

Always pair a new skip entry with a one-line entry in
`_documentation.skip_rationale` describing the upstream bug, so future
maintainers know whether the skip is still needed.

### Step 2d — Add a new smoke model

If you need to exercise a model in addition to the existing
mobilenet_v{1,2} / squeezenet_v1.{0,1} set:

```jsonc
"android": {
  "smoke_models": [
    "MobileNet/v1/mobilenet_v1.caffe.mnn",
    ...
    "MyModel/v1/my_model.caffe.mnn"          // <-- new
  ],
  ...
}
```

Then add the source caffe pair to the `SMOKE_SOURCES` array in `test_ci.sh`
so it gets fetched + on-device-converted alongside the existing ones, and
extend `_smoke_pair_for` to map the `.caffe.mnn` filename back to its
caffemodel + prototxt source pair.

### Step 2e — Add a new bench entry

```jsonc
"android": {
  ...
  "bench_stages": [
    ...
    {
      "name":    "bench/cpu-fp16",
      "filter":  "cpu",
      "comment": "10-iter benchmark on CPU, 4-thread, Precision_Low (FP16).",
      "binary":  "benchmark",
      "args":    ["{models_dir}", "10", "2", "0", "4", "2"]
    }
  ]
}
```

`{models_dir}` is substituted with `/data/local/tmp/MNN/public_models` at
dispatch time.

## Worked examples

### Example A — A new conv variant test

You've added an int4 grouped-conv kernel and want it covered at FP16 +
4-thread + memory-low + dynamicOption=2.

1. Create `test/op/Int4GroupConvTest.cpp` registering as
   `op/convolution/int4_group`.
2. Add a stage:
   ```jsonc
   {
     "name":            "lowmem/int4_group-d2-p2",
     "filter":          "lowmem",
     "comment":         "Int4 grouped conv: precision=Low, 4-thread, memory=Low, dyn=2.",
     "prefix":          "op/convolution/int4_group",
     "type":            0,
     "precision":       2,
     "threadOrGpuMode": 4,
     "tag":             "64",
     "memory":          2,
     "dynamicOption":   2
   }
   ```
3. Run:
   ```bash
   RUNS=lowmem ./test_ci.sh android <serial>
   ```

### Example B — Cross-backend numeric verification of a new op

You want your new op cross-checked between CPU and GPU.

1. Register the test as `op/myop`. Make sure the assertion passes on CPU
   first.
2. The default stages already cover it: `unit/cpu/all` (CPU correctness),
   `unit/opencl/op` (OpenCL IMAGE), `unit/opencl/op-buffer` (OpenCL BUFFER),
   `unit/vulkan/op` (Vulkan).
3. If only one backend's path needs broader coverage, add a dedicated stage
   with the desired `prefix` + `type` + `threadOrGpuMode`.

### Example C — Quarantining a flaky upstream bug

You merged an upstream sync that exposed a per-channel ~1-LSB drift in
`op/foo` on Vulkan, and you don't want it blocking CI.

1. Add `"op/foo"` to `unit/vulkan/op`'s `skip` array.
2. Add a key under `_documentation.skip_rationale`:
   ```jsonc
   "vulkan_foo_drift": "op/foo accumulates ~1 LSB drift per channel on Vulkan due to FP16 intermediates in upstream's foo kernel; standalone passes, the bulk run flags it. Tracking upstream issue #N."
   ```
3. Commit both edits together so `git blame` gives future you the full
   story.

## File-by-file map of recent CI changes

| File                                  | What it provides                                             |
|---------------------------------------|--------------------------------------------------------------|
| `test_ci.sh`                          | Bash driver. Reads `test_stages.json` for unit/lowmem/smoke/bench/llm. Pushes `libMNNConvertDeps.so` so on-device caffe→mnn conversion works. |
| `test_stages.json`                    | Every stage parameter, skip lists, smoke model list, bench entries. Self-documenting via `_documentation`. |
| `test/MNNTestSuite.{h,cpp}`           | `MNN_TEST_SKIP` env-var support; `Status.dynamicOption` so tests can adapt tolerance to the runtime hint. |
| `test/main.cpp`                       | Propagates `dynamicOption` from argv into `Status`.          |
| `test/op/AttentionTest.cpp`           | Test 3 skipped on OpenCL/Vulkan (CPUAttention `kv_cache=false` upstream TODO). |
| `test/op/BroadcastToTest.cpp`         | Loosens absolute tolerance to `0.002f` for non-CPU forwardType (FP16 intermediates). |
| `test/op/ConvolutionTest.cpp`         | `errorScale=200` for `memory=Low + dynamicOption=1` (1-LSB systematic offset). |
| `source/backend/opencl/execution/{buffer,image}/Unary*` + `cl/unary*` | Native OpenCL `ERFINV` (TF two-branch polynomial). |
| `source/geometry/GeometryBinary.cpp`  | Insert broadcast-to on Vulkan when input/output rank differ (fixes AddBroast). |
