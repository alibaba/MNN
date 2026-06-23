# 测试与 CI

本文说明 MNN 当前统一测试入口 `./test.sh` 和测试矩阵配置
`test_stages.json`，并给出新增算子测试的推荐流程。

## 快速开始

```bash
# 静态检查：文档完整性、cppcheck、PyMNN wrapper 资源校验
./test.sh static

# 本机回归：CPU 构建 + 单元测试 + smoke + LLM smoke
./test.sh local

# Android 设备完整矩阵
./test.sh android <serial>

# Android 子集。第三个参数是可选 filter：
./test.sh android <serial> cpu            # CPU + lowmem + llm
./test.sh android <serial> opencl-image   # 只跑 OpenCL IMAGE 阶段
./test.sh android <serial> vulkan
./test.sh android <serial> lowmem
./test.sh android <serial> android-ci     # bench + smoke + llm，不跑 unit/lowmem
```

测试矩阵由 `test_stages.json` 描述。forward type、precision、gpuMode、
线程数、tag、memory mode、动态量化选项、KleidiAI 开关、分阶段 skip
列表、smoke 模型和 benchmark 参数都放在 JSON 中。新增、删除或调整
unit / lowmem / smoke / bench 阶段时，优先改 `test_stages.json`，不要直接
改 shell。

## 文件结构

```text
test.sh                         # 统一测试入口：static / local / android
test_stages.json                # 可执行回归阶段的声明式配置
docs/testing.md                 # 本文档
skills/test-ci/SKILL.md         # Agent 使用入口，引用本文档
test/                           # C++ 测试框架和测试用例
  MNNTestSuite.{h,cpp}          # 测试注册、运行、MNN_TEST_SKIP 支持
  main.cpp                      # argv -> BackendConfig / RuntimeHint
  op/                           # 每个算子一个或一组测试文件
```

`static` 模式不使用 `test_stages.json`。它只做轻量检查：

- `doc_check`：检查 CMake 选项、可执行文件、部分 PyMNN API 是否在文档中出现
- `static_check`：对本次变更涉及的 C++ 源码跑 `cppcheck`
- `py_check`：PyMNN wrapper 资源校验，仅在对应 Python 文件变更时运行

`local` 和 `android` 模式使用 `test_stages.json`。

## Android 模式流程

`./test.sh android <serial> [filter]` 的执行流程：

1. 构建 `arm64-v8a`（调用 `project/android/build_64.sh` 后再用本机核心数增量 `make`）。
2. 推送 `ANDROID_BIN_LIST` 中的库和可执行文件到 `/data/local/tmp/MNN`。
3. 在需要 smoke/bench 时下载 Caffe 源模型，并在设备端使用刚构建出的
   `MNNConvert` 转成 `.mnn`。
4. 按 `test_stages.json` 运行阶段：
   `unit/* -> lowmem/* -> smokeA/* -> smokeB/* -> bench/* -> llm/*`。
5. 打印 summary：总数、通过、失败、跳过，以及每个阶段的结果。

每个阶段的 stdout/stderr 会保存到 `logs/test-<UTC timestamp>/` 下。退出码
非 0 表示至少有一个阶段失败；`SKIP` 不算失败。

## LLM 模型准备

`llm` 阶段使用 `llm_demo` 跑一个 MNN 格式 LLM 模型。模型准备是懒加载的：
只有跑到 `llm` 阶段时才会检查或下载模型；如果模型不可用，只跳过 `llm`
阶段，不影响 unit / smoke / bench。

| 环境变量 | 模式 | 含义 |
|----------|------|------|
| `LLM_MODEL_DIR` | local/android | 指向已有 MNN LLM 模型目录。设置后不会下载。 |
| `LLM_MODEL_REPO` | local/android | 模型仓库，默认 `taobao-mnn/Qwen2.5-0.5B-Instruct-MNN`。 |
| `LLM_MODEL_SOURCE` | local/android | 下载源：`huggingface` 或 `modelscope`。 |
| `LLM_MODEL_URL_BASE` | local/android | 直接覆盖下载 URL 前缀，优先级最高。 |

```bash
# 已有模型目录，不触发下载
LLM_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct-MNN ./test.sh local

# HuggingFace 不通时使用 ModelScope
LLM_MODEL_SOURCE=modelscope ./test.sh android <serial>
```

内置默认模型在 ModelScope 上会自动把 `taobao-mnn/*` 映射为 `MNN/*`；
如果显式设置了 `LLM_MODEL_REPO`，则按用户指定值使用。

## `test_stages.json` 结构

```text
{
  "android": {
    "stages":          [ ... ], // run_test.out unit + lowmem 阶段
    "smoke_models":    [ ... ], // smoke/bench 使用的模型路径
    "smoke_a_stages":  [ ... ], // 每个模型做一次前向
    "smoke_b_stages":  [ ... ], // CPU vs backend 数值对比
    "bench_stages":    [ ... ]  // benchmark.out 调用
  },
  "local": {
    "stages":          [ ... ],
    "smoke_a_stages":  [ ... ]  // 本机 CPU-only，不做 smokeB
  },
  "llm": {
    "model_repo":   "...",
    "config_file":  "config.json",
    "prompt_file":  "prompt.txt",
    "stage":        { ... }
  },
  "_documentation": { ... }
}
```

### stage 字段

| 字段 | 含义 |
|------|------|
| `name` | 阶段名，也用于日志文件名。 |
| `filter` | 过滤标签，和命令行第三个参数匹配。 |
| `comment` | 说明这个阶段为什么存在、覆盖什么。 |
| `binary` | smoke/bench 使用：`run_test`、`v2basic`、`backendtest`、`benchmark`。 |
| `prefix` | 传给 `run_test.out` 的测试名前缀，例如 `all`、`op`、`op/convolution`。 |
| `type` | forward type：`0` CPU，`3` OpenCL，`7` Vulkan。 |
| `precision` | `BackendConfig::PrecisionMode`：`0` Normal，`1` High，`2` Low。 |
| `threadOrGpuMode` | CPU 下是线程数；GPU 下是 `MNN_GPU_TUNING_*` 和 `MNN_GPU_MEMORY_*` bitmask。 |
| `tag` | 传给 `run_test.out` 的报告标签。 |
| `memory` | `BackendConfig::MemoryMode`：`0` Normal，`1` High，`2` Low。可省略。 |
| `dynamicOption` | `RuntimeHint::dynamicQuantOption`。可省略。 |
| `kleidiAi` | argv[8]，`1` 表示启用 KleidiAI。可省略。 |
| `skip` | 精确测试名列表，通过 `MNN_TEST_SKIP` 跳过。 |
| `args` | smoke/bench 使用的位置参数。支持 `{model}` 和 `{models_dir}` 替换。 |

### filter 标签

| 标签 | 含义 |
|------|------|
| `cpu` | CPU 阶段；`cpu` filter 也会包含 lowmem 和 llm。 |
| `opencl-image` | OpenCL IMAGE 内存模式。 |
| `opencl-buffer` | OpenCL BUFFER 内存模式。 |
| `vulkan` | Vulkan backend。 |
| `lowmem` | 低内存模式矩阵。 |
| `smoke-opencl` | OpenCL smokeA/smokeB。 |
| `smoke-vulkan` | Vulkan smokeA/smokeB。 |
| `llm` | LLM smoke。 |

快捷 filter：

- `opencl` = `opencl-image` + `opencl-buffer` + `smoke-opencl`
- `gpu` = OpenCL + Vulkan
- `unit` = CPU/OpenCL/Vulkan unit 阶段，不含 lowmem/smoke/llm
- `android-ci` = bench + smoke + llm，不跑 unit/lowmem

## 阶段类型

### `unit/cpu/*`

CPU 单元测试，覆盖注册在 `MNNTestSuite` 中的 C++ 测试。常见阶段：

- `unit/cpu/all`：单线程、`Precision_Normal`，覆盖面最广
- `unit/cpu/op-mt`：4 线程、只跑 `op` 前缀，覆盖线程池路径
- `unit/cpu/op-fp16-*`：窄范围 FP16 精度回归

### `unit/opencl/*` / `unit/vulkan/*`

GPU correctness 阶段使用 `TUNING_NONE`，避免单次正确性测试消耗大量调优时间。
性能相关阶段才使用更宽的 tuning 模式。

- `unit/opencl/op`：OpenCL IMAGE
- `unit/opencl/op-buffer`：OpenCL BUFFER
- `unit/vulkan/op`：Vulkan

每个 GPU 阶段都可以有自己的 `skip` 列表，用于隔离已知设备或驱动问题。
新增 skip 时必须在 `_documentation.skip_rationale` 里解释原因。

### `lowmem/*`

低内存模式矩阵，主要覆盖 `op/lowMemory/*` 和 weight-i8/i4 量化卷积。
组合维度包括 precision、线程数、dynamicOption 等。

### `smokeA/<backend>/<model>`

用 `MNNV2Basic.out` 对模型做一次前向。主要覆盖模型加载、shape infer 和基础
执行路径，不做数值对比。

### `smokeB/<backend>/<model>`

用 `backendTest.out` 做 CPU vs backend 数值对比，默认 tolerance 为 `0.05`。

### `bench/<backend>`

用 `benchmark.out` 做简单性能 sanity。只要程序能跑完并输出时间，阶段就通过；
不根据具体耗时判定失败。

### `llm/<model_name>`

用 `llm_demo` 跑配置好的 `config.json` 和 `prompt.txt`。

## 新增算子测试

新增算子测试通常分两步：

1. 在 `test/op/` 下写 C++ 测试，并用 `MNNTestSuiteRegister` 注册。
2. 如需特定 backend / precision / memory / dynamicOption，再在
   `test_stages.json` 中增加专门阶段。

### 编写测试

最小示例：

```cpp
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class MyOpTest : public MNNTestCase {
public:
    virtual ~MyOpTest() = default;
    virtual bool run(int precision) override {
        auto x = _Input({1, 4}, NCHW, halide_type_of<float>());
        x->writeMap<float>()[0] = 1.0f;
        x->writeMap<float>()[1] = 2.0f;
        x->writeMap<float>()[2] = 3.0f;
        x->writeMap<float>()[3] = 4.0f;
        auto y = _Multiply(x, _Scalar<float>(2.0f));

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

约定：

- 测试名（如 `op/myop`）会被 `test_stages.json` 中的 `prefix` 匹配。
- 相关测试尽量放在共同前缀下，例如 `op/binary/foo`、
  `op/convolution/bar`。
- 绝对误差用 `checkVector`，相对误差用 `checkVectorByRelativeError`。
- 需要读取运行配置时，可使用 `MNNTestSuite::get()->pStaus`。

新增文件后可先本地验证：

```bash
cd build
make -j$(nproc) run_test.out
./run_test.out op/myop
```

### 不改 JSON 的情况

如果测试名能被已有阶段覆盖，通常不用改 JSON：

- `op/myop` 会进入 `unit/cpu/all`、`unit/cpu/op-mt`、
  `unit/opencl/op`、`unit/opencl/op-buffer`、`unit/vulkan/op`
- `op/convolution/myop` 还会进入卷积相关 FP16 窄范围阶段

### 增加专门阶段

当已有阶段无法覆盖所需配置时，在 `test_stages.json -> android.stages`
增加一项：

```json
{
  "name":            "unit/cpu/myop-fp16-mt",
  "filter":          "cpu",
  "comment":         "MyOp FP16 precision, 4 threads.",
  "prefix":          "op/myop",
  "type":            0,
  "precision":       2,
  "threadOrGpuMode": 4,
  "tag":             "fp16myop64",
  "memory":          0
}
```

随后运行：

```bash
./test.sh android <serial> cpu
```

### 跳过已知问题测试

如果某个测试暴露的是已知 backend / 驱动问题，而不是当前变更要解决的问题，
可以把精确测试名加入对应阶段的 `skip`：

```json
{
  "name": "unit/opencl/op-buffer",
  "skip": [
    "op/myop"
  ]
}
```

同时必须在 `_documentation.skip_rationale` 中写清楚原因，便于后续判断是否
还需要保留 skip。

## 新增 smoke 模型或 bench 阶段

新增 smoke 模型：

1. 在 `android.smoke_models` 中加入 `.mnn` 相对路径。
2. 在 `test.sh` 的 `SMOKE_SOURCES` 中加入源模型下载信息。
3. 在 `_smoke_pair_for` 中补充 `.mnn` 到 `caffemodel/prototxt` 的映射。

新增 bench 阶段：

```json
{
  "name":    "bench/cpu-fp16",
  "filter":  "cpu",
  "comment": "CPU FP16 benchmark, 4 threads.",
  "binary":  "benchmark",
  "args":    ["{models_dir}", "10", "2", "0", "4", "2"]
}
```

`{models_dir}` 在 Android 模式下会替换为
`/data/local/tmp/MNN/public_models`。
