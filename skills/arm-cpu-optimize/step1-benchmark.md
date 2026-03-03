# 步骤 1：建立性能基准

> **目标**：编写性能基准测试，获取优化前的基线数据。
>
> **前置条件**：明确待优化的算子和目标参数。
>
> **复杂度**：低（需要编译运行）

---

## 1.1 确定优化目标

```
待优化算子：____（例如 MatMul, Attention, Convolution）
目标数据类型：____（FP32 / FP16 / INT8）
典型参数：____（例如 M=1, K=4096, N=4096）
目标平台：____（例如 Cortex-A78, Apple M1）
当前性能：____（如已知）
目标性能：____（如已知，理论峰值的 xx%）
```

---

## 1.2 编写性能测试

在 `test/speed/` 下创建 `XxxSpeedTest.cpp`，参考 `MatMulSpeed.cpp` 的模板：

```cpp
//
//  XxxSpeedTest.cpp
//  MNNTests
//

#include <math.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

using namespace MNN::Express;

class XxxSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        // ===== 1. 定义测试用例 =====
        // 用真实场景的参数组合
        struct TestCase {
            int param1, param2, param3;
            const char* name;
        };
        std::vector<TestCase> cases = {
            {1, 4096, 4096, "decode_1token"},     // LLM decode
            {128, 4096, 4096, "prefill_128"},      // LLM prefill
            {1024, 4096, 4096, "prefill_1024"},    // LLM long prefill
            {1, 4096, 11008, "ffn_decode"},        // FFN decode
        };

        for (auto& tc : cases) {
            auto res = _runBenchmark(tc.param1, tc.param2, tc.param3, tc.name);
            if (!res) return false;
        }
        return true;
    }

    bool _runBenchmark(int M, int K, int N, const char* name) {
        // ===== 2. 创建算子 =====
        // 根据具体算子修改
        std::unique_ptr<MNN::OpT> op(new MNN::OpT);
        op->type = MNN::OpType_MatMul;
        op->main.type = MNN::OpParameter_MatMul;
        op->main.value = new MNN::MatMulT;
        auto param = op->main.AsMatMul();
        param->transposeA = false;
        param->transposeB = false;

        auto x0 = _Input({}, NHWC, halide_type_of<float>());
        auto x1 = _Input({}, NHWC, halide_type_of<float>());
        x0->resize({M, K});
        x1->resize({K, N});
        auto y = Variable::create(Expr::create(op.get(), {x0, x1}));
        Variable::prepareCompute({y});

        // ===== 3. 正确性验证（首次运行）=====
        // 用小规模数据验证正确性
        {
            auto ptr0 = x0->writeMap<float>();
            auto ptr1 = x1->writeMap<float>();
            for (int i = 0; i < M * K; ++i) ptr0[i] = ((float)(i % 100)) / 10000.0f;
            for (int i = 0; i < K * N; ++i) ptr1[i] = ((float)(i % 100)) / 10000.0f;
            y->readMap<float>();
            // TODO: 添加正确性检查（对比参考实现）
        }

        // ===== 4. 性能测试（多次运行取平均）=====
        const int warmup = 3;
        const int repeat = 10;

        // Warmup
        for (int t = 0; t < warmup; ++t) {
            x0->writeMap<float>();
            x1->writeMap<float>();
            y->readMap<float>();
        }

        // Benchmark
        MNN::Timer _t;
        for (int t = 0; t < repeat; ++t) {
            x0->writeMap<float>();
            x1->writeMap<float>();
            y->readMap<float>();
        }
        float avgMs = _t.durationInUs() / 1000.0f / (float)repeat;

        // ===== 5. 输出结果 =====
        float gflops = 2.0f * (float)M * (float)K * (float)N / avgMs / 1e6f;
        MNN_PRINT("[%s] M=%d K=%d N=%d, Avg time: %.3f ms, GFLOPS: %.2f\n",
                  name, M, K, N, avgMs, gflops);

        return true;
    }
};

MNNTestSuiteRegister(XxxSpeedTest, "speed/XxxSpeed");
```

### 测试模板要点

1. **正确性 + 性能**：先验证结果正确，再做性能计时
2. **Warmup**：前几次运行不计入（排除 cache 冷启动）
3. **多次运行取平均**：至少 10 次，减少波动
4. **输出 GFLOPS**：方便与理论峰值对比
5. **多参数组合**：覆盖 decode（M=1）和 prefill（M 大）场景

### 非标准算子的 benchmark

上述模板基于 `Expr::create` 创建算子，适用于标准 OpType。对于 Fuse 算子或多输入/有状态算子（如 LinearAttention、Mamba），如果无法通过 Expr 直接创建，可以用 `CPUBackend` + `Tensor` 手动构造输入输出，直接调用 `onResize` + `onExecute` 来 benchmark。具体做法参考对应算子的单元测试代码。

---

## 1.3 编译并运行

```bash
# 编译（确保开启 ARM 优化）
cd build
cmake .. -DMNN_BUILD_TEST=ON -DMNN_ARM82=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
make -j$(nproc)

# 运行性能测试
./run_test.out speed/XxxSpeed
```

### macOS Apple Silicon（本身就是 ARM64）

```bash
cd build
cmake .. -DMNN_BUILD_TEST=ON -DMNN_ARM82=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
make -j$(sysctl -n hw.ncpu)
./run_test.out speed/XxxSpeed
```

### 交叉编译到 Android ARM 设备

```bash
mkdir build_arm && cd build_arm
cmake .. -DCMAKE_TOOLCHAIN_FILE=<toolchain> \
    -DMNN_BUILD_TEST=ON -DMNN_ARM82=ON -DMNN_LOW_MEMORY=ON

# push 到设备
adb push run_test.out /data/local/tmp/
adb shell "cd /data/local/tmp && ./run_test.out speed/XxxSpeed"
```

### 无 ARM 设备时

如果当前环境无法编译运行 ARM 代码，可以先完成代码优化和分析，将性能测试标注为"待实测"并写好测试命令，后续在 ARM 设备上补测。

---

## 1.4 记录基线数据

将结果保存为基准，后续每次优化都与之对比：

```markdown
## 基线数据

平台: Cortex-A78 (4x big + 4x little)
日期: <填写实际日期>
编译选项: MNN_ARM82=ON, MNN_LOW_MEMORY=ON

| 用例 | M | K | N | 时间(ms) | GFLOPS |
|------|---|---|---|---------|--------|
| decode_1token | 1 | 4096 | 4096 | xx.xx | xx.xx |
| prefill_128 | 128 | 4096 | 4096 | xx.xx | xx.xx |
| ... | | | | | |

理论峰值: xx GFLOPS (FP32) / xx GFLOPS (FP16)
当前效率: xx%
```

---

## 步骤 1 测试标准

### 通过标准

- [ ] 性能测试文件已创建并可编译
- [ ] `./run_test.out speed/XxxSpeed` 能稳定运行（如无 ARM 设备，标注"待实测"并给出完整测试命令）
- [ ] 基线数据已记录（至少包含 3 组不同参数）
- [ ] 多次运行结果波动 < 5%（如无法实测，此项跳过）

### 常见问题

| 问题 | 原因 | 修复 |
|------|------|------|
| 性能数据波动大 | CPU 频率调度、后台任务 | 锁频 + 绑核 + 增加 repeat 次数 |
| 编译找不到头文件 | cmake 选项未开启 | 确认 `MNN_BUILD_TEST=ON` |
| 设备上运行段错误 | 交叉编译工具链不匹配 | 检查 ABI 和系统版本 |

---

## 下一步

**步骤 1 通过后，进入 `step2-analyze.md`（步骤 2：分析瓶颈与制定方案）。**
