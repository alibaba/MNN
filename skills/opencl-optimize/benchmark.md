# 步骤 0：建立性能基准

> **目标**：编写性能基准测试，获取优化前的基线数据，识别性能瓶颈。
>
> **前置条件**：明确待优化的算子和目标参数。
>
> **复杂度**：低（需要编译运行）
>
> **参考**：真机测试和正确性验证的详细说明，见 `SKILL.md` "正确性验证" 章节。

---

## 0.1 确定优化目标

```
待优化算子：____（例如 LinearAttention, MatMul, Convolution）
目标数据类型：____（FP32 / FP16）
典型参数：____（例如 B=1, H=4, d=64, L=1/16/128）
目标平台：____（例如 Android SM8350-Adreno 660, Mali-G78）
当前性能：____（如已知）
目标性能：____（如已知，理论峰值的 xx%）
```

---

## 0.2 定位目标 kernel

改 kernel 前必须先确认 dispatch 路径（详见 `SKILL.md` "入口定位"）：

```bash
grep -rn "OpType_<MyOp>" source/backend/opencl/execution/buffer/
```

读 `onResize` 把目标 shape 代入，确定落到哪个 `buildKernel(...)`，再找对应的 `.cl` 文件。

---

## 0.3 编写性能测试

### 检查现有测试

首先检查是否已有测试文件：
- `test/op/XxxTest.cpp` - 正确性测试
- `test/speed/XxxSpeedTest.cpp` - 性能测试

已有测试文件也**必须先核对配置**再用——配错后端会静默跑到 CPU 上，数据毫无意义：

1. **后端为 OpenCL**：`config.type = MNN_FORWARD_OPENCL`（命令行 `3`），并检查测试代码没硬编码别的后端
2. **执行模式和生产一致**：`numThread` LLM 用 68（强制 buffer），非 LLM 用 4（跟随生产模式）——含义见 SKILL.md numThread 与原则 2
3. **编译带 `MNN_GPU_TIME_PROFILE=ON`**：开启后打印每 kernel 真实耗时，此时只需跑 1 次

### 测试模板

```cpp
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include "MNNTestSuite.h"

class XxxSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        // 1. 配置 OpenCL 后端
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_OPENCL;
        config.numThread = 68;  // 仅 LLM: 64(强制buffer)+4(线程)。非 LLM 用 4(auto，跟随生产模式)

        // 2. 创建输入数据
        auto input = _Input({batch, seq_len, dim}, NCHW);

        // 3. 构建计算图
        auto output = _Xxx(input, ...);

        // 4. 执行并计时
        output->readMap<float>();

        return true;
    }
};

MNNTestSuiteRegister(XxxSpeedTest, "speed/XxxSpeed");
```

多参数组合覆盖：Decode（L=1，小 batch）、Prefill（L=16/64/128/2048）、不同 head（H=4/8/16）与维度（d=64/128）。

---

## 0.4 编译并运行

编译（带 `-DMNN_GPU_TIME_PROFILE=ON`）+ 推送命令见 SKILL.md「编译与真机运行」。确保 `adb devices` 有设备后运行 speed 测试：

```bash
adb shell "cd /data/local/tmp/MNN && LD_LIBRARY_PATH=. ./run_test.out speed/XxxSpeed 3 1 68"
# 末位 numThread：LLM 用 68（强制 buffer），非 LLM 用 4（跟随生产模式）——见 SKILL.md numThread 含义
```

`MNN_GPU_TIME_PROFILE` 开启后会打印每个 kernel 的真实耗时：

```
kernel time = 5    us kernel_name_1
kernel time = 123  us kernel_name_2
kernel time = 45   us kernel_name_3
```

**以这些打印的值为准**，这是 kernel 的真实耗时。

---

## 0.5 记录基线数据

创建基线数据文档，记录不同输入尺寸下每个 kernel 的耗时：

```markdown
## 基线数据

**平台**: Android SM8350 (Adreno 660)
**日期**: <填写实际日期>
**编译选项**: MNN_OPENCL=ON, MNN_LOW_MEMORY=ON, MNN_GPU_TIME_PROFILE=ON

### 场景1: Decode (B=1, H=4, d=64, L=1)

| Kernel | 时间(us) | 占比 |
|--------|---------|------|
| kernel_1 | xx.xx | xx% |
| kernel_2 | xx.xx | xx% |
| **总计** | **xx.xx** | 100% |
```

---

## 0.6 分析性能瓶颈

### 瓶颈识别清单

```
□ 哪个 kernel 占用时间最多？（优先优化目标）
□ 该 kernel 的计算特点？（内存密集 / 计算密集 / 同步密集）
□ 该 kernel 的并行度如何？（Work-group 数量、每个 work-item 的工作量）
□ 是否有明显的性能异常？（某些场景特别慢、某些参数组合性能差）
□ 计算强度是多少？属于 compute-bound 还是 memory-bound？（参考 `optimization-handbook.md` §1）
```

### 拿到瓶颈后去哪

本步只负责**识别**瓶颈,「怎么优化」交给下游,不在此重复判断表:

- **kernel 级 还是 算子级**（单 kernel 占比高 vs 多 kernel 合计高 / 格式转换开销）→ SKILL.md「Step A.2 判断优化级别」
- **memory-bound 还是 compute-bound + 选哪个技巧** → `optimization-handbook.md` §1（roofline）+ §5 速查表,或 `kernel-opt.md` §1.1 决策树

---

## 通过标准

- [ ] 性能测试文件已创建并可编译
- [ ] `./run_test.out speed/XxxSpeed` 能稳定运行
- [ ] 基线数据已记录（包含每个 kernel 的具体耗时和占比）
- [ ] 已识别出主要性能瓶颈

### 常见问题

| 问题 | 原因 | 修复 |
|------|------|------|
| 编译找不到头文件 | cmake 选项未开启 | 确认 `MNN_BUILD_TEST=ON` |
| 设备上运行段错误 | 交叉编译工具链不匹配 | 检查 ABI 和系统版本 |
| 看不到 kernel 耗时 | 未开启性能分析 | 确认 `MNN_GPU_TIME_PROFILE=ON` |
| 性能数据波动大 | warmup 不足 | 增加 warmup 次数 |
