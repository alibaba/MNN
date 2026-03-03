---
name: arm-cpu-optimize
description: MNN ARM CPU 算子性能优化。涵盖计算拆解、函数复用、多线程、数据排布、ARM 汇编编写。采用"先正确，再加速"原则，基于性能基准测试驱动优化。
---

# MNN ARM CPU 性能优化 SKILL

> **触发条件**：当用户请求优化某个算子/内核在 ARM CPU 上的性能时触发。常见表述包括："优化xxx的ARM性能"、"加速xxx算子"、"写xxx的NEON实现"、"用SME2实现xxx"等。

## 概述

本 SKILL 指导 AI Agent 对 MNN 的 ARM CPU 后端进行性能优化。遵循 **"先正确，再加速"** 原则，每次优化都要保证结果不变。

### 核心原则

1. **正确性第一**：任何优化都必须通过正确性验证
2. **有数据支撑**：每次优化前后都要有**实测**性能数据对比
3. **优先复用已有函数（最重要）**：见下方详细说明
4. **替换前验证语义**：用 MNN 函数替换循环之前，必须确认函数的**精确数学语义**与原始代码一致（参数含义、归一化方式、边界行为等）。不要只看函数名就假设可以替换
5. **考虑数据规模**：函数调用和 Pack/Unpack 有固定开销。对小规模数据，朴素循环（编译器自动向量化）可能比调用 MNN 函数更快
6. **渐进式优化**：复用已有函数 → 多线程 → 数据排布 → 汇编（仅在必要时）

### ⚠️ 最重要的原则：优先复用已有函数

MNN 的 CoreFunctions 中已经包含了**经过汇编深度优化的高性能基础函数**。这些函数已经针对不同指令集（NEON/FP16/SDOT/I8MM/SME2）编写了专门的汇编内核，性能远超任何 C++ 循环或 Vec4 包装。

**优化的核心思路是：将算子的计算逻辑拆解为这些已有函数的组合调用，而不是自己写循环。**

#### 已有高性能函数清单

| 函数 | 作用 | 替代了什么 |
|------|------|----------|
| `gcore->MNNPackedMatMul` | 矩阵乘 C = A × B（已有 NEON/FP16/SME2 汇编） | 任何双重循环的矩阵乘 |
| `gcore->MNNPackedMatMulRemain` | 矩阵乘余数处理 | MatMul 的尾部处理 |
| `gcore->MNNComputeMatMulForE_1` | 矩阵向量乘 y = A × x（E=1 时专用） | 循环实现的 MatVec |
| `gcore->MNNComputeMatMulForH_1` | 向量矩阵乘 y = x × B（H=1 时专用） | 循环实现的 VecMat |
| `MNNScaleAndAddBiasScalar` | y = x * scale + bias | 循环乘标量/加标量 |
| `gcore->MNNScaleAndAddBias` | 按通道 scale + bias | 循环乘/加 |
| `MNNExp` | 批量 exp(x) | 循环调用 expf() |
| `MNNSiLu` / `MNNSiLuLowp` | 批量 SiLU 激活 | 循环 x*sigmoid(x) |
| `MNNSoftmax` | Softmax（含 Flash Attention 支持） | 循环 exp + sum + div |
| `MNNNorm` | LayerNorm / RMSNorm | 循环求范数 |
| `gcore->MNNPackCUnit` / `MNNUnpackCUnit` | NC4HW4 Pack/Unpack | 循环数据重排 |
| `gcore->MNNPackC4ForMatMul_A` | MatMul 的 A 矩阵 Pack | 循环重排 A |
| `gcore->MNNPackForMatMul_B` | MatMul 的 B 矩阵 Pack | 循环重排 B |
| `gcore->MNNConvRunForLineDepthwise` | Depthwise 卷积 | 循环卷积 |
| `MNNMatrixAdd` / `MNNMatrixSub` | 矩阵加减 | 循环加减 |
| `MNN_CONCURRENCY_BEGIN/END` | 多线程并行 | 单线程循环 |

#### ⚠️ 替换前必须验证的两件事

**1. 函数语义完全匹配**

在用 MNN 函数替换循环前，必须阅读函数签名和实现，确认数学语义完全一致。常见陷阱：
- 函数的归一化方式（sum vs mean）与你的需求不同
- 函数的参数含义（转置？通道顺序？）与你的数据布局不同
- 函数的输出格式与下游代码不兼容

**不要只看函数名就假设可以替换。** 如果语义不匹配，应寻找其他函数、调整数据，实在不行才退回手动实现并注释说明。

**2. in-place 安全性**

部分 MNN 函数不支持 `dst == src`（in-place 调用），因为内部实现会先写 dst 再读 src：

| 函数 | in-place (dst==src) | 说明 |
|------|:---:|------|
| `MNNScaleAndAddBiasScalar` | ✅ 安全 | 逐元素操作 |
| `MNNSiLu` / `MNNSiLuLowp` | ❌ 不安全 | 内部先写 dst 再读 src |
| `MNNExp` | ❌ 不安全 | 同上 |
| `MNNNorm` | ✅ 安全 | 只读 src，只写 dst |
| `gcore->MNNComputeMatMulForE_1` | ✅ 安全 | 输出独立于输入 |

> 当函数不支持 in-place 时，需要 scratch buffer（可复用 onResize 预分配的 buffer）。
> 判断方法：阅读函数实现或写小测试验证 `dst==src` 时结果是否正确。

#### ❌ 反面案例：避免以下做法

```cpp
// ❌ 错误1：已有 MNN 函数时用 Vec4 替代（MNN 函数有汇编优化，Vec4 只是 intrinsic 包装）
using Vec4 = Math::Vec<float, 4>;
for (int i = 0; i + 3 < size; i += 4) {
    Vec4 v = Vec4::load(data + i);
    v = v * Vec4(scale);
    Vec4::save(data + i, v);
}
// ✅ 正确：直接调用已有函数
MNNScaleAndAddBiasScalar(data, data, 0.0f, scale, size);

// ❌ 错误2：不考虑数据规模，总是用重量级函数
// MNNPackedMatMul 需要 Pack/Unpack，对小矩阵开销可能大于计算
// ✅ 正确：根据数据规模选择策略
//   大规模 → 用 MNN 函数（Pack 开销可摊薄）
//   小规模 → 保持朴素循环（编译器自动向量化，零额外开销）
//   MatVec（一个维度为1） → 用 MNNComputeMatMulForE_1（无需 Pack）

// ❌ 错误3：用循环实现 MatVec（S^T @ q）
for (int j = 0; j < dv; ++j) {
    float sum = 0;
    for (int i = 0; i < dk; ++i)
        sum += S[i*dv+j] * q[i];
    out[j] = sum;
}
// ✅ 正确：直接调用
gcore->MNNComputeMatMulForE_1(q, S, out, nullptr, &matParam, 0);

// ❌ 错误4：用循环实现 exp
for (int i = 0; i < size; ++i) dst[i] = expf(src[i]);
// ✅ 正确：
MNNExp(dst, src, offset, size);

// ❌ 错误5：用 std::vector 分配临时缓存
std::vector<float> temp(size);  // 每次调用都 malloc/free
// ✅ 正确：在 onResize 中用 Backend 的内存池
mTemp.reset(Tensor::createDevice<float>({size}));
backend()->onAcquireBuffer(mTemp.get(), Backend::DYNAMIC);
backend()->onReleaseBuffer(mTemp.get(), Backend::DYNAMIC);

// ❌ 错误6：用裸 NEON intrinsic 写循环（仅在没有 MNN 函数且性能敏感时才考虑，且更推荐写 .S 汇编）
#include "core/SimdHeader.h"
float32x4_t vsum = vdupq_n_f32(0.0f);

// ❌ 错误7：不验证 in-place 安全性就用 dst==src 调用
// 部分 MNN 函数不支持 in-place，会静默产生错误结果
// ✅ 正确：查阅上方 in-place 安全性表，不确定时使用 scratch buffer

// ❌ 错误8：不验证函数语义就替换（函数名匹配 ≠ 数学语义匹配）
// ✅ 正确：替换前阅读函数签名/实现，确认参数含义和计算逻辑完全一致
```

#### 🚫 分层实施策略

**强烈不建议的做法：**

| 避免使用 | 原因 | 应该用什么 |
|---------|------|----------|
| `#include "core/SimdHeader.h"` | 裸 NEON intrinsic 性能不一定最优且绑定平台 | 优先用 MNN 已有函数 或 编写 .S 汇编文件 |
| `std::vector<float>` 在 onExecute 中 | 每次运行都 malloc/free 开销巨大 | `Tensor + Backend 内存池` 用在 onResize |
| `#ifdef MNN_USE_NEON ... #else ... #endif` | 增加代码分支，维护困难 | 封装到底层函数，上层统一调用 |

**有条件允许 —— `Vec4` 循环：**

| 场景 | 是否允许 | 说明 |
|------|:------:|------|
| 已有 MNN 函数能覆盖 | ❌ | 必须用 MNN 函数，禁止用 Vec4 替代 |
| 没有对应 MNN 函数，且是性能热点 | ✅ | Vec4 作为中间方案，优于朴素循环 |
| 没有对应 MNN 函数，且计算量极小 | ❌ | 保持朴素循环，编译器自动向量化即可 |

> Vec4（`#include "math/Vec.hpp"`）本质是 intrinsic 的跨平台包装，性能不如专门调优的汇编函数，但**远优于朴素标量循环**。当 MNN 没有对应的已有函数且评估手写汇编成本过高时，Vec4 是合理的优化手段。

#### 优化决策树

```
看到一段循环代码 →
  ├─ 是双重循环的乘加？
  │   ├─ 有一个维度为 1（MatVec）？ → 用 MNNComputeMatMulForE_1（无需 Pack）
  │   ├─ 数据规模大（值得 Pack 开销）？ → 用 MNNPackedMatMul
  │   └─ 数据规模小（Pack 开销 > 计算）？ → 保持朴素循环
  ├─ 是单循环乘标量/加标量？ → 用 MNNScaleAndAddBiasScalar
  ├─ 是循环调用 expf/silu/sigmoid？ → 用 MNNExp/MNNSiLu
  ├─ 是循环做卷积？ → 用 MNNConvRunForLineDepthwise
  ├─ 是循环做数据重排？ → 用 MNNPackCUnit/MNNUnpackCUnit
  ├─ 是循环做 softmax？ → 用 MNNSoftmax
  ├─ 是循环做范数/归一化？ → 用 MNNNorm（先验证语义匹配！）
  ├─ 以上都不匹配？
  │   ├─ 能拆解为已有函数的组合？ → 组合调用（如外积 = MatMul 的特例）
  │   ├─ 计算量极小？ → 保持朴素循环（编译器自动向量化）
  │   ├─ 是性能热点且可向量化？ → 用 Vec4 循环作为中间方案
  │   └─ Vec4 也不够且是核心热点？ → 写新的 .S 汇编 kernel
  │
  ⚠️ 替换前：1) 验证函数语义匹配  2) 确认 in-place 安全性
```

### 注意事项

> **核心限制**：`schema/private/` 和 `source/internal/` 目录不应对 AI 公开或被随意修改。

> MNN 使用 **NC4HW4** 数据格式作为默认 CPU 布局，pack 大小由 `CoreFunctions::pack` 决定（FP32=4, FP16=8）

> **参考学习**：在开始优化前，强烈建议阅读 `source/backend/cpu/CPUAttention.cpp`，学习它如何调用 `gcore->MNNComputeMatMulForE_1`、`MNNScaleAndAddBiasScalar`、`MNNSoftmax` 等函数。你的优化代码应该通过调用 CoreFunctions 来实现高性能。

> **报告文件**：优化完成后请将性能报告写入 `<算子名>_optimization.md`，而不是仅在终端打印。

---

## ARM 指令集层级

从低到高，每级都向下兼容：

| 指令集 | 编译宏/检测 | Pack 大小 | 典型场景 | 代表芯片 |
|--------|-----------|----------|------------|---------|
| **ARMv7 NEON** | `__arm__` | 4 (FP32) | 基础 SIMD | Cortex-A7/A15 |
| **ARMv8 NEON** | `__aarch64__` | 4 (FP32) | 标准 64 位 | Cortex-A53/A72 |
| **ARMv8.2 FP16** | `MNN_ARM82` / `supportFp16arith` | 8 (FP16) | 半精度加速 | A55/A76/A78 |
| **ARMv8.2 SDOT** | `supportSDot` | - | INT8 点积加速 | A75+, A55r1+ |
| **ARMv8.6 I8MM** | `MNN_ARM86` / `supportI8mm` | - | INT8 矩阵乘加速 | A78C/X2/X3 |
| **ARMv9 SME2** | `MNN_SME2` / `supportSME2` | 可变 | 矩阵扩展引擎 | X4/X925 |

---

## 核心架构元素

### CoreFunctions 结构

所有 CPU 函数指针都注册在 `CoreFunctions` 中（`source/backend/cpu/compute/CommonOptFunction.h`）。运行时根据 CPU 能力选择最优实现：

```cpp
struct CoreFunctions {
    // CPU 特性检测
    bool supportFp16arith;
    bool supportSDot;
    bool supportI8mm;
    bool supportSME2;

    // Pack 参数
    int pack;     // FP32=4, FP16=8
    int bytes;    // FP32=4, FP16=2

    // 关键函数指针
    void(*MNNPackedMatMul)(...);         // 矩阵乘主核心
    void(*MNNPackedMatMulRemain)(...);   // 矩阵乘余数处理
    void(*MNNPackC4ForMatMul_A)(...);    // 输入数据 Pack
    void(*MNNPackForMatMul_B)(...);      // 权重数据 Pack
    void(*MNNGetMatMulPackMode)(&eP, &lP, &hP); // 获取 Pack 参数
    // ...
};
```

### 数据排布

| 排布 | 说明 | 使用场景 |
|------|------|---------  |
| **NC4HW4** / **NC8HW8** | 通道方向 Pack4/8，SIMD 友好 | 卷积、Pooling 等 |
| **[e/eP, l/lP, eP, lP]** | MatMul 的 A 矩阵 Pack | GEMM 优化 |
| **[h/hP, l/lP, lP, hP]** | MatMul 的 B 矩阵 Pack | 权重重排 |
| **NCHW** | 标准线性布局 | 形状计算、非 Pack 算子 |

### 文件组织

```
source/backend/cpu/
├── CPUXxx.cpp/.hpp              ← 算子主逻辑（调度、多线程）
├── compute/
│   ├── CommonOptFunction.h      ← CoreFunctions 定义
│   ├── CommonOptFunction.cpp    ← 默认 C++ 实现
│   ├── ConvOpt.h                ← 卷积相关函数声明
│   └── ...
├── arm/
│   ├── arm64/
│   │   ├── MNNPackedMatMul.S            ← FP32 NEON 矩阵乘
│   │   ├── MNNPackedMatMulFP16.S        ← FP16 矩阵乘
│   │   ├── MNNGemmInt8AddBiasScale_*.S  ← INT8 GEMM
│   │   ├── MNNPackedMatMul_int8.S       ← SDOT 矩阵乘
│   │   ├── MNNPackedMatMulRemain_int8.S ← SDOT 余数
│   │   └── ...
│   └── arm32/
│       ├── MNNGemmInt8*.S               ← 32 位 INT8 GEMM
│       └── ...
└── x86_64/
    └── ...                              ← x86 SSE/AVX 实现
```

---

## 分步流程总览

```
┌──────────────────────────────────────────────────────────┐
│  步骤 0: 计算拆解 (step0-decompose.md) ★ 最关键的步骤     │
│  输入: 算子源码                                           │
│  输出: 每个计算逻辑对应的 MNN 已有函数映射表               │
│  测试: 所有可复用的函数已识别，不存在遗漏                   │
├──────────────────────────────────────────────────────────┤
│  步骤 1: 建立性能基准 (step1-benchmark.md)               │
│  输入: 待优化的算子名和参数                                │
│  输出: test/speed/ 下的基准测试 + 基线数据                 │
│  测试: 基准测试能稳定运行，数据可复现                       │
├──────────────────────────────────────────────────────────┤
│  步骤 2: 分析瓶颈与制定方案 (step2-analyze.md)            │
│  输入: 步骤 0 的映射表 + 步骤 1 的基线数据                 │
│  输出: 优化方案（哪些换函数、哪些改排布、哪些需要新汇编）    │
│  测试: 方案可行性论证                                     │
├──────────────────────────────────────────────────────────┤
│  步骤 3: C++ 优化（函数替换 + 多线程 + 排布）              │
│  输入: 步骤 2 的优化方案                                  │
│  输出: 用 MNN 函数替换循环 + 多线程 + 内存池               │
│  测试: 正确性验证 + 性能对比                               │
├──────────────────────────────────────────────────────────┤
│  步骤 4: ARM 汇编优化 (step4-asm.md)（仅在必要时）        │
│  输入: 步骤 3 中无法用已有函数覆盖的热点                    │
│  输出: ARM NEON/FP16/SDOT/I8MM/SME2 汇编实现             │
│  测试: 正确性验证 + 性能对比 + 多指令集覆盖               │
├──────────────────────────────────────────────────────────┤
│  步骤 5: 集成与回归测试 (step5-integrate.md)              │
│  输入: 步骤 3 或 4 通过                                   │
│  输出: 全量回归测试                                       │
│  测试: 所有相关算子测试通过 + 性能报告                     │
└──────────────────────────────────────────────────────────┘
```

**说明**：步骤 4（汇编）往往不是必须的。如果通过步骤 3（利用现有 MNN 工具、重排版数据并应用多线程）已经达到了优异的性能提升，便可以直接进入集成测试阶段（步骤 5）。

### 失败回退

| 情况 | 处理方式 |
|------|---------|
| step3 某个替换导致性能下降 | 回退该替换，保持原实现，重新评估原因（如小数据规模开销大） |
| step3 实施时发现 step0 映射有误 | 修正映射表，选择更适合的函数或退回到手动循环 |
| 无法在 ARM 设备上实测性能 | 明确标记"待实测"，给出测试指令，并在力所能及的平台上做验证 |
| step4 汇编写出来但正确性不过 | 回退到安全 C++/Vec4 实现，确保可用性为先 |

---

## 参考文件

| 文件 | 参考价值 | 优先级 |
|------|---------|-------|
| `source/backend/cpu/CPUAttention.cpp` | **推荐阅读**：多线程 + CoreFunctions 调用的标杆实现 | ★★★ |
| `source/backend/cpu/compute/CommonOptFunction.h` | **必备**：CoreFunctions 定义及可用函数签名 | ★★★ |
| `source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp` | MatMul Tiling + Pack + 多线程参考 | ★★ |
| `test/speed/MatMulSpeed.cpp` | 性能基准测试参考模板 | ★★ |
| `source/backend/cpu/arm/arm64/MNNPackedMatMul.S` | 汇编编写参考案例 | ★ |

---

## 开始执行

**现在请打开 `skills/arm-cpu-optimize/step0-decompose.md`，开始步骤 0。**
