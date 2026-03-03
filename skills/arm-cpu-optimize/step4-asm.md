# 步骤 4：ARM 汇编优化

> **目标**：用 ARM 汇编实现算子的核心内循环，充分利用 NEON/FP16/SDOT/I8MM/SME2 指令。
>
> **前置条件**：步骤 3 已通过（C++ 优化完成，已确定热点函数）。
>
> **复杂度**：高（需要编译运行验证）
>
> **注意**：优先保证正确性，性能精调可后续迭代。

---

## 4.1 汇编文件规范

### 文件位置和命名

```
source/backend/cpu/arm/arm64/MNNXxxKernel.S       ← AArch64 NEON FP32
source/backend/cpu/arm/arm64/MNNXxxKernelFP16.S   ← AArch64 FP16
source/backend/cpu/arm/arm32/MNNXxxKernel.S       ← AArch32

# 如果与矩阵乘相关：
source/backend/cpu/arm/arm64/MNNPackedMatMul_xxx.S
```

### 文件模板

```asm
//
//  MNNXxxKernel.S
//  MNN
//
//  Created by MNN on 2026/03/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

// void MNNXxxKernel(float* dst, const float* src, size_t count)
asm_function MNNXxxKernel

// 寄存器分配说明
// x0: dst
// x1: src
// x2: count

// 保存 callee-saved 寄存器（如需要）
stp d14, d15, [sp, #-16]!
stp d12, d13, [sp, #-16]!
stp d10, d11, [sp, #-16]!
stp d8,  d9,  [sp, #-16]!

// 主循环
L_Loop:
    // NEON 指令
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
    // ... 计算 ...
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64

    subs x2, x2, #16
    bgt L_Loop

// 余数处理
L_Remain:
    // 处理不足一个向量的尾部
    cbz x2, L_End
    // ...

L_End:
// 恢复 callee-saved 寄存器
ldp d8,  d9,  [sp], #16
ldp d10, d11, [sp], #16
ldp d12, d13, [sp], #16
ldp d14, d15, [sp], #16
ret

#endif // __aarch64__
```

---

## 4.2 汇编编写指南

### AArch64 寄存器规划

```
通用寄存器：
  x0-x7:   参数传入 / 返回值
  x8-x18:  临时寄存器（caller-saved）
  x19-x28: callee-saved（用了必须保存/恢复）
  x29(fp), x30(lr): 帧指针和链接寄存器

NEON/FP 寄存器：
  v0-v7:   参数 / 返回值 / 临时
  v8-v15:  callee-saved 的低 64 位（d8-d15）
  v16-v31: 临时（caller-saved，可以自由使用）
```

### 常用 NEON 指令

#### FP32 基础操作
```asm
ld1 {v0.4s}, [x0]           // 加载 4 个 float
st1 {v0.4s}, [x0]           // 存储 4 个 float
fmla v0.4s, v1.4s, v2.4s    // v0 += v1 * v2（FMA）
fmla v0.4s, v1.4s, v2.s[0]  // v0 += v1 * v2[0]（broadcast FMA）
fadd v0.4s, v1.4s, v2.4s    // 加
fmul v0.4s, v1.4s, v2.4s    // 乘
fmax v0.4s, v1.4s, v2.4s    // max
```

#### FP16（ARMv8.2）
```asm
// 需要 .arch armv8.2-a+fp16
ld1 {v0.8h}, [x0]           // 加载 8 个 fp16
fmla v0.8h, v1.8h, v2.8h    // v0 += v1 * v2（fp16 FMA）
fmla v0.8h, v1.8h, v2.h[0]  // broadcast FMA
fcvtl v0.4s, v1.4h          // fp16 → fp32（低 4 元素）
fcvtn v0.4h, v1.4s          // fp32 → fp16
```

#### SDOT（ARMv8.2 DotProduct）
```asm
// 需要 .arch armv8.2-a+dotprod
sdot v0.4s, v1.16b, v2.16b  // int8 点积：每 4 字节一组求和
sdot v0.4s, v1.16b, v2.4b[0] // broadcast 点积
```

#### I8MM（ARMv8.6）
```asm
// 需要 .arch armv8.6-a+i8mm
smmla v0.4s, v1.16b, v2.16b  // int8 矩阵乘：2×8 × 8×2 → 2×2
```

#### SME2（ARMv9）
```asm
// SME2 是矩阵扩展，使用 ZA tile 寄存器
// 需要特殊的编译支持和运行时检测
smstart                      // 进入 streaming SVE mode
fmopa za0.s, p0/m, z0.s, z1.s  // outer product 结果累加到 ZA tile
smstop                       // 退出 streaming mode
```

---

## 4.3 MatMul 汇编内核示例

以 FP32 NEON 的 4×8 Tile 为例（简化版）：

```asm
// 计算 C[4×8] += A[4×K] × B[K×8]
// x0: C (4×8 packed)
// x1: A (4×K packed)
// x2: B (K×8 packed)
// x3: K (loop count)

// 初始化累加器
eor v16.16b, v16.16b, v16.16b  // C[0,0:4]
eor v17.16b, v17.16b, v17.16b  // C[0,4:8]
eor v18.16b, v18.16b, v18.16b  // C[1,0:4]
eor v19.16b, v19.16b, v19.16b  // C[1,4:8]
eor v20.16b, v20.16b, v20.16b  // C[2,0:4]
eor v21.16b, v21.16b, v21.16b  // C[2,4:8]
eor v22.16b, v22.16b, v22.16b  // C[3,0:4]
eor v23.16b, v23.16b, v23.16b  // C[3,4:8]

L_K_Loop:
    // 加载 A 的 4 个元素
    ld1 {v0.4s}, [x1], #16    // A[0:4, k]

    // 加载 B 的 8 个元素
    ld1 {v4.4s, v5.4s}, [x2], #32  // B[k, 0:8]

    // FMA: C[i, :] += A[i, k] * B[k, :]
    fmla v16.4s, v4.4s, v0.s[0]   // C[0,0:4] += A[0,k] * B[k,0:4]
    fmla v17.4s, v5.4s, v0.s[0]   // C[0,4:8] += A[0,k] * B[k,4:8]
    fmla v18.4s, v4.4s, v0.s[1]   // C[1,0:4] += A[1,k] * B[k,0:4]
    fmla v19.4s, v5.4s, v0.s[1]   // C[1,4:8]
    fmla v20.4s, v4.4s, v0.s[2]   // C[2,0:4]
    fmla v21.4s, v5.4s, v0.s[2]   // C[2,4:8]
    fmla v22.4s, v4.4s, v0.s[3]   // C[3,0:4]
    fmla v23.4s, v5.4s, v0.s[3]   // C[3,4:8]

    subs x3, x3, #1
    bgt L_K_Loop

// 存储结果
st1 {v16.4s, v17.4s}, [x0], #32
st1 {v18.4s, v19.4s}, [x0], #32
st1 {v20.4s, v21.4s}, [x0], #32
st1 {v22.4s, v23.4s}, [x0], #32
```

---

## 4.4 注册汇编函数

在 C++ 中声明外部汇编函数并注册到 CoreFunctions：

```cpp
// 声明（在 CommonOptFunction.h 或对应的头文件中）
extern "C" {
void MNNXxxKernel(float* dst, const float* src, size_t count);
}

// 注册到 CoreFunctions（在 CPU 初始化代码中）
// 位置取决于具体优化的函数，通常在 arm64 的初始化函数中
gCoreFunctions->MNNPackedMatMul = MNNXxxKernel;
```

---

## 4.5 多指令集版本

为不同指令集创建不同版本的汇编：

```
MNNXxxKernel.S         → 基础 NEON FP32
MNNXxxKernelFP16.S     → ARMv8.2 FP16
MNNXxxKernel_sdot.S    → ARMv8.2 SDOT
MNNXxxKernel_i8mm.S    → ARMv8.6 I8MM
```

运行时检测 CPU 能力并选择最优版本：

```cpp
if (gCoreFunctions->supportI8mm) {
    // 使用 I8MM 版本
} else if (gCoreFunctions->supportSDot) {
    // 使用 SDOT 版本
} else if (gCoreFunctions->supportFp16arith) {
    // 使用 FP16 版本
} else {
    // 使用基础 NEON 版本
}
```

---

## 步骤 4 测试标准

### 测试方法

```bash
# 1. 编译并链接汇编
cd build
cmake .. -DMNN_BUILD_TEST=ON -DMNN_ARM82=ON
make -j$(nproc)

# 2. 正确性测试
./run_test.out op/Xxx

# 3. 性能测试
./run_test.out speed/XxxSpeed
```

### 通过标准

- [ ] **汇编编译无错误**
- [ ] **正确性测试通过**：`./run_test.out op/Xxx` 全部 passed
- [ ] **性能提升**：相对 C++ 版本有明显提升
- [ ] 记录优化后的性能数据

### 常见汇编错误

| 错误 | 原因 | 修复 |
|------|------|------|
| `undefined symbol` | 函数名不匹配 | 检查 `asm_function` 名与 C++ 声明一致 |
| Crash 在 `ret` | 未恢复 callee-saved 寄存器 | 检查 stp/ldp 配对 |
| 结果错误 | FMA 累加器没清零 | 循环前确保 `eor vN.16b, vN.16b, vN.16b` |
| 结果部分错误 | 余数处理有误 | 检查 `L_Remain` 分支 |
| `.arch` 编译错误 | SDOT/I8MM 需要指定 arch | 添加 `.arch armv8.2-a+dotprod` 等 |

---

## 下一步

**步骤 4 通过后，进入 `step5-integrate.md`（步骤 5：集成与回归测试）。**
