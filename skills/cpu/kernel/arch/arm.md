# AArch64 kernel 实现参考

> **何时读**：要在 AArch64 上写一条新 kernel 的 intrinsic 或 `.S`（NEON / SDOT / I8MM / SME2）、
> 给低 bit 权重加一条 unpack 路径、或迁移一个已有 kernel 到更高档 ISA 之前。
> **本文只写 AArch64 专属事实**：目录与命名、`asm_function` 与指令编码方式、ABI 寄存器分区、
> 三档矩阵指令的语义与错位、低 bit unpack 的指令预算。
>
> **不在本文**：标量 oracle 从哪来、分层比较点、跨 ISA × 精度的正确性矩阵
> → [`../correctness-gate.md`](../correctness-gate.md)；什么时候才该下沉到 asm
> → [`../SKILL.md`](../SKILL.md) 铁律 1；tile/packer/cell stride 这五个同源量与 `QuanPostTreatParameters`
> 字段语义 → [`../pack-and-abi.md`](../pack-and-abi.md)；函数表注册与快照时序
> → [`../dispatch-and-register.md`](../dispatch-and-register.md)；ARM 侧派发路径全景与 `MNN_CPU_TARGET`
> → [`../../optimize/arch/arm.md`](../../optimize/arch/arm.md)；benchmark 与测试命令
> → [`../../shared/build-test-and-benchmark.md`](../../shared/build-test-and-benchmark.md)。
> x86_64 侧对应 [`x86_64.md`](x86_64.md)，RISC-V 侧 [`riscv.md`](riscv.md)。
>
> 术语：ISA 写 **AArch64**，目录名写 **arm64**。
> **本文所有 `.S` 路径相对 `source/backend/`**，`.h`/`.cpp` 路径相对 `source/backend/cpu/`。

## 〇、三条必须先纠正的认知

这三点是关于 AArch64 asm 的常见误解（本 skill 的前身 `step4-asm.md` 正是这么写的，已随重构删除），
与仓库实测不符，先记住，否则后面全错：

| # | 常见误解 | 仓库实测（HEAD） |
|---|---|---|
| 1 | 用 `.arch armv8.2-a+dotprod` / `.arch armv8.6-a+i8mm` 打开指令 | **全仓库 `.S` 里没有任何 `.arch` 指示符**（grep = 0）。`sdot` / `udot` / `smmla` / SME2 指令**一条都不以助记符形式出现**，全部写成 `.inst <hex> // <助记符>`。见 §2.2 |
| 2 | fp16 kernel 是 `arm64/MNNXxxKernelFP16.S` | fp16 的 `.S` **不在** `cpu/arm/arm64/` 下，而在**另一棵树** `arm82/asm/arm64/`（另一个 object lib、另一份 `MNNAsmGlobal.h`）。见 §1.1 |
| 3 | int8 kernel 用 `_int8.S` 后缀 | **全仓库没有一个 `*_int8.S`**（find = 0）。int8 kernel 靠**函数名**区分（`MNNGemmInt8AddBiasScale_*`），不靠文件名后缀 |

## 一、文件位置与命名：实测目录清单

### 1.1 目录矩阵

`.S` 计数用 `ls <dir>/*.S | wc -l` 实测（HEAD）：

| 目录 | `.S` 数 | 承载 | object lib | 编译门 |
|---|---|---|---|---|
| `cpu/arm/arm64/` | 105 | NEON 基线 + SDOT + I8MM 的 fp32/int8 kernel | `MNNARM64` | `__aarch64__`（`cpu/arm/CMakeLists.txt`） |
| `cpu/arm/arm64/low_memory/` | 17 | 低 bit（w2/w3/w4）+ 动态量化 + AbsMax | 同上 | **`MNN_LOW_MEMORY`** |
| `cpu/arm/arm64/bf16/` | 6 | bf16 matmul / pack | 同上 | `MNN_SUPPORT_BF16` |
| `cpu/arm/arm64/sme2_asm/` | 13 | **进 streaming mode 的** SME2 kernel（fp32 **与 fp16 都在这里**） | 同上 | `MNN_SME2` |
| `cpu/arm/arm32/` | 89 | AArch32 NEON | `MNNARM32` | `^armv7` |
| `cpu/arm/arm32/bf16/` | 4 | AArch32 bf16 | 同上 | `MNN_SUPPORT_BF16` |
| `arm82/asm/arm64/` | 15 | **fp16** matmul / pack / activation | `MNN_Arm82` | `MNN_USE_ARMV82`（`arm82/CMakeLists.txt`） |
| `arm82/asm/arm64/low_memory/` | 17 | **fp16** 低 bit gemm + 动态量化 | 同上 | `MNN_LOW_MEMORY` |
| `arm82/asm/arm64/sme2_asm/` | **1** | 只有 `MNNPackedMatMulRemainFP16_SME2.S` | 同上 | `MNN_SME2` |
| `arm82/asm/arm32/` | 10 | AArch32 fp16 | 同上 | `^armv7` |

> `cpu/arm/CMakeLists.txt` 与 `arm82/CMakeLists.txt` 都是被 `include()`（不是 `add_subdirectory()`）
> 拉进来的（都在 `cpu/CMakeLists.txt` 里 `add_subdirectory`），**共享变量作用域**，且 arm 在前、arm82 在后。
> 这条时序在 §2.2 会咬人。

### 1.2 命名模式实测

| 模式 | 含义 | 真实例子 |
|---|---|---|
| `MNNXxx.S` | NEON 基线，fp32 | `cpu/arm/arm64/MNNPackedMatMul.S`、`MNNRankOneUpdate.S` |
| `..._ARMV82_...` | **SDOT** 档（不是 "armv8.2 泛指"） | `cpu/arm/arm64/MNNGemmInt8AddBiasScale_ARMV82_Unit.S` |
| `..._ARMV86_...` | **I8MM** 档 | `cpu/arm/arm64/MNNGemmInt8AddBiasScale_ARMV86_Unit.S` |
| `..._w2_ / _w3_ / _w4_` | 低 bit 权重位宽，**必在 `low_memory/`** | `low_memory/MNNGemmInt8AddBiasScale_ARMV82_w3_Unit.S` |
| `..._FP16.S` | fp16，**必在 `arm82/asm/` 树下** | `arm82/asm/arm64/low_memory/MNNGemmInt8AddBiasScale_ARMV86_w4_Unit_FP16.S` |
| `..._SME2.S` / `..._Sme2_HpNN.S` | SME2 | `sme2_asm/MNNPermuteSumWeightInt4Sme2_Hp128.S` |
| `..._SME2_w4_Fp32.S` / `_Fp16.S` | SME2 int8 gemm，**精度在文件名尾部** | `sme2_asm/MNNGemmInt8AddBiasScale16x32_SME2_w4_Fp16.S` |
| `ARMV86_MNNXxx_BF16.S` | bf16，**档位是前缀不是中缀** | `arm64/bf16/ARMV86_MNNPackedMatMul_BF16.S` |

四条实测出来的不一致，抄命名时别被带偏：

- **精度大小写不统一**：`arm82/` 树用 `_FP16`（全大写），`sme2_asm/` 用 `_Fp16` / `_Fp32`（首字母大写）。
  **SME2 的 fp16 int8-gemm kernel 在 `cpu/arm/arm64/sme2_asm/`，不在 `arm82/` 树下**——
  这是 fp16 唯一一处不在 arm82 树里的例外。
- **bf16 的档位标记是前缀**（`ARMV86_MNNPackedMatMul_BF16.S`），其余全是中缀/后缀。
- **同名 kernel 跨目录同时存在**：`MNNGemmInt8AddBiasScale_16x4_w4_Unit.S` 在
  `cpu/arm/arm64/low_memory/`（受 `MNN_LOW_MEMORY` 门控）**和** `cpu/arm/arm32/` 根目录
  （**不受门控，恒编译**）各有一份。arm32 侧没有 `low_memory/` 子目录。
- **int8 kernel 没有 `_int8.S` 后缀**：数据类型编码在函数名里而不是文件名里，所以
  `MNNPackedMatMul_int8.S` 这类文件名**并不存在**（凭 `MNNPackedMatMul.S` 类推会踩空）。
  int8 的 packed matmul 参考看 `MNNGemmInt8AddBiasScale_ARMV82_Unit.S` / `_ARMV86_Unit.S`。

### 1.3 `sme2_asm/` 的准入判据：是否进 streaming mode

放不放 `sme2_asm/` **不看文件名里有没有 SME2**，看**是否执行 `smstart`**。实测（HEAD，全命中）：

| 文件 | `smstart` | 目录 |
|---|---|---|
| `sme2_asm/` 下 13 个文件 | **13/13 都有** | `sme2_asm/` |
| `cpu/arm/arm64/MNNPackC4Int8ForMatMulA_SME2.S` | **0** | arm64 根目录 |
| `cpu/arm/arm64/MNNPackC4Int8ForMatMulA_SME2_Hp64.S` | **0** | arm64 根目录 |

这两个 packer 名字里带 `SME2`，但只是**tile 参数按 SME2 布局**的普通 NEON 代码（无 `z`/`za` 寄存器），
所以留在根目录。

- **规则**：新写的 SME2 kernel 若用 `z*` / `za*` / 谓词寄存器 → 必须 `smstart`/`smstop` → 进 `sme2_asm/`；
  若只是"为 SME2 的 tile 做重排"的纯 NEON 代码 → 放 arm64 根目录。
- **放错的后果**：放进 `sme2_asm/` 会被套上 `MNN_SME2` 构建门，非 SME2 构建里符号消失；
  反之纯 SME2 指令放在根目录会在所有 aarch64 构建里被编译（本仓库靠 `.inst` 绕过汇编器检查，
  所以**不会编译报错**，只会在不支持的机器上执行时 SIGILL）。

### 1.4 arm32 侧

存在（89 + 4 个 `.S`），命名与 arm64 **基名相同**（`MNNPackedMatMul.S`、`MNNGemmInt8AddBiasScale_16x4_Unit.S` …），
但：

- **没有** `_ARMV82_` / `_ARMV86_` / `SME2` 任何文件 —— arm32 侧无 SDOT / I8MM / SME2 asm 路径。
- **没有** `low_memory/` 子目录，低 bit 只有 `MNNGemmInt8AddBiasScale_16x4_w4_Unit.S` 一个（w4，无 w2/w3）。
- fp16 在 `arm82/asm/arm32/`（10 个），用 `-mfpu=neon-fp-armv8 -mfloat-abi=softfp`（`arm82/CMakeLists.txt`）。
- arm32 的 `GEMM_INT8_DST_XUNIT` 是 **2**，不是 4（`compute/Int8FunctionsOpt.h` 的 `#else` 分支）。

## 二、`asm_function` 与文件骨架

### 2.1 `MNNAsmGlobal.h`：全文 13 行，只做符号可见性

`cpu/arm/MNNAsmGlobal.h`（完整内容）：

```asm
.macro asm_function fname
#ifdef __APPLE__
.globl _\fname
_\fname:
#else
.global \fname
#ifdef __ELF__
.hidden \fname
.type \fname, %function
#endif
\fname:
#endif
.endm
```

它做四件事，一件不多：

| 平台 | 下划线前缀 | 导出指示符 | 可见性 | 符号类型 |
|---|---|---|---|---|
| Apple（`__APPLE__`） | **加 `_`** | `.globl` | 默认（可导出） | 不声明 |
| ELF | 不加 | `.global` | **`.hidden`** | `.type ... %function` |
| 其它（如 MSVC/ARM64EC） | 不加 | `.global` | 默认 | 不声明 |

三条实操后果：

- **它不生成 prologue / epilogue、不切 section、不对齐**。`.text` / `.align` / `stp` / `ret` 全要自己写。
- **ELF 上符号是 hidden 的**。想用 `nm` 自证「新 kernel 编进来了」，要用 `nm -a` 或
  `readelf -s`（看 `LOCAL`/`HIDDEN`），`nm -D`（动态符号表）查不到。Apple 上则要记得**多一个下划线**。
- 有**三份内容等价的副本**：`cpu/arm/MNNAsmGlobal.h`、`arm82/asm/MNNAsmGlobal.h`、
  `cpu/x86_x64/MNNAsmGlobal.h`（三份 diff 只差一行创建日期注释）。
  include 路径由各自的 `target_include_directories` 提供：
  `cpu/arm/CMakeLists.txt`（→ `cpu/arm/`）、`arm82/CMakeLists.txt`（→ `arm82/asm/`）。
  **`.S` 里一律写 `#include "MNNAsmGlobal.h"`，靠 include 路径选中正确那份**——
  不要写相对路径，否则文件在两棵树之间搬移时会拉到错的那份。

对应的 C++ 侧声明必须 `extern "C"`（无 mangling 才能连上 asm 符号），例如
`compute/Int8FunctionsOpt.h` 的 `#ifdef __cplusplus extern "C" {`。

### 2.2 没有 `.arch`：本仓库用 `.inst` 手工编码

**这是 AArch64 侧最重要的一条本地约定。** 实测：

| 检查项 | 结果 |
|---|---|
| `.S` 中 `.arch` 指示符 | **0** 处 |
| `.S` 中 `sdot` / `udot` / `smmla` / `usmmla` 以**助记符**出现 | **0** 处 |
| `.S` 中 `.inst` 行（`source/backend/` 全量） | **6006** 行 |

按目录分布：`sme2_asm/` 4581、`arm82/asm/arm64/low_memory/` 392、`cpu/arm/arm64/low_memory/` 387、
`cpu/arm/arm64/` 359、`arm82/asm/arm64/` 5。

三档的真实写法（**直接摘自仓库**，可复制）：

```asm
// SDOT —— cpu/arm/arm64/MNNGemmInt8AddBiasScale_ARMV82_Unit.S
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]

// I8MM —— cpu/arm/arm64/low_memory/MNNGemmInt8AddBiasScale_ARMV86_w3_Unit.S
    .inst 0x4e88a46c // smmla v12.4s, v3.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a46d // smmla v13.4s, v3.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    .inst 0x4e8aa46e // smmla v14.4s, v3.16b, v10.16b // tile0-oc4, tile0-oc5, tile1-oc4, tile1-oc5
    .inst 0x4e8ba46f // smmla v15.4s, v3.16b, v11.16b // tile0-oc6, tile0-oc7, tile1-oc6, tile1-oc7

// SME2 —— cpu/arm/arm64/sme2_asm/MNNGemmInt8AddBiasScale16x32_SME2_w4_Fp32.S
.inst 0xd503477f  // smstart
.inst 0xc00800ff  // zero {za}
    .inst 0xc0080033  // zero {za0.s, za1.s}
    .inst 0xa400bd60  // ld1b {z0.b}, p7/z, [x11]       // src
    .inst 0xa400ac41  // ld1b {z1.b}, p3/z, [x2]        // weight
    .inst 0xc08a4022  // luti4 {z2.b-z3.b}, zt0, z1[0]  // int4->int8
    .inst 0xa0827c00  // smopa za0.s, p7/m, p3/m, z0.b, z2.b
    .inst 0xa0837c01  // smopa za1.s, p7/m, p3/m, z0.b, z3.b
```

**为什么是 `.inst`**：CMake 从未给这些文件加过对应的 `-march`。

- `cpu/arm/CMakeLists.txt` 那条会给 SME2 asm 加
  `-march=armv8.6-a+sve+sve2+sme+sme2+fp16` 的语句是**被注释掉的**。
- 紧接着的那条 `set_source_files_properties` 作用在 `${MNN_SME2_SRCS_ASM_FP16}` 上——**这个变量在该文件里从未定义**，
  它定义在 `arm82/CMakeLists.txt`，而 arm82 是在 `cpu/CMakeLists.txt` 被 include 的，
  **晚于** arm 子目录。所以 `cpu/arm/CMakeLists.txt` 执行时变量是空的，整句是 no-op。
- 结论：**`cpu/arm/arm64/sme2_asm/` 下的文件没有任何额外 `-march`**，只继承 `MNNARM64` 的默认基线。
  `arm82` 树最高只给到 `-march=armv8.2-a+fp16`（`arm82/CMakeLists.txt`）。

**所以写新 kernel 时：**

| 你要用的指令 | 怎么写 |
|---|---|
| 基线 NEON（`ld1` / `fmla` / `tbl` / `ushl` / `scvtf` / `ld1r` …） | 直接写助记符 |
| `sdot` / `udot` / `smmla` / `usmmla` / 任何 SME/SVE 指令 | **必须 `.inst 0xXXXXXXXX // <助记符>`**，注释里写全等价助记符 |

- **注释是唯一的可读性来源**，格式跟随仓库：`.inst <8位小写 hex>  // <助记符及操作数>`，
  必要时再追加语义注释（见上面 smmla 例子里的 `tile0-oc0, ...`）。**注释写错不会有任何报错**，
  这是本仓库 asm 最脆的一环——编码前后各人工核对一次。
- 得到 hex 的可靠办法：写一个只含目标指令的临时 `.s`，用带 `-march=armv8.6-a+i8mm+sme2` 的
  `clang`/`as` 汇编，再 `objdump -d` 抄回编码。**不要手算**。
- **也不要"顺手加上 `.arch`"**：加了之后该文件里的 `.inst` 仍然有效，但你打破了
  "全仓库零 `.arch`" 的一致性，且部分工具链（Xcode 集成汇编器、旧 NDK）对 `.arch` 支持不一致，
  这正是本仓库选 `.inst` 的原因。

### 2.3 最小 `.S` 模板（按实测校准）

以 `cpu/arm/arm64/MNNRankOneUpdate.S`（仓库里最新、最干净的一个）为骨架：

```asm
//
//  MNNXxxKernel.S
//  MNN
//
//  Created by MNN on 20XX/XX/XX.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__            // 或 #if defined(__aarch64__)，见下表
#include "MNNAsmGlobal.h"

.text
.align 5                      // 2^5 = 32 字节；arm64 根目录 105/105 个文件都有

// 可选：宏定义放在 asm_function 之前
.macro ADD_BIAS_FLOAT d0, d1, d2, d3, z0
    fadd \d0\().4s, \d0\().4s, \z0\().4s
    ...
.endm

// void MNNXxxKernel(float* dst, const float* src, size_t n)
// x0:dst  x1:src  x2:n
// 寄存器计划（必答的五个 live range 问题见本文 §3.5）：
//   v16-v25: accumulator，compute → store
//   v30-v31: fp32 min/max，仅 postprocess 存活
asm_function MNNXxxKernel

// 只在真的用到 callee-saved 时才开栈帧
stp d14, d15, [sp, #(-16 * 10)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]
// ...

.LXxx_LoopRow:
    // load / unpack / compute / postprocess / store
    subs x3, x3, #1
    bne .LXxx_LoopRow

.LXxx_End:
// 逆序恢复
ldp x19, x20, [sp, #(16 * 5)]
ldp x21, x22, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 10)
ret

#endif // __aarch64__
```

模板里每一行的实测依据：

| 元素 | 实测 |
|---|---|
| `#ifdef __aarch64__` vs `#if defined(__aarch64__)` | 两种都在用，**118 : 17**（arm64 三个目录共 135 个 `.S`）。`#ifdef` 是多数派；**135/135 全都有守卫，一个不缺** |
| `#include "MNNAsmGlobal.h"` | 在 `#ifdef` **之内**（守卫先、include 后） |
| `.text` | 全部有 |
| `.align 5` | arm64 根目录 **105/105** 有；`sme2_asm/` 只 **5/13** 有。新文件按 105 那边写，加上 |
| `ret` | 写在列首（不缩进），紧接 `#endif` |
| 局部标签 | 新文件用 `.L` 前缀（`.LRou_LoopRow`，不进符号表）；老文件用裸标签（`LoopDz8_TILE_10:`）。**新代码用 `.L`** |

## 三、AArch64 ABI 与寄存器分区

### 3.1 ABI 事实表（AAPCS64，与本仓库无关的硬约束）

| 寄存器 | 角色 | 保存责任 |
|---|---|---|
| `x0`-`x7` | 整型/指针参数 1-8，`x0`(-`x1`) 返回值 | caller-saved |
| `x8` | 间接结果位置（本仓库当普通临时用） | caller-saved |
| `x9`-`x15` | 临时 | caller-saved |
| `x16` / `x17` | IP0 / IP1，linker veneer 可能改写 | caller-saved |
| `x18` | 平台寄存器（**Darwin / Windows 保留**） | **不要用** |
| `x19`-`x28` | 长生命周期 | **callee-saved：必须存-恢复** |
| `x29` (FP) / `x30` (LR) | 帧指针 / 返回地址 | callee-saved |
| `sp` | 栈指针，**必须 16 字节对齐** | — |
| `v0`-`v7` | 浮点/向量参数与返回值 | caller-saved |
| `v8`-`v15` | — | **只有低 64 位（`d8`-`d15`）callee-saved**；高 64 位是 caller-saved |
| `v16`-`v31` | 临时 | caller-saved |
| `z0`-`z31` / `p0`-`p15` / `za` / `zt0` | SVE / SME 状态 | 全部 caller-saved（streaming 边界另有规则，见 §4.4） |

两条最容易翻车的：

- **`v8`-`v15` 只保低 64 位。** 用 `v8.16b` 存 128 位 unpack 中间量并期望它跨调用存活 → 高 64 位丢。
  仓库的存法正是 `stp d8, d9, ...`（64 位一个），**这就是 ABI 只要求低 64 位的直接体现**，
  不是省事。要在这段里放 128 位长生命周期值，就得自己额外开栈槽。
- **`x18` 在 Darwin / Windows 上是保留的**，绝不能当临时寄存器用。仓库 kernel 里普遍从 `x19` 起用，
  临时集中在 `x8`-`x17`。

### 3.2 本仓库的实际分区惯例（以及它的反例）

老文档给的分配（accumulator `v16`-`v25`、scale/min/max `v26`-`v31`）是一个**合理起点，但不是仓库不变量**。
实测：

| 文件 | accumulator | fp32 min/max |
|---|---|---|
| `low_memory/..._ARMV86_w3_Unit.S` TILE_10 | **`v12`-`v31`（20 个）** | 无空闲寄存器，postprocess 时才腾出 |
| `low_memory/..._ARMV86_w4_Unit.S` | — | **`v30`/`v31`** |
| `low_memory/..._ARMV82_w4_Unit.S` | — | **`v26`/`v27`** |
| `low_memory/..._ARMV82_w4_Unit.S` | — | **`v0`/`v1`**（大 tile 分支，v26/v27 已被占） |
| `low_memory/..._ARMV82_w3_Unit.S` | — | `v26`/`v27` |

**同一个文件里不同 tile 分支用不同寄存器装 min/max**（`_ARMV82_w4_Unit.S` 里 `v0/v1` 与 `v26/v27` 并存），
这正是 §3.5 那五个 live range 问题要逐个 tile / tail 分支回答的原因。

AArch64 上的分区**建议**（tile 数大时会被迫破例，破例就要重做 live range 表）：

| 用途 | 建议寄存器 | 依据 |
|---|---|---|
| src / 已 unpack 的 weight / unpack 临时 | `v0`-`v7`（+ `v8`-`v11` 若已存 `d8`-`d11`） | `_ARMV86_w3_Unit.S` 用 `v0`(main) `v1`(aux) `v2`/`v3`(mask) `v4`/`v5`(shift) `v8`-`v11`(unpacked) `v3`-`v7`(src) |
| accumulator | `v16`-`v31`，不够时向下扩到 `v12` | 同上，TILE_10 占 `v12`-`v31` |
| 常量 mask / shift 表 | `v2`-`v5` 之类的低位，且**每次 loop 重建**（`movi`）或从 `adr` 常量池 `ld1` | `_ARMV86_w3_Unit.S` `adr x16, .L_w3_unpack_consts_fp32` |
| fp32 min/max | `v30`/`v31`（默认）或 `v26`/`v27`；tile 满时用 `v0`/`v1` | 见上表 |

### 3.3 prologue / epilogue 的两种实测形态

| | NEON / SDOT / I8MM | SME2 |
|---|---|---|
| 帧大小 | **160 字节**：`stp d14, d15, [sp, #(-16 * 10)]!` | **320 字节**：`stp x29, x30, [sp, #-320]!` |
| 帧指针 | 不建 | **`mov x29, sp`** |
| 保存内容 | `d8`-`d15` + `x19`-`x28` | `x29`/`x30` + `x19`-`x28` + `d8`-`d15` |
| streaming | — | `smstart` 在**保存之后**、`smstop` 在**恢复之前** |
| 坐标 | `cpu/arm/arm64/MNNGemmInt8AddBiasScale_ARMV82_Unit.S` / 入口 `stp` → 尾部逐条逆序 `ldp` → `ret` | `sme2_asm/MNNGemmInt8AddBiasScaleHp128_SME2_w4_Fp32.S` / 尾部 `smstop` → `ldp` → `ret` |

- 恢复**必须严格逆序**（`_ARMV82_Unit.S`、`_ARMV86_w3_Unit.S` 都是逐条逆序）。
- SME2 那 320 字节里有大量未用槽位（`#96`-`#160` 之间），是给 streaming 下的 spill 预留的；
  改 SME2 kernel 时**不要顺手压缩帧**，先确认没有 `st1w`/`str` 往这些槽写。

### 3.4 `QuanPostTreatParameters` 的硬编码字节偏移（asm 侧独有的雷）

int8 kernel 的 asm 侧**只认字节偏移，不认字段名**。`x6` 是结构体指针，实测取法：

| 偏移 | 字段 | 仓库取法（`_ARMV86_w3_Unit.S`） |
|---|---|---|
| `#0` | `scale` | 各 tile 分支内 `ldr`；**但 `_16x4_Unit_FAST.S` 完全不读 `#0`**——它的 scale 已折进权重流 |
| `#8` | `biasFloat` | `ldr x9, [x6, #8]` |
| `#16` | `maxValue`（后接 `#20 minValue`） | `add x23, x6, #16  // int8 max ptr`（`_ARMV82_Unit.S`，**取地址不是取值**，两个 int32 连读） |
| `#24` | `useInt8` | 决定输出 int8 还是 fp32 |
| `#28`/`#32` | `roundValuePos`/`roundValueNeg` | |
| `#40` | `srcKernelSum` | `ldr x8,  [x6, #40]` |
| `#48` | `weightKernelSum` | `ldr x28, [x6, #48]` |
| `#56` | `fp32minmax` | `ldr x14, [x6, #56]` |
| `#64` | `blockNum` | `ldr x26, [x6, #64]` |
| `#72` | `bias` | |
| `#80` | `inputScale` | `ldr x23, [x6, #80]` |
| `#88` | `inputBias` | `ldr x27, [x6, #88]` |
| `#96` | `accumBuffer` | `ldr x10, [x6, #96]` |
| `#104` | `indices` | `ldr x8, [x6, #104]`（仅 `sme2_asm/` 八个 kernel） |

对照结构体声明 `compute/Int8FunctionsOpt.h`（各 `.S` 顶部的注释块是它的逐字镜像）。字段**语义**（`useInt8` 两条输出路径、
`fp32minmax` 可能为 `nullptr`、哪些字段没有默认值）见
[`../pack-and-abi.md`](../pack-and-abi.md) §五，这里只讲 asm 侧的两个坑：

1. **往结构体中间插字段 = 静默改所有 asm 的 ABI。** 加字段只能加**尾部**，且要 grep
   `\[x6, #` 核对每个 arm64 / arm32 / arm82 kernel，并同步各 `.S` 顶部的镜像注释块。编译器不会报任何错。
2. **`maxValue`/`minValue` 是取地址连读的**（`add x23, x6, #16`），不是两次 `ldr`。
   这两个 int32 相邻的假设被烘焙进了 asm。

### 3.5 加 unroll / hoist 常量之前：必须回答的五个 live range 问题

写在 `.S` 对应 macro 附近的注释里（模板见 §2.3）。这五问不是形式主义，每一问都对应本文已记录的真实事故：

1. **fp32 min/max 什么时候加载？会不会被 unpack 覆盖？**
   §3.2 实测：同一个 `_ARMV82_w4_Unit.S` 里不同 tile 分支分别用 `v0`/`v1` 与 `v26`/`v27` 装 min/max。
   `_16x4_Unit_FAST.S` 更极端——它**根本没加载** `#56 fp32minmax`，却拿 `v26`/`v27` 去 clamp（见 §六 第 4 条）。
2. **scale / zero point / bias 是否跨 K loop 或 tile loop 存活？** 存活就不能进 §3.2 表里
   「每次 loop 重建」的那一档寄存器。
3. **accumulator 与 unpack 临时寄存器，在每一个 tile / tail 分支上都不冲突？**
   `_ARMV86_w3_Unit.S` TILE_10 占掉 `v12`-`v31`，此时已无空闲寄存器可借。
4. **如果 hoist 常量，所有 postprocess 路径是否仍拿到正确值？** 必须逐条走
   `useInt8` 两条出口、`fp32minmax == nullptr` 分支、以及每个 tail 分支（字段语义见
   [`../pack-and-abi.md`](../pack-and-abi.md) §五）。
5. **用了哪些 callee-saved 寄存器，保存与恢复配对了吗？** §3.3：恢复必须严格逆序；
   SME2 还要求 `smstop` 在恢复之前。

各 ISA 的 ABI 硬约束见 §3.1（AArch64）、[`x86_64.md`](x86_64.md) §三、[`riscv.md`](riscv.md)。

## 四、SDOT / I8MM / SME2 的指令语义与常见错位

### 4.1 三档的 tile 参数

写 kernel 循环之前先确认这三个数（取值来源是 `compute/Int8FunctionsOpt.h` 的宏，
`compute/Int8FunctionsOpt.cpp` 的 getter 只是把宏转发出去）：

| 档 | UNIT (hP，OC) | SRC_UNIT (lP，IC) | DST_XUNIT (eP，tile) | 宏 | getter |
|---|---|---|---|---|---|
| NEON 基线（aarch64） | 4 | 16 | 4 | `GEMM_INT8_*` | `MNNGetGemmUnit` |
| NEON 基线（arm32） | 4 | 16 | **2** | 同上宏的 `#else` 分支 | 同上 |
| **SDOT** | 8 | **4** | 12 | `*_ARM82` | `MNNGetGemmUnitSdot` |
| **I8MM** | 8 | **8** | 10 | `*_ARM86` | `MNNGetGemmUnitI8mm` |
| **SME2** | 32 | 4 | 16 | `*_SME2` | `MNNGetGemmUnitSme2_HP32` |
| SME2 decode-max | **128** | — | — | `GEMM_INT8_UNIT_SME2_128` | 无（executor 里覆盖） |

**`SRC_UNIT` 就是矩阵指令在 IC 方向一次吃掉的字节数**：SDOT 4、I8MM 8。这不是巧合，
是三档 tile 差异的根源。getter 与宏的绑定关系、`DST_XUNIT` 被当 ISA 身份用等契约问题全在
[`../pack-and-abi.md`](../pack-and-abi.md) §二，本文不重复。

### 4.2 SDOT：本仓库用的是 **indexed** 形式

`sdot Vd.4s, Vn.16b, Vm.4b[i]`：

- `Vd` 的**每个 int32 lane** = 该 lane 对应的 `Vn` 的 4 个 int8 × `Vm` 中**第 `i` 组 4 字节**（广播到全部 lane）
  的点积，累加进原值。一条指令 = 16 次 MAC。
- 所以 `Vn` 提供 **4 组 OC 方向的 4 个 IC**，`Vm` 的一个 4 字节 lane 提供 **1 个 tile 的 4 个 IC**，
  `[0]`-`[3]` 四条指令扫完一个 src 寄存器里的 4 个 tile：
  `_ARMV82_Unit.S` 就是 `v0.4b[0..3]` 配 `v1.4b[0..1]`，对应 `DST_XUNIT=12` 里的前 6 个 tile。
- **C++ 模拟必须同分组**：`simulateSdot4(acc, weightReg, srcReg, laneIdx)` 里
  weight 按 `[4 OC][4 IC]` 取、src 按 `srcReg[laneIdx*4 .. laneIdx*4+3]` **广播**。
  写成"src 也按 lane 走"是最常见的错法——它在 `laneIdx==0` 时恰好正确，所以小 case 测不出来。
- **错位症状**：能跑、accumulator 数值量级正常、模型输出质量下降。
  分层比较点用 int32 accumulator（[`../correctness-gate.md`](../correctness-gate.md) §2.1 那张表）才能定位。

### 4.3 I8MM：`smmla` 是 2×8 乘 8×2

`smmla Vd.4s, Vn.16b, Vm.16b`：

- `Vn` 视为 **2×8 的 int8 矩阵**（2 行 tile × 8 个 IC），`Vm` 视为 **8×2**（8 个 IC × 2 个 OC），
  结果是 **2×2 的 int32**，按 `[row0col0, row0col1, row1col0, row1col1]` 落进 `Vd` 的 4 个 lane。
- 仓库的注释把这个 2×2 语义写清楚了，抄这个格式：
  `_ARMV86_w3_Unit.S` `// smmla v12.4s, v3.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1`。
- **`Vm` 的 B layout 错位是"能跑但质量差"的典型**：把 SDOT 的 `[4 OC][4 IC]` 布局直接喂给 `smmla`，
  形状（16 字节）完全合法，只是把 OC/IC 两个方向转置了一半。**没有任何断言会触发。**
- 三条必查：
  1. weight reorder 是否按 `[8 IC][2 OC]` 交织（不是 `[2 OC][8 IC]`）；
  2. src packer 是否按 `[2 tile][8 IC]` 交织，且 `SRC_UNIT=8` 而非 4；
  3. accumulator 到输出的 **de-interleave**：4 个 lane 是 2×2 而不是 4 个连续 tile，
     store 前必须重排（`_ARMV86_w3_Unit.S` 里 postprocess 前的 `zip`/`uzp`/`trn` 段）。
- I8MM 档的 `DST_XUNIT=10`（不是 8 或 12），是 5 个 `smmla` 行对（2 tile/对）——
  tail 分支从 TILE_10 → TILE_8 → … → TILE_1 逐级 fall through（`_ARMV86_w3_Unit.S`）。

### 4.4 SME2：streaming mode、ZA tile、`luti4`

本仓库 SME2 的**现状实证**（13/13 文件）：

| 事实 | 坐标 |
|---|---|
| `smstart` = `.inst 0xd503477f`，紧跟在 GPR/FPR 保存**之后** | `sme2_asm/MNNGemmInt8AddBiasScaleHp128_SME2_w4_Fp32.S` |
| `smstop` = `.inst 0xd503467f`，在 `ldp` 恢复**之前** | 同文件尾部 `End:` 标签下 |
| 谓词在 `smstart` 后立即建好，并**整函数存活** | `ptrue p0.b, #4` / `ptrue p5.b, vl16` / `ptrue pn8.b` / `ptrue p1.s` |
| 动态尾部用 `whilelt pn9.s/pn10.s`，不是分支树 | 同上文件 |
| ZA 双层清零：外层 `zero {za}`，内层 `zero {za0.s, za1.s}` | `16x32_SME2_w4_Fp32.S` |
| 两种累加指令**并存**：`smopa`（外积，16x32 档）与 `sdot za.s[..., VGx4]`（多向量，Hp128 档） | `16x32_..._Fp32.S` / `Hp128_..._w4_Fp16.S` |
| ZA → Z 用 `mova {z8.s-z11.s}, za.s[w8, 0, VGx4]` | `Hp128_..._w4_Fp16.S` |
| **int4 unpack 用 `luti4` 查表**，不用移位/掩码 | `16x32_SME2_w4_Fp32.S` `luti4 {z2.b-z3.b}, zt0, z1[0]` |

写 SME2 kernel 的五条：

1. **`smstart` / `smstop` 必须配对，且包住整个向量段。** 所有出口（含早退分支）都要经过 `smstop`；
   仓库的做法是所有路径 `b End`，只有一个 `smstop`。**新加早退分支时别绕过 `End`。**
2. **`smstop` 在恢复 `d8`-`d15` 之前。** streaming 进出会改变有效向量长度并使 Z/P 状态失效，
   顺序反了会把垃圾恢复回 `d8`-`d15`。
3. **向量长度是运行时量。** 不要把 `z` 寄存器宽度写死。仓库靠 `ptrue`/`whilelt` + `MUL VL`
   寻址（`ld1w {z28.s-z31.s}, pn10/z, [x20, #4, MUL VL]`）而不是常量字节偏移。
4. **`zt0`（`luti4` 的查表寄存器）也是 streaming 状态**，只在 `smstart`/`smstop` 之间有效，
   且要先装表再用。w4 走 `luti4` 是 SME2 独有的省指令手段，SDOT/I8MM 档没有对应指令，
   **不要把 SME2 的 w4 unpack 思路移植到 sdot/i8mm 档**。
5. **构建门与目录**：文件必须放 `sme2_asm/`（受 `MNN_SME2`），且**不会拿到任何 `-march`**（§2.2），
   所以每条 SME/SVE 指令都得 `.inst`。`smeCoreNumber` / `supportSME2` 的注册时序见
   [`../dispatch-and-register.md`](../dispatch-and-register.md) §四。

**本仓库未见 / 待确认**：SME2 `.S` 里没有 tile-slice 冲突检测、没有 `MSR SVCR` 的手工写法
（统一走 `smstart`/`smstop` 编码），也没有对 ZA 做 lazy save/restore 的代码。
非 streaming 与 streaming 之间的函数调用在本仓库不发生（SME2 kernel 是叶子函数），
所以 ZA 跨调用保存的问题未被触及。

### 4.5 kernel 选择判据是 ARM 专属的，不与 x86_64 同构

`compute/ConvInt8TiledExecutor.cpp` 按架构分叉，**两侧语义完全不同**：

```cpp
mGemmKernel = mRelatedFunctions.Int8GemmKernel;
#ifdef MNN_USE_SSE
    // x86_64：判据是 nbits() <= 7
#else
    // ARM：判据是 symmetricQuan()->method() == QuantizeAlgo_OVERFLOW_AWARE
    if (mResourceInt8->mWeightBits == 4) { mGemmKernel = mRelatedFunctions.Int8GemmKernel_W4; }
#endif
```

两条要点：

- 选 `Int8GemmKernelFast` 的前置条件在 ARM 上是 **`QuantizeAlgo_OVERFLOW_AWARE`**，
  不是位宽判断。你新写的 Fast 变体如果不满足 overflow-aware 的溢出假设，就不能挂在这个字段上。
- **`mWeightBits == 4 → Int8GemmKernel_W4` 这一行只在 ARM 分支里**，x86_64 永远走不到。
  所以 ARM 侧新增 `_W2` / `_W3` 变体时，要在这里补对应分派；照抄 x86_64 的结构会漏掉整条低 bit 路径。
  x86_64 侧的判据见 [`x86_64.md`](x86_64.md)。

## 五、低 bit（w2/w3/w4）的 AArch64 专属要点

### 5.1 unpack 指令预算：先数指令，再谈带宽

w2/w3 的有效带宽本来就低，**瓶颈通常不是 DRAM，是 unpack 的 issue slot**。
实测各 unpack 宏的指令构成（逐条数，不含 load）：

| 宏 | 位置 | 指令构成 | 合计 | 产出 |
|---|---|---|---|---|
| `UNPACK_W2_SDOT` | `low_memory/..._ARMV82_w2_Unit.S` | 2 `tbl` + 1 `ext` + 2 `ushl` + 1 `movi` + 2 `and` | **8** | 1 cell（8B → 8OC×4IC） |
| `UNPACK_W3_SDOT` | `low_memory/..._ARMV82_w3_Unit.S` | 4 `tbl` + 2 `ext` + 4 `ushl` + 2 `movi` + 4 `and` + 2 `orr` | **18** | 1 cell（12B+4B pad） |
| `W3_UNPACK_SERIAL` | `low_memory/..._ARMV86_w3_Unit.S` | 4 `ushl` + 3 `ushr` + 7 `and` + 7 `add` | **21** | 1 cell（16B main + 8B aux → 4 个输出寄存器） |

两条从实测里读出来的方向：

- **I8MM 的 w3 unpack 完全没有 `tbl`/`ext`**，是纯 `ushr`/`ushl`/`and`/`add` 的移位流水；
  SDOT 的 w3 则是 `tbl`/`ext` 流水。**同一个位宽在两档上是两套不同策略**，
  说明"哪种更快"取决于 tile 数和寄存器压力，不能跨档照抄。
- `tbl` 在多数 AArch64 微架构上吞吐低于简单的 `ushr`/`and`，且**独占 shuffle 端口**；
  当 unpack 与 `sdot`/`smmla` 争同一端口时，把 `tbl` 换成移位链常有净收益（I8MM w3 就是这么做的）。
  改之前先在 C++ 模拟版上对齐（分组要求见 §4.2）。

### 5.2 aux plane 的 64→128 位复制：`ld1r {.2d}`

w3 的 cell 由 "main plane（2bit）+ aux plane（1bit）" 组成，aux 只有 8 字节但要参与 128 位运算。

**正确写法（仓库实测，10 处全部一致）**：

```asm
// low_memory/MNNGemmInt8AddBiasScale_ARMV86_w3_Unit.S
    ld1  {v0.16b}, [x2], #16      // main plane
    ld1r {v1.2d},  [x2], #8       // aux plane replicated to 16B
```

`ld1r {v1.2d}, [x2], #8` 一条指令完成"读 8 字节 + 复制到高低两个 64 位半"，**并带 post-index**。
替代写法 `ld1 {v1.8b}, [x2], #8` + `mov v1.d[1], v1.d[0]` 是 2 条指令，
且第二条对同一寄存器读写、拉长依赖链。

- 全仓库 `ld1r {vN.2d}` 只出现在这一个文件的 10 处（都是 aux plane），**这就是该模式的唯一用法**；
  `ld1r {vN.4s}` 则广泛用于广播 fp32 标量（min/max、scale），见 §3.2 表。
- **别把 aux plane 的 8 字节和 padding 搞混**：SDOT 的 w3 cell 是
  "8B main + 4B aux + **4B zero padding**"（`_ARMV82_w3_Unit.S` 的 layout 注释），
  padding 存在的原因是 **cell stride 必须是 `SRC_UNIT` 的整数倍**。
  I8MM 的 w3 cell 是 "16B main + 8B aux = 24B，无 padding"（`_ARMV86_w3_Unit.S`
  注释 `advance x2 by 24`）。**两档的 cell stride 不同，指针步进不能共用常量。**
  cell stride 契约本身见 [`../pack-and-abi.md`](../pack-and-abi.md) §3.1。

### 5.3 不要靠 prefetch 或加 unroll 救 unpack-heavy kernel

实测：**`source/backend/` 全部 `.S` 里 `prfm` 只出现 1 次**——
`arm82/asm/arm64/low_memory/MNNGemmInt8AddBiasScale_ARMV86_w2_Unit_FP16.S`
（`prfm pldl1keep, [x12, #512]`，在 `W2_TILE1_LU4_STEP` 宏里，TILE_1 单 batch 路径，一次取 4 个 cell）。

- **解读**：仓库演进到 HEAD，`prfm` 只在**一个** tile=1 的低 bit 分支上被认为值得加。
  低 bit kernel 的权重流是顺序的，硬件 stride prefetcher 已经覆盖；
  加 `prfm` 通常只是多占一个 slot，在 issue-bound 的 unpack 循环里是**净负**。
- **加 unroll 同理**：unroll 摊薄的是 loop 开销，而 unpack-heavy kernel 的开销在指令数本身，
  unroll 只会把寄存器压力推高到迫使 spill。`_ARMV82_w2_Unit.S` 的注释
  `TILE_12/8/4 fall through to TILE_1 single-batch` 就是这个取舍的产物：
  **低 bit 的大 tile 分支被有意做成退到 TILE_1**，而不是各写一份大 unroll。
- **要做的是数指令**：把 §5.1 那张表对你的 kernel 填一遍，
  算 `unpack 指令数 / dot 指令数` 的比值。比值 > 1 就先降 unpack，不要碰 prefetch/unroll。

### 5.4 优化方向优先级

按这个顺序试，**不要从"扩大 packed bytes"开始**（那是改 ABI，要用户明确接受，
且会同时打破 [`../pack-and-abi.md`](../pack-and-abi.md) 里的五个同源量）：

| 优先级 | 方向 | 本仓库实证 |
|---|---|---|
| 1 | **bit-plane 方向**：换 main/aux 的划分或 OC-major/IC-major | w3 = "2bit main + 1bit aux, aux 是 OC-major"（`_ARMV82_w3_Unit.S`）；这是刻意选的，因为 aux 能一次 `ld1r` 广播 |
| 2 | **常量复用**：mask / shift 表 hoist 出 loop，或放常量池一次 `adr` | `_ARMV86_w3_Unit.S` `adr x16, .L_w3_unpack_consts_fp32` + 注释 `x16 stays valid throughout the function` |
| 3 | **换 unpack 指令族**：`tbl`/`ext` ↔ `ushr`/`ushl`/`and`/`add` | SDOT w3 用 tbl 流水，I8MM w3 用移位流水（§5.1） |
| 4 | **block64 专用路径**：为默认 block 写快路径，保留 block32 / per-channel 旧路径 | 两种粒度都要单独验（[`../correctness-gate.md`](../correctness-gate.md) §1.4） |
| 5 | 扩大 packed bytes / 改 cell stride | **最后手段**，等于改 ABI |

`movi` 重建常量 vs 占一个寄存器 hoist，是 2 与 3 之间的实际权衡：
`UNPACK_W2_SDOT` 里 `movi \v_w\().16b, #3` 是**每 cell 重建**的（因为 `v_w` 已被 `ext` 用作 scratch，
注释明写 `v_w is destroyed`）——省一个寄存器换一条指令。tile 数多、寄存器紧时这是对的选择。

## 六、fp16 与 fp32 是两套 kernel，互不能推断

AArch64 上 fp16 走 `Arm82Backend` + 第二张函数表，**kernel 是不同的 `.S` 文件、不同的 object lib、
不同的 `MNNAsmGlobal.h` 副本**（§1.1、§2.1）。

| | fp32 | fp16 |
|---|---|---|
| int8 gemm `.S` | `cpu/arm/arm64/(low_memory/)MNNGemmInt8AddBiasScale_*` | `arm82/asm/arm64/low_memory/MNNGemmInt8AddBiasScale_*_FP16` |
| SME2 int8 gemm `.S` | `cpu/arm/arm64/sme2_asm/..._Fp32.S` | `cpu/arm/arm64/sme2_asm/..._Fp16.S`（**同目录**，不在 arm82 树） |
| object lib | `MNNARM64` | `MNN_Arm82` |
| `-march` | 基线 | `armv8.2-a+fp16`（`arm82/CMakeLists.txt`） |
| accumulator | fp32（`v*.4s`） | **也是 fp32（`v*.4s`）** |
| postprocess 尾部 | 直接 `st1 {v*.4s}` | `fcvtn` / `fcvtn2` 降到 `.8h` 再 `st1 {v*.8h}`（`arm82/.../MNNGemmInt8AddBiasScale_ARMV82_Unit_FP16.S`） |
| `fp32minmax` | fp32 | **仍是 fp32**，在 `fcvtn` **之前**施加（结构体偏移 `#56` 两侧一致，见 §3.4） |
| float pack | 4 | **8** |

三条硬性要求：

1. **`fp16 正确` 推不出 `fp32 正确`，反之亦然。** 两条走不同 `.S`、不同函数表。
   op 单测必须两个精度各跑一遍（`precision` argv 位；命令见
   [`cpu/shared/build-test-and-benchmark.md`](../../shared/build-test-and-benchmark.md)）。
2. **改了 fp32 的 `.S`，同名 fp16 `.S` 不会自动跟着变。** 两棵树是**手工镜像**的——
   `_ARMV82_w2_Unit.S` 的注释 `mirrors FP16 sdot w2`、
   `_ARMV82_w3_Unit.S` 的 `Mirrors UNPACK_W3_SDOT from the FP16 sdot w3 kernel`
   就是这层人工同步关系的自述。**改一边必须同步改另一边**，没有任何机制保证一致。
3. **`fcvtn` 的舍入是额外误差源。** fp16 输出与 fp32 输出对不上一定量级的差异是预期的；
   做 bit-exact 比较只能在**同精度内**做（分层比较点见 [`../correctness-gate.md`](../correctness-gate.md) §2.1）。
   fp16 的 `Arm82Functions` 是逐字段赋值的二级表（唯一例外），加字段的坑见
   [`../dispatch-and-register.md`](../dispatch-and-register.md) §3.2。
4. **上表「`#56` 两侧一致」是接口约定，不是自动成立的事实。** `_16x4_Unit_FAST.S` 在
   `7b49c22772` 之前就是活的反例：它**从未加载 `#56`**，float 出口却拿 `v26`/`v27` 去 clamp——
   那两个寄存器里当时还是 int16 gemm 累加器。同一提交还修掉它把 `#96 accumBuffer` 当
   `#80 inputScale` 读、以及 `realCount == 3` 前导覆盖循环计数器后越界读 `srcKernelSum[2]`。
   三个错都藏了很久，因为这个 kernel **只在无 dotprod 的 armv8 上派发**，现代机器上跑不到。
   **回退档 / 基线档的 kernel 要单独按 §3.4 的偏移表核一遍**，别指望被主档的测试覆盖。

## 七、AArch64 侧改动前自查表

- [ ] 新 `.S` 放对了目录：低 bit → `low_memory/`；进 streaming mode → `sme2_asm/`；fp16 → `arm82/asm/arm64/`（SME2 int8 gemm 例外，见 §1.2）
- [ ] 文件有 `#ifdef __aarch64__` 守卫、`#include "MNNAsmGlobal.h"` 在守卫内、`.text`、`.align 5`（§2.3）
- [ ] 用了 `asm_function`，C++ 侧声明在 `extern "C"` 内（§2.1）
- [ ] **没有加 `.arch`**；`sdot`/`smmla`/SME 指令全部写成 `.inst hex // 助记符`，注释与 hex 已核对（§2.2）
- [ ] 没有用 `x18`；`x19`-`x28` 与 `d8`-`d15` 已存-恢复且**逆序**（§3.1、§3.3）
- [ ] 需要跨调用存活的 128 位值没有放在 `v8`-`v15`（只保低 64 位，§3.1）
- [ ] `[x6, #N]` 的偏移对照 `compute/Int8FunctionsOpt.h` 核过，`.S` 顶部的镜像注释块与头文件仍一致（§3.4）
- [ ] 新增结构体字段只加在**尾部**，并 grep 过所有 `\[x6, #`（§3.4）
- [ ] SDOT 的 indexed 分组 / I8MM 的 2×2 de-interleave 在 C++ 模拟版上对齐过（§4.2、§4.3）
- [ ] SME2：`smstart`/`smstop` 配对、所有出口经过 `smstop`、`smstop` 在 `ldp` 之前、没有写死向量长度（§4.4）
- [ ] 低 bit：unpack 指令数已数过，`unpack / dot` 比值合理；没有靠 `prfm` 或加 unroll 顶（§5.1、§5.3）
- [ ] 低 bit：两档的 cell stride 分别确认（SDOT 有 padding，I8MM 无，§5.2）
- [ ] fp16 与 fp32 两棵树都改了并**分别测过**（§六）
- [ ] 新 `.S` 已进 CMake 的源文件列表；函数指针只在对应 ISA 能力位为真时注册（§4.5、[`../dispatch-and-register.md`](../dispatch-and-register.md) §五）
- [ ] `MNN_CPU_TARGET` 降档后的 fallback 路径**可达且结果与改动前逐位一致**（[`../correctness-gate.md`](../correctness-gate.md) §1.6、§2.2）
- [ ] tile 契约七处同改：[`../pack-and-abi.md`](../pack-and-abi.md) §四
- [ ] 注册面九步清单：[`../dispatch-and-register.md`](../dispatch-and-register.md) §五

## 八、相关文档

| 我要找 | 去哪 |
|---|---|
| 该不该下沉到 asm（投入判据） | [`../SKILL.md`](../SKILL.md) 铁律 1 |
| tile / packer / cell stride 五个同源量、`QuanPostTreatParameters` 字段语义、`MNNGetGemmUnit` 消费者 | [`../pack-and-abi.md`](../pack-and-abi.md) |
| 三层表结构、构建门 vs 运行门、二级表构造、快照时序、新 ISA 注册清单 | [`../dispatch-and-register.md`](../dispatch-and-register.md) |
| 标量 oracle 来源、分层比较点、跨 ISA × 精度正确性矩阵、症状 → 退回哪一层 | [`../correctness-gate.md`](../correctness-gate.md) |
| ARM 侧派发路径全景、`MNN_CPU_TARGET` 降级自证、事故台账 | [`../../optimize/arch/arm.md`](../../optimize/arch/arm.md) |
| 构建选项、`run_test.out` 用法、benchmark 与性能报告格式 | [`../../shared/build-test-and-benchmark.md`](../../shared/build-test-and-benchmark.md) |
| x86_64 侧的对应实现参考 | [`x86_64.md`](x86_64.md) |
| RISC-V（RVV / 厂商矩阵扩展）实现参考 | [`riscv.md`](riscv.md) |
