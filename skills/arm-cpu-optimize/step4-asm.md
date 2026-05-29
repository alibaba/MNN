# 步骤 4：ARM 汇编优化

> **目标**：把已经用 C++ 验证过的热点 kernel，迁移为 NEON/SDOT/I8MM/SME2 汇编实现。
>
> **前置条件**：步骤 0-3 已完成，已经知道热点路径、输入 layout、pack mode、正确性 oracle 和性能 baseline。
>
> **原则**：不要直接写汇编。先标量、再 C++ SIMD 模拟、再寄存器计划、最后写 `.S`。

---

## 4.0 进入汇编前的门禁

只有同时满足下面条件，才进入 asm：

- 已有 CoreFunctions、Vec4 或 C++ 优化无法覆盖该热点。
- 已有可运行的 C++ 标量 oracle，并能和当前正确实现对齐。
- 已有目标 shape/ISA 的 speed baseline。
- 已明确 pack layout、cell stride、metadata 格式、tail 规则、postprocess 规则。
- 已知道本次要优化的是 E=1 decode、E>1 prefill、block32、block64、per-channel 还是 tail。

如果这些条件不满足，先回步骤 0-3。

---

## 4.1 四阶段实现流程

### 阶段 A：C++ 标量 oracle

写一个最直接、最容易读懂的参考实现：

- 权重读取必须按真实 pack layout 和真实 cell stride，不要用理想化公式代替。
- w2/w3 要显式解 bit，不要隐藏在复杂宏里。
- metadata 读取、scale、zero point、bias、min/max、add-dst 要拆开，方便定位差异。
- 对低 bit GEMV，优先用小 case 对齐：单 block、单 OC group、单 K tile、block64。

对比建议：

| 对比点 | 目的 |
|--------|------|
| unpack 后的 int 值 | 确认 bit layout |
| dot accumulator | 确认 `sdot`/`smmla` 数学分组 |
| dequant 后 FP32 | 确认 scale/zp |
| postprocess 后 dst | 确认 bias/minmax/add-dst |

### 阶段 B：C++ SIMD/寄存器模拟

在 C++ 中模拟目标指令和寄存器，而不是直接写 `.S`：

- 用 `std::array<int8_t, 16>`、`std::array<int32_t, 4>` 或局部数组模拟 NEON lanes。
- 写小 helper 模拟 `sdot`、`smmla`、`smlal`、`zip/uzp/ext/tbl` 等关键行为。
- 用变量名模拟 asm 寄存器，例如 `vAcc0`、`vW0`、`vAux`、`vMin`、`vMax`。
- 每次改变 unroll、pack 顺序或 unpack 方法，都先让模拟版通过 oracle。

低 bit 示例要点：

```cpp
// 伪代码：表达意图，不要求直接复制
std::array<int8_t, 16> vW0 = unpackW2Plane0(bytes);
std::array<int8_t, 16> vW1 = unpackW2Plane1(bytes);
std::array<int32_t, 4> vAcc = sdot4(vAcc, vInput, combine(vW0, vW1));
```

这一步的价值是把“数学/pack 错”和“汇编寄存器/ABI 错”分开。

### 阶段 C：寄存器生命周期表

写 asm 前必须先列寄存器表。建议直接放在 `.S` 对应 macro 附近的注释中。

| 寄存器 | 用途 | live 范围 | 可复用条件 | 风险 |
|--------|------|-----------|------------|------|
| `x0-x7` | ABI 参数 | 函数入口到重映射 | 可复制到 temp | 参数被覆盖后难恢复 |
| `x8-x18` | 临时指针/loop | 当前 loop | loop 结束后 | tail 分支跳转 |
| `x19-x28` | 长生命周期指针 | 整个函数 | 保存/恢复后 | callee-saved |
| `v0-v7` | input/unpack tmp | 当前 K step | dot 后 | 被 postprocess 误用 |
| `v8-v15` | 长生命周期值 | 整个函数 | 保存低 64 位 | callee-saved |
| `v16-v25` | accumulator | compute 到 store | store 后 | unroll 增大压力 |
| `v26-v31` | scale/min/max/constants/tmp | postprocess 前后 | 明确分支后 | 最容易被 clobber |

必须显式回答：

- min/max 是什么时候加载的？是否可能被 unpack 覆盖？
- scale/zp/bias 是否跨 K loop 或 tile loop 存活？
- accumulator 和 unpack tmp 是否在所有 tile/tail 分支都不冲突？
- 如果 hoist 常量，所有 postprocess 路径是否仍然能拿到正确值？
- 用了哪些 callee-saved 寄存器？是否保存和恢复？

### 阶段 D：最小 asm 实现

- 一次只迁移一个 tile 或一个 ISA 路径。
- 先保持和 C++ SIMD 模拟完全同构，再做指令级精简。
- 优先让 block64/default case 正确，保留 block32/per-channel/fallback 的旧安全路径。
- 每加一个 unroll 或新分支，立即跑小测试和目标 op test。

---

## 4.2 AArch64 ABI 和文件规范

### 文件位置

```text
source/backend/cpu/arm/arm64/MNNXxxKernel.S
source/backend/cpu/arm/arm64/MNNXxxKernelFP16.S
source/backend/cpu/arm/arm64/MNNXxxKernel_int8.S
source/backend/cpu/arm/arm64/sme2_asm/MNNXxxKernel_SME2.S
```

### 基本规则

- 使用现有 `MNNAsmGlobal.h` 和 `asm_function` 风格。
- `x0-x7` 是参数，`x8-x18` caller-saved，`x19-x28` callee-saved。
- `v0-v7` caller-saved，`v8-v15` 的低 64 位 callee-saved，`v16-v31` caller-saved。
- 需要 `.arch armv8.2-a+dotprod`、`.arch armv8.6-a+i8mm` 或 SME2 特性时，跟随仓库已有 asm 文件写法。
- 不要为了省几条指令破坏可读性。低 bit kernel 的 layout 注释和寄存器注释比普通 kernel 更重要。

### 最小模板

```asm
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

// void MNNXxxKernel(...)
asm_function MNNXxxKernel
    // Register plan:
    // x0: dst
    // x1: src / packed weight
    // x2: params
    // v16-v19: accumulators
    // v30-v31: fp32 min/max, only live in postprocess

L_loop:
    // load
    // unpack / compute
    // postprocess
    // store
    ret

#endif
```

---

## 4.3 SDOT/I8MM/低 bit 特别检查

### w2/w3 pack 和 stride

- OC 分线程的 `weightPtr` 偏移必须使用真实 packed cell 字节数。
- block32、block64、per-channel 的 metadata 步进要分别确认。
- 如果每个 cell 有 padding，kernel 和 packer 必须都按 padded stride 前进。
- 验证必须覆盖 `tId>0` 的 OC chunk。

### unpack 指令预算

w2/w3 的有效带宽低，常见不是 DRAM 慢，而是 unpack 太重：

- 检查是否有连续 `tbl/ext/ushr/shl/and` 导致 issue bound。
- aux plane 如果需要 64-bit 复制到 128-bit，优先考虑 `ld1r {.2d}`，避免 `ld1 {.8b}` + `mov d[1]`。
- 不要只靠 prefetch 或加 unroll 解决 unpack-heavy kernel。
- 如果不允许扩大字节，优先优化 bit-plane 方向、常量复用、block64 专用路径。

### accumulator 和后处理

- `sdot` 每 4 byte 形成一个 int32 lane，C++ 模拟必须和这个分组一致。
- `smmla` 的 2x8 by 8x2 分组容易因为 B layout 错位而“看起来能跑但质量差”。
- min/max、bias、scale、zp 最好在 postprocess 前短生命周期加载；如果提前加载，必须确认 unpack 不会 clobber。
- FP32 后处理和 FP16 后处理要分别验证，不能用 FP16 正确推断 FP32 正确。

---

## 4.4 dispatch 和注册

在 C++ 中声明并注册 asm symbol 时，同时检查 pack mode：

```cpp
extern "C" {
void MNNXxxKernel(...);
}
```

检查清单：

- arm init 中的函数指针是否只在对应 ISA 支持时注册。
- `supportI8mm`、`supportSDot`、`supportSME2` 的优先级是否符合预期。
- SME2 如果改变 `UNIT/SRC_UNIT/DST_XUNIT`，packer 和 kernel 必须一起切换。
- mixed/online reorder 不要复用错误 ISA 的低 bit kernel 指针。

---

## 4.5 测试标准

### 编译

```bash
cmake --build build -j 8
```

### op 正确性

```bash
cd build
./run_test.out op/lowMemory/blockConv 0 1 4
./run_test.out op/lowMemory/HybridConv 0 1 4
./run_test.out op/lowMemory/blockConv 0 1 4  # 在 SDOT 目标设备/构建配置上复跑
./run_test.out op/lowMemory/HybridConv 0 1 4  # 在 SDOT 目标设备/构建配置上复跑
```

根据算子替换为更精确的 op test；低 bit LLM kernel 至少保留上面两类。

### 模型级 sanity

```bash
./llm_demo /path/to/w3/config.json prompt.txt 64 1
./llm_demo /path/to/w3/config.json prompt.txt 64 1  # 在 SDOT 目标设备/构建配置上复跑
./llm_demo /path/to/w2/config.json prompt.txt 64 1
./llm_demo /path/to/w2/config.json prompt.txt 64 1  # 在 SDOT 目标设备/构建配置上复跑
```

如果输出重复或异常：

- 先跑 greedy/no-thinking，固定采样变量。
- 比较 FP16 和 FP32。
- 分别比较 I8MM、SDOT、SME2 等目标路径。
- 如果只有 mixed sampling 复读，而 no-thinking/greedy 正常，不要直接判定 kernel 错。

### 性能

```bash
./run_test.out speed/GemvBW 0 2
```

报告至少包含：

- before/after `us/iter`
- weight bytes 或 bytes/elem
- effective GB/s 和 `%peak`
- GFLOPS/AI
- ISA、线程数、block size、precision

---

## 4.6 失败回退

| 现象 | 处理 |
|------|------|
| C++ SIMD 模拟和标量 oracle 不一致 | 先修 pack/unpack 数学，不写 asm |
| asm op test 过但 LLM 乱码 | 查 E=1、block64、fp32 min/max、multi-block 和 postprocess |
| i8mm 正常但 sdot 错，或反过来 | 分开检查各自 packer、kernel 注册和 runtime flag |
| 性能低于预期 | 看 unpack 指令数、寄存器压力、load/store、postprocess，不先扩大 packed bytes |
| 新 unroll 导致质量变差 | 回退到上一个正确 unroll，重新做 live range 表 |
