---
name: cpu-optimize
description: MNN CPU 后端性能归因分支（`skills/cpu/` 下，另一分支是 `cpu/kernel` kernel 开发）。按五层（Runtime 线程 / Executor 调度 / Layout 内存 / Dispatch 函数表 / Kernel ISA）定位瓶颈，含 bound 类型判定、op 级实验回路、跨框架 op 对 op 对比、跨层不一致的事后定位，以及 ARM / x86_64 / RISC-V 三侧「我到底跑在哪条 ISA 路径上」的诊断面。CPU 上算子或 LLM prefill/decode 慢、线程数或内存占用异常、出现性能回归、要与 llama.cpp 等框架逐算子对比时使用。
---

# MNN CPU 后端性能优化

> **触发条件**：优化或评审 CPU 上的 MNN 算子、低 bit GEMM/GEMV、LLM prefill/decode 性能；
> 排查 CPU 侧线程 / 调度 / 布局 / 内存 / dispatch 的性能回归。ARM、x86_64、RISC-V 均适用。

## 本分册回答什么

**「为什么慢，该改哪一层」。**

本分册是 [`skills/cpu/`](../SKILL.md) 的性能优化分支。ISA 路径的**诊断面**在本分册内
（[`arch/arm.md`](arch/arm.md) / [`arch/x86_64.md`](arch/x86_64.md) / [`arch/riscv.md`](arch/riscv.md)）；构建测试与 env 开关是两个分支
共用的**工具**，在 [`cpu/shared/`](../shared/build-test-and-benchmark.md)。

不回答「这个 kernel 怎么写」——那是 [`cpu/kernel`](../kernel/SKILL.md)。
不回答「这个 op 框架里有没有」——那是 [`add-new-op`](../../add-new-op/SKILL.md)。

| 你的问题 | 去哪 |
|---------|------|
| 慢，不知道原因 | **本分册**，从 [`diagnose-and-route.md`](diagnose-and-route.md) 开始 |
| 已定位到某个 kernel，要动 SIMD/asm | [`cpu/kernel`](../kernel/SKILL.md)（**必须先有性能数据**） |
| 结果不对（不是慢） | [`bugfix.md`](bugfix.md)，或框架级 [`general-debug`](../../general-debug/SKILL.md) |
| 这个 op MNN 里还没有 | [`add-new-op`](../../add-new-op/SKILL.md) |
| RISC-V / RVV / 厂商矩阵扩展（IME2）在哪条路径上 | [`arch/riscv.md`](arch/riscv.md) |
| 要在 RISC-V 开发板上交叉编译、跑正确性与性能 | [`cpu/shared/riscv-remote-validation.md`](../shared/riscv-remote-validation.md) |
| 要跑 CI / 加测试阶段 | [`test-ci`](../../test-ci/SKILL.md) |

## 前置纪律

三条各侧共用的纪律（先确认路径再谈性能 / 先正确再加速 / 数字必须带维度标签）在
[`cpu/SKILL.md`](../SKILL.md)「三条共用的前置纪律」，全树只有那一份，本文不复述。

本分册只补一条落地方式：**「先确认路径」在性能归因里是第一个动作，不是背景知识。**
开跑之前先用 [`arch/arm.md`](arch/arm.md) / [`arch/x86_64.md`](arch/x86_64.md) / [`arch/riscv.md`](arch/riscv.md) 的自证方法确认
这次到底跑在哪条 ISA 路径、哪张函数表上——走错路径不报错，而基于错路径测出的数字全部作废。

## 起点

**先读 [`diagnose-and-route.md`](diagnose-and-route.md)**，它做三件事：排除「路径不对 / 数字不可复现」这两个最廉价的解释；把实验回路从端到端缩到 **op 级**（按「时间占比 ÷ 工作量占比」挑最不划算的 op，为它建贴场景的用例，跨框架对比时两边都建同一个 op 的用例）；然后按 bound 类型把你路由到下面某一层。

## 分层文档

五层坐标系（每层管什么、典型改动是什么、去读哪份文档，含 L4/L5 的诊断面与实现面两个去处）
在 [`diagnose-and-route.md`](diagnose-and-route.md) §三——**那张表全树唯一一份**，本文不复述。
本分支自己的文档就是它「去哪读」一列里的 [`runtime-and-scheduling.md`](runtime-and-scheduling.md)（L1/L2）、
[`layout-and-memory.md`](layout-and-memory.md)（L3）、[`arch/`](arch/)（L4 诊断面）。

**跨层交界处（L2↔L3 的 stride、L3↔L4 的 pack 与 tile、L4↔L5 的 ABI）是最贵的坑集中地**，
表现多为「不崩、结果略差」，统一收在 [`bugfix.md`](bugfix.md)。

## 工具文档（父级共享）

| 文档 | 用途 |
|------|------|
| [`cpu/shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md) | 构建开关、`run_test.out` 使用规则、真实测试名注册表、`llm_demo` / `llm_bench`、验证矩阵模板、实验纪律、结果记录规范 |
| [`cpu/shared/env-registry.md`](../shared/env-registry.md) | 环境变量 / 编译宏 / backend flag / `constexpr` 四种机制的区分与逐项语义。**用任何开关做 A/B 前先查这里** |

## CoreFunctions 复用清单

优化的第一选择是复用而不是新写。每处复用都要验证精确语义（参数含义、layout、转置、归一化方式、
in-place 安全性、tail 行为、量化后处理），**不能只看函数名**。

> **这张表全树唯一一份**，两个分支共用：`kernel` 的「该不该写 kernel」门禁
> （[`kernel/SKILL.md`](../kernel/SKILL.md) 铁律 2）直接引用它，不再另抄一份。

| 函数 | 优先用途 | 注意点 |
|------|----------|--------|
| `gcore->MNNPackedMatMul` | 大规模 GEMM | Pack 开销要能摊薄 |
| `gcore->MNNPackedMatMulRemain` | GEMM tail | 和主 kernel layout 一致 |
| `gcore->MNNComputeMatMulForE_1` | E=1 GEMV/decode | LLM decode 优先看这里 |
| `gcore->MNNComputeMatMulForH_1` | H=1 VecMat | 确认矩阵方向 |
| `gcore->MNNScaleAndAddBias` / `MNNScaleAndAddBiasScalar` | scale+bias | 检查 in-place |
| `MNNSoftmax` | softmax | 确认 axis/layout |
| `MNNNorm` | LayerNorm/RMSNorm | 确认 mean/rms 语义 |
| `gcore->MNNNormPacked` | NC4/NC8 LayerNorm/RMSNorm | 确认 pack、batch stride、residual fusion、tail 和线程分片 |
| `MNNExp` / `MNNSiLu` | 激活 | 部分函数不支持 in-place |
| `gcore->MNNPackCUnit` / `MNNUnpackCUnit` | NC4/NC8 重排 | pack size 由 runtime 决定 |
| `gcore->MNNPackC4ForMatMul_A` / `MNNPackForMatMul_B` | MatMul pack | 和 kernel pack mode 配套 |
| `MNN_CONCURRENCY_BEGIN/END` | 多线程 | 注意 per-thread pointer 偏移 |

两条反向提醒：**小 shape 上重型 pack+matmul 可能比朴素循环慢**；**为了复用现成 kernel 把融合算子
拆成多遍访存是净亏**，优先扩展现有 `CoreFunctions` 入口以保留融合语义。

## 参考文件

| 文件 | 用途 |
|------|------|
| `source/backend/cpu/compute/CommonOptFunction.h` | `CoreFunctions` / `MatmulRelatedFunctions` 定义与签名 |
| `source/backend/cpu/CPUAttention.cpp` | MatMul/Softmax/Norm/多线程复用参考 |
| `source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp` | pack、tiling、线程拆分参考 |
| `source/backend/cpu/compute/ConvInt8TiledExecutor.cpp` | 低 bit int8 的 tile / 分片 / kernel 选择主战场 |
| `source/backend/cpu/arm/arm64/MNNPackedMatMul.S` | AArch64 asm 风格参考 |
| `test/speed/MatMulSpeed.cpp` | speed test 组织方式参考 |

## 复盘

非平凡任务结束后，如果产生了可复用的教训，走 [`retrospective`](../../retrospective/SKILL.md)：
把可复用的结论上提到对应层文档或 [`bugfix.md`](bugfix.md)，实验过程与原始数字不进本仓。
