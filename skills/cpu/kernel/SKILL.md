---
name: cpu-kernel
description: MNN CPU 后端 kernel 开发分支（`skills/cpu/` 下，另一分支是 `cpu/optimize` 性能归因）。覆盖标量 oracle → C++ SIMD → intrinsic → 汇编的四级实现阶梯、pack/kernel ABI 契约（tile、cell stride、后处理参数）、CoreFunctions 派发表注册与二级表安全构造、跨 ISA × 精度的正确性门禁，以及 AArch64 / x86_64 / RISC-V 三份实现参考。为 CPU 后端新写或移植 kernel（NEON / SSE / AVX / RVV intrinsic 或 .S 汇编）、新增一层 ISA、改 pack mode 或 tile 参数、把 kernel 挂进函数表时使用。已定位到 kernel 才进本分支。
---

# MNN CPU Kernel 开发 Skill

> **触发条件**：要为 CPU 后端**写一个 kernel**（intrinsic 或汇编）、新增一层 ISA 支持、
> 改 pack mode / tile 参数、或把新函数挂进 `CoreFunctions` 派发表。

## 这个 skill 与相邻 skill 的分界

三份 skill 回答三个不同的问题，混用会导致在错的地方找答案：

| skill | 回答的问题 | 独占内容 |
|---|---|---|
| [`add-new-op`](../../add-new-op/SKILL.md) | **这个算子在框架里存在吗** | `MNN.fbs` schema、`source/shape/Shape*.cpp`、`source/geometry/Geometry*.cpp` 几何拆解、`test/op/*Test.cpp`、`register.py` |
| **`cpu/kernel`（本分册）** | **这条 kernel 怎么写对、怎么被选中** | 实现阶梯、pack/ABI 契约、tile 契约、派发表注册与二级表安全写法、跨 ISA × 精度正确性门禁 |
| [`cpu/optimize`](../optimize/SKILL.md) | **为什么慢、该改哪一层** | 分层诊断、线程/调度、布局/内存、benchmark 纪律、env 开关、跨层不一致的事后定位 |

两条最容易走错的边界：

- **算子还不存在** → 先去 `add-new-op` 把 schema / shape / geometry / op test 建起来，
  再回本 skill 写 kernel。反过来先写 kernel 会发现无处注册。
- **"运行时到底选中了哪条路径、为什么退到慢路径、能力位是不是 0"** → 这是**诊断**，
  在诊断面文档 [`cpu/optimize/arch/arm.md`](../optimize/arch/arm.md) §三 / [`cpu/optimize/arch/x86_64.md`](../optimize/arch/x86_64.md) §三
  / [`cpu/optimize/arch/riscv.md`](../optimize/arch/riscv.md) §三。
  本 skill 只管**"怎样正确注册，使它能被选中"**。

**没有性能数据就不要进本 skill。** 手写 kernel 是最贵的一层。先用 [`cpu/optimize`](../optimize/SKILL.md) 的
投入决策门量化"改这一层的天花板在哪"——已经跑到 82% roofline 的 kernel 再写一版汇编是白干。

## 五条 ISA 无关铁律

三条各侧共用的纪律（先确认路径再谈性能 / 先正确再加速 / 数字必须带维度标签）在
[`cpu/SKILL.md`](../SKILL.md)「三条共用的前置纪律」，本文不复述。下面五条是 kernel 开发独有的：

1. **「先正确再加速」在本分册的落地形式是四级阶梯，每级有明确退出条件，不许跳级。**

   | 级 | 做什么 | 退出条件（满足才允许进下一级） |
   |---|---|---|
   | A | 标量参考实现（oracle） | 在四个分层比较点上对齐——[`correctness-gate.md`](correctness-gate.md) §2.1。int8 多数场景不用自己写，仓库自带标量版 |
   | B | C++ SIMD / 寄存器模拟 | 与 A 在同样四个点上一致。**分组必须与目标指令同构**（`sdot` 的 lane 广播语义见 [`arch/arm.md`](arch/arm.md) §4.2）。简单 elementwise 可豁免此级 |
   | C | intrinsic 实现 | 目标 ISA 档与 oracle 对齐 + [`correctness-gate.md`](correctness-gate.md) 最小矩阵通过。**多数 kernel 应该停在这里**，x86_64 侧更是以 intrinsic 为主力形态 |
   | D | 最小汇编 | [`correctness-gate.md`](correctness-gate.md) 完整矩阵（ISA × 精度 × 线程档）+ [`dispatch-and-register.md`](dispatch-and-register.md) §五 注册面逐条核过 |

   **进 D 的判据只有一条**：intrinsic 已实测不够（编译器寄存器分配不理想、需精确控制 unpack
   指令预算、需手写软流水），而不是"汇编应该更快"。进 D 之前先回答
   [`arch/arm.md`](arch/arm.md) §3.5 的五个 live range 问题；一次只迁一个 tile 或一条 ISA 路径，
   保留旧的安全路径。性能已达标就停在 C。
2. **优先复用已有 `CoreFunctions`。** 普通算子先拆成现有入口的组合，只有覆盖不了热点才新增
   Vec/intrinsic/asm。入口清单见 [`cpu/optimize/SKILL.md`](../optimize/SKILL.md)「CoreFunctions 复用清单」
   （全树唯一一份）。复用前必须**逐项核对语义**（参数含义、layout、转置、归一化方式、
   in-place 安全性、tail 行为、量化后处理），不能只看函数名。
3. **Executor 只编排，ISA kernel 必须下沉。** `CPUXxx.cpp` 负责参数解析、buffer、线程划分和 fallback，
   **不直接写 NEON/SSE/AVX intrinsic 或汇编**。SIMD 实现放 `source/backend/cpu/compute/` 或对应架构目录，
   经 `CoreFunctions` 入口分发。扩签名时同步检查**所有**架构实现和调用点。
4. **pack mode、tile 参数与 kernel 指针必须同时改。** packer、cell stride、`MNNGetGemmUnit`、
   weight reorder、mixed/online reorder 选择、kernel 注册是一个原子集合。改一个不改其余，
   形状仍然"合法"，症状是**能跑、不崩、单测过、只是模型质量变差**——本仓库最难发现的一类 bug。
   必须同源的五个量与要落笔的七处，见 [`pack-and-abi.md`](pack-and-abi.md) §一、§四。
5. **寄存器生命周期表先于 unroll。** 加 unroll、hoist 常量、复用临时寄存器之前先写 live range 表。
   min/max、scale、bias、zero point、accumulator、unpack 常量都不能被 postprocess 之前的临时逻辑覆盖。

## 任务 → 文档索引

| 我要做的事 | 先读 |
|---|---|
| 从零实现一个热点 kernel | 铁律 1 的阶梯表定顺序，再按目标架构读 [`arch/`](arch/) 对应那份 |
| 决定"该不该进汇编" | 铁律 1（进 D 的判据） |
| 改 pack layout / tile / cell stride / weight reorder | [`pack-and-abi.md`](pack-and-abi.md) |
| 低 bit（w2/w3/w4）权重读取与 metadata 步进 | [`pack-and-abi.md`](pack-and-abi.md) §三 |
| 把新函数挂进函数表 / 新增一层 ISA | [`dispatch-and-register.md`](dispatch-and-register.md) |
| 给 `CoreFunctions` 加字段或改签名 | [`dispatch-and-register.md`](dispatch-and-register.md) §三（二级表构造） |
| 决定要跑哪些正确性组合 | [`correctness-gate.md`](correctness-gate.md) |
| 写 AArch64 intrinsic / `.S` | [`arch/arm.md`](arch/arm.md) |
| 写 x86_64 intrinsic / `.S` | [`arch/x86_64.md`](arch/x86_64.md) |
| 写 RVV / 厂商矩阵扩展 kernel | [`arch/riscv.md`](arch/riscv.md) |
| kernel 写对了但慢 | 出本分册，去 [`cpu/optimize`](../optimize/SKILL.md) |
| kernel 写错了、要定位 | [`cpu/optimize/bugfix.md`](../optimize/bugfix.md) + [`general-debug`](../../general-debug/SKILL.md) |

## ISA 选择表：先确认目标是哪一档

同一份 C++ 代码在不同档位下 tile、pack、精度语义甚至 kernel 前置条件都不同。
动手前先写清楚这次覆盖哪几格，**不要用一格代表全部**：

| 平台 | 轴 | 档位 | 详见 |
|---|---|---|---|
| ARM | 架构级别 | aarch32 / aarch64 低于 v8.2 / v8.2 `sdot` / v8.6 `+smmla` / v9.2 `+SME2`（**累积**） | [`cpu/optimize/arch/arm.md`](../optimize/arch/arm.md) §1.1、§2.2 |
| ARM | 精度 | fp32 / fp16（arm82，第二张表）/ bf16 | 同上 §1.2、§2.1 |
| x86_64 | ISA | SSE / AVX2 / AVX+FMA / AVX512 No-VNNI / AVX512 VNNI | [`cpu/optimize/arch/x86_64.md`](../optimize/arch/x86_64.md) §一、§2.1 |
| RISC-V | ISA | 标量 / 标准 RVV / 厂商矩阵扩展（IME2 等，**构建期互斥**） | [`cpu/optimize/arch/riscv.md`](../optimize/arch/riscv.md) §一、§二；实现面 [`arch/riscv.md`](arch/riscv.md) |

**ARM 侧两条轴是相乘关系**，要覆盖的是格子不是档位：架构级别决定 int8 tile，精度决定
`bytes` / `pack` / 走哪张表，fp16 的 int8 档是从当前架构级别**继承**来的（不是第五档 ISA）。
x86_64 侧只有一条 ISA 轴，原因见下。

**这些档位各由哪张表承载、三侧结构差在哪里**（第二张表按什么分、二级表怎么构造、
`Precision_Low` 语义、ISA A/B 怎么做、`MNN_CPU_USE_DEFAULT_BACKEND` 行为）见
[`cpu/SKILL.md`](../SKILL.md)「三侧不同构对照表」——全树只有那一份，本文不复述，也不要把一侧的心智模型套到另一侧。

## 相关 skills

- [`add-new-op`](../../add-new-op/SKILL.md)：算子在框架里的注册面（schema / shape / geometry / op test）。
- [`cpu/optimize`](../optimize/SKILL.md)：性能诊断、线程与调度、布局与内存、跨层不一致的事后定位。
- [`cpu/shared/riscv-remote-validation.md`](../shared/riscv-remote-validation.md)：RISC-V 开发板交叉编译、板端正确性与性能实验的专属纪律。
- [`general-debug`](../../general-debug/SKILL.md)（skill 名 `bugfix`）：按症状分流的排查入口，下分七册（内存别名、导出/量化、fp16 值域、GPU 越界、kernel 隐式假设、陈旧缓存、逐 run 非确定）。
- [`test-ci`](../../test-ci/SKILL.md)：跑回归 / CI / 真机 benchmark。
