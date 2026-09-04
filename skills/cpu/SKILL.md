---
name: cpu
description: MNN CPU 后端（ARM / x86_64 / RISC-V 三侧）的总入口，只做分流不承载技术内容。下分 `optimize/`（为什么慢、该改哪一层）与 `kernel/`（这条 kernel 怎么写对、怎么被选中）两个分支，`shared/` 放两者共用的构建测试跑分命令、env 开关注册表与 RISC-V 开发板远端验证纪律。做 CPU 侧的工作但还不确定该进哪个分支，或需要三侧结构差异对照（第二张函数表按什么分、二级表怎么构造、`Precision_Low` 语义、ISA A/B 怎么做——「三侧不同构对照表」全树唯一一份，就在本文件）时读这里。
---

# MNN CPU 后端 Skill（入口）

> **触发条件**：CPU 上的性能优化或 kernel 开发，ARM / x86_64 / RISC-V 均适用。

本文件只做分流，不承载技术内容。**先选分支，再进对应入口。**
唯一的例外是下面「三侧不同构对照表」——那张表全树只有这一份，其他文档一律引用它、不许再抄。

## 第一步：先分清你的问题属于哪一类

| 你的问题 | 去哪 |
|---|---|
| **慢**，还不知道原因，也不知道该改哪个文件 | [`optimize/SKILL.md`](optimize/SKILL.md) → 从它的 `diagnose-and-route.md` 开始 |
| 要和 llama.cpp 等别的框架比性能，或对手框架里找不到 MNN 融合算子的对应物 | [`optimize/diagnose-and-route.md`](optimize/diagnose-and-route.md) §二（op 对 op、口径对齐、融合算子拆解） |
| 已有性能数据、已定位到某个 kernel，要动 SIMD / 汇编 / pack / 派发表 | [`kernel/SKILL.md`](kernel/SKILL.md) |
| 「我到底跑在哪条 ISA 路径上」「为什么退到慢路径」 | [`optimize/arch/arm.md`](optimize/arch/arm.md) / [`optimize/arch/x86_64.md`](optimize/arch/x86_64.md) / [`optimize/arch/riscv.md`](optimize/arch/riscv.md) |
| kernel 写完了，要决定跑哪些组合才算「过」 | [`kernel/correctness-gate.md`](kernel/correctness-gate.md) |
| **结果不对**，且出现在做完 CPU 性能改动之后，稳定地错在某一档 | [`optimize/bugfix.md`](optimize/bugfix.md) |
| 结果不对，但同一输入每次跑都不一样 | [`general-debug/nondeterminism.md`](../general-debug/nondeterminism.md)（§9 未初始化内存 / §10 多线程非确定） |
| 要跑构建 / 测试 / benchmark，或查某个开关的语义 | [`shared/build-test-and-benchmark.md`](shared/build-test-and-benchmark.md) / [`shared/env-registry.md`](shared/env-registry.md) |
| 这个算子 MNN 里还没有（缺 schema / shape / geometry） | [`add-new-op`](../add-new-op/SKILL.md) |
| RISC-V / RVV / SpacemiT IME2 厂商矩阵扩展 | 诊断 [`optimize/arch/riscv.md`](optimize/arch/riscv.md)；实现 [`kernel/arch/riscv.md`](kernel/arch/riscv.md) |
| 要在 RISC-V 开发板上编译 / 跑正确性 / 跑性能 | [`shared/riscv-remote-validation.md`](shared/riscv-remote-validation.md) |
| 要跑 CI / 加测试阶段 / 真机 benchmark | [`test-ci`](../test-ci/SKILL.md) |

**最常见的走错**：性能数据还没有就直接进 `kernel/` 手写 kernel。手写 kernel 是最贵的一层，
先在 `optimize/` 侧过投入决策门——已经跑到 82% roofline 的 kernel 再写一版汇编是白干。

## 目录结构

```
skills/cpu/
├── SKILL.md                             ← 本文件，只分流
├── optimize/                            分支一：为什么慢，该改哪一层
│   ├── SKILL.md                         分支入口
│   ├── diagnose-and-route.md            bound 类型判定与路由（慢的第一站）
│   ├── runtime-and-scheduling.md        L1/L2：线程数、ThreadPool、tiling、划分
│   ├── layout-and-memory.md             L3：pack 格式、weight reorder、peak RSS
│   ├── bugfix.md                        跨层交界处的正确性坑
│   └── arch/{arm,x86_64,riscv}.md       L4 诊断面：我在哪条路径上 + 事故台账
├── kernel/                          分支二：这条 kernel 怎么写对，怎么被选中
│   ├── SKILL.md                         分支入口（含四级实现阶梯与退出条件）
│   ├── pack-and-abi.md                  tile / cell stride / 后处理 ABI 契约
│   ├── dispatch-and-register.md         函数表注册、二级表安全写法、快照时序
│   ├── correctness-gate.md              跨 ISA × 精度 × 线程 × tail 的门禁
│   └── arch/{arm,x86_64,riscv}.md       实现面：目录命名、指令编码、寄存器分区
└── shared/                              两个分支共用的工具层
    ├── build-test-and-benchmark.md      构建开关、run_test.out、llm_demo/llm_bench、跑分纪律
    ├── riscv-remote-validation.md       RISC-V 开发板：远端构建矩阵、板端正确性与性能实验
    └── env-registry.md                  env / 编译宏 / backend flag / constexpr 四种机制
```

三个一级目录各自只有一种「东西」：`optimize/` 是诊断，`kernel/` 是实现，`shared/` 是工具。
**架构维度不构成一级目录**，它是每个分支内部的 `arch/` 子层——因为「ARM 的什么事」取决于你在诊断还是在实现，
这两件事的内容完全不同（诊断面只讲路径与自证，实现面只讲怎么写对）。

两处结构约定，不要去「改齐」：

- **`optimize/arch/` 与 `kernel/arch/` 各有三份、一一对应**（arm / x86_64 / riscv）。
  同一架构在两侧的内容完全不同：诊断面只讲路径与自证，实现面只讲怎么写对。
  新增一个架构要么两侧都加，要么明确说明为什么只有一侧。
- `shared/` 下只有**工具**，没有事实层。ISA 路径事实归 `optimize/arch/`（诊断视角），
  注册与 ABI 事实归 `kernel/`。`riscv-remote-validation.md` 放在这里是因为它讲的是
  **怎么跑**（远端构建矩阵、板端实验纪律），不是 RISC-V 的技术事实。
  曾经放在顶层的 `arm.md` / `x86_64.md` 是分支拆分前的遗留，已按视角一分为二。

## 两个分支的分界线

| 分支 | 回答的问题 | 独占内容 |
|---|---|---|
| [`optimize/`](optimize/SKILL.md) | **为什么慢，该改哪一层** | bound 类型判定与路由、L1/L2 线程与调度、L3 布局与 peak RSS、跨层不一致 bugfix、两侧派发路径诊断面与事故台账 |
| [`kernel/`](kernel/SKILL.md) | **这条 kernel 怎么写对，怎么被选中** | 实现阶梯、pack/ABI 契约、派发表注册与二级表安全写法、跨 ISA × 精度正确性门禁、三份 ISA 实现参考 |

分界线是**「诊断 vs 实现」**，不是「性能 vs 正确性」：两个分支都要过正确性门禁，
但「运行时到底选了哪条路、为什么退到慢路径」属于诊断，「怎样正确注册使它能被选中」属于实现。

**正确性内容按第二条轴分：事前门禁 vs 事后定位。** 这条轴与「诊断 vs 实现」正交，
所以正确性文档在两个分支下各有一份，那不是重复：

| 你在做什么 | 去哪 | 它的产出 |
|---|---|---|
| kernel 写完了，要决定跑哪些组合才算「过」 | [`kernel/correctness-gate.md`](kernel/correctness-gate.md) | 必测矩阵（六条会切换代码路径的轴）+ 两条判定标准 |
| 已经错了，要从「哪一档错」倒推到哪一层 | [`optimize/bugfix.md`](optimize/bugfix.md) | 六类跨层不一致的机制、真实提交证据、预检清单 |

两份的症状表也按这条轴分工：**交付前**的静默失败症状在 `correctness-gate.md` §三，
**事后**的跨层症状路由在 `bugfix.md` §一。同一症状允许在两处各出现一行，
但**必测取值只写在门禁里，机制与预检只写在 bugfix 里**——不要在另一处复述。

同一个事实经常两侧都要提，规则是：**诊断面只写「怎么看出来」，实现面只写「怎么写对」。**
例如 `Int8GemmKernelFast`——「它为什么被选中」的判据写在 `kernel/arch/`，
「我这次到底选中了谁」的自证方法写在 `optimize/arch/`。

## 五层模型（两个分支共用的坐标系）

L1 Runtime 线程 / L2 Executor 调度 / L3 Layout 内存 / L4 Dispatch 函数表 / L5 Kernel ISA，
每层管什么、典型改动是什么、去哪份文档（含 L4/L5 的诊断面与实现面两个去处），见
[`optimize/diagnose-and-route.md`](optimize/diagnose-and-route.md) §三——**那张表全树唯一一份**，本文不复述。
跨层交界处（L2↔L3 的 stride、L3↔L4 的 pack 与 tile、L4↔L5 的 ABI）是最贵的坑集中地，
症状多为「不崩、不报错、结果只是略差」，统一收在 [`optimize/bugfix.md`](optimize/bugfix.md)。

## 三侧不同构对照表（全树唯一一份）

**不要把一侧的心智模型套到另一侧。** 三侧不是同一个结构换了指令名，是五处根本不同：

| 维度 | ARM | x86_64 | RISC-V |
|---|---|---|---|
| 第二张函数表按什么分 | **精度**：fp16 是另一张表 + 另一个 Backend（`Arm82Functions` / `Arm82Backend`），fp32 留在基表 | **ISA**：SSE 直接打进基表，AVX2/AVX512 才是另一张表 + 另一个 Backend（`AVX2Functions` / `AVX2Backend`） | **没有第二张表，也没有第二个 Backend**：RVV 与 vendor 都在基表上逐字段覆盖 |
| 二级表怎么构造 | arm82 是**逐字段赋值**（约 120 条），漏字段 → `nullptr` 或**不确定值** → 崩溃/乱码/随构建抖动 | `new` 后**整体拷贝**基表再打补丁，漏字段 → 继承 SSE 实现 → **慢但对** | 就地覆盖基表字段，漏字段 → 保留标量实现 → **慢但对** |
| `Precision_Low` 语义 | 真 fp16：`bytes=2`、`pack=8`，是一条独立路径 | **不是 fp16**，`bytes` 仍为 4；而且被 `AVX2Backend` 构造函数写死，用户请求的精度无效 → **precision 轴在 x86_64 上是死的** | **没有 RVV fp16 路径**，precision 轴不切换函数表 |
| ISA 档之间怎么做 A/B | `MNN_CPU_TARGET` 降档（需 `-DMNN_PIPELINE_PROFILE=ON`），clamp 0..3 | 同左，clamp 0..4 | **只能用两个独立 build 目录**：`MNN_RVV_SPACEMIT_IME2` ON/OFF 两个注册 TU 定义同名符号，是**构建期互斥**，没有运行时开关 |
| `MNN_CPU_USE_DEFAULT_BACKEND` | 不影响 fp16（fp16 分支在它**之前**） | **静默绕过整条 AVX2/AVX512**，pack 从 8/16 掉回 4 | 不改路径（它后面没有第二个 Backend），只是跳过多线程初始化 |

推论，三侧都常被踩：

- **「结果错」在 ARM 上要先怀疑表（漏字段）；在 x86_64 与 RISC-V 上几乎不可能是表的问题**
  （最坏也只是退回慢实现），去查 pack/ABI 与 kernel 门禁。
- **A/B 精度扫描在 x86_64 上不是两条路**——两次跑的是同一条。x86_64 上做 ISA A/B 只能靠
  `MNN_CPU_TARGET` 降档，而它**默认构建下是彻底空操作**（`getenv` 与能力位屏蔽整段都被
  `#ifdef MNN_PIPELINE_PROFILE` 包住），必须 `-DMNN_PIPELINE_PROFILE=ON` 重建。
- **RISC-V 上「同一个二进制里翻开关做 A/B」这条路不存在**，别去找 env；
  也别在同一个 build 目录里改 `MNN_RVV_SPACEMIT_IME2` 后增量构建就比性能。

坐标与自证方法分别在 [`optimize/arch/arm.md`](optimize/arch/arm.md) §二/§三、
[`optimize/arch/x86_64.md`](optimize/arch/x86_64.md) §二/§三 与
[`optimize/arch/riscv.md`](optimize/arch/riscv.md) §二/§三；注册面的安全写法在
[`kernel/dispatch-and-register.md`](kernel/dispatch-and-register.md) §三。

## 命名约定

| 场合 | 写什么 |
|---|---|
| 平台 / ISA 术语（正文、结论、commit message） | **x86_64**、**AArch64**、**RISC-V**、**RVV** |
| 目录名、宏名、CMake 选项名（引用时保持字面） | `source/backend/cpu/x86_x64/`、`cpu/arm/arm64/`、`cpu/riscv/rvv/`、`MNN_USE_SSE`、`MNN_AVX2`、`MNN_AVX512`、`MNN_AVX512_VNNI`、`MNN_X86_USE_ASM`、`MNN_USE_ARMV82`、`MNN_SME2`、`MNN_USE_RVV`、`MNN_RVV_SPACEMIT_IME2`、`MNN_RVV_MARCH` |

`x86_x64`、`arm64`、`riscv` 是仓库里的既有目录名，**不要"修正"它们**；反过来也不要在正文里写 `x86_x64` 或 `riscv` 当术语。

## 三条共用的前置纪律

1. **先确认路径，再谈性能。** CPU 后端有多张函数表、多条 ISA 路径、多重构建门，走错路径不报错。
   「编进来了没有」与「运行时选不选」是两个独立维度，任何性能结论必须同时报这两组。
2. **先正确，再加速。** 任何改动都要有可复现的正确性门禁；LLM 低 bit kernel 还要做模型级 sanity，
   不能只靠 op 单测（op 单测的形状分布系统性漏掉 decode 路径）。
3. **数字必须带维度标签。** 收益不随线程数 / shape / 精度 / ISA 外推，甚至会反号。
   没有标签的性能数字不可复现，等于没有。

## 复盘

非平凡任务结束后，如果产生了可复用的教训，走 [`retrospective`](../retrospective/SKILL.md)：
结论上提到对应的层文档或 [`optimize/bugfix.md`](optimize/bugfix.md)，实验过程与原始数字不进本仓。
