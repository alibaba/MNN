# RISC-V 侧路径与派发：诊断面

> **何时读**：在 RISC-V CPU 上做性能归因，需要先确认「我到底跑在哪条路径上」的时候。
> 这是 [`optimize/`](../SKILL.md) 分支的 **L4（Dispatch / 函数表）诊断面**。
>
> **不在本文**：
> - RVV / IME2 指令语义、W4B64 布局、TCM 流水、注册面写法 → [`../../kernel/arch/riscv.md`](../../kernel/arch/riscv.md)
> - 与 ARM / x86_64 的结构差异 → [`../../SKILL.md`](../../SKILL.md)「三侧不同构对照表」（**只有那一份**）
> - 开发板交叉编译、板端正确性与性能实验纪律 → [`../../shared/riscv-remote-validation.md`](../../shared/riscv-remote-validation.md)
> - env 开关语义 → [`../../shared/env-registry.md`](../../shared/env-registry.md)
>
> **命名**：目录名是 `source/backend/cpu/riscv/`（下分 `rvv/` 与 `rvv/spacemit_ime2/`），
> CMake 选项名 `MNN_USE_RVV` / `MNN_RVV_SPACEMIT_IME2` / `MNN_RVV_MARCH` 保持字面写法，
> 平台术语写 RISC-V，向量扩展写 RVV。

## 〇、动手前先锁定现状

RISC-V 侧的路径分叉比 ARM / x86_64 多一层（构建门 × 架构门 × vendor shape 门），
**没写清坐标就开跑，测出来的数字后面无法归因**。改第一行代码前把这七行填满：

| 维度 | 必须确认 |
|---|---|
| 入口 | Execution、函数表字段、注册入口 TU、最终 kernel symbol |
| 数据 | dtype、量化位宽、block size、对称/非对称、scale/offset 类型 |
| shape | prefill / decode、M/N/K、tail、head/GQA、context 范围 |
| ISA | 标准 RVV 还是 vendor、运行时 VLEN/SEW/LMUL、实际编译参数 |
| 存储 | pack layout、cache/TCM、临时 buffer、每 token 读写字节数 |
| 并发 | 线程池、核亲和性、共享矩阵单元、barrier/dispatch 次数 |
| 验收 | op test、短生成、长 prompt、纯 RVV smoke、端到端 benchmark |

填法：从 op / Execution 一路 grep 到注册点、packer、kernel，确认**真实热路径**而不是
「看起来该走的路径」；同时记录当前分支与远端工作区状态，避免覆盖别人的在制品
（远端纪律见 [`../../shared/riscv-remote-validation.md`](../../shared/riscv-remote-validation.md) §一）。
再取一组**未改代码**的正确性与性能基线。

## 一、三条路径矩阵

| 路径 | 表怎么装 | 构建门 | 编译参数 | 运行时门 |
|---|---|---|---|---|
| **纯 C++ / 通用 CPU** | 基表 `gCoreFunction` 里未被下面两级覆盖的字段就留在标量实现 | 无 | — | — |
| **标准 RVV** | 在**基表上逐字段覆盖**（`CommonOptFunction.cpp` 的 `#if defined(__riscv) && defined(MNN_USE_RVV)` 块、`Int8FunctionsOpt.cpp` 的 `#ifdef __riscv` + `#ifdef MNN_USE_RVV` 块） | `MNN_USE_RVV`（根 `CMakeLists.txt`，默认 **OFF**）**且** `CMAKE_SYSTEM_PROCESSOR` 匹配 `riscv64` | object lib `MNNRVV`，`-march=${MNN_RVV_BASE_MARCH}`（默认 `rv64gcv`）`-mabi=lp64d` | `CoreFunctions::supportRVV`（`compute/CommonOptFunction.h`），由 `gCPUInfo.rvv` 赋值 |
| **SpacemiT IME2（vendor）** | 通过 `MNNSpacemitIme2FastPathRegistration.cpp` 这个注册入口接管 fast path；kernel 在 `rvv/spacemit_ime2/` | `MNN_RVV_SPACEMIT_IME2`（`riscv/CMakeLists.txt`，默认 **OFF**），依赖前一行已成立 | 两个 object lib：`MNNSpacemitIme2Runtime`（基线 march + `MNN_USE_SPACEMIT_IME2` 宏）与 `MNNSpacemitIme2`（**基线 march + `_xsmtvdotii`**，另加 `-fno-stack-protector`） | 见 §2.2：vendor 路径由 shape / layout 门禁在 fast-path hook 内部判定，不满足就回退 |

**`MNN_LOW_MEMORY` 会改变 vendor 侧的文件集合**：开启时 `MNNSpacemitIme2ConvInt8Executor.cpp`
才会被编进 `MNNSpacemitIme2Runtime`。低比特 conv 的 vendor 路径在 `MNN_LOW_MEMORY=OFF` 的构建里
根本不存在，不要拿这种构建的数据谈 W4 收益。

## 二、诊断需要的结构常识

### 2.1 RISC-V 既没有第二张表，也没有第二个 Backend

RVV 与 vendor 都是**在基表 `gCoreFunction` / `gCoreFunc` 上逐字段覆盖**，不像 ARM fp16（arm82）
或 x86_64 AVX2 那样另起一张表 + 一个 Backend。

推论：**RISC-V 上「结果错」几乎不可能是表的问题**（最坏只是某个字段没被覆盖、退回标量，慢但对），
和 x86_64 同类；出现数值错要去查 pack/ABI 与 vendor 门禁，不要花时间核对函数表。
三侧的完整结构差异（含二级表构造方式、`Precision_Low` 语义、A/B 手段）见
[`../../SKILL.md`](../../SKILL.md)「三侧不同构对照表」——**全树只有那一份，本文不重复。**

### 2.2 两个 fast-path 注册 TU 是**构建期互斥**，不是运行时二选一

`riscv/CMakeLists.txt` 在 `MNN_RVV_SPACEMIT_IME2=ON` 时会把
`rvv/MNNRvvFastPathRegistration.cpp` 从 `MNNRVV` 的源文件列表里 `REMOVE_ITEM`，
由 vendor target 提供同一个注册入口。

归因时的含义：**vendor ON 的构建里没有「标准 RVV fast path」这条对照路径**。
想做 IME2 vs 纯 RVV 的 A/B，只能用两个独立配置的 build 目录，不能在同一目录里翻选项。
这也是为什么性能对照必须准备两套产物，见 [`../../shared/riscv-remote-validation.md`](../../shared/riscv-remote-validation.md) §二。

### 2.3 vendor ISA 不允许渗进标准 RVV object

`riscv/CMakeLists.txt` 用 `string(REPLACE "_xsmtvdotii" "" MNN_RVV_BASE_MARCH "${MNN_RVV_MARCH}")`
把厂商扩展从基线 ISA 串里剥掉，**即使旧的 build cache 里 `MNN_RVV_MARCH` 还带着合并写法**，
并打印 `Restricting xsmtvdotii to the SpacemiT IME2 target`。

只有 `MNNSpacemitIme2` 这一个 object lib 带 `_xsmtvdotii`。
若在 `MNNRVV` 或 runtime lib 里看到厂商指令，说明有人绕过了这条隔离——那是构建问题，不是性能问题。

## 三、自证：我这次真的跑在 RVV / IME2 上吗

三个维度必须分开报，任何一个不成立，性能数字都不可用：

1. **架构对不对**：`uname -m` 应为 `riscv64`。
   `MNN_USE_RVV=ON` 但 `CMAKE_SYSTEM_PROCESSOR` 不是 `riscv64` 时，`riscv/CMakeLists.txt`
   只打印 `WARNING: RVV optimizations are only supported on riscv64 architecture` 就整段跳过——
   **构建仍然成功，只是什么都没启用**。这和 x86 上 `MNN_USE_SSE` 恒开、ARM 上 arm82 有独立 Backend
   都不同，是 RISC-V 侧最容易被忽略的静默退化。
2. **编进来了没有**：configure 阶段应有 `Enabling RVV Optimizations`；
   `MNNRVV` 目标存在；vendor 档还要有 `MNNSpacemitIme2Runtime` 与 `MNNSpacemitIme2`。
3. **运行时选不选**：`CoreFunctions::supportRVV` 来自 `gCPUInfo.rvv`。
   为 false 时上面两块 `#if` 里的赋值全部不生效，跑的是标量基表。

「编进来了」与「运行时选中」是两个独立维度，这条纪律见 [`../../SKILL.md`](../../SKILL.md)
「三条共用的前置纪律」第 1 条。

## 四、瓶颈判定：先 profile 再选方案

| 现象 | 优先检查 |
|---|---|
| prefill 慢 | pack / 动态量化遍数、M tile、权重复用、矩阵单元利用率 |
| 首次 prefill 慢、后续每换一个 shape 又慢一次 | Geometry 是否重建子 `Op`/Execution、不可变权重是否重复 reorder。机制与 `onRecompute` 修法是架构中立的，见 [`../runtime-and-scheduling.md`](../runtime-and-scheduling.md) §2.6 |
| 短 prompt 汇总速度异常低 | 是否把首次 prefill 冷启动混进了平均值；逐条记 cold/hot 时间与 token 数 |
| decode 慢 | packed weight 字节数、持续带宽、dispatch/barrier、epilogue |
| kernel 快但模型不快 | 调用次数、Attention/KV、layout conversion、线程池 |
| 增加线程反而慢 | 共享矩阵单元、内存带宽、核拓扑、同步成本 |
| TCM 无收益 | 工作集、copy/compute 是否重叠、启动成本、真实 TCM 可用性 |
| 数值只在 vendor 路径错 | pack ABI、signedness、scale/zero-point 修正、tail（去 [`../../kernel/arch/riscv.md`](../../kernel/arch/riscv.md)） |

decode 上限估算：

```text
decode tokens/s 上限 ≈ 持续有效带宽 / 每 token 必读权重与元数据字节数
```

**不要把接口峰值带宽、稀疏 TOPS 或单条指令峰值当成模型可达吞吐。**
1024-bit 向量寄存器宽度也不代表每个普通 RVV 算术操作都有 1024-bit/cycle 吞吐——
分析时区分架构 VLEN、执行管线宽度、load channel 和矩阵单元吞吐。

## 五、prefill 与 decode 分开优化

同一个 kernel 或同一个线程数不会同时最优，两侧要分别立目标、分别报数。

**prefill 优先项**：多行 M tile 提高权重复用；合并 absmax / 动态量化 / A pack / `sum(A)`；
strided-row 或连续 row-block 调度以减少细任务 dispatch；register blocking；
direct-layout epilogue 省掉中间 C buffer 和二次转换。
先确认 activation 行数足以摊薄 pack 与 barrier，小 M 保留轻量路径。

**decode 优先项**：M1/GEMV 专用 kernel；连续输出 panel 分片形成顺序权重访问；
persistent worker 减少每层 dispatch；direct output 把 scale、bias/clamp 与最终 layout 写入融合；
减少 packed-B 元数据与冗余读取；在谈计算峰值前先测持续有效内存带宽。

**不要默认用满全部核心**：增加 worker 可能只增加共享矩阵单元争抢、DRAM 竞争与 barrier 成本。

## 六、三层实现划分（归因时的坐标系）

```text
通用 CPU Execution
  -> 标准 RVV Execution / 函数表
       -> vendor Execution / kernel target
```

| 层 | 可依赖能力 | 典型职责 |
|---|---|---|
| 通用 CPU | 标量、通用线程与 Tensor layout | 参数解析、通用 buffer、fallback |
| 标准 RVV | RVV 1.0 与运行时 VLEN | 向量量化、归约、pack、通用 GEMM/Attention |
| Vendor | 专用编译 target 与运行时资源 | IME2 kernel、TCM、核拓扑、专用 layout |

**厂商指令、TCM 和 shape 门禁只进入厂商 target；标准 RVV 始终保留为可独立构建、可运行的 fallback。**
归因时先判定「这次的热点落在哪一层」，再决定改哪一层——把 vendor 的问题拿到通用层改，
会同时污染另外两个平台。实现侧的分层落地写法在 [`../../kernel/arch/riscv.md`](../../kernel/arch/riscv.md)。

**vendor 路径由构建能力隔离，不要为它加 `getenv` 调优开关**：运行时只保留 shape、layout、
资源和正确性门禁。env 机制的选择依据见 [`../../shared/env-registry.md`](../../shared/env-registry.md)。

## 七、口径类常见错误

| 错误 | 修正 |
|---|---|
| 把 60 TOPS 稀疏峰值用于 dense W4 | 使用匹配的 dense/sparse 口径 |
| 把 LPDDR 接口峰值当持续带宽 | 用板端 microbenchmark 或模型字节数反推 |
| 只验证 vendor ON 构建 | 再编译、运行纯 RVV OFF 变体 |
| 增加线程就假设更快 | 做线程数 sweep 并检查共享单元与带宽 |
| 拿本机 ARM/x86 交叉编译结果代替板端结论 | 本机编译只能发现通用接口污染，ISA / VLEN / 核拓扑 / TCM / 持续带宽必须在目标板验证 |
| 在同一 build 目录翻 `MNN_RVV_SPACEMIT_IME2` 后直接比性能 | 两个独立 build 目录，见 §2.2 |
