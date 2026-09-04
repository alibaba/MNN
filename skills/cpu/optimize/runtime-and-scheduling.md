# L1 Runtime / 线程 与 L2 Executor / 调度

> **何时读**：[`diagnose-and-route.md`](diagnose-and-route.md) 把问题定位到 L1（线程、等待、大小核、barrier）
> 或 L2（tiling、分片、resize 频率、launch 开销）之后。
>
> **不在本文**：
> 具体命令与测试名 → [`shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md)；
> 开关语义 → [`shared/env-registry.md`](../shared/env-registry.md)；
> ISA 路径与函数表 → [`arch/arm.md`](arch/arm.md) / [`arch/x86_64.md`](arch/x86_64.md)；
> kernel 内部 → [`cpu/kernel/SKILL.md`](../kernel/SKILL.md)；
> 布局与内存 → [`layout-and-memory.md`](layout-and-memory.md)；
> 正确性 bug → [`bugfix.md`](bugfix.md)。

---

## 一、L1：先搞清楚你在哪套线程实现上

### 1.1 `MNN_CONCURRENCY_*` 有四套互不相同的后端

`source/core/Concurrency.h` 用宏在**构建期**选定四条路径之一。这不是实现细节——四条路径的线程模型、
调度语义、可调项完全不同，所有 ThreadPool 相关的优化只在其中一条上有意义。

| 分支 | 条件 | 实际行为 | ThreadPool 调优是否适用 |
|------|------|---------|----------------------|
| 串行 | `MNN_FORBIT_MULTI_THREADS`（`Concurrency.h`） | 退化成普通 `for` 循环 | 不适用 |
| **MNN 自有 ThreadPool** | `MNN_USE_THREAD_POOL` | `CPUBackend::enqueue` → `ThreadPool` | **适用，且这是默认** |
| Apple GCD | `__APPLE__` 且未开 ThreadPool | `dispatch_apply` 到 global queue | 不适用，线程由系统管 |
| OpenMP | MSVC/ 其他 | `#pragma omp parallel for` | 不适用，调 `OMP_*` |

默认值：`CMakeLists.txt` 里 `MNN_USE_THREAD_POOL` 默认 **ON**，且同文件后面会强制把
`MNN_OPENMP` 置 OFF（本身默认也是 OFF）。所以**常规构建走的是 MNN 自有 ThreadPool**。

两个容易踩的细节：

- CMake 选项名是 `MNN_FORBID_MULTI_THREAD`，而它 `add_definitions` 出来的宏是
  `MNN_FORBIT_MULTI_THREADS`（`CMakeLists.txt`）——**拼写不一致**，grep 找不到时换另一个拼写再试。
- `MNN_CONCURRENCY_BEGIN_CONDITION`（`Concurrency.h`）**只在 OpenMP 分支里有定义**，
  ThreadPool / GCD / 串行分支都没有。全仓库目前无人使用它——想用之前先补齐四个分支的定义，
  否则默认构建直接编不过。

### 1.2 线程数是四段决策，不是一个数字

| 段 | 位置 | 决定什么 |
|---|------|---------|
| 用户配置 | `ScheduleConfig::numberThread` / `Interpreter` hint | 期望的并发度 |
| Runtime | `CPURuntime::mThreadNumber`，亲和性 `mCpuIds`/`mCpuMask` | 池子实际有多少 worker、绑在哪些核 |
| Backend | `CPUBackend::threadNumber()`（`CPUBackend.hpp`） | 各 executor 默认拿到的并发度 |
| Executor | `computeThreadNumber(workItems)`（`CPUBackend.cpp`） | **按 workload 形状下调** |

**`computeThreadNumber` 是 opt-in 的**，全仓库只有 5 个调用点：
`CPUAttention.cpp`、`ConvInt8TiledExecutor.cpp`（两处）、`CPURoPE.cpp`、
`CPULayerNorm.cpp`。其余 executor 用的都是裸 `threadNumber()`。
新写 executor 时**不会自动获得 P 核上限保护**，得自己调。

它的逻辑很短，但两个前提都容易漏：

```cpp
// CPUBackend.cpp
int CPUBackend::computeThreadNumber(int workItems) const {
    int perfCores = mCoreFunctions->perfCoreNumber;
    if (workItems > 1 && perfCores > 0 && mThreadNumber > perfCores) {
        return perfCores;
    }
    return mThreadNumber;
}
```

- `workItems > 1`：**`workItems == 1`（decode 形状）不降档**。decode 是带宽 bound，
  多一个 E 核仍能多搬一点数据；prefill 是计算 bound，E 核只会在 barrier 上拖尾。
- `perfCores > 0`：`perfCoreNumber` **只在 Apple 平台被赋值**
  （`CPURuntime.cpp`，`sysctlbyname("hw.perflevel0.physicalcpu")`）。
  Linux/Android 分支里没有任何赋值点，它恒为 0 → **整个 P 核上限机制在 Android 上是空操作**。
  在 Android 异构 SoC 上复现 prefill 线程悬崖时，不要以为这层保护已经生效。

`perfCoreNumber` 挂在 `CoreFunctions` 上，所以二级表要显式带过去：
`CommonOptFunction.cpp` 从 `gCPUInfo` 拷入基表，`Arm82Functions.cpp` 单独拷一次。
新加这类"运行时拓扑信息"字段时同理，机制见
[`cpu/kernel/dispatch-and-register.md`](../kernel/dispatch-and-register.md) §3.2。

### 1.3 大小核有两套机制，而且互斥

MNN 对异构核有两条完全不同的应对，**它们不能叠加**：

| 机制 | 做法 | 触发 |
|------|------|------|
| A. 降并发 | 只用 P 核数量个线程（§1.2） | executor 主动调 `computeThreadNumber` |
| B. 不等分 | 用满所有线程，但给弱核少分活（`mGroupWithComputeRate`） | 构造 `CPUBackend` 时满足四个条件 |

机制 B 的四个门（`CPUBackend.cpp`，任一不满足就整段跳过）：

1. `mThreadNumber > 1` 且 `mPower != Power_Low`
2. `hint().cpuDecreaseRate` 落在 `(0, 100)`（默认 50，见 `Backend.hpp`，
   可由 `Session.cpp` 的 hint 改）
3. `MNNGetCPUInfo()->groups.size() >= 2`（只有一个核簇就没有异构问题）
4. 计算强度门：`mComputeI` 按能力位取 28 / 14 / 7（i8mm / dot / 都没有）

第 4 条的用法在 `computeDivideSizes` 里：

```cpp
// CPUBackend.cpp
void CPUBackend::computeDivideSizes(int size, int* dst, float avgDiv, int threads) const {
    if (threads <= 0 || threads > mThreadNumber) {
        threads = mThreadNumber;
    }
    // Group rates are defined over the full thread set; a capped caller needs an even split.
    if (mGroupWithComputeRate.size() <= 1 || (avgDiv > 0 && avgDiv < mComputeI) || threads != mThreadNumber) {
        // 等分
        ...
        return;
    }
    // 按 group rate 加权分
```

三条回到等分的路：group 只有一个、**调用方传进来的 `avgDiv` 低于 `mComputeI`**（低计算强度的活
不值得做非对称划分，加权只会让弱核成为新瓶颈）、以及 **`threads != mThreadNumber`**。

最后一条是 A 与 B 互斥的根因：一旦调用方用 `computeThreadNumber` 把 `threads` 压到 P 核数，
`threads != mThreadNumber` 成立，**加权划分整段失效，退回等分**。这是刻意的（的注释写了
group rate 是按全线程集定义的），但读代码时很容易以为两个机制会同时生效。

调查大小核问题时，先确认自己在哪个机制上：打印 `mGroupWithComputeRate.size()`、
实际 `threads`、以及调用方传的 `avgDiv`。三个值定了，划分结果就是确定的。

### 1.4 ThreadPool 的等待与唤醒：四条正确性要点

`ThreadPool.cpp`。这块被专门调过，**改自旋/睡眠策略时四条要求必须同时满足，缺任一条各有独特故障**：

| 要点 | 代码 | 违反后果 |
|------|------|---------|
| notify 前必须持锁一次 | （`{ std::lock_guard _l(mQueueMutex); }` 空临界区后再 `notify_all`） | 漏唤 → 死锁：全 worker 停在 `__psynch_cvwait`，主线程 barrier 空转 |
| 空闲计时状态（计数**和**时间戳）随每次完成整体重置 | （`worked` 后同时清 `spin`、`idleYields`、`idleStart`） | 只重置计数 → worker 在 64 次 yield 后误睡 |
| 空闲预算必须**时间制** | （`kWorkerIdleTimeout = 8ms`，每 64 次 yield 拿 `steady_clock::now() - idleStart` 比较） | 次数制 → decode worker 睡死（yield 成本随负载差一个数量级），kv2048 −13% |
| 预算耗尽后**不清零** | （`MNNThreadPoolRelax` 里 `spin` 到顶后不复位，本次等待剩余时间一直走 `yield()`） | 清零 → 退化成"spin 512 + yield 1"循环，仍在满速自旋 |

背景机制（这是为什么要有 `kThreadPoolSpinBudget` 的原因）：

- `std::this_thread::yield()` 在 ARM/Darwin 是 `swtch_pri` **系统调用**。每次自旋都调它，
  在 LLM decode（每 token 上百个极小并行任务）下会主导每-op barrier 开销。
- 所以用 `isb sy` 做 512 次有界退避（`kThreadPoolSpinBudget = 512`，
  `MNNThreadPoolRelax`）。RISC-V 用 Zihintpause 裸编码，x86_64 直接落到 `yield()`。
- 睡眠的另一半动机不是省电而是**调度器**：异构 SoC 上不持有活的 worker 热自旋会毒化 OS 的核放置，
  抢走工作线程的 P 核时间。

主线程侧的 barrier：先自己跑 slice 0，再轮询其余 flag，中间也走 `MNNThreadPoolRelax`。
`MNN_THREAD_POOL_MAX_TASKS`为 2，意味着同时在飞的 task 槽只有两个。

改这块之前先记住两条已经付过学费的结论：barrier 开销由 **op 数 × 单次成本**放大，
所以只在 decode 这种"每 token 上百个小 op"的形态下才显著；而空闲自旋与 E 核拖尾是**两个独立机制**，
必须分开量化、一起修，只修一个只能恢复一部分。ARM 侧的事故记录见 [`arch/arm.md`](arch/arm.md) §四。

### 1.5 判断 barrier 是不是瓶颈

不要从"我觉得同步开销大"开始。可用的判据：

- **op 数 × 单次 barrier 成本**估个量级。decode 每 token 上百个小 op，barrier 成本会被放大上百倍；
  prefill 每层就几个大 op，barrier 占比通常可忽略。
- **线程数扫描**：如果 t1 → t2 收益远小于 1.8x，且 t4 → t8 反而变慢，优先怀疑 L1 而不是 kernel。
- **替换法**：把并行段临时改成串行（`MNN_CONCURRENCY_BEGIN` 手写成 for），看总时间。
  如果串行反而更快，问题一定在 L1/L2，不在 kernel。

---

## 二、L2：划分与编排

### 2.1 三个划分工具的语义不同，别混用

| 工具 | 位置 | 语义 | 注意 |
|------|------|------|------|
| `computeDivideSizes(size, dst, avgDiv, threads)` | `CPUBackend.cpp` | 写出**前缀和边界**（`dst[i]` 是第 i 段的结束位置，不是长度），可加权 | `dst` 长度必须 ≥ `threadNumber()`；`threads` 语义见 §1.3 |
| `multiThreadDivide(size)` | `CPUBackend.cpp` | 返回 `(sizeDivide, scheduleNumber)`，按 `pack` 对齐 | 用的是**裸 `threadNumber()`**，不参与 P 核上限；`scheduleNumber` 可能 > 线程数 |
| 手写 `UP_DIV(size, threads)` | 散落各 executor | 等分 | 最容易和实际并发度脱钩 |

### 2.2 铁律：划分用的线程数必须与实际并发度同源

这是 L1↔L2 交界处最贵的一类 bug。**凡是"以线程数为参数、结果被持久化或跨阶段复用"的量都要审**：

- 权重打包边界（`mOcMain` / `ocMainThreads`）：`ConvInt8TiledExecutor.cpp` 的 `reorderWeight()`、
  `packWeightAndQuantInfo()`、`calculateSmeNeonWorkDivision()`。
  注意该字段旁的注释——`ocMain` 是**构造期**按池子满线程数烘焙的，
  所以推导时必须用 `ocMainThreads` 而不是 execute 期被 cap 后的 `threads`，
  否则 SME 线程会越过边界去读 NEON 布局的权重。
- `mDivides` 之类的划分数组，以及任何"按线程数开的数组 / buffer"。
- SME/NEON 非对称划分：`ConvInt8TiledExecutor.cpp`。

审的方法：确认 `computeDivideSizes(size, dst, avgDiv, threads)` 的 `threads`
与调用方实际开的并发度**是同一个表达式**，而不是"看起来都是线程数"。
ARM 侧的真实事故（`812e1bed34`：t5–t8 输出全废、越界写）记在 [`arch/arm.md`](arch/arm.md) §四。

### 2.3 分片轴的选择会切换代码路径

`ConvInt8TiledExecutor::onResize` 里，分片轴不是配置项，是被 shape 和线程数**算出来**的：

```
ConvInt8TiledExecutor.cpp   mSplitByOc = true;            // 默认按 OC 分
                        // planeSize 足够大 / 满足 preferLinearPlaneSplit
                                  // → mSplitByOc = false，改按输出 nhw 分
```

由此产生两个必须分别测的形状：

- **`planeSize == 1`（decode）**：`threads` 不降档（§1.2），倾向按 OC 分。
- **`planeSize >> threads`（prefill）**：`threads` 被 cap 到 P 核，倾向按 plane 分。

同一段代码在这两个形状下走的是不同分支。**只测 prefill 或只测单线程，覆盖率是结构性不足，
不是"测得少"**——完整的六个轴见
[`cpu/kernel/correctness-gate.md`](../kernel/correctness-gate.md) §一。

与之绑在一起的还有量化路径：`mUseBatchQuan`也是按 `inputPlane` / `planeSize`
选的，消费点在同文件的三处。**decode 和 prefill 是两条量化路径**，
不能用一条的结论推另一条。

### 2.4 launch 开销与"小 shape 不该并行"

`Concurrency.h` 定义了阈值 `LAUNCH_MULTI_THREADS_WORKLOAD = 1e+5`，
调用点在 `CPUBinary.cpp`、`CPURaster.cpp`。
低于这个规模就不开线程——因为 enqueue + barrier 的固定成本会超过并行收益。

L2 层典型的 launch/编排问题与对策：

| 现象 | 判据 | 对策 |
|------|------|------|
| profile 里时间很分散，没有单个热点 | 单 op 耗时小但 op 数量大 | 减少 launch 次数（合并 op、扩大每次并行的粒度），不要去优化"最慢的那个 op" |
| 小 shape 上比朴素实现还慢 | pack + enqueue 成本 ≈ 计算成本 | 加规模门限，小规模保留朴素/Vec4 路径 |
| 每次 forward 都在重算 shape | `onResize` 被反复调用 | 查 shape 是否真的变了；变的只是 seq_len 时看能否复用 |

`onResize` 的代价常被低估：它会重算 tiling、重开 buffer，在 LLM decode 里如果每 token 都触发一次，
占比会很可观。判定方法是在 `onResize` 入口计数打印，跑 N 个 token 看计数是不是 N。

### 2.5 融合语义优先于 kernel 复用

带宽敏感场景下，把一个融合算子拆成几次 `CoreFunctions` 调用**会净变慢**——省下的是开发量，
付出的是多遍访存。优先**扩展现有入口的签名**以保留融合语义；扩签名要同步改所有架构实现和调用点，
清单见 [`cpu/kernel/dispatch-and-register.md`](../kernel/dispatch-and-register.md) §3.3。
相邻 A/B 必须实测，不要凭"少了一次函数调用"推断。

### 2.6 动态 shape：换一次 shape 可能把所有权重重排一遍

**症状**：首次 prefill 特别慢，之后每换一个新 shape 又慢一次，而同 shape 重复请求正常。

**机制**：shape 变了要重跑 Geometry。若这个算子在 Geometry 层分解（融合 op 尤其如此），
`onCompute` 会**重新生成子 `Op` 与 Command**；Execution cache 以子 `Op*` 为键，新指针一律 miss，
于是每个子卷积重建 Execution，**把不可变权重从头 reorder 一遍**。
这是秒级代价，与 activation arena 的毫秒级调整完全不是一个量级——先分清是哪一个，
再决定改哪里。**不要用预留大内存或固定 padding 掩盖它。**

**修法**：当拓扑、权重、参数都不变，只有 shape 与 binding 变时，实现
`GeometryComputer::onRecompute`（`source/geometry/GeometryComputer.hpp`）。它默认返回 `false`，
调用点在 `GeometryComputerUtils.cpp`（受 `GEOMETRCOMPUTEMASK_OPENCACHE` 与 `!hasWrap` 双重门禁），
返回 `false` 就退回 `onCompute` 走完整重建——**所以它是纯优化，写错只会变慢不会变错**。四步契约：

1. 先校验子 Command 数、临时 Tensor 数、输入输出数**仍然匹配**；任一不符立即 `return false`；
2. 只更新临时 Tensor 的 shape / dtype / layout 与 Command 的输入输出绑定；
3. **保留**原 Command、`BufferStorage`、子 `Op` 指针和 Execution——已重排的权重就是靠这个复用的；
4. 验证：首次 cold prefill、至少两个从未出现过的新 shape、主 tile 两侧的 tail。

树里第一个实现是 `source/geometry/GeometryFusedProj.cpp` 的 `GeometryFusedProj::onRecompute`
（`MNN_SUPPORT_TRANSFORMER_FUSE`），可照它的校验粒度写。

**闭合证据链**用三样东西，不要靠推断：Execution 构造次数、子 `Op` 地址、weight-reorder 计时。

**decode tuning 不是 prefill warmup**：`Llm::tuning(OP_ENCODER_NUMBER)`（`transformers/llm/engine/src/llm.cpp`）
把 `gen_seq_len` 置 1 后跑的是 **M=1 的 decode forward**，prefill 可能用另一个 Module/Pipeline
和独立的 Execution cache，tuning 一遍不会替 prefill 预热。产品若允许启动预热，
就在接第一个用户请求前跑一次 prefill-only forward 再 reset KV；长期方案是让不同 Module
共享不可变的 packed-weight Resource。

---

## 三、改动前自查

L1 改动（线程数 / 等待策略 / 亲和性）：

- [ ] 确认在 ThreadPool 分支上（§1.1），不是 GCD/OpenMP
- [ ] §1.4 四条要点逐条对照，说清自己改的是哪一条、为什么不破坏另外三条
- [ ] 至少在 t1 / t2 / t4 / t(max) 上各测一遍；异构机器额外测 t = P核数 与 t = P+E
- [ ] 确认没有让 `perfCoreNumber == 0` 的平台（Android/Linux）出现新的行为差异
- [ ] 死锁是这类改动的主要风险：长跑一次，卡住就抓栈看有没有 worker 停在条件变量上

L2 改动（划分 / tiling / 编排）：

- [ ] 列出所有"以线程数为参数"的量，逐个确认与实际并发度同源（§2.2）
- [ ] decode（`planeSize == 1`）与 prefill（`planeSize >> threads`）分别测
- [ ] 确认改动没有把某个 shape 推到未测过的分支上（`mSplitByOc`、`mUseBatchQuan`）
- [ ] 越界写是这类改动的主要风险：按线程数开的数组，长度用的是 cap 前还是 cap 后的值

两类共同：

- [ ] 收益带完整维度标签（ISA / 线程数 / precision / shape），见
      [`shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md) §七
- [ ] 收益不外推：只在单线程有收益就写清楚只在单线程有收益

---

## 四、代码坐标速查

| 内容 | 位置 |
|------|------|
| 并行宏四分支 | `source/core/Concurrency.h` |
| ThreadPool 常量与退避 | `source/backend/cpu/ThreadPool.cpp` |
| worker 主循环（自旋 → 计时 → 睡眠） | `ThreadPool.cpp` |
| 主线程 notify + barrier | `ThreadPool.cpp` |
| 线程数上限 | `CPUBackend.cpp computeThreadNumber()` |
| P 核数探测（**仅 Apple**） | `CPURuntime.cpp` |
| 异构加权划分构造 | `CPUBackend.cpp` |
| 划分实现（含三条等分回退） | `CPUBackend.cpp computeDivideSizes()` |
| pack 对齐划分 | `CPUBackend.cpp multiThreadDivide()` |
| 功耗模式 → 亲和性 | `CPUBackend.cpp` |
| 分片轴选择 | `ConvInt8TiledExecutor.cpp` |
| 量化路径选择 | `ConvInt8TiledExecutor.cpp`（消费点） |
| 打包边界与线程数 | `ConvInt8TiledExecutor.cpp` |
| 并行门限 | `Concurrency.h`；调用点 `CPUBinary.cpp`、`CPURaster.cpp` |
| `onRecompute` 默认实现（返回 `false`） | `source/geometry/GeometryComputer.hpp` |
| `onRecompute` 调用点与双重门禁 | `source/geometry/GeometryComputerUtils.cpp` |
| 树里第一个 `onRecompute` 实现 | `source/geometry/GeometryFusedProj.cpp` |
| decode tuning（不预热 prefill） | `transformers/llm/engine/src/llm.cpp Llm::tuning()` |
