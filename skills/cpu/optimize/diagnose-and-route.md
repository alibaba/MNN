# 分层归因与路由：为什么慢、该改哪一层

> **何时读**：拿到一个「CPU 上慢」的问题，还没决定动哪个文件之前。这是 `cpu/optimize` 的第一站。
> 要和别的框架比性能、或对手框架里找不到 MNN 融合算子的对应物时，直接看 §二。
>
> **不在本文**：具体命令与测试名在 [`shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md)；
> 开关语义在 [`shared/env-registry.md`](../shared/env-registry.md)；ISA 路径全景在 [`arch/arm.md`](arch/arm.md) / [`arch/x86_64.md`](arch/x86_64.md) / [`arch/riscv.md`](arch/riscv.md)；
> 正确性 bug 在 [`bugfix.md`](bugfix.md)；kernel 怎么写在 [`cpu/kernel/SKILL.md`](../kernel/SKILL.md)。

---

## 零、先做两件事，否则后面全是浪费

### 0.1 确认「慢」这个数字可复现

一个不可复现的数字会把后面所有归因带偏。最低要求：同一命令连跑三次，波动小于你要追的收益。做不到就先解决测量问题（见 [`shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md) §六）。

**汇总平均会吃掉冷启动**：首次 prefill、以及每一个从未出现过的 shape，都可能付一次性的权重重排代价
（机制见 [`runtime-and-scheduling.md`](runtime-and-scheduling.md) §2.6）。短 prompt 场景下一个冷请求就能
把平均值压下去，看起来像 kernel 慢。所以**逐请求记录时间与 token 数、把 cold 和 hot 分开报**，
不要只看一个汇总速度。

### 0.2 确认你以为在跑的路径确实在跑

这一条排在分层归因之前，因为它是**最廉价也最常中**的解释。

CPU 后端有多张函数表、多条 ISA 路径、多重构建门。「AVX512 机器上测出 AVX2 的性能」「fp16 没开」「低 bit kernel 没编进来」都不会报错。

| 要确认的 | 怎么确认 |
|---------|---------|
| 编进来了吗 | 看 CMake 实际的编译命令 / `nm` 查符号。`MNN_LOW_MEMORY`、`MNN_AVX512`、`MNN_SME2`、`MNN_USE_ARMV82` 都是构建门 |
| 运行时选中了吗 | 打印能力位；或用 `MNN_CPU_TARGET` 逐档 A/B（**注意默认构建下它是空操作**，见 [`shared/env-registry.md`](../shared/env-registry.md)） |
| 这个 op 真的落在那个 kernel 上吗 | 在候选 kernel 入口各插一次一次性打印。**不要用「CPU 支持 i8mm」推断「这个 op 走了 i8mm」** |

具体两侧的自证方法见 [`arch/arm.md`](arch/arm.md) §三 与 [`arch/x86_64.md`](arch/x86_64.md) §三。

**如果这一步发现路径不对，问题到此结束**——把路径修对再重新测，不要进入分层归因。

---

## 一、先把实验回路缩到 op 级

**性能优化从「为目标场景建一个 op 级用例」开始，不是从读 kernel 源码开始。**
端到端 benchmark 只能告诉你「慢」，它的回路太长：一次 `llm_bench` 几十秒、混着 200 个 op、
掺着调度与内存噪声，改一行 kernel 看不出差别。op 级用例把回路缩到秒级，且只留一个自变量。

### 1.1 五步顺序

| 步 | 做什么 | 去哪 |
|---|---|---|
| 1 | 端到端 profile，拿到**每个 op 类型的时间占比** | §1.2 |
| 2 | 挑出**最不划算**的 op —— 不是占比最大的那个 | §1.3 |
| 3 | 为它建（或复用）一个贴目标场景的 op 级用例 | §1.4 |
| 4 | 在这个用例里迭代：判 bound、定层、改代码 | §四、§五 |
| 5 | **回到端到端复核** | [`shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md) §五、§七 |

第 5 步不能省。**op 级用例是放大镜，不是验收标准**——它刻意排除了调度、内存、cache 干扰，
而那些干扰在真实模型里是存在的。op 级 +30% 在模型里只剩 +2% 是常态。
只有 op 级收益、没有端到端数字的结论不予采信。

### 1.2 拿 op 时间占比：`llm_bench --profile`

`--profile` 挂上 `Profiler` 的前后回调（`llm_bench.cpp`），结束时调
`printTimeByType(1)`。表头是
**`Node Type | Avg(ms) | % | Called times | Flops Rate`**（`tools/cpp/Profiler.cpp`），
按耗时排序，末尾打印 `total time` 与 `total mflops`。

- 要看**单个节点**而不只是 op 类型汇总：置 `MNN_LLM_BENCH_PROFILE_NAME`（**存在即生效，不看值**）
  → 额外调 `printTimeByName(1)`，表头 `Node Name | Op Type | Avg(ms) | % | Flops Rate`。
  判断「同一类 op 里是哪一层慢」（lm_head vs 每层 proj）必须用它，类型汇总看不出来。
- **prefill 与 decode 必须分两次 profile**（`-p` / `-n`，见
  [`shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md) §四）。
  两者热点分布是两套：同一个 Qwen3-0.6B 在 decode 下 Convolution 72.3% / Attention 16.0%，
  换到 kv2048 混合场景 Attention 就涨到 45.1%。拿一个场景的占比去指导另一个场景是错的。
- `llm_bench` 只验速不验内容（合成 prompt 全是 token id 16），profile 结论要配同档正确性门禁。

### 1.3 挑哪个 op：看「时间占比 ÷ 工作量占比」

占比最大的 op 往往已经接近硬件上限，改它的天花板很低。真正的机会在**效率最差**的 op，
`Flops Rate` 这一列就是为此存在的：

- 真实例子（Qwen3.5-0.8B tg128）：`LinearAttention` 时间占 **22.0%**、flops 只占 **5.8%**
  → 计算效率约为 MatMul 的 **1/4**，是当轮 ROI 最高的目标。而占 66% 的 Convolution（GEMV）
  在大 shape 上已经跑到 82% roofline，继续优化基本是白干。
- ⚠ **decode 的 GEMV 是带宽 bound，`Flops Rate` 对它是错的尺子。** 那一档的尺子是
  `speed/GemvBW` 的 `eff GB/s` / `%peak`。同一个 w4 kernel 在 lm_head（151936×1024）是 82% roofline，
  在每层小 proj（1024×1024）只有 16%——**先确认这个 op 的正确尺子是什么，再谈它效率低**。
- 还有一类不归任何 op 名下：时间分散在大量小 op、没有明显热点。那是 dispatch/launch 开销
  （§四 的 L2 行），建 op 级用例救不了它。

### 1.4 op 级用例必须贴真实场景

用例参数不对，测出来的数字与模型内无关。四件必须对齐的事：

| 要对齐的 | 怎么做 | 不对齐的后果 |
|---|---|---|
| **shape** | 用目标模型的真实 M/K/head_dim/seq_len；`speed/GemvBW` 用 `MNN_GEMVBW_M` / `MNN_GEMVBW_K` 覆盖，不用重编 | GEMV 效率强 shape 相关，默认大形状的 %peak 不代表模型内实际效率 |
| **precision** | `argv[3]`；ARM 上 `2` 才是真 fp16 第二张表 | 拿 fp32 数据解释 fp16 模型 |
| **线程数** | `argv[4]`，固定跑 1 / 4 / 超过 P 核数三档 | ★ `speed/GemvBW` 的线程只在 `argc > 4` 时才被赋值，`... speed/GemvBW 0 2` 实测的是 4 线程 |
| **量化 / 内存档** | `argv[6]`（memory），`2` = `Memory_Low` | ★ 不给 `memory=2`，`op/lowMemory/*` 根本不进低 bit int8 executor，**测的不是你改的 kernel** |

argv 全是位置参数、错位不报错；完整语义与真实测试名注册表在
[`shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md) §二、§三。
**判用例是否真的跑了看 `passed` 数，不看退出码**——名字不匹配时照样打印
`all tests passed` 加 `"passed":0`。

---

## 二、跨框架对比：两边都要有同一个 op 的用例

只有自己的数字说明不了「已经够快」。跨框架对比的**唯一有效形式是 op 对 op**：
在对手框架里也建一个同 shape、同量化、同线程的用例。整模型 tokens/s 的对比里混着
tokenizer、采样策略、KV 管理、内存分配器的差异，归因不到 kernel。

### 2.1 口径先对齐，否则数字不可比

本仓库已有可抄的范例：`test/speed/GemvBWTest.cpp` 就是**对标 llama.cpp 的 `gemv_roofline.cpp`** 建的，
默认形状 4096×14336 取自 Llama-3-8B FFN。它显式声明了口径：**W bytes 只算权重 + per-block
（alpha + zp，fp16）元数据，不算输入向量与输出**——两边用同一个分母，`eff GB/s` 才能直接比。

于是能得到这类可交付的结论（`docs/perf/arm_low_bit_gemm.md`）：

| | MNN eff GB/s | vs llama.cpp |
|---|---|---|
| W8 | 109.4 | 超过 Q8_0（109） |
| W4 | 100.7 | **+29%** vs Q4_0（78） |
| W3 | 50.2 | 与 Q3_K 持平 |
| W2 | 64.5 | **+58%** vs Q2_K（41） |

要逐项对齐的：分母口径（算不算输入/输出/元数据）、block size、量化方案是否等价
（`Q4_0` ≠ 任意 w4）、线程数与绑核、cache 状态（冷/热）、迭代次数与取值方式（best-of vs 均值）、
机器热态。**任何一项没对齐，看到的差异都可能全部来自它。**

### 2.2 融合算子让 op 对不齐时：先拆解，再选一侧做基准

MNN 把一串计算融成一个 op（`Attention`、`LinearAttention = 305`、`FusedLinear = 307`、
`GatedRMSNorm`、`RoPE`），对手框架里没有对应物——profile 里就是一个 22% 的黑盒，对不上任何东西。

**第一步：查清这个融合 op 里面到底有哪些计算。** 读 executor，不要猜：

- `CPUAttention.cpp` = QK matmul + scale/mask（`_maskQK`）+ softmax + PV matmul + KV cache 管理
- `CPULinearAttention.cpp`（GDN）= depthwise Conv1D(k=4) + SiLU（有显式 Step 注释）
  → L2Norm + qk dot→ decay/state 递推→ gate/beta

**第二步，二选一，优先第一条：**

**（a）在 MNN 侧把融合关掉**，拿到分解后的逐 op 占比，再与对手框架逐个对。
导出期开关全部**默认 ON**（`transformers/llm/export/llmexport.py`）：

| 开关 | 关掉之后 |
|---|---|
| `--disable_fuse_qkv_proj` | q/k/v（含 linear-attn）投影不再合成一个 `FusedLinear` |
| `--disable_fuse_gate_up_proj` | SwiGLU 的 gate/up 投影不再合成一个 `FusedLinear` |
| `--disable_fuse_ln_proj` | 块输入的 binary RMSNorm 不再折进投影（`has_ln` 变体） |
| `--disable_fuse_linear_attn_gate` | gate/beta 常量不再折进 `LinearAttentionParam`，恢复导出 softplus/sigmoid 链 |
| `--disable_transformer_c4` | 连带关掉**所有**依赖 `MNN_SUPPORT_TRANSFORMER_FUSE` 的融合导出（即上面四项） |

⚠ 两个陷阱：①拆开导出的模型**必须重跑正确性门禁**——`--disable_fuse_linear_attn_gate`
在不支持 gate_fold 的引擎上会被忽略，并把原始 `a` 投影当 decay gate 用，
症状是**结果错，不是加载失败**。②运行时侧的 `MNN_SUPPORT_TRANSFORMER_FUSE=OFF`
也能去掉融合 op（`ShapeRegister.cpp`），但它同时让 attention 全家测试编不进去、
**匹配 0 个用例**，不要用它当拆解手段。

**（b）在没有融合的那一侧建最小端到端用例**：把融合 op 覆盖的那几步——**只有那几步**——
串成一个能独立跑的小图，两边跑同一个小图。边界要精确到与融合 op 完全一致：
含不含最后那次 norm、含不含 KV cache 写入，都会改变结论。多包一个 op 或少包一个 op，
得到的就是两个不可比的数。

无论走哪条：**拆解只用于归因，不作为交付形态。** 拆开跑出来的总时间通常比融合版慢
（那正是融合的收益），不要拿拆解后的数字代表 MNN 的性能。

---

## 三、五层模型：按「改哪一层」组织，而不是按「第几步」

CPU 侧的性能问题几乎都能归到这五层之一。选错层的代价是把时间花在完全不相关的代码上。

> **这张表全树唯一一份。** [`cpu/SKILL.md`](../SKILL.md) 与 [`SKILL.md`](SKILL.md) 都引用它，不再另抄一份。

| 层 | 名字 | 管什么 | 典型改动 | 去哪读 |
|---|------|--------|---------|--------|
| **L1** | Runtime / 线程 | 线程数决策、ThreadPool 等待与唤醒、大小核、barrier | 改等待策略、改线程数上限、改 relax 时机 | [`runtime-and-scheduling.md`](runtime-and-scheduling.md) |
| **L2** | Executor / 调度 | tiling、`mSplitByOc`、per-thread 划分、resize 频率、命令编排 | 改分片策略、减少 re-resize、合并 launch | [`runtime-and-scheduling.md`](runtime-and-scheduling.md) |
| **L3** | Layout / 内存 | pack 格式、weight reorder、cache 复用、缓存增长与 peak RSS、访存次数 | 改 block 粒度、改布局、消除中间 buffer | [`layout-and-memory.md`](layout-and-memory.md) |
| **L4** | Dispatch / 表 | 哪条 ISA 路径、哪张函数表、能力位、回退 | 换 kernel 指针、加一层 ISA、修回退 | 诊断面 [`arch/arm.md`](arch/arm.md) / [`arch/x86_64.md`](arch/x86_64.md) / [`arch/riscv.md`](arch/riscv.md)；实现面 [`cpu/kernel/dispatch-and-register.md`](../kernel/dispatch-and-register.md) |
| **L5** | Kernel / ISA | 单个 kernel 内部：指令选择、unroll、寄存器、unpack 预算 | 改 asm/intrinsic、改 unpack 方案 | 实现面 [`cpu/kernel/arch/`](../kernel/arch/)；该不该动手 [`cpu/kernel/SKILL.md`](../kernel/SKILL.md) |

**跨层交界处是最贵的坑集中地**（L2↔L3 的 stride、L3↔L4 的 pack 与 tile、L4↔L5 的 ABI）。这类问题的表现往往是「不崩、结果略差」，统一记在 [`bugfix.md`](bugfix.md)。

---

## 四、先判 bound 类型，再选层

不要从「我觉得 kernel 可以更快」开始。先用数字把瓶颈类型定下来，这决定了该动哪层。

| bound 类型 | 判据 | 主责层 | 常见误判 |
|-----------|------|--------|---------|
| **DRAM 带宽 bound** | eff GB/s 接近机器 peak；线程数增加收益近线性直到饱和 | L3（减少权重流量 / 改布局） | 低 bit kernel 的 eff BW 低**不等于**带宽不足，见下一行 |
| **unpack / issue bound** | eff GB/s 远低于 peak，但单核已经打满；unpack 指令数 / 有效字节比高 | L5 | 误当成带宽 bound 去加 prefetch 或扩大 packed bytes，两者都无效 |
| **postprocess bound** | 把 postprocess 换成空实现后耗时明显下降 | L5（或 L3 若能融合） | 归因到 GEMM 主循环 |
| **dispatch / launch bound** | 单 op 耗时小但 op 数多；profile 里时间分散在大量小 op | L2（减少编排开销）| 去优化最耗时的那个 op，但它只占 5% |
| **同步 / barrier bound** | 线程数增加反而变慢；或空闲线程占了大量 CPU | L1 | 当成"线程数没调好"只改数字，不查等待策略 |
| **re-resize bound** | 每 token / 每次 forward 都在重算 shape | L2 | 当成 kernel 慢 |
| **内存占用问题（不是慢）** | peak RSS 超标但吞吐正常 | L3 | 以为是 buffer 大小问题，实际是分配器 free-list 复用落空 |

判 bound 的通用手法是**替换法而非推理法**：把某一环换成空操作或朴素实现，看总时间怎么变。推理容易错，替换不会。

---

## 五、症状 → 优先怀疑（路由表）

| 症状 | 优先怀疑的层 | 第一个动作 |
|------|-------------|-----------|
| 改了 kernel，benchmark 完全没变 | L4 | 确认 kernel 被调用（§0.2）。绝大多数是没生效 |
| 单线程还行，多线程扩展性差 | L1 | 看空闲 worker 是否在热自旋；看 barrier 次数 |
| 线程数超过某个值后**反而崩塌** | L1 + L2（两个机制叠加） | 大小核拖尾与调度器毒化要分开量化、一起修 |
| prefill 快、decode 慢得不成比例 | L2 + L3 | decode 是 `E == 1`，走的量化路径和 tile 都不同（`ConvInt8TiledExecutor.cpp`） |
| profile 里没有明显热点，时间很分散 | L2 | dispatch/launch 开销，不是任何单个 kernel |
| 低 bit（w2/w3）有效带宽很低 | L5 | 数 unpack 指令 / 有效字节，别先假设 DRAM 慢 |
| 加了 unroll 反而变慢 | L5 | 寄存器压力、load/store、branch，不是"unroll 不够多" |
| 融合算子改成复用现成 kernel 后变慢 | L3 | 被拆成了多遍访存，融合语义丢了 |
| peak RSS 超标但速度正常 | L3 | 分配器 free-list 复用，与 buffer 大小无关 |
| 换机器结论就反过来 | L1 + L4 | 核拓扑与能力位都变了，两台机器的 bound 类型可能不同 |
| 小 shape 上比朴素实现还慢 | L2 | pack 开销没摊薄，小规模该保留朴素路径 |
| 只有某一档 ISA 慢 | L4 | 该档的 packer / kernel 配对，或回退到了更低档 |
| 首次 prefill 慢，之后每换一个新 shape 又慢一次 | L2 | 查 Geometry 是否重建子 `Op`/Execution、不可变权重是否被重复 reorder（[`runtime-and-scheduling.md`](runtime-and-scheduling.md) §2.6） |
| 短 prompt 的汇总速度异常低 | 先怀疑测量口径 | 是否把首次 cold prefill 混进了平均值；逐请求记 cold/hot 时间与 token 数（§0.1） |

---

## 六、一条重要的元规则：收益不随维度外推

同一个改动在不同线程数、不同 shape、不同精度下的收益**可以差一个数量级，甚至反号**。

真实例子（记在 case 文档里）：宽 KV block 在单线程上 +67%，主要来自单核 K/V 流长效应（35→49 GB/s）；到多核就只剩 +1.4%，因为多核下流已经交错、softmax 在小 qRows 上是 exp 吞吐 bound 而不是调用 bound。

所以：

- **报收益必须带完整维度标签**（ISA / 线程数 / precision / block size / shape），否则数字不可复现也不可外推。
- **在你关心的那个维度上实测**，不要从相邻维度推断。
- 如果一个改动只在单线程上有收益，要明确说清楚——它可能仍然值得做（decode 常是低并发），但不能写成通用收益。

---

## 七、进入下一步

定位到层之后：

- L1 / L2 → [`runtime-and-scheduling.md`](runtime-and-scheduling.md)
- L3 → [`layout-and-memory.md`](layout-and-memory.md)
- L4 → [`arch/arm.md`](arch/arm.md)、[`arch/x86_64.md`](arch/x86_64.md) 或 [`arch/riscv.md`](arch/riscv.md)
- L5 → [`cpu/kernel/SKILL.md`](../kernel/SKILL.md)（有性能数据才进）
- 发现是正确性问题而不是性能问题 → [`bugfix.md`](bugfix.md)
- 改完要交付 → [`shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md) §五 验证矩阵 + §七 结果记录
