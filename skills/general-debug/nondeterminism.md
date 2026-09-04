# 逐 run 不同：未初始化内存与多线程非确定

> **归属**：[`general-debug`](SKILL.md) 的分类分册之一，先在入口的分流表确认类别再读本文。
> 本册收**同一输入每次跑结果都不一样**的两类根因：§9 读到未初始化内存（行为随堆内容漂移）、
> §10 归约顺序或 kernel 变体被线程调度时序决定。
>
> **不在本文**：稳定地错在某一档（只有 t4 错、只有超过某长度错）不属于本册——
> CPU 侧见 [`cpu/optimize/bugfix.md`](../cpu/optimize/bugfix.md)，跨后端别名见 [`memory-aliasing.md`](memory-aliasing.md)
> （其 §1.6 收 GPU 单 dispatch 内 threadgroup 竞争导致的逐次不同）。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

## §9 未初始化内存 / 堆垃圾依赖

**触发**（满足以下之一强烈怀疑本类）：
- 同一二进制多次跑，结果在"正常"与"乱码"之间漂移（相位漂移），或前几 token 正常随后崩坏；
- 加 printf、改无关代码、换 build 目录就"好了"或换个地方坏（堆布局敏感）；
- 只在特定线程数 / 特定机型触发（如 t1 乱码、t4 正常）；
- 输出大片 ±65504（fp16 饱和）或单字符刷屏（`!!!`）；
- bisect 结论不稳定，同一 commit 两次判定不同。

### 9.1 核心心法

**行为依赖堆内容 = 读了未初始化内存。** 这类 bug 的"大多数时候正常"是假象：干净堆（零页、复用块里碰巧无害的旧数据）让垃圾值恰好不致病。不要用"跑几次好像没问题"证否它。

**第一武器：`MallocPreScribble=1 MallocScribble=1`**（macOS；把新分配/已释放内存填 0xAA）。它把潜伏的未初始化读变成**按需确定性复现**：
- scribble 下 5/5 乱码、干净堆 5/5 正常 ⇒ 未初始化读实锤；
- **scribble 下做 bisect**：每个 commit 的 verdict 变成确定性的，消除相位漂移 confound（干净堆 bisect 会把同一 commit 判成两种结论）；
- 0xAA 的语义：按 fp16 解释 = -0.052（小但合法，悄悄污染）；按 fp32 位型 = denormal → 取倒数爆炸 → ±65504 饱和。看到 ±65504/`!!!` 先想垃圾内存。

### 9.2 排查流程

1. **scribble 复现 + 确定性二分**：worktree 并行构建候选 commit，全部在 scribble 下判定。
2. **trace hook 对拍定位首个分叉 op**（手法见 [`memory-aliasing.md`](memory-aliasing.md) §1.3）。⚠️ **张量内容 hash 必须按物理字节算**：`precision=low` 时 float 型张量实际是 fp16 存储（2 字节/元素），但 `getType().bytes()` 返回 4；按 `elementSize * getType().bytes()` hash 会越界读进相邻 allocation，所有 hash 退化成堆布局噪声，制造"op 输出非确定"的假线索。
3. **怀疑到具体类后，列出全部构造路径**：MNN 的 Execution 除主 ctor 外还有 `onClone`/`createClone` 的拷贝 ctor。主 ctor 初始化了不代表对象安全——拷贝 ctor 漏拷的成员就是堆垃圾。给已有类加成员时 grep 所有构造函数。
4. **`this` 指针对拍区分两种机制**：ctor 出口打印 `this`+成员值，使用点再打印一次。ctor 打过但使用点值变 = 被越界写；使用点有值但该 `this` 从没出现在 ctor 日志 = 构造路径绕过（clone）。

### 9.3 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的原因 |
|------|-------------|
| 干净堆偶发乱码、scribble 下 100% 复现 | 未初始化成员/堆垃圾依赖，实锤 |
| 只在某线程数坏 | 未初始化 flag 改变了线程数 clamp 或工作划分 |
| trace hash 显得"非确定" | hash 按逻辑字节算了 fp16 张量（越界读邻分配） |
| 主 ctor 有初始化却仍读到垃圾 | clone/拷贝 ctor 漏拷，或 ctor 后被越界写（用 `this` 对拍区分） |

### 9.4 参考案例：DenseConvInt8TiledExecutor 拷贝 ctor 漏拷 mMixedKernel（2026-08-26）

**症状**：Qwen3-1.7B/4B t1 乱码（scribble 下 5/5 确定复现），t4 与 master 正常。scribble 下 bisect 定位到新增 `if (mMixedKernel) threads = ALIMAX(threads, 4)` clamp 的 commit。

**根因链**：`DenseConvInt8TiledExecutor(bn, op, exe)` 拷贝 ctor 只拷了 `mGemmKernel`，`mMixedKernel` 是堆垃圾 → clone 实例在 t1 读到 truthy 垃圾 → clamp 把 threads 抬到 4 → `_updateMixedKernelFlag` 顺势把 SME/NEON mixed kernel 合法化 → 在 1 线程池上按 4 线程划分、且对按非 mixed 路径打包的权重用 mixed kernel → conv 输出 ±65504 饱和 → 整模型 `!!!`。主 ctor 路径 `mMixedKernel` 被正常赋 false，所以 t4 和非 clone 实例都不踩。

**修复**：拷贝 ctor 补拷 `mMixedKernel` + 头文件默认初始化 `= false`。验证：修复后同二进制干净堆/scribble 输出逐 token 一致（10/10 SAME）。

**避坑要点**：
- 新成员 = 头文件默认初始化 + 检查每条构造路径，两件事都做；
- 验证修复的标准是"输出不再依赖堆内容"（干净/scribble 逐 token 一致），不是"这次跑看着正常"。

### 9.5 相关文件索引

| 文件 | 作用 |
|------|------|
| `source/backend/cpu/compute/ConvInt8TiledExecutor.hpp` | 成员默认初始化（本案例漏 `mMixedKernel`） |
| `source/backend/cpu/compute/ConvInt8TiledExecutor.cpp` | 拷贝 ctor / `createClone` / `onClone`（本案例修复处） |
| `transformers/llm/engine/demo/llm_demo.cpp` | trace hook 落点；hash 参考实现需按物理字节 |

---

## §10 CPU 多线程数值非确定（贪心同输入逐 run 分叉）

**触发**：
- 贪心解码同输入连跑，输出逐 run 分叉——不是乱码，内容连贯但彼此不同；
- 只在部分线程数触发（如 t4/t8 分叉、t1/t2 稳定），只在部分模型触发（KV head 少、工作项少的 GQA 模型更敏感）；
- 分支新引入"per-thread 异构 kernel"（SME/NEON 混合、不同精度路径）或"原子计数器动态工作分发"之后出现。

### 10.1 核心心法

**逐 run 数值不同 = 累加/归约顺序或 kernel 变体被调度时序决定。** 时序源有三：①`fetch_add` 动态分发的工作项→线程映射；②跨线程归约的合并顺序；③按 tId 选择的 kernel/精度变体。①×③ 的组合最隐蔽——单看各自都是合理的优化（负载均衡 / 异构加速），组合后"哪个工作项算在哪条变体上"由时序决定。先分流"采样非确定"还是"前向数值非确定"：**确定性测试必须显式钉死 sampler**（MNN 模型 config 常见默认 `sampler_type:"mixed"` 本就是非确定采样，不钉死连 master 都分叉，整轮归因会被采样噪声带偏）。

### 10.2 排查流程

1. **钉死 sampler**（config 置 `"sampler_type":"greedy"`），同 prompt 连跑 ≥3 次对输出文本 hash；仍分叉才是前向问题。
2. **env gate 归因矩阵**：给每个可疑优化加临时开关（各 kernel 变体、分发策略各一个），组合开关分别测确定性，收敛到"开 A 且开 B 才分叉"的最小组合；对矩阵中"关了也没用"的组件做静态审查确认其分发/kernel 选择本就确定，正式排除。
3. **静态审查命中组合**：分发用共享原子计数器？kernel/精度选择依赖 tId？两者皆是即本类实锤。barrier 后按固定序合并的归约本身确定，不用改。
4. **修复模式：分组计数器**——按 kernel 变体把工作项划成固定区间，每组一个计数器，线程只从本组区间抢：item→变体映射确定（逐 run 一致），组内仍动态（慢线程少拿，负载均衡保留）。

### 10.3 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的原因 |
|------|-------------|
| 钉死 greedy 后 master 也分叉 | 测试方法问题（sampler 没生效），不是引擎 bug |
| 共享单计数器动态分发 + 按 tId 选 kernel 变体 | 本类实锤：item→变体映射由时序决定 |
| 分发静态（按 tId 划片）或 kernel 全员一致 | 非本类，回查 [`memory-aliasing.md`](memory-aliasing.md) §1.6（别名竞争）与本册 §9（未初始化） |

### 10.4 参考案例：Qwen3.5 t≥4 贪心输出分叉（2026-08-26）

- **根因**：`CPUAttention` kvSplit flash-decoding 用单一 `splitNext.fetch_add` 动态分发 (unit, chunk) 工作项，mixed SME/NEON matmul 按 `tId >= mSmeThreadCount` 选变体 → 工作项的数值路径随时序漂移。numUnits=2 的 GQA 模型 kvSplit≥2 才踩；dense 模型 numUnits=8 经 gcd 后 kvSplit=1 走静态分片，天然幸免。
- **归因**：attn-mix 关 → DET；conv-mix 关 → 仍 NONDET（其 ocIndex/kernel 选择全静态，排除）；kvSplit 关 → DET。唯一源头 = attention mixed × kvSplit 动态分发。
- **修复与验证**：SME/NEON 两组各一计数器、item 边界按线程数比例固定，组内动态。t4/t8 贪心 ×3-4 连跑 hash 全一致；decode 性能差 ≤±1.8%（噪声带内）。

### 10.5 相关文件索引

| 文件 | 作用 |
|------|------|
| `source/backend/cpu/CPUAttention.cpp` | kvSplit 动态分发循环、mixed matmul 变体选择（本案例修复处） |
