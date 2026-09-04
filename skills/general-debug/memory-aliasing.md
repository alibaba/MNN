# §1 内存别名 / 生命周期错误

> **归属**：[`general-debug`](SKILL.md) 的分类分册之一，先在入口的分流表确认类别再读本文。
>
> **不在本文**：同一输入**每次跑都不一样**且与别名无关（未初始化成员、动态分发时序）见
> [`nondeterminism.md`](nondeterminism.md)；所有后端**一致地**错见 [`export-and-quant.md`](export-and-quant.md)；
> 越界写把 GPU 打崩见 [`gpu-oob.md`](gpu-oob.md)。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

**触发**（满足以下之一强烈怀疑本类）：
- 输出乱码 / 数值明显错乱，但 kernel / op 单跑对；
- CPU 后端错、其它后端（Metal / GPU）对，或反之；
- 关掉某个新加的 op / 优化 pass 就好，看着又没写错逻辑；
- `onResize` / buffer 分配相关代码改过后开始回归；
- 加 `printf` 或改 buffer size 就"好了"。
- 同一 workload 反复创建/销毁后，物理内存按固定步长增长，且常规 backend GC 无效。

## 1.1 核心心法

**"地址看着都对但结果错" ≈ 内存别名或生命周期错误。** 遇到这种症状**优先怀疑内存复用**，不要先去怀疑算法本身。这类 bug 的共同特征：

- 单个 kernel / op 单独跑对，串起来错；
- 打印指针发现"都是合法地址"，但内容互相污染；
- 加 `printf` 或改 buffer 大小就"好了"（其实只是位置错开了）；
- 关线程或改线程数症状变化（并发 + 别名一起放大）。

**方法论一句话**：**别信"这块 buffer 应该是独立的"，用地址等式去证明**。

## 1.2 MNN 内存模型快速回顾

进入排查前先牢记 MNN 的三条内存复用语义（拿不准就去读 `source/core/BufferAllocator.hpp`、`source/core/Backend.hpp`）：

### (a) `BufferAllocator` 是 arena，`free` **不释放**

- `alloc(size)`：如果 free pool 里有相同 size 的 chunk，**直接复用地址**；
- `free(chunk)`：**只是把这块标记为可复用**，物理内存保留；
- 交错的 `alloc(); free(); alloc(); free()` 模式，只要 size 相同，第二次 `alloc` 必然拿回第一块。

### (b) Backend tensor buffer 会被跨 op 复用

- `onResizeBegin` / `onResizeEnd` / `compute()` 之间，pipeline 会根据 tensor 生命周期把互不重叠的 tensor 分配到同一块 backend buffer；
- 同一个 `Tensor*` 在不同 op 里的物理 buffer 可能不同；不同 `Tensor*` 也可能共享同一物理 buffer；
- 这在 GPU 后端（Metal `id<MTLBuffer>`、CUDA `void*`）同样会发生。

### (c) `MemChunk` 的 `ptr()` 仅在 lifetime marker in-use 期间稳定

- `onResize` 里申请的 chunk，`onExecute` 期间用它的 `ptr()` 是安全的（op 内时序保证）；
- 但同一 op 内**两个 MemChunk 之间**是否别名，完全取决于 alloc/free 次序。

## 1.3 排查流程

### Step 1: 复现并最小化

- 定住一个能稳定复现的最小 case（最短 prompt、最小 shape、单线程）。多线程先关掉，避免并发遮盖别名症状。
- 记录基线：换个后端跑同 case 是不是对？把新加的 op / pass / fusion 关掉（用 env 开关 / cmake option）是不是对？**"另一个后端对、这个后端错"是内存类 bug 的强烈信号** —— 同一段算法在两个后端上的差异，很多时候只在于内存模型不同。

### Step 2: 用地址等式定位别名（关键手法）

在怀疑的 op 的 `onExecute` 入口打印所有 scratch chunk / tensor 的**物理地址**：

```cpp
MNN_PRINT("[MyOp] cos=%p sin=%p qTmp=%p kTmp=%p out=%p\n",
          mTmpCos.ptr(), mTmpSin.ptr(), mTmpQC4.ptr(), mTmpKC4.ptr(), output->host<void>());
```

判据：

- **两个"逻辑上独立"的 buffer 地址相等 → 100% 别名**，直接进 Step 3；
- 地址不等但差得很近（相邻 chunk）→ 可能是越界写而不是别名，改看 Step 5；
- 地址完全无关但结果仍错 → 别名可能不在这几个 buffer 里，扩大打印范围（覆盖 `onResize` 里所有 `alloc`，包括子调用 `MNNNorm` / `MNNLowpToFp32` 之类内部若也走 allocator 需要一起看）。

### Step 3: 从 `onResize` 找根因

如果确认了别名，直接看该 op 的 `onResize`：

- **反模式**：`a = alloc(N); free(a); b = alloc(N); free(b);` — size 相同 + 交错就一定别名。
- **正确模式**：先把所有 chunk `alloc` 完（此时前面的 chunk 都还是 in-use），再一起 `free`：

  ```cpp
  mA = buf->alloc(N);
  mB = buf->alloc(N);   // 此时 mA 未 free，allocator 只能给新地址
  mC = buf->alloc(M);
  ...
  buf->free(mA);
  buf->free(mB);
  buf->free(mC);
  ```

  `onExecute` 里继续用 `ptr()` 依然合法 —— `free` 只是"生命周期声明结束"，物理内存仍在。参考实现：`source/backend/cpu/CPULayerNorm.cpp`、`source/backend/cpu/CPURoPE.cpp::onResize`。

### Step 4: 跨 op tensor 别名（fusion / 图重排场景）

当症状出现在多个 op 之间（比如做了 QKV / Gate-Up fusion 之后），要检查 backend 侧的 tensor buffer 是否被 pipeline 分配到了同一物理 buffer：

- Metal 侧参考 `MetalBackend.mm` 里 `matchQKVFusions` 的做法 —— **在 `onResizeEnd` 的 `compute()` 之后**再检查 output buffer 是否重叠，重叠就 fallback；
- 调用顺序至关重要：只有 `compute()` 之后才知道实际分配结果，反过来做检查等于空跑；
- 若图重排后仍重叠，需要在 converter 侧调整 op 顺序（如 `reorderQKVProjections`），或直接放弃这次 fusion。

### Step 5: 越界写 / 未初始化

如果地址不别名但结果仍错，考虑：

- **越界写**：某个 kernel 按 `numHead * headDim` 写，但 `alloc` 只按 `headDim` 或漏了 `threadNumber` 倍数。快速验证：把 scratch buffer size 翻倍再跑，如果症状消失就是越界。
- **未初始化 + arena reuse**：新 `alloc` 拿回来的是别的 op 用过的脏数据。fp32 / fp16 里的脏数据可能不是 NaN 而是"看着正常的小数"，症状是"结果略微不对"。
- **多线程 tId 分片错**：`chunk.ptr() + tId * stride` 里 `stride` 少乘一个维度，导致相邻线程互踩。单线程能过就是这个。

### Step 6: 生命周期错位（栈上临时 buffer）

- 不要在 `onResize` 里把 `std::vector` / 栈数组的地址存到成员里，然后在 `onExecute` 里用 —— `onResize` 返回后那块内存已失效。
- scratch 一律走 `BufferAllocator` 或成员变量的 `std::vector`（且要 `resize` 而不是 `reserve`）。

## 1.4 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的原因 |
|------|-------------|
| 单 op 对，串起来错；从第一步就错 | Scratch buffer 别名（Step 3） |
| 一个后端错另一个后端对 | 有别名的一侧走了 arena reuse（Step 1/3） |
| 关掉 fusion / 优化 pass 就好 | 跨 op tensor buffer 重叠（Step 4） |
| 加 printf / 改 buffer size 就好 | 越界写或别名（Step 5） |
| 单线程对，多线程错 | tId 分片 stride 少算了维度（Step 5） |
| 每次运行结果不一样、有时对有时错 | 未初始化 + arena reuse 的脏数据（Step 5） |
| 结果全 NaN / 全 0 | 生命周期错位（Step 6）或未写就读 |
| **GPU 上逐次不同，且把「前驱算子折进后继算子」的融合关掉就稳** | **融合引入的别名竞争：前驱的输入被分配器复用成了后继的输出（§1.6）** |
| **逐次不同，但强制逐 op commit 串行化后仍不稳** | 污染发生在**单个 dispatch 内**（TG 间竞争 / 别名），不是跨 dispatch hazard（§1.6） |

## 1.5 参考案例：CPU inv_freq RoPE scratch 别名

**症状**：Qwen3 c4-head 模型，Metal 后端正常，CPU 后端从 prefill 第一步就吐乱码 token。

**排查路径**：

1. 复现最小化：单线程、单 prompt 仍乱码 → 不是并发问题。
2. 换 Metal 正常 → 强怀疑 CPU 特有的内存模型问题。
3. 地址打印：`mTmpCosFloat.ptr() == mTmpSinFloat.ptr()`，`mTmpCos.ptr() == mTmpSin.ptr()`。别名坐实。
4. 回看 `CPURoPE::onResize`：`alloc(); free(); alloc(); free();` 交错模式，size 又相同。
5. 读 `BufferAllocator.hpp` 注释确认语义：`free` 只标记可复用、`alloc` 命中相同 size 直接复用地址。

**为什么 Metal 没事**：`MetalRope.mm` 用 `id<MTLBuffer>`（ARC），且 cos/sin 在 shader 内当场算，不走 arena reuse，天然没有别名机会。

**修复**（对齐 `CPULayerNorm.cpp` 惯用法）：`CPURoPE::onResize` 里把所有 scratch chunk 一次性 alloc 完，最后统一 free。`onExecute` 里继续用 `ptr()` 依然合法。

**避坑要点**：这个 bug 无法通过 review 逻辑代码发现 —— 代码逻辑完全正确，`cosFloat[j] = c` 也确实写到了 `cosFloat` 指向的地址，只是这个地址恰好也是 `sinFloat`。**必须靠"打印地址、找相等对"这一步来揭穿**。

## 1.6 参考案例：融合引入的别名竞争（LayerNorm 折进 Conv1x1，2026-08-03）

**症状**：Qwen3.5-2B Metal decode 输出**逐次不同**（5 连跑 5 种 hash），偶尔整段退化成重复字符；
0.8B 同配置看起来稳定；关掉 LN 融合就稳。单测全过，`MTL_SHADER_VALIDATION` 无报告。

**根因**：动态分配器把某投影的输出复用到了 LayerNorm 的 **residual 输入**同一字节区间。
在 LN 还是**独立、更早**的 dispatch 时这个复用完全合法（LN 读完 residual 才轮到 conv 写）。
LN 折进投影 dispatch 后，**同一个 kernel 内既读 residual 输入又写该输出** ⇒ 先写出的
threadgroup 覆盖掉其他 threadgroup 仍要读的数据 ⇒ 结果取决于 TG 调度顺序。

**决定性的两步定位手法**（本类 bug 通用，比 debugger capture 便宜得多）：

1. **先分离"单 dispatch 内"还是"跨 dispatch"**：用 `MNN_METAL_COMMIT_NUM=1` 强制**逐 op 一次
   commit**，把所有 dispatch 串行化。
   - 串行化后**变稳** ⇒ 跨 dispatch hazard（缺 barrier / 资源提前复用 / untracked 资源）；
   - 串行化后**仍不稳** ⇒ 污染在**单个 dispatch 内部**，只剩 TG 间竞争、别名、未初始化读三种可能。
     本例正是这一支，一个实验就把假设空间砍掉一大半。
2. **再用字节区间别名探针一次命中**：在融合 dispatch 的 encode 处（env 门控的临时代码），
   把它**写**的每个张量与**读**的每个张量都换算成 `(MTLBuffer*, offset, offset+bytes)`，
   两两判重叠并打印。别名会直接以
   `out_q[0,8192) overlaps ln_res_in[0,8192)` 的形式暴露，本例 18 层全命中。
   > 关键：**必须比"写集合 × 读集合"，而不是只比几个可疑张量**。此前只核对了 LN 自己的
   > 三元组（hidden/resIn/resOut）互不重叠就误判"无别名"，漏掉了 residual 输入与**投影输出**
   > 这一对 —— 而那才是真正的冲突对。

**修复模式**：在融合匹配处（`matchLNFusions`）挂载前做上述别名检测，重叠即把冲突输出
`onAcquireBuffer(..., Backend::STATIC)` re-home（STATIC 内存动态池永不复用），
re-home 失败则**保守跳过该次融合**（fail-safe 方向）。

**排除项与它们为什么误导**：
- "关掉 A 就稳、关掉 B 也稳 ⇒ 是 A×B 的交互" —— **未必**。本例 "LN-only 稳定" 的真实原因是
  LN 融合对这些层**根本没生效**（4 个消费者使 sole-consumer 条件不成立，没有 leader 可挂载），
  而不是"LN 单独是安全的"。**先确认某配置下这条优化到底有没有命中**（加临时计数打印），
  再据此推断，否则会把判别维度搞错（本例真正的维度是**层类型**，不是融合路数）。
- `MTL_SHADER_VALIDATION` 查不出这类问题 —— 所有访问都在合法绑定范围内，它只抓越界。

**⚠️ 对拍口径陷阱（本次一度误判默认态也坏）**：`llm_demo` 的 stdout 内嵌 `cost time` 行与末尾
性能统计块。直接 `shasum` 整个输出会让**本来确定的配置也"每次不同"**。对拍必须先剔除计时行，
例如 `awk '/^#####/{exit} !/cost time/{print}'`，只 hash 生成文本。

## 1.7 参考案例：验证「融合是否数学等价」——用 fp32 当 oracle（2026-08-04）

**场景**：把一串算子折进一个新 kernel（本例：per-head RMSNorm × SiLU 门控 + 两次 C4 重排，
7 个 dispatch → 1）。fp16 下输出 token 与原链路分叉，需要判断是**逻辑/索引写错**还是
**rounding 顺序差异**。

**第一步一定是用 fp32 跑一遍**（`precision: "high"`）。fp16 与 fp32 的差别只在存储与中间
舍入，索引、布局、控制流完全相同，所以：

- **fp32 bit-identical** ⇒ 索引、内存布局、数学表达式全部正确，问题必定只在 fp16 舍入；
- **fp32 也不同** ⇒ 是真 bug（索引/布局/漏写元素），别再纠缠精度。

本例 fp32 一次就 bit-identical，直接把假设空间从"可能哪儿都错"缩到"只是 rounding"，
省掉了所有对索引的反复怀疑。**这一步应该排在 token 对拍之前。**

**第二步：分阶段 env 探针二分，定位是哪一半算术不同。** 每个阶段只改一件事（临时代码，
定位完删除）：

| 阶段 | 内容 | 本例结果 |
|---|---|---|
| 0 | matcher 关掉（注册仍在） | = 基线 ⇒ 注册本身惰性 |
| 1 | 只做内存提升，不装融合 | = 基线 ⇒ 提升无副作用 |
| 2 | 只装融合 leader，不 claim 任何 op | ≠ 基线 ⇒ 差异出自新 kernel 自身输出 |
| 3 | leader 退化成**纯搬运**（只做索引重排，不算数） | = 基线 ⇒ 读/写索引与 leader 机制全对 |
| 4 | 用链路的中间结果替换新 kernel 的**前半**计算 | = 基线 ⇒ 后半（SiLU + 乘法）精确 |
| 5 | 用链路的中间结果替换新 kernel 的**后半**计算 | ≠ 基线 ⇒ **差异只在前半（RMSNorm）** |

阶段 3 尤其值得单列：**先证明"纯搬运能精确复现基线"**，之后所有差异都可归给算术，
不必再怀疑索引。

**⚠️ 最大的坑：替换读源的探针必须给那个中间张量做 STATIC 提升。**
链路中间张量的生命周期在它原本的消费者处就结束，动态内存池随后可以回收；探针在更晚的位置
去读它，读到的是**已被覆写的脏数据**。本例第一轮因此拿到三个互相矛盾的 hash
（同一个问题测出三种结论），补上 `onAcquireBuffer(t, Backend::STATIC)` 后结论立刻自洽。
**探针不可靠时得到的一切结论都要作废重来。**

**其它省时经验**：
- 先把假设**算清**再测：本例怀疑 `fma` 收缩，但归约循环里 `channelUnit == SIMD_GROUP_WIDTH`
  意味着每 lane 只迭代一次，`0 + d*d` 与 `fma(d,d,0)` 恒等 —— 该实验注定无信息量，白跑一轮。
- **冷/热**：刚删 pipeline cache（`mnn_cachefile.bin`）的第一次运行与后续结果不同。
  本例一度把"第一次跑"与"后续跑"直接对比，得出"HEAD 自己都不确定"的错误结论。
  **所有对拍前先预热一次。**
- 结论落地时区分口径：若最终判定为编译器 codegen 层面的等价重排，验收口径就写成
  「fp32 bit-identical + fp16 确定性 + 质量/回归」，并在文档里明确它与「byte-identical」的差别，
  不要含糊带过。

## 1.8 物理内存增长：先按 VM 区域和分配栈归因

GPU workload 的 `phys_footprint` 增长不等于 GPU 资源泄漏。若 backend 自报的活跃分配已回落，
但进程物理内存仍线性增长，应先用系统 VM/heap 工具回答“活着的是哪类区域、谁分配的”，再改
context、allocator 或缓存策略：

1. 在每轮完整 teardown 后同时记录进程物理内存和 backend 活跃分配量；两者分离时不要继续只查 GPU。
2. 用 VM 区域汇总区分 IOAccelerator/Metal 映射与 `MALLOC_LARGE` 等 CPU heap；用 heap 大小分布找出
   与每轮增量匹配的重复 allocation。
3. 对一个代表性活块抓分配回溯，沿栈检查返回缓冲的所有权。调用方提供了外部 buffer，**不代表**
   callee 返回的指针一定与它别名；所有权判断必须基于实际返回指针或显式契约，而不能只看“曾传入非空指针”。
4. 修复后既要验证多轮 teardown 平台化，也要补覆盖“返回外部 buffer”和“返回新临时 buffer”两条路径的单测。

## 1.9 相关文件索引

| 文件 | 作用 |
|------|------|
| `source/core/BufferAllocator.hpp` | Arena 语义注释，是否复用 freed chunk 的官方描述 |
| `source/core/Backend.hpp` | Backend tensor buffer 生命周期接口 |
| `source/backend/cpu/CPULayerNorm.cpp` | 正确的批量 alloc-then-free-all 模式参考 |
| `source/backend/cpu/CPURoPE.cpp` | 参考案例的修复实现（onResize 注释里写了原因） |
| `source/backend/metal/MetalBackend.mm` | 跨 op tensor buffer 重叠检查（`matchQKVFusions`）；`matchLNFusions` 内的"写集合 × 读集合"字节区间别名检测 + STATIC re-home 是 §1.6 修复模式的参考实现 |
