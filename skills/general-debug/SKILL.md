---
name: bugfix
description: MNN 各类正确性/回归 bug 的排查方法论集合。按 bug 分类组织：当前覆盖 (1) 内存别名/生命周期错误（buffer aliasing、arena reuse、`MemChunk` / tensor buffer 生命周期）、(2) 量化误差/导出侧权重损坏（低 bit 打包、导出分块、PyTorch MPS/CUDA 大张量静默错误）、(3) 持久化缓存误信（weight-mmap sync 标记自我污染、陈旧/跨模型缓存复用）、(4) 数值精度/fp16 表示能力不足（长序列复读、position 塌缩、以及「实时计算→预计算查表」重构的三类陷阱），后续会补充并发竞争、图优化回归、Codegen/Shader 错等类别。
---

# MNN Bugfix 排查 Skill

> **触发**：MNN 中出现正确性 bug、单测/golden 对不上、回归；或做完改动（新 op、fusion、图 pass、后端 kernel）后行为异常。
>
> **使用方法**：先按下方 [Bug 分类导航](#bug-分类导航) 匹配症状定位到对应章节，每个章节独立可读；如果症状横跨多类，按导航表列出的顺序逐一排查。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

---

## Bug 分类导航

| # | 类别 | 典型症状 | 章节 |
|---|------|---------|------|
| 1 | **内存别名 / 生命周期** | 数值错乱、乱码 token、NaN；代码逻辑看着正确、指针地址合法；换后端结果不同；关掉某优化就好；单/多线程行为差异 | [§1](#1-内存别名--生命周期错误) |
| 2 | **量化误差 / 导出侧权重损坏** | 低 bit（Q4）下输出乱码或退化、Q8/更高 bit 正常；**所有推理后端都错**（CPU/Metal 一致地错）；torch 侧 `--test` 正常但 MNN 推理错；只有大模型/大 vocab 触发 | [§2](#2-量化误差--导出侧权重损坏) |
| 3 | **并发 / 线程竞争**（GPU 侧已覆盖） | 结果不稳定、每次运行不同；单线程稳定但多线程随机错。**GPU 上逐次不同**已有专门案例（融合引入的别名竞争 + 逐 op commit 二分法） | [§1.6](#16-参考案例融合引入的别名竞争layernorm-折进-conv1x12026-08-03)；CPU 多线程分片类仍待补 |
| 4 | *（待补：图优化回归）* | 某个 converter pass 之后模型跑错，disable 该 pass 后正常 | *待补* |
| 5 | **数值精度 / fp16 表示能力不足** | 长 prompt 输出重复/漂移/退化，短 prompt 正常；fp16 后端错、fp32（`precision=high`）对；**所有 fp16 后端一致地错**；出错阈值是 2 的幂（2048/4096） | [§5](#5-数值精度--fp16-表示能力不足) |
| 6 | **GPU Shader 越界 / Command Buffer 故障** | `[METAL] command buffer error` 后速度假快数百倍；只在某 shape 阈值之上触发；关某条 kernel 路径 env 后消失；或同越界在别的模型上表现为静默数值损坏 | [§6](#6-gpu-shader-越界--command-buffer-故障) |
| 7 | **后端 kernel 隐式假设违反** | 某类模型（如 SWA/prefix LM/bidirectional）静默输出乱码或语义偏移；标准 causal LLM (Qwen/Llama) 正常；调整某个"看起来无关"的性能开关（如 `MNN_METAL_QK_CAUSAL_TRI=0`）后就好 | [§7](#7-后端-kernel-隐式假设违反) |
| 8 | **持久化缓存误信（weight-mmap / 陈旧缓存）** | 开 `use_mmap` 才乱码（单字符刷屏）；App 内错、`llm_demo` 对；真机错、host 对；清缓存目录/换 `tmp_path` 就好；换模型不换目录后乱码 | [§8](#8-持久化缓存误信weight-mmap-cache--陈旧缓存) |

> 新增章节时同步在这张表里补一行；每个章节命名为 `## <编号> <类别名>`，保持编号递增。

> **后端专属 skill 优先**：若问题明确落在 **QNN / NPU（高通 HTP）** 后端——结果不对/精度差、报错 `1002/6000/1003/6004`、`graphFinalize/graphExecute` 失败、`validateOpConfig failed`、某算子 QNN 不支持、LLM 在 NPU 上乱码、或要给 QNN 新增/适配算子——**直接转 [`qnn-debug`](../qnn-debug/SKILL.md) skill**（含 QNN 两条执行路径、中间张量 dump 定位法、误差模式与算子约束速查），不要在本文件里从头排查。本文件覆盖的是**跨后端通用**的正确性 bug 方法论。

---

## 通用排查原则（所有类别共用）

1. **先复现最小化**：定住能稳定复现的最小 case（最短 prompt、最小 shape、单线程），然后逐步添加变量。
2. **两向 A/B**：换后端、换开关、换编译选项、换线程数 —— 任一维度上"这边错那边对"都是强线索。
3. **别信"这块应该是独立的"，用直接观察去证明**：地址、值、时序都要用打印/断点去看，不要靠推理。
4. **修改先做假设，验证后再改代码**。改一版跑一版，避免"多改一起观察但不知道哪个生效"。
5. **改完记录**：如果本次 bug 有可复用的教训，追加到本文件对应章节的"参考案例"里；如果发现新的 bug 类别，新开一节并更新导航表。
6. **导出/序列化崩溃先查 I/O 边界**：用回溯定位第一个文件写入点；converter 入口先创建并验证父目录，`fopen` 失败必须立即返回，禁止把空 `FILE*` 传给 `fwrite`。不要把这种晚发的 SIGSEGV 误判成模型或后端问题。
7. **区分建图与执行阶段的资源绑定**：对外部 backend API，先确认 graph tensor 创建时允许哪些字段，client buffer、memory handle 等运行时资源只在 API 要求的阶段绑定；在至少两个 SDK/SoC 组合上检查建图日志，不能以某个版本的宽容行为代替契约。

---

## §1 内存别名 / 生命周期错误

**触发**（满足以下之一强烈怀疑本类）：
- 输出乱码 / 数值明显错乱，但 kernel / op 单跑对；
- CPU 后端错、其它后端（Metal / GPU）对，或反之；
- 关掉某个新加的 op / 优化 pass 就好，看着又没写错逻辑；
- `onResize` / buffer 分配相关代码改过后开始回归；
- 加 `printf` 或改 buffer size 就"好了"。

### 1.1 核心心法

**"地址看着都对但结果错" ≈ 内存别名或生命周期错误。** 遇到这种症状**优先怀疑内存复用**，不要先去怀疑算法本身。这类 bug 的共同特征：

- 单个 kernel / op 单独跑对，串起来错；
- 打印指针发现"都是合法地址"，但内容互相污染；
- 加 `printf` 或改 buffer 大小就"好了"（其实只是位置错开了）；
- 关线程或改线程数症状变化（并发 + 别名一起放大）。

**方法论一句话**：**别信"这块 buffer 应该是独立的"，用地址等式去证明**。

### 1.2 MNN 内存模型快速回顾

进入排查前先牢记 MNN 的三条内存复用语义（拿不准就去读 `source/core/BufferAllocator.hpp`、`source/core/Backend.hpp`）：

#### (a) `BufferAllocator` 是 arena，`free` **不释放**

- `alloc(size)`：如果 free pool 里有相同 size 的 chunk，**直接复用地址**；
- `free(chunk)`：**只是把这块标记为可复用**，物理内存保留；
- 交错的 `alloc(); free(); alloc(); free()` 模式，只要 size 相同，第二次 `alloc` 必然拿回第一块。

#### (b) Backend tensor buffer 会被跨 op 复用

- `onResizeBegin` / `onResizeEnd` / `compute()` 之间，pipeline 会根据 tensor 生命周期把互不重叠的 tensor 分配到同一块 backend buffer；
- 同一个 `Tensor*` 在不同 op 里的物理 buffer 可能不同；不同 `Tensor*` 也可能共享同一物理 buffer；
- 这在 GPU 后端（Metal `id<MTLBuffer>`、CUDA `void*`）同样会发生。

#### (c) `MemChunk` 的 `ptr()` 仅在 lifetime marker in-use 期间稳定

- `onResize` 里申请的 chunk，`onExecute` 期间用它的 `ptr()` 是安全的（op 内时序保证）；
- 但同一 op 内**两个 MemChunk 之间**是否别名，完全取决于 alloc/free 次序。

### 1.3 排查流程

#### Step 1: 复现并最小化

- 定住一个能稳定复现的最小 case（最短 prompt、最小 shape、单线程）。多线程先关掉，避免并发遮盖别名症状。
- 记录基线：换个后端跑同 case 是不是对？把新加的 op / pass / fusion 关掉（用 env 开关 / cmake option）是不是对？**"另一个后端对、这个后端错"是内存类 bug 的强烈信号** —— 同一段算法在两个后端上的差异，很多时候只在于内存模型不同。

#### Step 2: 用地址等式定位别名（关键手法）

在怀疑的 op 的 `onExecute` 入口打印所有 scratch chunk / tensor 的**物理地址**：

```cpp
MNN_PRINT("[MyOp] cos=%p sin=%p qTmp=%p kTmp=%p out=%p\n",
          mTmpCos.ptr(), mTmpSin.ptr(), mTmpQC4.ptr(), mTmpKC4.ptr(), output->host<void>());
```

判据：

- **两个"逻辑上独立"的 buffer 地址相等 → 100% 别名**，直接进 Step 3；
- 地址不等但差得很近（相邻 chunk）→ 可能是越界写而不是别名，改看 Step 5；
- 地址完全无关但结果仍错 → 别名可能不在这几个 buffer 里，扩大打印范围（覆盖 `onResize` 里所有 `alloc`，包括子调用 `MNNNorm` / `MNNLowpToFp32` 之类内部若也走 allocator 需要一起看）。

#### Step 3: 从 `onResize` 找根因

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

#### Step 4: 跨 op tensor 别名（fusion / 图重排场景）

当症状出现在多个 op 之间（比如做了 QKV / Gate-Up fusion 之后），要检查 backend 侧的 tensor buffer 是否被 pipeline 分配到了同一物理 buffer：

- Metal 侧参考 `MetalBackend.mm` 里 `matchQKVFusions` 的做法 —— **在 `onResizeEnd` 的 `compute()` 之后**再检查 output buffer 是否重叠，重叠就 fallback；
- 调用顺序至关重要：只有 `compute()` 之后才知道实际分配结果，反过来做检查等于空跑；
- 若图重排后仍重叠，需要在 converter 侧调整 op 顺序（如 `reorderQKVProjections`），或直接放弃这次 fusion。

#### Step 5: 越界写 / 未初始化

如果地址不别名但结果仍错，考虑：

- **越界写**：某个 kernel 按 `numHead * headDim` 写，但 `alloc` 只按 `headDim` 或漏了 `threadNumber` 倍数。快速验证：把 scratch buffer size 翻倍再跑，如果症状消失就是越界。
- **未初始化 + arena reuse**：新 `alloc` 拿回来的是别的 op 用过的脏数据。fp32 / fp16 里的脏数据可能不是 NaN 而是"看着正常的小数"，症状是"结果略微不对"。
- **多线程 tId 分片错**：`chunk.ptr() + tId * stride` 里 `stride` 少乘一个维度，导致相邻线程互踩。单线程能过就是这个。

#### Step 6: 生命周期错位（栈上临时 buffer）

- 不要在 `onResize` 里把 `std::vector` / 栈数组的地址存到成员里，然后在 `onExecute` 里用 —— `onResize` 返回后那块内存已失效。
- scratch 一律走 `BufferAllocator` 或成员变量的 `std::vector`（且要 `resize` 而不是 `reserve`）。

### 1.4 常见对照表：症状 → 优先怀疑

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

### 1.5 参考案例：CPU inv_freq RoPE scratch 别名

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

### 1.6 参考案例：融合引入的别名竞争（LayerNorm 折进 Conv1x1，2026-08-03）

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

### 1.7 参考案例：验证「融合是否数学等价」——用 fp32 当 oracle（2026-08-04）

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

### 1.8 相关文件索引

| 文件 | 作用 |
|------|------|
| `source/core/BufferAllocator.hpp` | Arena 语义注释，是否复用 freed chunk 的官方描述 |
| `source/core/Backend.hpp` | Backend tensor buffer 生命周期接口 |
| `source/backend/cpu/CPULayerNorm.cpp` | 正确的批量 alloc-then-free-all 模式参考 |
| `source/backend/cpu/CPURoPE.cpp` | 参考案例的修复实现（onResize 注释里写了原因） |
| `source/backend/metal/MetalBackend.mm` | 跨 op tensor buffer 重叠检查（`matchQKVFusions`）；`matchLNFusions` 内的"写集合 × 读集合"字节区间别名检测 + STATIC re-home 是 §1.6 修复模式的参考实现 |

---

## §2 量化误差 / 导出侧权重损坏

**触发**（满足以下之一强烈怀疑本类）：
- 低 bit（Q4）导出输出乱码/退化，Q8 或 fp 导出完全正常；
- **换任何推理后端都错**（CPU / Metal / master 分支 runtime 一致地错）—— 与 §1 的"一个后端错一个后端对"正好相反；
- `llmexport.py --test "<query>"`（torch 侧 rebuilt 模型，不走量化打包）输出正常；
- 只在特定模型（大 vocab、大 hidden）上触发，同流程导小模型正常；
- MNN op 单测（`run_test.out`）全过。

### 2.1 核心心法

**"所有后端一致地错 + torch 侧对 + 单测过" ≈ 导出产物（权重文件）本身坏了。** 此时不要在推理引擎里找 bug，而是：

1. 用导出参数做**量化 bisect**，定位哪个权重坏；
2. **离线反量化导出文件，与 HF 原始权重逐行比对**，把"导出坏"与"运行时 dequant 坏"分开；
3. 警惕**框架静默错误**：PyTorch 加速后端（MPS/CUDA）在超大张量上可能不报错、直接给错结果。

### 2.2 相关背景

- 导出量化入口：`transformers/llm/export/utils/torch_utils.py::quant`（大权重按 `_QUANT_MAX_ELEMENTS`=256M 元素沿 oc 分块），实际量化在 `_quant_on_device`（优先 CUDA→MPS→CPU）。
- 权重写文件：`utils/mnn_converter.py::build_weight`（header + q_weight + alpha [+ bias]）；lm_head/tie_embeddings 的 offset 信息写进 `llm_config.json` 的 `tie_embeddings` 字段。
- Q4 打包格式：每 byte 两个权重，**高 nibble 在前**；asym alpha 布局为每 (oc,block) 一对 `[zero, scale]`；引擎 dequant 公式 `w = q * scale + (zero + offset * scale)`，`offset = -(1 << (bit-1))`（见 `transformers/llm/engine/src/diskembedding.cpp`）。
- 可用的 bisect 旋钮：`--quant_bit/--quant_block`（body）、`--lm_quant_bit/--lm_quant_block`（lm_head 单独控制）、`--quant_config <json>`（任意 op 级覆盖，如 `{"/lm/lm_head/Linear": {"bits": 8, "block_size": 0}}`）、`--hqq`、`--seperate_embed`。

### 2.3 排查流程

#### Step 1: 确认是导出侧而非推理侧

三个证据凑齐即可确认：① CPU 和 GPU 后端**一致地**错；② `--test` torch 侧输出正常（权重映射没问题）；③ op 单测全过（kernel 没问题）。

#### Step 2: 量化 bisect —— 用导出参数二分定位坏权重

每次只动一个变量，导出后跑 CPU greedy 与 HF golden 对比：

| 实验 | 命令要点 | 用于区分 |
|------|---------|---------|
| 全 Q8 | `--quant_bit 8 --quant_block 0` | 是否 4bit 特有 |
| 去 hqq | 去掉 `--hqq` | 是否 hqq 引入 |
| body Q4 + lm_head Q8 | `--hqq --lm_quant_bit 8 --lm_quant_block 0` | 是否 lm_head |
| body Q8 + lm_head Q4 | `--quant_bit 8 --lm_quant_bit 4` | 反向确认 lm_head |
| 某类层强制 8bit | `--quant_config` 指定 op 列表 | 是否特定层类 |
| block 0 vs 64 | `--lm_quant_block 0/64` | 是否 block 量化特有 |

本案例结论链：Q8 全对 → body Q4 + lm_head Q8 对 → body Q8 + lm_head Q4 错 → **锁定 lm_head Q4**；block 0/64 都错 → 与 block 无关，是 4bit 打包本身。

#### Step 3: 离线反量化，与 HF 权重逐行比对（关键手法）

写脚本读 `llm.mnn.weight`，按 `llm_config.json` 的 `tie_embeddings`（`weight_offset/alpha_offset/alpha_size/quant_bit/quant_block/alpha_dtype`）反量化若干行，与 HF `embed_tokens.weight` 算 cosine：

- **先用一个已知正确的导出（如 Q8）验证脚本方法学**（cos 应全部 ≈1），再去测坏的导出 —— 否则分不清是权重坏还是自己反量化约定写错；
- 抽样行要覆盖头、中、尾（本案例：row 0/1/100 全乱、row 151645/248319 正常 —— "部分行坏"是关键线索）；
- 对坏区做**二分搜索找边界**：本案例第一个正常行 = 131072，边界字节偏移 = 2^27，精确的 2 的幂 → 几乎必然是溢出/截断/框架 bug，不是量化算法问题；
- 看坏区字节内容：本案例全 0（不是随机垃圾）→ 说明是"某一步整体输出了 0"，不是错位/串行。

#### Step 4: 在导出代码里复现最小 case

边界 2^27 字节 = 2^28 个 int4 元素，恰好等于分块大小 256M 元素 → 怀疑 `_quant_on_device` 对满块输入出错。直接用随机张量复现：

```python
qw_mps, _ = _quant_on_device(w.to('mps'), 4, 0, False, False, False)   # 131072 x 2048
qw_cpu, _ = _quant_on_device(w, 4, 0, False, False, False)
# mps 输出全 0，cpu 正常 → 框架 bug 坐实
```

再逐 op 拆：`(q.reshape(-1,2) * m).sum(axis=1)` —— **乘法结果正常，`sum(axis=1)` 在 MPS 上对 ≥2^28 uint8 元素静默返回全 0**（uint8→int64 归约溢出类 bug，2^27 元素正常）。

#### Step 5: 修复原则

- **绕开出错的框架 op**，用等价的安全写法（本案例：uint8 逐列 `packed |= col << shift` 按位累加替代 `sum`），不要指望升级框架版本；
- 修复后必须验证 **加速后端与 CPU 输出逐字节相等**（`torch.equal`），再全量重导 + 端到端跑 CPU 和 GPU greedy。

### 2.4 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的原因 |
|------|-------------|
| 所有后端一致错 + torch `--test` 对 | 导出产物权重坏（本节） |
| Q4 错 Q8 对 | 4bit 打包路径 bug（Q8 不走打包） |
| 只有大 vocab / 大 hidden 模型触发 | 大张量分块边界 / 框架大张量静默 bug |
| 反量化比对"部分行坏部分行好"，边界是 2 的幂 | 溢出 / 截断 / 框架归约 bug |
| 坏区全 0（非随机垃圾） | 某步整体输出 0（归约/拷贝失败），非量化误差 |
| 反量化比对全部行都乱 | 先怀疑自己的反量化约定（nibble 序、zero/scale 布局），用 Q8 导出验证脚本 |

### 2.5 参考案例：Qwen3.5-2B Q4 lm_head 全零（PyTorch MPS sum 归约 bug）

**症状**：Qwen3.5-2B Q4+hqq 导出，Metal 和 CPU greedy 都输出乱码；Q8 导出完全正常；`--test` torch 侧正常；op 单测全过。最初被当作"Metal 乱码问题"排查。

**排查路径**：
1. CPU 也错、master runtime 也错 → 排除 Metal 后端和分支回归；
2. 量化 bisect（Step 2 表格逐项）→ 锁定 lm_head Q4（body Q8 + 仅 lm_head Q4 即崩）；block 0/64 都崩 → 非 block 问题；
3. 离线反量化（先用 Q8 导出验证脚本，cos 全 ≈1）→ Q4 导出 row 0~131071 全坏、之后全好；
4. 二分边界 = 精确 131072 行 = 2^27 字节；坏区字节全 0；
5. 131072 = `_QUANT_MAX_ELEMENTS // ic` = 第一个量化分块 → 随机张量复现：MPS 上 2^28 元素 Q4 打包输出全 0，CPU 正常；
6. 逐 op 拆解 → `sum(axis=1)` 是罪魁：**MPS 对 ≥2^28 个 uint8 元素的归约静默返回全 0**（alpha 正常，所以只有权重坏）。

**根因**：lm_head 248320×2048 ≈ 508M 元素 > 256M 分块上限，第一块恰好 2^28 元素；`torch_utils.py` Q4 打包用 `(q_weight * multipliers).sum(axis=1)`，该 sum 在 MPS 上触发框架 bug → 前 131072 行 lm_head 权重全零 → logits 大面积错乱 → 乱码。小模型（如 qwen3-0.6b）vocab 小、不触发分块满 2^28，因此从未暴露。

**修复**（`transformers/llm/export/utils/torch_utils.py`）：

```python
# 旧：q_weight = (q_weight * multipliers).sum(axis=1).to(torch.uint8)   # MPS 大张量静默全 0
# 新：uint8 逐列按位累加，全程不发生 dtype 提升，绕开归约
packed = torch.zeros(q_weight.shape[0], dtype=torch.uint8, device=q_weight.device)
for i in range(group_size):
    shift = quant_bit * (group_size - 1 - i)
    packed |= q_weight[:, i] << shift
q_weight = packed
```

验证：`torch.equal(mps结果, cpu结果) == True`；重导 Q4+hqq 后 CPU（58 tok/s）与 Metal（90 tok/s）greedy 输出均正确。

**避坑要点**：
- "乱码"不一定是推理引擎/后端问题 —— **先用"所有后端是否一致地错"分流**：一致错查导出，不一致错查后端内存（§1）；
- 加速后端（MPS/CUDA）的大张量 op 可能**不报错、给全 0** —— 关键路径的量化/打包结果要有与 CPU 的一致性校验意识；
- 反量化比对脚本一定要先在已知正确的导出上自校准。

### 2.6 相关文件索引

| 文件 | 作用 |
|------|------|
| `transformers/llm/export/utils/torch_utils.py` | 量化 + 低 bit 打包（本案例修复处，含 256M 分块逻辑） |
| `transformers/llm/export/utils/mnn_converter.py` | `build_weight` 写权重文件、`write_header`、tie_embeddings 信息 |
| `transformers/llm/export/utils/hqq_quantizer.py` | hqq 量化实现 |
| `transformers/llm/engine/src/diskembedding.cpp` | 引擎侧 Q4/Q8 dequant 参考（nibble 序、alpha 布局、offset 公式） |
| `transformers/llm/engine/src/llmconfig.hpp` | `tie_embeddings` 字段解析 |

---

## §5 数值精度 / fp16 表示能力不足

**触发**（满足以下之一强烈怀疑本类）：
- 长 prompt / 长上下文输出**重复、漂移、退化**，短 prompt 完全正常；
- fp16 后端错、强制 fp32（config `"precision": "high"`）对；
- **所有 fp16 后端一致地错**（Metal 和 CPU arm82 同样错）—— 与 §1 的"一个后端错一个对"相反；
- 出错阈值恰好是**2 的幂**（2048 / 4096 / 8192）；
- torch 侧 `--test` 正常（torch 用 fp32 跑）。

### 5.1 核心心法

**"fp32 对 / fp16 错 + 只有长序列错 + 阈值是 2 的幂" ≈ 图里存在动态范围过大的中间张量。**

fp16 只有 10 bit 尾数，**整数的精确表示上限是 2048**（2^11）：

| 数值区间 | fp16 可表示的最小间隔 |
|---|---|
| < 2048 | 1（精确） |
| 2048 ~ 4096 | 2 |
| 4096 ~ 8192 | 4 |
| 8192 ~ 16384 | 8 |

也就是说 fp16 里 `2048.0` 和 `2049.0` 是**同一个数**。任何把"大整数"或"大整数 × 系数"作为中间张量烘进计算图的做法，在 fp16 后端都会让相邻取值塌缩成 bit 完全相同的结果。

而 LLM 的 `position_ids` 可以到 128k，远超这个上限。

**方法论一句话**：**沿数据流找"绝对值最大的中间张量"，把它改写成值域小的等价形式**——不要指望后端"精度高一点"能解决，这是表示能力的硬上限。

### 5.2 相关背景

- MNN 的 fp16 由 backend precision 决定：Metal 默认 fp16、CPU 走 arm82 时也是 fp16。config 里 `"precision": "high"` 可强制 fp32。
- 导出侧任何写成 `x.float() * const` 的表达式都会变成图里一个真实的中间张量，**它的值域就是后端要承受的动态范围**。导出期用 torch fp32 验证是发现不了的。
- LLM 里典型的大值域中间量：`position_ids`（0~128k）、`position * inv_freq`（RoPE 角度，可达上万弧度）、未归一化的 logits、累加型 reduce 的中间和。

### 5.3 排查流程

#### Step 1: 用 precision 开关分流（第一步，代价最小）

```bash
# 在 config.json 里加 "precision": "high" 强制 fp32
./llm_demo config_fp32.json prompt.txt
```

fp32 对、fp16 错 → 基本坐实本类，不用再去查 kernel 逻辑或内存。

#### Step 2: 长度扫描找阈值

```bash
for L in 512 1024 2048 3000 4096 8192; do
  echo "=== len=$L ==="; ./llm_demo config.json /tmp/prompt_${L}.txt 32
done
```

**阈值落在 2 的幂上是最强的信号**。若阈值 = 2048/4096，直接对照 5.1 的间隔表反推是哪个量越过了精确表示区间。

#### Step 3: 逐 step logits 对拍，定位第一个发散点

用 `llm_logits_diff` 工具（`transformers/llm/engine/demo/llm_logits_diff.cpp`）做**teacher-forced** 对比——A 后端的 argmax 同时喂给两个模型，保证每一步比较的是同一 KV/history 状态下的 logits：

```bash
./llm_logits_diff config_fp32.json config_fp16.json prompt.txt 64
```

输出每步的 argmax 是否一致、margin、maxAbsDiff、KL(A||B) 和 teacher-forced NLL。关键读法：
- **前 N 步 byte-identical、第 N+1 步突然发散** → 找出第 N+1 步对应的绝对位置，往往正好是阈值；
- KL 逐步单调放大 → 累积型误差；KL 在某步跳变 → 表示能力塌缩（本类）。

#### Step 4: 定位是哪个中间张量

在导出侧 dump 候选中间量的绝对值上界，找出超过 2048 的那个。RoPE 场景直接看 `position * theta` 的量级：position 到 128k、theta 最大接近 1 → 角度可达 1e5 弧度，远超 fp16 的整数精确区。

#### Step 5: 改写成小值域等价形式

不是降精度要求，而是**做数学等价变换让所有中间量都落进 fp16 的舒适区**。常用手法：

- **整数-余数拆分 + 查表**：`p = S*q + r`（整数运算精确），再用角度和公式合成，所有中间量落在 `[-1, 1]`；
- **提前折叠周期**：三角函数按 `mod 2π` 折叠（在 float64 下算完再降精度）；
- **减去最大值再 exp**：softmax 的标准做法，同类思路；
- **保持整数就是整数**：不要过早 `.float()`，整数索引类的量一路用 int 传到 Gather。

### 5.4 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的原因 |
|------|-------------|
| fp32 对 / fp16 错，长序列才错 | 中间张量动态范围超 fp16（本节） |
| 阈值恰为 2048 / 4096 | 整数在 fp16 里塌缩（5.1 间隔表） |
| 所有 fp16 后端一致地错 | 图结构问题（导出侧），不是某后端 kernel |
| 输出"大段重复" | RoPE / 位置编码相关（相邻位置塌缩成同一个） |
| 误差随步数单调放大 | 累积误差，非表示能力（可能是正常的 fp16 噪声） |
| 换 fp32 仍错 | 不是本类，回查 §2（导出权重）或 §7（kernel 假设） |

### 5.5 参考案例：RoPE position 在 fp16 下塌缩（长中文 prompt 大段重复）

**症状**：长中文 prompt 回答出现大段重复；短 prompt 正常；fp32 正常；Metal 与 CPU arm82 一致地错。

**根因**：导出侧 RoPE 把 `position_ids.float() * theta` 直接烘进图。position ≥ 2048 后 fp16 无法精确表示，相邻位置产生 **bit 完全相同**的 cos/sin —— 模型对"第 3000 个 token"和"第 3001 个 token"的位置感知完全一致，4096 以后（间隔 4）彻底失去位置区分能力 → attention 退化 → 复读。

**修复**：把位置保持为整数，拆成 `p = 2048*q + r`（整数运算精确），用两张预计算表 + 角度和公式合成：

```
angle(p) = q*(2048*theta) + r*theta   (mod 2pi)
cos(p)   = cosH[q]*cosL[r] - sinH[q]*sinL[r]
sin(p)   = sinH[q]*cosL[r] + cosH[q]*sinL[r]
```

表的角度在 **float64** 下折叠进 `[0, 2π)` 后再降 fp32。这样后端接触到的每个张量都在 `[-1, 1]`，fp16 精度 ~6e-4，只剩无害的相位抖动。

**避坑要点**：这个 bug 无法通过 review 逻辑代码发现——`position * theta` 数学上完全正确，torch fp32 下验证也完全正确。**必须靠"fp32/fp16 A/B + 阈值是否 2 的幂"来揭穿。**

### 5.6 「实时计算 → 预计算查表」重构的三类陷阱（重要）

5.5 的修复把"每次实时算"改成了"构造期预计算 + 运行期查表"。这是一类通用优化手法（也见于各种 LUT 化、常量折叠、预烘焙），**它本身会引入三个新的 bug 类别**，全部在 code review 阶段才被发现。改这类代码时逐条自查：

#### 陷阱 A：构造期固化的数据与后续参数修改脱钩（最严重）

**机制**：改造前 `forward()` 读 `self.theta`，外部改 `theta` 立刻生效；改造成查表后，表在 `__init__` 里烘焙，`forward()` 不再读 `self.theta`——**任何在构造之后修改输入参数的代码路径都会静默失效**。

**实例**：`utils/model.py` 的 Gemma3/Gemma4 dual-RoPE 路径先 `Rotary(full_config)` 建表（`head_dim=256` → 表宽 128），再按 `partial_rotary_factor=0.5` 改 `rotary_dim=128` 和 `theta`。表还是旧的 → 生成全宽 RoPE，把本该 pass-through 的维度也旋转了，且 theta 频率分布也错 → 导出模型输出胡言乱语，**不报错、不崩溃**。

**自查清单**：
- `grep` 所有对该对象属性的外部赋值（`\.theta\s*=`、`\.rotary_dim\s*=`），确认没有发生在构造之后；
- 若必须允许后置修改，把建表提成**公开方法**（如 `build_rope_tables()`）并要求调用方改完显式重建；注释里写明"Must be re-called by anyone mutating X"；
- 检查**子类**：子类 `__init__` 里 `super().__init__()` 之后改参数，是同一个陷阱。（本案例中 `DitRotary`/`OmniRotary`/`VisionRotary` 恰好都完整覆写了 `forward()` 不走查表路径，才躲过一劫——这是运气不是设计。）

#### 陷阱 B：导出图丢失了框架的边界检查

**机制**：查表 = `embedding` / `Gather`。**PyTorch 的 `embedding` 有边界检查会抛 `IndexError`，但导出成 MNN 图后的 Gather 算子不做边界检查**——越界索引直接读表外内存，把垃圾当数据用。

**后果特征**：导出期一切正常（torch 会拦），线上真机长上下文才爆，且不是崩溃而是静默读脏数据。这是最难排查的组合。

**自查清单**：
- 表的行数是否覆盖索引的**理论最大值**，而不只是"典型值"？（`max_position_embeddings` 不是硬上限，NTK / 用户改 config 外推都会超）；
- 高表很小的时候直接**放大预留**：本案例高表预留 `4 * max_pos`，增量成本仅 ~96KB（d=128/128k 模型高表 33KB → 129KB），预留范围内的角度是数学精确的；
- 注意算清**哪张表是大头**：低表行数固定 2048、与 `max_pos` 无关，`2048 × (rotary_dim/2) × 4B` 才是主要开销（d=128 时两张低表共 1MB，d=256 时 2MB）；4 张表合计 d=128/128k 约 1.1MB、d=256/128k 约 2.25MB。放大高表预留几乎免费，放大低表则不然；
- 再在 `forward()` 里加 `clamp` 作为**最后兜底**——但注意陷阱 C。

#### 陷阱 C：加 clamp 会破坏与之配对的运算

**机制**：为修陷阱 B 给索引加 `clamp`，会让原本互相配对的两个量**解耦**。

**实例**：原代码 `q = floor(pos/2048)`、`r = pos - q*2048`。对负 `pos` 是安全的（floor 除法和减法天然配对：`pos=-1` → `q=-1` → `r=2047` ✓）。但一旦给 `q` 加 `clamp(0, ...)`，clamp 把 `q` 从 -1 抬到 0，`r = -1 - 0*2048 = -1` → **负索引越界**。修 B 的动作直接制造了新的越界。

**修法**：让配对量各自独立成立，不要依赖对方。本案例 `r` 改用 `torch.remainder(pos, 2048)`——数学模运算，结果恒在 `[0, 2048)`，与 `q` 怎么 clamp 完全无关。

**自查清单**：给某个量加钳制/饱和后，`grep` 所有用到它的表达式，逐个确认不变量是否仍成立。

#### 验证方式

这三类陷阱都不产生异常，必须主动验证：

```python
# 1. 表宽/表长与最终参数一致（陷阱 A）
print(r.rotary_dim, r.rope_cos_low.shape[1])   # 应为 rotary_dim//2

# 2. 极端索引不越界（陷阱 B/C）
for p in [0, 2047, 2048, max_pos+5000, 999999, -1]:
    q, r_ = ...; assert 0 <= q < high_entries and 0 <= r_ < split

# 3. 范围内与实时计算逐点对齐（保证等价变换没写错）
ref = torch.cos(pos.double().reshape(-1,1) * theta.double()).float()
assert (table_out - ref).abs().max() < 1e-6
```

### 5.7 相关文件索引

| 文件 | 作用 |
|------|------|
| `transformers/llm/export/utils/transformers.py` | `Rotary.build_rope_tables()` / `forward()`——本案例修复处，注释里写了 fp16 拆分原理 |
| `transformers/llm/export/utils/model.py` | Gemma3/Gemma4 dual-RoPE 后置修改 theta 的位置（陷阱 A 实例） |
| `transformers/llm/engine/demo/llm_logits_diff.cpp` | 逐 step teacher-forced logits 对拍工具（Step 3） |
| `transformers/llm/engine/src/llm.cpp` | `gen_position_ids`——超过 `max_position_embeddings` 时打 warning |
| `transformers/llm/engine/src/llmconfig.hpp` | `max_position_embeddings()` 等 config 字段解析 |

---

## §6 GPU Shader 越界 / Command Buffer 故障

**触发**（满足以下之一强烈怀疑本类）：
- 运行中出现 `[METAL] command buffer error`（`InnocentVictim` / `SubmissionsIgnored` / GPU restart），此后速度数字**假快数百倍**（forward 实际没跑）；
- 只在**某个 shape 阈值之上**触发（某 kv 长度、某模型 head_dim、某 batch），阈值之下完全正常；
- 关掉某条 kernel 路径的 env 开关后消失；
- ⚠️ 反面：同样的越界在别的模型上可能**不崩而是静默数值损坏**（踩到的是已映射内存）——"没崩"不等于"没越界"。

### 6.1 排查流程（按成本从低到高）

1. **先看第一条错误，别被 victim 骗**：`InnocentVictim`/`SubmissionsIgnored` 都是**受害者**代码，真凶 buffer 常常不在日志里。不要基于 victim 的 op 去猜。
2. **用 env 开关把"路径"与"触发变量"解耦**（判别性探针，代价一次 run）：逐个关可疑路径（如 `MNN_METAL_DECODE_SDPA=0`、`MNN_METAL_DISABLE_REPLAY=1`）看谁消失。⚠️ 注意"关 A 消失"不等于"A 是根因"——replay 常只是放大面；要看**最小共同集**（本案例：0.8B 关 replay 也好，2B 只有关 splitkv 才好 ⇒ 根在 splitkv）。
3. **解耦相关变量**：阈值型触发常有多个共变量（kv 长度 ↔ nwg=ceil(kv/256)）。用 pin 类旋钮做 2×2（当年是 `MNN_METAL_DECODE_SPLITKV_NWG=19/20` × 安全/故障 kv）——本案例一轮就锁定"nwg>16 而非 kv 本身"。⚠️ 本案例的 split-KV 路径及其 `MNN_METAL_DECODE_SPLITKV` / `_NWG` 两个 env **已于 2026-07-30 删除**（收敛到单 pass `MNN_METAL_DECODE_SDPA`）；方法论照用，具体开关名以 `metal-optimize/env-registry.md` 现表为准。
4. **Metal Shader Validation 拿实锤**（最有力，几分钟）：
   ```bash
   MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
   MTL_SHADER_VALIDATION_FAIL_MODE=allow <重现命令，n 可缩到 32>
   ```
   直接报 **kernel 名 + 越界 offset**（`Invalid device store at offset N, executing kernel "xxx"`）。
5. **对 offset 做算术反推**：拿第一个非法 offset 除以已知 stride，反推 kernel 以为的 buffer 尺寸 vs 实际分配尺寸。本案例：非法 offset ≈133120B = `8×32×(128+2)×4B`——正好是"元素数对、字节数减半"，直指 fp16 后端 `createDevice<float>` 按 2B 存储的陷阱（`metal-optimize/kernel-dev-and-optimize.md` 陷阱 F）。
6. **加一次性尺寸日志坐实**（分配处 + dispatch 处各一行，打印 elementSize / MTLBuffer.length / 索引参数），修复后删除。
7. **修复验证矩阵**：validation 0 OOB + 原故障配置 e2e 零错误 + greedy 对拍（⚠️ 用 metal+greedy config，见 `metal-optimize/build-and-test.md` Step 1.5——本案例曾被默认 mixed-sampler config 污染出一个假 bug）+ `run_test.out` 全过。

### 6.2 测试覆盖教训

阈值型 bug 能长期潜伏是因为**测试矩阵恰好停在结构阈值上**：splitkv 的 nwg 在 kv=4096 时恰为 16（越界临界），而历史性能/对拍全部 ≤p2048~p4096。**改 kernel 后的覆盖至少要跨过它的每个结构常数边界**（nwg cap、tile 对齐、tg mem 档位），各取"边界±1"各测一档。

### 6.3 参考案例：split-KV partial buffer 半长分配（2026-07-29，`6975fa71e7`）

`mTempSplitKV` 用 `createDevice<float>` 分配、shader 按 `device float*` 写：fp16 后端下存储 2B/元素 ⇒ buffer 半长，nwg>16（kv>4096）越界。HD=256（Qwen3.5）撞未映射页 → GPU 故障链；HD=128（Qwen3）同条件仅静默损坏。修复 = 按字节分配（`createDevice<uint8_t>`，公式显式 `* sizeof(float)`）。完整陷阱条目见 `metal-optimize/kernel-dev-and-optimize.md` 陷阱 F。

---

## §7 后端 kernel 隐式假设违反

**触发**（满足以下之一强烈怀疑本类）：
- 加载**非标准 causal LLM**（Mistral SWA、Gemma-2 SWA、prefix LM、encoder-decoder cross-attn、BERT-family bidirectional 等）时**静默输出乱码或语义偏移**，Qwen/Llama/Phi 等纯 causal LLM 完全正常；
- 换后端**一致地错**（Metal + CPU 都错，或 Metal 三段路径与 Metal FA 路径都错），但 torch 侧 rebuilt 模型 `--test` 输出正常；
- 短 prompt 不明显、长 prompt 越来越错（尤其超过 SWA window size 后）；
- 输出前 N 个 token byte-identical，后续开始发散。

> **2026-07-31 更新**：Metal 的 causal 假设已改为**数据驱动**（`mCausalLayout`，见 `MetalAttention.mm` `_computePathFlags`）——真实 mask 张量（`mHasMask=true`）自动关掉 causal-tri/bound/FA-v1/faNax 并逐元素 honor mask，标量哨兵/无 mask + kvcache 才走 causal 优化。CPU/hexagon 早已如此。**因此 Metal 上的非 causal 模型不再需要手动设 env**（`MNN_METAL_QK_CAUSAL_TRI` 已删除）。本节的 Metal 部分主要作为历史方法论保留；若仍遇非 causal 乱码，先确认 `gen_attention_mask` 是否给该模型正确产出了**真实 mask 张量**（而非误走标量分支），根因多在导出/mask 生成侧而非 kernel。

### 7.1 核心心法

**"kernel 逻辑正确，只是它假设的模型语义与实际模型不符" ≈ 隐式假设违反。**

这类 bug 的共同特征：
- shader / kernel 代码**逻辑上完全正确**（review 挑不出错），单独跑 op 测试也过；
- 但 kernel 编写时**默认了一个模型层面的约定**（如"attention 是因果下三角"、"tensor layout 是 NC4HW4"、"KV 尾插"），一旦模型不遵守就静默错；
- 通常一个后端的多条路径**共享**这个假设 —— 例如 Metal 三段+CAUSAL_TRI/BOUND 与 Metal FA kernel 都硬编码了"causal mask"，SWA 模型两条路径**都错**，用户以为是 Metal 后端问题去查 kernel 反而找不到根因；
- 与 §1（别名）区别：地址/内存都对；与 §2（导出）区别：权重完好、torch 侧正常；与 §6（shader/codegen 错）区别：没有崩溃、shape 都在支持范围内。

**方法论一句话**：**遇到"这个模型错、那个模型对"，先查 kernel 的隐式假设，再查具体代码逻辑**。

### 7.2 已知的隐式假设清单（MNN Metal LLM）

| Kernel / 路径 | 隐式假设 | 违反后现象 | 相关开关 |
|---|---|---|---|
| `prefill_qk[_tensor]` CAUSAL_TRI 分支 + host 侧梯形 dispatch | mask 是 causal lower-triangular（下三角内 mask=0/pass，上三角内 mask=-inf/0） | 上三角"应参与"的位置被 host 侧完全跳过 dispatch，QK 值为脏值/未初始化 → softmax 归约错误 | `MNN_METAL_QK_CAUSAL_TRI=0` 完全回退 |
| softmax `softmax_plane[_sg]` CAUSAL_BOUND 分支 | 每行 q 只归约 `[0, causal_base + q_local)` valid prefix，之后 zero-pad 32-align | 若实际语义中 `k > q + kv_off` 处仍应 valid，其归约值为 0 → attention 分布偏移 | 同上 |
| `prefill_qkv[_tensor]` `av_k_upper` 早退 | AV K 循环截断到 tile 内最大 valid q 对应的 causal 上界 | k 超出 av_k_upper 位置的 P·V 贡献被忽略 | 同上 |
| Fused `prefill_flash_attn`（MetalFlashAttnShader.hpp） | `in_bounds = (kv_col_abs <= q_abs + kv_valid_offset)` 硬编码 causal | 非 causal 位置直接被 `-INFINITY` mask 掉 | `MNN_ENABLE_FLASH_ATTN_PREFILL=0` 也无用 —— **FA 本身就有此假设** |
| `decode_qk_softmax` fused decode kernel | KVCache 场景 decode = 单 token, 自回归 = causal | decode 不用因果判定（seq_q=1 天然 causal），此假设通常自然成立 | — |
| **通用**：Attention op / RoPE / KVCache 路径 | tensor NC4HW4 layout（c 维按 4 打包）；某些模型的 export 层未适配 | 换 layout 导出后 kernel 按 NC4HW4 stride 读到错误位置 → 乱码 | Attention_C4 宏（编译期） |

### 7.3 排查流程

#### Step 1: 用"模型分类"分流

问自己：这个模型是 **causal** 还是 **non-causal**？

- **Causal-only**（标准 LLM）：Qwen 全系列、Llama 全系列、Phi、Mistral 7B v0.3+（改回 full-window 部分）、Yi、DeepSeek、Baichuan → 一般不会踩此类
- **含 SWA / mixed window**：Mistral 7B v0.1 (前 3 层 full window, 后 SWA)、Gemma-2 (每层交替 SWA / full)、Ministral → 高概率踩
- **Prefix LM / bidirectional**：Baichuan-Base、UL2 前缀部分、encoder 类 → 必踩
- **不确定**：读 HF 模型的 `config.json`，看 `sliding_window` / `attention_bias` / `is_encoder_decoder` 字段；或读 modeling 源码里 attention_mask 生成部分

#### Step 2: 数据驱动检测（Metal 现状）+ 单一 gate 消除法（历史手法）

**Metal（2026-07-31 起）**：causal 与否由 mask 张量形状自动判定，非 causal 模型走真实 mask 张量即自动 honor，无需任何 env。若非 causal 模型仍乱码，**先查 `gen_attention_mask` 是否为该模型走了正确分支**（真实张量 vs 误走标量），而非调 kernel 开关。

**历史手法（其他后端 / 旧分支）**：**只要"关掉某个开关就恢复"，几乎必然是隐式假设违反**。旧 Metal 分支上曾用：

```bash
# (已删除) 旧分支：关 CAUSAL_TRI/BOUND 回到矩形 grid
# MNN_METAL_QK_CAUSAL_TRI=0 ./llm_demo ...
```

不要一次关一堆开关（那样分不清哪个是罪魁），一个一个来。

#### Step 3: 长度扫描

对于 SWA 类模型，症状**随 kv_seq_len 演进**：

```bash
for L in 128 512 1024 2048 4096; do
  echo "=== kv=$L ==="
  ./llm_demo config.json /tmp/prompt_${L}.txt 20
done
```

- 若前几长度都对、超过某长度（往往 = model 的 SWA window size，Mistral 是 4096）开始乱 → **强 SWA 证据**
- 若从头就乱（哪怕 kv=128）→ 可能是 prefix LM 或完全 bidirectional
- Causal 假设违反的**特征时序**：因为 causal-tri 在对角线附近工作正确，只有上三角被误跳过，短 prompt（seq < KV_TILE）主要是对角线，看着还行；长 prompt 上三角占比大，错误累积

#### Step 4: 跨后端对拍（辅助）

- **理想 oracle**：CPU 后端。CPU attention 通常不做 causal 假设优化（走完整 mask 输入）→ 若 CPU 也错，问题不在此类（回去查 §2 导出，或看模型是否本来就该错）
- **Metal 内部两路**：`MNN_ENABLE_FLASH_ATTN_PREFILL=1` (FA) vs `=0` + `MNN_METAL_QK_CAUSAL_TRI=0` (纯 rectangular 三段)。**两者都错**（Metal 双错）→ FA 本身也有 causal 假设，模型本身非 causal
- **HF/torch 侧 sanity**：`llmexport.py --test <query>` 是不是也正常？若正常 → 模型是可跑的，MNN 侧假设不匹配

#### Step 5: 读 shader 里的假设注释（快速定位假设是什么）

MNN Metal shader 里的关键假设都有**明确注释**，`grep -n "Assumption\|causal-lower-triangular\|mask is a no-op\|hard-codes causal"`：

```
source/backend/metal/MetalAttentionShader.hpp:558:  Assumption: the mask provided ... is causal-lower-triangular
source/backend/metal/MetalAttentionShader.hpp:651:  causal ADD/SET masks are 0/pass in the valid region
source/backend/metal/MetalAttention.mm:531:  FA also hard-codes causal masking via `kv_valid_offset = seq_k - seq_q`
```

新增 kernel 优化时**必须**留下这种注释；review 时**必须**读这些位置。

#### Step 6: 加固方向（若确认此类）

- **Metal：已实现（2026-07-31）** —— `MetalAttention.mm` `_computePathFlags` 从 `inputs[3]` 形状派生 `mCausalLayout`（真实张量 mask ⇒ 非 causal ⇒ 逐元素 honor、关全部 causal 优化；标量/无 mask + kvcache ⇒ causal）。配套 `llm.cpp` 对 metal 后端也发标量哨兵 causal mask（同 cpu/hexagon）。`MNN_METAL_QK_CAUSAL_TRI` 已删除
- **其他后端 / 通用长期方向**：runtime 首次 attention encode 时抽样验证 mask 是否 lower-triangular 并缓存（工作量最大但通用）；或导出侧 `llm_config.json` 落 `attention_type` 字段权威标注

### 7.4 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的隐式假设 |
|------|-------------|
| SWA 模型（Mistral v0.1/Gemma-2）乱码，Qwen 正常 | attention mask 假设 causal（本节） |
| Prefix LM / BERT 类整段乱 | attention 假设 causal 或 KVCache 单向 |
| 短 prompt 对、长 prompt 错（错的位置在开头附近） | causal-tri 的上三角覆盖累积错 |
| 短 prompt 错 | prefix LM / bidirectional 从第一步就崩 |
| MNN_METAL_QK_CAUSAL_TRI=0 就对 | causal-tri/bound 假设 |
| MNN_ENABLE_FLASH_ATTN_PREFILL=0 后仍错 | FA + 三段都错，模型本身非 causal |
| 只有某几层错 | 层级差异（如 Gemma-2 交替 SWA / full） |
| 换 Metal → CPU 就对 | 后端优化（本节）；换 CPU → Metal 就对 = §1 或 §2 |

### 7.5 参考案例（占位）

**待补**：目前尚无生产 SWA 模型跑 Metal 后端出错的**已复现**案例入库（分支中 Qwen 系列均为 causal，未触发）。若未来第一次实测出现 SWA/Gemma-2/prefix LM 走 Metal 报错，务必按 Step 1-6 走完 + 补充参考案例到此节。

**预期案例形态**（供未来复现参考）：Mistral 7B v0.1 W4-b32 导出 → MNN Metal 后端 → 长 prompt (>4096 tokens) → 输出在 window 边界后开始重复/漂移；CPU 后端一致乱（因为 CPU attention 也可能不按 SWA 特化）；HF torch 侧正常；`MNN_METAL_QK_CAUSAL_TRI=0` **仅缓解 causal-tri/bound 部分**，FA 本身仍错 → 需要架构层加固。

### 7.6 相关文件索引

| 文件 | 作用 |
|------|------|
| `source/backend/metal/MetalAttentionShader.hpp` | CAUSAL_TRI / CAUSAL_BOUND 的假设注释位置（`grep Assumption`）；prefill_qk/prefill_qk_tensor/prefill_qkv 三个 kernel 的实现 |
| `source/backend/metal/MetalFlashAttnShader.hpp` | FA kernel（同样 hard-code causal） |
| `source/backend/metal/MetalAttention.mm` | mQkCausalTri / mCausalBound / mFlashAttnPrefill 的 gate 条件；FA 的 causal-only comment (`:531`) |
| `source/backend/metal/MetalSoftmaxShader.cpp` | softmax CAUSAL_BOUND 分支实现 |
| `skills/metal-optimize/env-registry.md` | `MNN_METAL_QK_CAUSAL_TRI` 等相关开关的完整语义登记 |
| `skills/metal-optimize/kernel-dev-and-optimize.md` | causal-tri / CAUSAL_BOUND 的设计文档（§2.3.1）|

---

## §8 持久化缓存误信（weight-mmap cache / 陈旧缓存）

**触发**（满足以下之一强烈怀疑本类）：
- 开启 `use_mmap`（权重 mmap 落盘）后输出乱码/单字符刷屏（`!!!`、连续换行），关掉 `use_mmap` 或 `use_cached_mmap` 就好；
- **同一二进制：App 内错、`llm_demo` 对**（或反之）；iOS/Android 真机错、Mac/host 对；
- 清空 tmp/缓存目录后第一次跑就好，之后又坏；换一个 `tmp_path` 就好；
- 前几个 token 正常、随后整段崩坏（部分权重是真的、部分是垃圾的典型混合特征）。

### 8.1 核心心法

**"缓存是否有效"只能在运行起点判定一次。** `use_cached_mmap` 的契约是"上一个进程写完整套权重并留下 sync 标记 → 本次按相同分配顺序直接复用磁盘内容"。这个契约有两个隐含前提，破坏任何一个都是静默乱码：

1. **标记不能被本次运行自己写的 sync 污染**（判定时机必须在 mmap 分配器创建时刻，之后不可变）；
2. **缓存文件必须属于同一个模型**（缓存文件名前缀 `0_0_0_0_` 只含 precision/memory/power，**不含模型标识**——换模型不换目录必然拿到错误权重）。

另外牢记：跳过权重读取的 execution 拿到的 STATIC buffer **必须真的来自 mmap 池**。首次 `onClearBuffer` 后静态分配器切回 RAW malloc（封池），此后任何被重建的带权重 execution 若仍处于"信任缓存"模式，就是在拿未初始化内存当权重。

### 8.2 排查流程

1. **配置对齐分流**：App 与 demo 的默认配置差异先列全（`use_mmap` / `use_cached_mmap` / `tmp_path` / 加载次数）。"App 错 demo 对"大概率不是平台问题，是配置或加载模式差异。
2. **缓存卫生三连**：换全新 `tmp_path` → 跑一次；同目录再跑一次（warm）；换另一个模型同目录跑（污染探测）。三个结果就能区分"自我污染 / warm 复用坏 / 跨模型污染"。
3. **复刻加载模式**：iOS App 是"启动预载 + 使用时重载"的**同进程双加载**；`llm_demo` 是单加载。双加载可疑时写 20 行的 double-load 复现器（load → destroy → 清目录 → load → generate），在 host 上复现比真机埋点便宜一个量级。
4. **埋点看 hint 演化**：在 `CPURuntime::onCreate` 的 weightMemoryPath 分支打印 `useCachedMmap`/`syncValid`，在各 `useCachedMmap > 1` 跳读点（`ConvInt8TiledExecutor` 等）打印命中。判据：hint 在**运行中途**从 1 变 2 = 自我污染坐实；跳读发生在首次真实推理 resize（而非装载期）= 重建的 execution 在拿野内存。
5. **修复方向**：判定移入分配器首次创建分支（`MetalBackend.mm:onCreate` 是正确参考实现）；更彻底的加固是跳读前校验 buffer 确实来自 mmap 池。

### 8.3 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的原因 |
|------|-------------|
| mmap 开着才乱码，冷启动+干净目录也乱 | 本次运行自我污染（sync 标记中途被自己看见） |
| 冷启动好、同目录第二次坏 | warm 复用路径的分配顺序/布局不匹配 |
| 换模型不换目录后乱码 | 缓存文件名无模型标识，跨模型污染 |
| App 错、demo 对 | App 双加载模式触发 + demo 单加载不触发 |
| 只有某类 op（如 geometry 分解的 fuse op）坏 | 该 op 的 execution 在封池后重建，跳读拿到 RAW 内存 |

### 8.4 参考案例：fuse 模型 iOS CPU 全乱码（useCachedMmap 自我污染，2026-08-06）

**症状**：4 个 FusedLinear 导出模型在 iPad/iPhone 上 CPU 后端（`use_mmap=true`）全部输出单字符刷屏（`!!!`/`\n`），偶发 SIGSEGV；同设备 Metal 正常；Mac `llm_demo` 单跑正常；非 fuse 模型任何配置都正常。

**排查路径**（两条红鲱鱼 + 一次真命中）：
1. Mac 上 `use_mmap=true` "复现"乱码 → 实为 `llm_demo` 强制 `tmp_path:"tmp"` + 缓存无模型标识，吃了之前另一个模型的缓存（**红鲱鱼一：跨模型污染**）。教训：对拍前 `rm -rf tmp`。
2. "非 fuse 模型也坏" → bisect 全 GOOD 才发现主 build 目录增量编译产物陈旧（**红鲱鱼二**）。教训：怀疑"分支回归"先开全新 build 目录验证，别信老增量目录。
3. 干净构建 + 干净缓存后锁定复现矩阵：**fuse × use_mmap × CPU × iOS**；给 App 加 `nommap` 判别开关 → mmap=false 立好。
4. 关键洞察：iOS App 是**同进程双加载**（启动预载 + benchfiles 重载并清 tmp 目录）。按此写 double-load 最小复现器 → **Mac 上完整复现**，真机问题降维成 host 调试。
5. 埋点两处：`useCachedMmap` 每次 resize `+= syncValid` 自增（单载 trace 1→2→3→4→5）；140 个 FusedLinear 分解出的成员 conv 在**首次真实推理 resize** 时重建并全部跳读权重。
6. 根因链闭合：首次 `onClearBuffer` 写出 sync.static 并封池 → 下一次 resize 重查看见**自己刚写的标记** → hint 1→2 → resize 重建的 conv 跳读 + STATIC buffer 来自 RAW malloc → 权重=未初始化内存。非 fuse conv 装载期创建一次且跨 resize 复用，永远踩不到；Metal 不分解 FusedLinear 且判定本来就只做一次，双重幸免。

**修复**（`4c50f4b12`）：CPU 的 sync 检查移入 `mStaticAllocatorMMap == nullptr` 创建分支，对齐 Metal。验证：Mac double-load 修复、warm 二进程正常、iPhone 13 Pro 真机 0.6b/0.8b/2b CPU 全部恢复。

**避坑要点**：
- "真机错 host 对"先对齐**配置与加载模式**（mmap 开关、双加载），不要先怀疑硬件/SIMD/平台；
- 复现器要**复刻加载模式**而不只是配置——单载复现不出双载 bug；
- `MNN_ASSERT` 在 release 构建是空操作，mmap 分配器里的断言不会救你；
- 直跑被 SIGKILL（rc=137）而 lldb 下正常时，先在 lldb 里拿结果，别死磕信号来源；
- 已知遗留：同进程重载且**不清缓存目录**（陈旧 sync + 旧权重文件）仍会误信；`llm_demo` 共享 `tmp/` 无模型标识。见 8.1 前提 2。

### 8.5 相关文件索引

| 文件 | 作用 |
|------|------|
| `source/backend/cpu/CPUBackend.cpp` | CPURuntime::onCreate 的 weightMemoryPath/sync 判定（本案例修复处） |
| `source/backend/metal/MetalBackend.mm` | Metal 的一次性判定正确参考（`onCreate` :1984 附近） |
| `source/core/BufferAllocator.cpp` | `MmapAllocator`：缓存文件命名（`prefix + allocTimes`）、sync() 写标记、autoRemove 语义 |
| `source/backend/cpu/compute/ConvInt8TiledExecutor.cpp` | Q4 conv 的 `useCachedMmap > 1` 跳读点 |
| `source/core/ConvolutionCommon.cpp` / `source/backend/cpu/CPULayerNorm.cpp` | 其余跳读点 |
| `transformers/llm/engine/demo/llm_demo.cpp` | 强制 `tmp_path:"tmp"` 的共享缓存陷阱（:274） |

---

<!--
新增类别模板（复制以下骨架，编号 +1，并同步更新顶部导航表）：

## §N <类别名>

**触发**：
- <典型症状 1>
- <典型症状 2>

### N.1 核心心法
### N.2 相关背景
### N.3 排查流程
    #### Step 1: ...
    #### Step 2: ...
### N.4 常见对照表：症状 → 优先怀疑
### N.5 参考案例
### N.6 相关文件索引

-->
