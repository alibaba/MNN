---
name: bugfix
description: MNN 各类正确性/回归 bug 的排查方法论集合。按 bug 分类组织：当前覆盖 (1) 内存别名/生命周期错误（buffer aliasing、arena reuse、`MemChunk` / tensor buffer 生命周期）、(2) 量化误差/导出侧权重损坏（低 bit 打包、导出分块、PyTorch MPS/CUDA 大张量静默错误），后续会补充数值精度、并发竞争、图优化回归、Codegen/Shader 错等类别。
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
| 3 | *（待补：并发 / 线程竞争）* | 结果不稳定、每次运行不同；单线程稳定但多线程随机错 | *待补* |
| 4 | *（待补：图优化回归）* | 某个 converter pass 之后模型跑错，disable 该 pass 后正常 | *待补* |
| 5 | *（待补：数值精度 / fp16 溢出）* | 结果"接近对"但存在系统性偏差，长序列尤其明显 | *待补* |
| 6 | *（待补：Shader / Codegen 错）* | GPU 后端某 shape 或某 quant 组合下崩溃或乱码；调整 threadgroup / kernel 参数症状变化 | *待补* |

> 新增章节时同步在这张表里补一行；每个章节命名为 `## <编号> <类别名>`，保持编号递增。

---

## 通用排查原则（所有类别共用）

1. **先复现最小化**：定住能稳定复现的最小 case（最短 prompt、最小 shape、单线程），然后逐步添加变量。
2. **两向 A/B**：换后端、换开关、换编译选项、换线程数 —— 任一维度上"这边错那边对"都是强线索。
3. **别信"这块应该是独立的"，用直接观察去证明**：地址、值、时序都要用打印/断点去看，不要靠推理。
4. **修改先做假设，验证后再改代码**。改一版跑一版，避免"多改一起观察但不知道哪个生效"。
5. **改完记录**：如果本次 bug 有可复用的教训，追加到本文件对应章节的"参考案例"里；如果发现新的 bug 类别，新开一节并更新导航表。
6. **导出/序列化崩溃先查 I/O 边界**：用回溯定位第一个文件写入点；converter 入口先创建并验证父目录，`fopen` 失败必须立即返回，禁止把空 `FILE*` 传给 `fwrite`。不要把这种晚发的 SIGSEGV 误判成模型或后端问题。

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

### 1.6 相关文件索引

| 文件 | 作用 |
|------|------|
| `source/core/BufferAllocator.hpp` | Arena 语义注释，是否复用 freed chunk 的官方描述 |
| `source/core/Backend.hpp` | Backend tensor buffer 生命周期接口 |
| `source/backend/cpu/CPULayerNorm.cpp` | 正确的批量 alloc-then-free-all 模式参考 |
| `source/backend/cpu/CPURoPE.cpp` | 参考案例的修复实现（onResize 注释里写了原因） |
| `source/backend/metal/MetalBackend.mm` | 跨 op tensor buffer 重叠检查（`matchQKVFusions`） |

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
