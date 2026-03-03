# 步骤 0：计算拆解（最关键的步骤）

> **目标**：阅读待优化算子的源码，将每一段计算逻辑映射到 MNN 已有的高性能函数。
>
> **前置条件**：明确待优化的算子文件路径。
>
> **复杂度**：中（纯代码分析，不需要编译运行）
>
> **这一步决定了优化质量的上限**。如果这一步做不好，后面的工作方向就是错的。

## 0.0 必读参考（在写任何代码之前）

**必须先阅读以下两个文件**，理解 MNN 的优化范式：

1. **`source/backend/cpu/CPUAttention.cpp`** — 学习它如何：
   - 通过 `gcore->MNNComputeMatMulForE_1` 做矩阵向量乘（而不是写循环）
   - 通过 `MNNScaleAndAddBiasScalar` 做标量乘（而不是写循环）
   - 通过 `MNNSoftmax` 做 Softmax（而不是写循环）
   - 在 `onResize` 中用 `Tensor + Backend` 分配缓存（而不是 std::vector）
   - 用 `MNN_CONCURRENCY_BEGIN/END` 做多线程

2. **`source/backend/cpu/compute/CommonOptFunction.h`** — 了解所有可用的高性能函数

> ⚠️ **强烈建议：你的优化代码应该像 `CPUAttention.cpp` 一样，优先通过调用 `CoreFunctions` 已有函数实现高性能。只有在确实无法使用已有函数时，才考虑利用 AI 能力编写 NEON intrinsic 或 Vec4 循环。**

---

## 0.1 阅读源码，列出所有计算逻辑

逐行阅读算子的 `onExecute` 函数，将每一段计算逻辑提取出来。

```markdown
## 计算逻辑清单

| # | 代码位置 | 计算描述 | 代码模式 |
|---|---------|---------|---------|
| 1 | Lxx-Lyy | 描述这段代码做什么计算 | 循环模式（如：双重循环乘加、单循环乘标量） |
| 2 | ... | ... | ... |
```

---

## 0.2 对每个计算逻辑，查找可用的 MNN 函数

在 `source/backend/cpu/compute/CommonOptFunction.h` 中查找匹配的函数。
使用 **优化决策树**：

```
看到一段循环代码 →
  ├─ 是双重循环的乘加？
  │   ├─ 有一个维度为 1（MatVec）？ → 用 MNNComputeMatMulForE_1（无需 Pack）
  │   ├─ 数据规模大（值得 Pack 开销）？ → 用 MNNPackedMatMul
  │   └─ 数据规模小（Pack 开销 > 计算）？ → 保持朴素循环
  ├─ 是单循环乘标量/加标量？ → 用 MNNScaleAndAddBiasScalar
  ├─ 是循环调用 expf/silu/sigmoid？ → 用 MNNExp/MNNSiLu
  ├─ 是循环做卷积？ → 用 MNNConvRunForLineDepthwise
  ├─ 是循环做数据重排/转置？ → 用 MNNPackCUnit/MNNUnpackCUnit/MNNTranspose32Bit
  ├─ 是循环做范数归一化？ → 用 MNNNorm（先验证语义匹配！）
  ├─ 是 std::vector 临时内存？ → 替换为 Tensor + Backend 内存池
  └─ 以上都不匹配？
      ├─ 能拆解为已有函数的组合？ → 组合调用
      ├─ 计算量极小？ → 保持朴素循环
      ├─ 是性能热点且可向量化？ → 用 Vec4 循环
      └─ 都不行且是核心热点？ → 标记为"需要新汇编 kernel"
```

---

## 0.3 输出映射表

```markdown
## 函数映射表

| # | 原始代码 | 替换为 MNN 函数 | 说明 |
|---|---------|----------------|------|
| 1 | memcpy 循环 | memcpy（保持不变） | 已经是最优 |
| 2 | XXX 循环 | `MNN 函数名` | 替换理由 |
| ... | ... | ... | ... |

### 无法直接替换的计算

| # | 计算 | 原因 | 建议方案 |
|---|------|------|---------|
| X | 某个循环 | 语义不匹配 / 规模太小 / 无对应函数 | A) 保持手动循环  B) 写新 kernel |
```

---

## 0.4 验证检查（每个替换都必须过这三关）

### ✅ 语义验证

对映射表中的每一条替换，回答以下问题：

```
□ 函数的数学语义与原始循环完全一致？
  - 参数含义是否匹配（如转置、通道顺序）？
  - 归一化方式是否匹配（如 sum vs mean、L1 vs L2）？
  - 输出格式是否与下游代码兼容？
□ 如果语义不完全一致，是否已标记为"保持手动实现"并注释原因？
```

### ✅ in-place 安全性检查

```
□ 替换后的代码中是否有 dst==src 的调用？
□ 如有，该 MNN 函数是否支持 in-place？
  （参考 SKILL.md 中的 in-place 安全性表）
□ 如不支持 in-place，是否已安排 scratch buffer？
```

### ✅ 数据规模检查

```
□ 对于双重循环替换为 MNNPackedMatMul 的情况：
  - 数据规模是否足够大，使 Pack/Unpack 开销值得？
  - 如果规模小，是否已标记为"保持朴素循环"？
□ 对于 MatVec，是否优先使用无需 Pack 的 MNNComputeMatMulForE_1？
```

---

## 0.5 额外全局检查

### 内存分配检查

```
是否使用了 std::vector 作为临时缓存？ → 替换为 onResize 中的 Tensor + Backend 内存池
是否在 onExecute 中有 new/malloc？ → 移到 onResize
是否有不必要的数据拷贝？ → 尝试 in-place 计算或虚拟 Tensor
多个小 buffer 是否可合并为一块预分配？ → 用连续内存 + 偏移量切分
```

### 多线程检查

```
算子是否已经使用了 MNN_CONCURRENCY_BEGIN/END？ → 检查划分是否合理
是否有可以按 Head/Batch 并行的循环？ → 添加多线程
线程间是否有写冲突？ → 确保每个线程有独立的写目标
```

### 数据排布检查

```
State 矩阵的排布是否对 MatMul 友好？ → 如果需要反复做 S^T @ vec，考虑存储 S^T
是否有频繁的转置操作？ → 考虑改变存储格式避免转置
是否有跨 stride 的不连续访问？ → 考虑预先 Pack 为连续内存
```

---

## 步骤 0 测试标准

### 通过标准

- [ ] 算子的所有计算逻辑已列出（无遗漏）
- [ ] 每个计算逻辑都已对应到 MNN 函数（或标记为"保持手动实现"并说明原因）
- [ ] 每条替换都通过了语义验证、in-place 检查、数据规模检查
- [ ] 内存分配、多线程、数据排布检查已完成
- [ ] 映射表已输出

### 常见错误

| 错误 | 原因 | 修复 |
|------|------|------|
| 把所有循环都用 Vec4 包装 | 没有查找已有函数 | 先查 CommonOptFunction.h |
| 用裸 NEON intrinsic (vmlaq_f32 等) 替代已有功能 | 以为 intrinsic 一定更快 | 优先检查已有函数，MNN 内置汇编往往经过更深入的调优 |
| 不验证函数语义就替换 | 函数名匹配 ≠ 语义匹配 | 阅读函数实现确认数学计算一致 |
| 不验证 in-place 安全性 | 假设所有函数都支持 dst==src | 查阅 in-place 表，不确定时用 scratch |
| 小规模数据用重量级函数 | 没考虑 Pack/Unpack 开销 | 小规模保持朴素循环 |
| 漏掉了 SiLU/Exp/Conv 的替换 | 没有逐行审查代码 | 逐行检查每个数学函数和循环 |
| 只在终端打印报告 | 没有写入文件 | 必须写入 `<算子>_optimization.md` |

---

## 下一步

**步骤 0 通过后，进入 `step1-benchmark.md`（步骤 1：建立性能基准）。**
