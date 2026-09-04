# §2 量化误差 / 导出侧权重损坏

> **归属**：[`general-debug`](SKILL.md) 的分类分册之一，先在入口的分流表确认类别再读本文。
>
> **不在本文**：只有**一个后端**错（另一个对）见 [`memory-aliasing.md`](memory-aliasing.md)；
> 权重完好但 fp16 长序列才错见 [`fp16-range.md`](fp16-range.md)；
> 权重在磁盘缓存里被换成垃圾见 [`stale-cache.md`](stale-cache.md)。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

**触发**（满足以下之一强烈怀疑本类）：
- 低 bit（Q4）导出输出乱码/退化，Q8 或 fp 导出完全正常；
- **换任何推理后端都错**（CPU / Metal / master 分支 runtime 一致地错）—— 与 [`memory-aliasing.md`](memory-aliasing.md) 的"一个后端错一个后端对"正好相反；
- `llmexport.py --test "<query>"`（torch 侧 rebuilt 模型，不走量化打包）输出正常；
- 只在特定模型（大 vocab、大 hidden）上触发，同流程导小模型正常；
- MNN op 单测（`run_test.out`）全过。

## 2.1 核心心法

**"所有后端一致地错 + torch 侧对 + 单测过" ≈ 导出产物（权重文件）本身坏了。** 此时不要在推理引擎里找 bug，而是：

1. 用导出参数做**量化 bisect**，定位哪个权重坏；
2. **离线反量化导出文件，与 HF 原始权重逐行比对**，把"导出坏"与"运行时 dequant 坏"分开；
3. 警惕**框架静默错误**：PyTorch 加速后端（MPS/CUDA）在超大张量上可能不报错、直接给错结果。

## 2.2 相关背景

- 导出量化入口：`transformers/llm/export/utils/torch_utils.py::quant`（大权重按 `_QUANT_MAX_ELEMENTS`=256M 元素沿 oc 分块），实际量化在 `_quant_on_device`（优先 CUDA→MPS→CPU）。
- 权重写文件：`utils/mnn_converter.py::build_weight`（header + q_weight + alpha [+ bias]）；lm_head/tie_embeddings 的 offset 信息写进 `llm_config.json` 的 `tie_embeddings` 字段。
- Q4 打包格式：每 byte 两个权重，**高 nibble 在前**；asym alpha 布局为每 (oc,block) 一对 `[zero, scale]`；引擎 dequant 公式 `w = q * scale + (zero + offset * scale)`，`offset = -(1 << (bit-1))`（见 `transformers/llm/engine/src/diskembedding.cpp`）。
- 可用的 bisect 旋钮：`--quant_bit/--quant_block`（body）、`--lm_quant_bit/--lm_quant_block`（lm_head 单独控制）、`--quant_config <json>`（任意 op 级覆盖，如 `{"/lm/lm_head/Linear": {"bits": 8, "block_size": 0}}`）、`--hqq`、`--seperate_embed`。

## 2.3 排查流程

### Step 1: 确认是导出侧而非推理侧

三个证据凑齐即可确认：① CPU 和 GPU 后端**一致地**错；② `--test` torch 侧输出正常（权重映射没问题）；③ op 单测全过（kernel 没问题）。

### Step 2: 量化 bisect —— 用导出参数二分定位坏权重

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

### Step 3: 离线反量化，与 HF 权重逐行比对（关键手法）

写脚本读 `llm.mnn.weight`，按 `llm_config.json` 的 `tie_embeddings`（`weight_offset/alpha_offset/alpha_size/quant_bit/quant_block/alpha_dtype`）反量化若干行，与 HF `embed_tokens.weight` 算 cosine：

- **先用一个已知正确的导出（如 Q8）验证脚本方法学**（cos 应全部 ≈1），再去测坏的导出 —— 否则分不清是权重坏还是自己反量化约定写错；
- 抽样行要覆盖头、中、尾（本案例：row 0/1/100 全乱、row 151645/248319 正常 —— "部分行坏"是关键线索）；
- 对坏区做**二分搜索找边界**：本案例第一个正常行 = 131072，边界字节偏移 = 2^27，精确的 2 的幂 → 几乎必然是溢出/截断/框架 bug，不是量化算法问题；
- 看坏区字节内容：本案例全 0（不是随机垃圾）→ 说明是"某一步整体输出了 0"，不是错位/串行。

### Step 4: 在导出代码里复现最小 case

边界 2^27 字节 = 2^28 个 int4 元素，恰好等于分块大小 256M 元素 → 怀疑 `_quant_on_device` 对满块输入出错。直接用随机张量复现：

```python
qw_mps, _ = _quant_on_device(w.to('mps'), 4, 0, False, False, False)   # 131072 x 2048
qw_cpu, _ = _quant_on_device(w, 4, 0, False, False, False)
# mps 输出全 0，cpu 正常 → 框架 bug 坐实
```

再逐 op 拆：`(q.reshape(-1,2) * m).sum(axis=1)` —— **乘法结果正常，`sum(axis=1)` 在 MPS 上对 ≥2^28 uint8 元素静默返回全 0**（uint8→int64 归约溢出类 bug，2^27 元素正常）。

### Step 5: 修复原则

- **绕开出错的框架 op**，用等价的安全写法（本案例：uint8 逐列 `packed |= col << shift` 按位累加替代 `sum`），不要指望升级框架版本；
- 修复后必须验证 **加速后端与 CPU 输出逐字节相等**（`torch.equal`），再全量重导 + 端到端跑 CPU 和 GPU greedy。

## 2.4 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的原因 |
|------|-------------|
| 所有后端一致错 + torch `--test` 对 | 导出产物权重坏（本册） |
| Q4 错 Q8 对 | 4bit 打包路径 bug（Q8 不走打包） |
| 只有大 vocab / 大 hidden 模型触发 | 大张量分块边界 / 框架大张量静默 bug |
| 反量化比对"部分行坏部分行好"，边界是 2 的幂 | 溢出 / 截断 / 框架归约 bug |
| 坏区全 0（非随机垃圾） | 某步整体输出 0（归约/拷贝失败），非量化误差 |
| 反量化比对全部行都乱 | 先怀疑自己的反量化约定（nibble 序、zero/scale 布局），用 Q8 导出验证脚本 |

## 2.5 参考案例：Qwen3.5-2B Q4 lm_head 全零（PyTorch MPS sum 归约 bug）

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
- "乱码"不一定是推理引擎/后端问题 —— **先用"所有后端是否一致地错"分流**：一致错查导出，不一致错查后端内存（[`memory-aliasing.md`](memory-aliasing.md)）；
- 加速后端（MPS/CUDA）的大张量 op 可能**不报错、给全 0** —— 关键路径的量化/打包结果要有与 CPU 的一致性校验意识；
- 反量化比对脚本一定要先在已知正确的导出上自校准。

## 2.6 相关文件索引

| 文件 | 作用 |
|------|------|
| `transformers/llm/export/utils/torch_utils.py` | 量化 + 低 bit 打包（本案例修复处，含 256M 分块逻辑） |
| `transformers/llm/export/utils/mnn_converter.py` | `build_weight` 写权重文件、`write_header`、tie_embeddings 信息 |
| `transformers/llm/export/utils/hqq_quantizer.py` | hqq 量化实现 |
| `transformers/llm/engine/src/diskembedding.cpp` | 引擎侧 Q4/Q8 dequant 参考（nibble 序、alpha 布局、offset 公式） |
| `transformers/llm/engine/src/llmconfig.hpp` | `tie_embeddings` 字段解析 |
