# §7 后端 kernel 隐式假设违反

> **归属**：[`general-debug`](SKILL.md) 的分类分册之一，先在入口的分流表确认类别再读本文。
>
> **不在本文**：QNN / NPU 后端的算子约束与误差模式见 [`qnn-debug`](../qnn-debug/SKILL.md)；
> 假设成立但数值不够用（fp16 动态范围）见 [`fp16-range.md`](fp16-range.md)；
> 假设成立但读到脏内存见 [`memory-aliasing.md`](memory-aliasing.md)。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

**触发**（满足以下之一强烈怀疑本类）：
- 加载**非标准 causal LLM**（Mistral SWA、Gemma-2 SWA、prefix LM、encoder-decoder cross-attn、BERT-family bidirectional 等）时**静默输出乱码或语义偏移**，Qwen/Llama/Phi 等纯 causal LLM 完全正常；
- 换后端**一致地错**（Metal + CPU 都错，或 Metal 三段路径与 Metal FA 路径都错），但 torch 侧 rebuilt 模型 `--test` 输出正常；
- 短 prompt 不明显、长 prompt 越来越错（尤其超过 SWA window size 后）；
- 输出前 N 个 token byte-identical，后续开始发散。

> **2026-07-31 更新**：Metal 的 causal 假设已改为**数据驱动**（`mCausalLayout`，见 `MetalAttention.mm` `_computePathFlags`）——真实 mask 张量（`mHasMask=true`）自动关掉 causal-tri/bound/FA-v1/faTc 并逐元素 honor mask，标量哨兵/无 mask + kvcache 才走 causal 优化。CPU/hexagon 早已如此。**因此 Metal 上的非 causal 模型不再需要手动设 env**（`MNN_METAL_QK_CAUSAL_TRI` 已删除）。本册的 Metal 部分主要作为历史方法论保留；若仍遇非 causal 乱码，先确认 `gen_attention_mask` 是否给该模型正确产出了**真实 mask 张量**（而非误走标量分支），根因多在导出/mask 生成侧而非 kernel。

## 7.1 核心心法

**"kernel 逻辑正确，只是它假设的模型语义与实际模型不符" ≈ 隐式假设违反。**

这类 bug 的共同特征：
- shader / kernel 代码**逻辑上完全正确**（review 挑不出错），单独跑 op 测试也过；
- 但 kernel 编写时**默认了一个模型层面的约定**（如"attention 是因果下三角"、"tensor layout 是 NC4HW4"、"KV 尾插"），一旦模型不遵守就静默错；
- 通常一个后端的多条路径**共享**这个假设 —— 例如 Metal 三段+CAUSAL_TRI/BOUND 与 Metal FA kernel 都硬编码了"causal mask"，SWA 模型两条路径**都错**，用户以为是 Metal 后端问题去查 kernel 反而找不到根因；
- 与 [`memory-aliasing.md`](memory-aliasing.md)（别名）区别：地址/内存都对；与 [`export-and-quant.md`](export-and-quant.md)（导出）区别：权重完好、torch 侧正常；与 [`gpu-oob.md`](gpu-oob.md)（shader 越界）区别：没有崩溃、shape 都在支持范围内。

**方法论一句话**：**遇到"这个模型错、那个模型对"，先查 kernel 的隐式假设，再查具体代码逻辑**。

## 7.2 已知的隐式假设清单（MNN Metal LLM）

| Kernel / 路径 | 隐式假设 | 违反后现象 | 相关开关 |
|---|---|---|---|
| `prefill_qk[_tensor]` CAUSAL_TRI 分支 + host 侧梯形 dispatch | mask 是 causal lower-triangular（下三角内 mask=0/pass，上三角内 mask=-inf/0） | 上三角"应参与"的位置被 host 侧完全跳过 dispatch，QK 值为脏值/未初始化 → softmax 归约错误 | `MNN_METAL_QK_CAUSAL_TRI=0` 完全回退 |
| `prefill_qk` simdgroup-matrix 整 tile causal skip（`!CAUSAL_TRI` 分支） | 门控曾写成 `DEFAULT_MASK \|\| ADD_MASK \|\| SET_MASK`，即"有 mask 输入就假定 causal" | 非 causal 张量 mask（ViT 全可见 mask）被静默改成 causal：q 的前 16 行 tile 看不到后面的 key | 已修：收窄为仅 `DEFAULT_MASK` |
| `attention_mask_offset` 的 `mask_q_len`（`MetalAttention.mm::_writeQKVParam`） | 曾从固定下标取 q 长度，即假定张量 mask 一定是 rank-4 `[b,h,q,k]` | rank-3 `[b,q,k]` 拿到 `mask_q_len=1` ⇒ 所有 q 行都读 mask 第 0 行；全零 mask 无害，逐行变化的 mask 静默错 | 已修：改取 mask 末两维 |
| softmax `softmax_plane[_sg]` CAUSAL_BOUND 分支 | 每行 q 只归约 `[0, causal_base + q_local)` valid prefix，之后 zero-pad 32-align | 若实际语义中 `k > q + kv_off` 处仍应 valid，其归约值为 0 → attention 分布偏移 | 同上 |
| `prefill_qkv[_tensor]` `av_k_upper` 早退 | AV K 循环截断到 tile 内最大 valid q 对应的 causal 上界 | k 超出 av_k_upper 位置的 P·V 贡献被忽略 | 同上 |
| Fused `prefill_flash_attn`（MetalFlashAttnShader.hpp） | `in_bounds = (kv_col_abs <= q_abs + kv_valid_offset)` 硬编码 causal | 非 causal 位置直接被 `-INFINITY` mask 掉 | `MNN_ENABLE_FLASH_ATTN_PREFILL=0` 也无用 —— **FA 本身就有此假设** |
| `decode_qk_softmax` fused decode kernel | KVCache 场景 decode = 单 token, 自回归 = causal | decode 不用因果判定（seq_q=1 天然 causal），此假设通常自然成立 | — |
| **通用**：Attention op / RoPE / KVCache 路径 | tensor NC4HW4 layout（c 维按 4 打包）；某些模型的 export 层未适配 | 换 layout 导出后 kernel 按 NC4HW4 stride 读到错误位置 → 乱码 | Attention_C4 宏（编译期） |

## 7.3 排查流程

### Step 1: 用"模型分类"分流

问自己：这个模型是 **causal** 还是 **non-causal**？

- **Causal-only**（标准 LLM）：Qwen 全系列、Llama 全系列、Phi、Mistral 7B v0.3+（改回 full-window 部分）、Yi、DeepSeek、Baichuan → 一般不会踩此类
- **含 SWA / mixed window**：Mistral 7B v0.1 (前 3 层 full window, 后 SWA)、Gemma-2 (每层交替 SWA / full)、Ministral → 高概率踩
- **Prefix LM / bidirectional**：Baichuan-Base、UL2 前缀部分、encoder 类 → 必踩
- **不确定**：读 HF 模型的 `config.json`，看 `sliding_window` / `attention_bias` / `is_encoder_decoder` 字段；或读 modeling 源码里 attention_mask 生成部分

### Step 2: 数据驱动检测（Metal 现状）+ 单一 gate 消除法（历史手法）

**Metal（2026-07-31 起）**：causal 与否由 mask 张量形状自动判定，非 causal 模型走真实 mask 张量即自动 honor，无需任何 env。若非 causal 模型仍乱码，**先查 `gen_attention_mask` 是否为该模型走了正确分支**（真实张量 vs 误走标量），而非调 kernel 开关。

**历史手法（其他后端 / 旧分支）**：**只要"关掉某个开关就恢复"，几乎必然是隐式假设违反**。旧 Metal 分支上曾用：

```bash
# (已删除) 旧分支：关 CAUSAL_TRI/BOUND 回到矩形 grid
# MNN_METAL_QK_CAUSAL_TRI=0 ./llm_demo ...
```

不要一次关一堆开关（那样分不清哪个是罪魁），一个一个来。

### Step 3: 长度扫描

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

### Step 4: 跨后端对拍（辅助）

- **理想 oracle**：CPU 后端。CPU attention 通常不做 causal 假设优化（走完整 mask 输入）→ 若 CPU 也错，问题不在此类（回去查 [`export-and-quant.md`](export-and-quant.md) 导出侧，或看模型是否本来就该错）
- **Metal 内部两路**：`MNN_ENABLE_FLASH_ATTN_PREFILL=1` (FA) vs `=0` + `MNN_METAL_QK_CAUSAL_TRI=0` (纯 rectangular 三段)。**两者都错**（Metal 双错）→ FA 本身也有 causal 假设，模型本身非 causal
- **HF/torch 侧 sanity**：`llmexport.py --test <query>` 是不是也正常？若正常 → 模型是可跑的，MNN 侧假设不匹配

### Step 5: 读 shader 里的假设注释（快速定位假设是什么）

MNN Metal shader 里的关键假设都有**明确注释**，`grep -n "Assumption\|causal-lower-triangular\|mask is a no-op\|hard-codes causal"`：

```
source/backend/metal/MetalAttentionShader.hpp:558:  Assumption: the mask provided ... is causal-lower-triangular
source/backend/metal/MetalAttentionShader.hpp:651:  causal ADD/SET masks are 0/pass in the valid region
source/backend/metal/MetalAttention.mm:531:  FA also hard-codes causal masking via `kv_valid_offset = seq_k - seq_q`
```

新增 kernel 优化时**必须**留下这种注释；review 时**必须**读这些位置。

### Step 6: 加固方向（若确认此类）

- **Metal：已实现（2026-07-31）** —— `MetalAttention.mm` `_computePathFlags` 从 `inputs[3]` 形状派生 `mCausalLayout`（真实张量 mask ⇒ 非 causal ⇒ 逐元素 honor、关全部 causal 优化；标量/无 mask + kvcache ⇒ causal）。配套 `llm.cpp` 对 metal 后端也发标量哨兵 causal mask（同 cpu/hexagon）。`MNN_METAL_QK_CAUSAL_TRI` 已删除
- **其他后端 / 通用长期方向**：runtime 首次 attention encode 时抽样验证 mask 是否 lower-triangular 并缓存（工作量最大但通用）；或导出侧 `llm_config.json` 落 `attention_type` 字段权威标注

## 7.4 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的隐式假设 |
|------|-------------|
| SWA 模型（Mistral v0.1/Gemma-2）乱码，Qwen 正常 | attention mask 假设 causal（本册） |
| Prefix LM / BERT 类整段乱 | attention 假设 causal 或 KVCache 单向 |
| 短 prompt 对、长 prompt 错（错的位置在开头附近） | causal-tri 的上三角覆盖累积错 |
| 短 prompt 错 | prefix LM / bidirectional 从第一步就崩 |
| MNN_METAL_QK_CAUSAL_TRI=0 就对 | causal-tri/bound 假设 |
| MNN_ENABLE_FLASH_ATTN_PREFILL=0 后仍错 | FA + 三段都错，模型本身非 causal |
| 只有某几层错 | 层级差异（如 Gemma-2 交替 SWA / full） |
| 换 Metal → CPU 就对 | 后端优化（本册）；换 CPU → Metal 就对 = [`memory-aliasing.md`](memory-aliasing.md) 或 [`export-and-quant.md`](export-and-quant.md) |
| **换 Metal → CPU 就对，且不崩不报错、只是输出质量差**；模型是编码器 / ViT / 无 mask 的 attention | 后端把"无 mask 输入"当因果。查 `IsAttentionMaskCausal` 与 kernel 里的**裸 `#else`** 因果分支（§7.5.1） |
| host 侧发了 `-DXXX_MASK` 但后端行为不像它生效 | 那个宏可能是 vestigial 的——去 `.cl`/`.metal` 源里 grep 宏名，确认真的被引用（§7.5.1） |

## 7.5 参考案例

### 7.5.1 CPU 把"无 mask 输入"无条件当因果，静默三角化所有 FuseFmhaV2 模型（2026-10-13，已复现）

**症状**：**没有任何报错**。走 `FuseFmhaV2` / `FuseSelfAttentionV2` / `FuseSelfAttentionV3`
转换出来的模型（源图 attention 子图里完全没有 mask 运算），在 CPU 上被算成因果注意力，
在 Metal / CUDA 上算对。输出有限、连贯、看似合理，只是模型被结构性地"只能看左上角"。
OpenCL 同样错，但**不是**因为回落到 CPU——它跑自己的 kernel，kernel 里有一个同构的
缺省因果 bug（见下面的"已闭合"小节）；Vulkan 本机无 libvulkan，未能验证。

**这个 bug 不是从故障现象发现的，是从审计约定一致性发现的** —— 起因是问"什么模型会出现
『无 mask + kv_cache=false』这一档"。所以 Step 1-6 的常规流程在这里不适用，下面记的是实际
走通的路径。

**排查路径**：

1. **查谁会产出这个输入形态，一直查到转换器层。** 全仓只有两个 pass 造 `OpType_Attention`，
   对 mask 的态度相反：
   - `FuseAttention.cpp:114-122`：mask **强制**——既不是 `IsSelect`（SET 型）也不是
     `IsBinaryAdd`（ADD 型）就 `return false` 不融合，所以它产出的 op 恒为 4+ 输入。
   - `FuseFmhaV2.cpp:250/501/706`：匹配的图案里**根本没有 mask 运算**
     （`Softmax ← MatMul ← mul(scale) ← MatMul ← slice`，QKV 同源），产出 `{q, k, v}`
     **3 输入** + `kv_cache=false`。

   旁证：`FuseTransformerC4.cpp` 多处用 `inputIndexes.size() < 3`（不是 `< 4`）做校验，
   说明 3 输入形态是流水线里预期存在的。
2. **确认两边后端的缺省语义。** CPU：`isLowerTriangular = (mask == nullptr)`，无条件因果。
   Metal：`DEFAULT_MASK` 门控在 `mKVCache` 上，`kv_cache=false` 时不发 mask key → 无遮挡。
3. **实测钉死。** `AttentionTest` 的 unit test 3 正是这一档（`kv_cache=false` + 只给 3 个输入，
   参考实现用空 mask 算即全可见），但被 `skipNoKvCache = cpuInfer` 在 CPU 上跳过、挂着
   `// TODO: CPU support kv_cache == false`。把跳过开关临时打开：
   `Error Position: Output[0][1][9]` + `Error: Attention without kv_cacheunit test failed!`；
   同一用例 Metal 14/14 通过。

**根因**：跨后端对"缺省输入"的语义契约不一致。有 KV cache ⟹ 自回归 LM 层 ⟹ 无 mask 时因果是
合理默认；无 KV cache ⟹ 一次性注意力（视觉 / 交叉 / 双向编码器）⟹ 没有依据假设因果，"无 mask"
必须意味着"不遮挡"。Metal 一直是后者，CPU 是前者。

**修复**：在 `core/AttentionMaskConvention.hpp` 里加 `IsAttentionMaskCausal(hasMask, dims, kvCache)`，
CPU 与 Metal 共用同一个谓词；解封 unit test 3 在 CPU 上的跳过。验证：CPU 家族 14/14、CPU
精度×线程四档 14/14、Metal 家族 4 轮交替无失败、`llm_demo`（LLM 与 VLM × CPU 与 Metal）四组
输出与基线**逐字节相同**（md5 一致），确认 LLM 因果路径（rank-0 + kv_cache=true）未受影响。

**已闭合（2026-10-13）：OpenCL 的同构 bug，根因在 kernel 而非 host 判据，已按对称方式修复并实测。**
解封后在 OpenCL 上实测，失败位置与 CPU 修复前**完全相同**（`Output[0][1][9]`），而且是错值不是 crash。先要排除一个
误判：macOS 上 Apple 的 OpenCL 仍提供 **CPU 设备**，所以后端会正常初始化、**并不回落到
CPUAttention**（若回落，CPU 的修复就会让它通过）——判据是"没有像 Vulkan 那样打印
`Don't support 3` / `Can't create executor`"。真正的原因在
`attention_flash_prefill_buf.cl`：因果分支是 `#if defined(ADD_MASK) || defined(SET_MASK)`
的**裸 `#else`**，所以只要没有逐元素 mask 宏，kernel 就是因果的；而
`AttentionBufExecution.cpp:1257` 为这种情况发的 `-DDEFAULT_MASK` **是 vestigial 的，没有任何
.cl 引用它**。因此修 OpenCL 必须动 kernel 源 + 跑 `opencl_codegen.py`（会重新生成**所有**
`*_mnn_cl.cpp`），不是改一个 build option 就行 —— 见 [`opencl-optimize`](../opencl-optimize/SKILL.md)
的"改 `.cl` 必跑 codegen"。

**修法（与根因对称）**：host 侧 `AttentionBufExecution` 接入共享形状分类
`IsAttentionMaskCausal(hasMask, dims, kvCache)`——`kvCache` 取 `AttentionParam::kv_cache`，
**不要**误用 `KVCacheCLManager` 那个 meta-presence 的 `mKVCache`——无 dense mask 且非因果时
显式发 **`-DNO_MASK`**（新增的第三个状态，不再落裸 `#else`）；kernel 侧三处 `#else` 收窄为
`#elif !defined(NO_MASK)`（kv_end clamp、`kv_valid_offset` 声明、逐元素因果判断），裸
`#else` 保持因果缺省。OpenCL 的 dense 边界保持 `dims>2` 未动（rank-2 mask 在 OpenCL 仍不算
dense，与 rank≥2 约定有意分歧）。

**验证**：Mali 安卓真机 fp16 buffer 模式——`attention_nocache_mask` 的 bidir_sentinel 误差从
20.6-20.8 降到 0.009-0.019（与 all-visible dense 值**逐位一致**），
`attention_cache_bidirectional` 的 sep 从 0.000000 升到 8.4/5.0/4.3（≫ tol）；
host macOS OpenCL fp32+fp16 下 op/attention 14 套件全过。fp32 腿必须由 host 补：这台 Mali
的 fp32 环境性坏掉（`trans_3d_buf` 海量 CL -36），而 CI 的 OpenCL 单测阶段跑的正是
precision=1。Vulkan 仍未验证；读代码它把 absent/rank-0 映射到
`maskMode 0`（其 shader 视为不遮挡），**方向上反而是对的**。

**教训（本案例最反直觉的一条）**：`-DXXX` 这样的 build option **存在不等于被消费**。
host 侧发了 `-DDEFAULT_MASK`、名字也贴切，很容易让人以为 kernel 里有个 `#ifdef DEFAULT_MASK`
分支在按它行事；实际 kernel 用的是裸 `#else`，那个宏删掉也不会有任何变化。排查"某后端行为
不符合 host 侧配置"时，**先去 .cl / .metal 源里 grep 那个宏名**，确认它真的被引用。

**排除项与它们为什么误导**：

1. **差点判定成"理论角落"。** 第一轮只在 `AttentionTest` 里找到手搓的 3 输入 op，据此以为没有
   真实模型会走到。**"谁会产出这个输入形态"必须查到转换器 / 导出器层，只看运行时引擎会漏掉
   整条来源。**
2. **差点漏掉 in-tree 生产者排查就动手。** 把"无 mask → 因果"改成"看 kv_cache"，会连带改变
   **rank-0 哨兵 + kv_cache=false** 的语义（从因果翻成双向）。所以动手前逐个查了所有发 rank-0
   哨兵的地方：`llm.cpp`（LLM 层 kv_cache=true）、`embedding.cpp:118-131`（注释明写
   "Standard lower-triangular causal mask"，走带 past_kv 的导出）、`omni.cpp` talker
   （`talker.py:279-281` 特意用零长度 past KV 让每层产生 `Concat(past,new)`，从而让
   FuseAttention 设 `kv_cache=true`）、`dflash.cpp:230-232`（喂 rank-4 稠密，不受影响）。
   全部安全才改。**改缺省语义前，先枚举所有依赖旧缺省的调用方。**
3. **修复时最容易写错的一点**：`noElementMask` **不能**跟着改成含 kv_cache 的版本。它必须继续用
   不含 kv_cache 的哨兵判据（`scalarCausalSentinel || isBidirectional`），否则 rank-0 +
   kv_cache=false 会同时失去因果身份**和**哨兵身份，掉进 `_maskQK` 的 dense 路径按
   `maskCols = seqLen+pad` 逐行读一个 1 元素缓冲 → 越界。两个标志职责不同：
   `isLowerTriangular` 决定"要不要遮"，`noElementMask` 决定"要不要读 mask 内存"。

**避坑要点**：

- **跨后端的缺省值语义属于契约，要主动对照，不能等它出问题。** 这类 bug 不崩、不 NaN、没有日志，
  只有模型质量下降，从故障现象反推几乎不可能。
- 判据是"这个后端在**输入缺失/歧义**时假设了什么"。逐个后端列一遍，分歧就会自己浮出来。本例中
  Vulkan / CUDA 把 absent/rank-0 一律读成全可见（不看 kv_cache），与 CPU/Metal 仍不一致 ——
  已知、已记录在 `AttentionMaskConvention.hpp`，未修。
- 单测里被 `skip` + `TODO` 挡住的用例，往往正对应一个真实的后端能力缺口。看到
  `skipXxx = cpuInfer` 这类按后端跳过的逻辑，值得追一次"为什么跳"。

### 7.5.2 （占位）SWA / prefix-LM 走 Metal 的生产案例

**待补**：目前尚无生产 SWA 模型跑 Metal 后端出错的**已复现**案例入库（分支中 Qwen 系列均为
causal，未触发）。若未来第一次实测出现 SWA/Gemma-2/prefix LM 走 Metal 报错，务必按 Step 1-6
走完 + 补充参考案例到此节。

**预期案例形态**（供未来复现参考）：Mistral 7B v0.1 W4-b32 导出 → MNN Metal 后端 → 长 prompt
(>4096 tokens) → 输出在 window 边界后开始重复/漂移；CPU 后端一致乱（因为 CPU attention 也可能
不按 SWA 特化）；HF torch 侧正常；`MNN_METAL_QK_CAUSAL_TRI=0` **仅缓解 causal-tri/bound 部分**，
FA 本身仍错 → 需要架构层加固。

## 7.6 相关文件索引

| 文件 | 作用 |
|------|------|
| `source/backend/metal/MetalAttentionShader.hpp` | CAUSAL_TRI / CAUSAL_BOUND 的假设注释位置（`grep Assumption`）；prefill_qk/prefill_qk_tensor/prefill_qkv 三个 kernel 的实现 |
| `source/backend/metal/MetalFlashAttnShader.hpp` | FA kernel（同样 hard-code causal） |
| `source/backend/metal/MetalAttention.mm` | mQkCausalTri / mCausalBound / mFlashAttnPrefill 的 gate 条件；FA 的 causal-only comment (`:531`) |
| `source/backend/metal/MetalSoftmaxShader.cpp` | softmax CAUSAL_BOUND 分支实现 |
| `skills/metal-optimize/env-registry.md` | `MNN_METAL_QK_CAUSAL_TRI` 等相关开关的完整语义登记 |
| `skills/metal-optimize/kernel-dev-and-optimize.md` | causal-tri / CAUSAL_BOUND 的设计文档（§2.3.1）|
| `source/core/AttentionMaskConvention.hpp` | mask 三分类 + `IsAttentionMaskCausal(hasMask, dims, kvCache)` 的权威定义，以及各后端对 absent/rank-0 的分歧现状（§7.5.1） |
| `tools/converter/source/optimizer/merge/FuseAttention.cpp` | mask **强制**（`:114-122`，非 Select/Add 就 `return false`），产出 4+ 输入 op |
| `tools/converter/source/optimizer/merge/FuseFmhaV2.cpp` | 三个 pass（`:250/501/706`）产出 **3 输入 + kv_cache=false** 的 op，即"源图无 mask 运算"那一档 |
| `source/backend/cpu/CPUAttention.cpp` | `isLowerTriangular` / `noElementMask` / `useMaskInSoftmax` 三个标志的职责划分（§7.5.1 排除项 3） |
| `source/backend/opencl/execution/cl/attention_flash_prefill_buf.cl` | 无 mask 时因果曾落在 `#if defined(ADD_MASK)\|\|defined(SET_MASK)` 的**裸 `#else`**（`-DDEFAULT_MASK` 无人消费）；2026-10-13 起非因果档由 host 显式发 `-DNO_MASK`，三处 `#else` 收窄为 `#elif !defined(NO_MASK)` |
| `source/backend/opencl/execution/buffer/AttentionBufExecution.cpp` | OpenCL host 侧 mask 分类：`mHasMask`（dense 边界 `dims>2`）/ `mCausalLayout`（`IsAttentionMaskCausal` + `AttentionParam::kv_cache`）→ `-DADD_MASK` / `-DSET_MASK` / `-DNO_MASK` 三态 |
| `source/backend/opencl/execution/cl/opencl_codegen.py` | 改 `.cl` 后必跑，重新生成所有 `*_mnn_cl.cpp` |
| `test/op/AttentionTest.cpp` | unit test 3（`kv_cache=false` + 3 输入）及其按后端跳过的 `skipNoKvCache` 逻辑 |
