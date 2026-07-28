# Metal 推理性能优化 Playbook（Kernel + 系统架构 + 失败实验 + 基建）

> **配套 SKILL.md 的 sub-doc**：把 [`kernel-basics.md`](./kernel-basics.md) 的方法论具象化为可复用的性能优化知识库。
> 案例以 LLM decode/prefill 为主战场（优化密度最高、数据最全），但 GEMM/GEMV/attention kernel 技巧、管线同步消除、测量方法论对 CNN / Diffusion 等其他模型同样适用。
> 内容分四大块：**一、Kernel 层优化（GEMM/GEMV/Attention）**；**二、系统架构层级优化**；**三、无效/负收益实验记录**（避免重试）；**四、基础设施与测量方法论**。文末附前瞻路线图。

## 0. 优化总纲

LLM decode 每步生成一个 token，核心链路：

```
RMSNorm → Q/K/V Linear(GEMV) → RoPE → Attention(QK+Softmax+AV) → O Linear(GEMV)
       → RMSNorm → Gate/Up Linear(GEMV) → SiLU*mul → Down Linear(GEMV) → Residual
```

- **Decode**：GEMV（矩阵-向量乘）占 60-80% 时间，是优化主战场；其次是 Attention 和 RMSNorm。但小模型（0.6B 级）decode 受**串行 CPU+同步管线**约束（GPU 利用率 ~60%），kernel 级 GPU 节省不一定兑现为 wall——见 §二、§四方法论。
- **Prefill**：GEMM 占 ~50%；Attention 三段中间物化（mTempQK / mTempSoftMax）是长 prompt 显存/带宽瓶颈，causal-bound / flash-attention 是主杠杆。
- **战略**（2026-07 修订）：**Prefill 走 kernel 深化，decode 走管线深化**。

### 累计总账（分支 feature/metal-causal-skip-v2 vs master `342fdcb18`，同机交替配对）

**Qwen3-0.6B, M4 Pro, Metal**：

| 指标 | master | 分支 | 提升 |
|---|---|---|---|
| tg128 decode | 306-310 | 322-324 | **+4.7%** |
| pp2048 prefill | 4077 | 4303 | **+5.5%** |

- **M5 上 prefill 收益更大**（causal-tri 系列对 tensor-API 路径的扩展）：0.6B pp2048 **+51.5% vs master**。
- **iPhone 17 Pro（iOS 26.5）**：Metal4 tensor API 探测修复后 Qwen3.5-2B prefill **953 → 1884 tok/s（+97%）**。
- Decode 实验批次（2026-07-24）：0.6B mergedqkv tg128 347 → ~360（+5.2%）；mergedqkv 链路已删除（`515f077247`），基线改用 head-b32：tg128 = 340 tok/s。

### vs llama.cpp Metal Q4_1（M4 Pro, 4 线程）

| Model | pp512 | pp2048 |
|---|:---:|:---:|
| Qwen3-0.6B | **1.014× ✅** | **1.025× ✅** |
| Qwen3-4B | 0.983× | 0.980× |
| Qwen3-8B | 0.911× | 0.948× |

8B 差 5-9% 是 **base GEMM perf 问题**（OFF baseline 已经 0.94-0.92×），非 attention 范畴。

---

# 一、Kernel 层优化（GEMM / GEMV / Attention）

## 1.1 GEMV（decode 主战场）

### 1.1.1 Q4 GEMV Deferred Dequantization（+28%，系列最大单项）

**问题**：标准 Q4 GEMV kernel 中，每个 simdgroup 线程在累积循环的内层同时做 nibble 解包 + 反量化（乘 scale + bias）+ FMA 累积。反量化涉及 fp16 乘加，是瓶颈。

**优化**：延迟反量化——内层循环只做整数累积（int8 × int8 → int32），循环结束后一次性反量化：

```metal
// 旧：每步反量化
for (k) {
    half w = dequant(packed_w[k]);  // 每步 fp16 乘加
    sum += input[k] * w;
}

// 新：延迟反量化
int32_t isum = 0;
for (k) {
    int8_t w = unpack(packed_w[k]);  // 只做整数解包
    isum += int32_t(input_quant[k]) * int32_t(w);
}
sum = half(isum) * scale + bias;  // 循环外一次反量化
```

**实现要点**（`source/backend/metal/ConvSimdGroupShader.hpp`）：
1. **输入也需要量化**：input 从 fp16 动态量化为 int8，在 host 端（`MetalConvolution1x1.mm`）分配量化 buffer 和 scale buffer
2. **双 buffer**：`mTempInput`（量化后 int8）+ `mInputScales`（per-row scale）
3. **kernel 内部**：先对 input 做 per-row absmax 量化，然后整数 GEMV，最后 `result = isum * input_scale * weight_scale + weight_bias * input_sum`
4. **input_sum 修正**：weight 是非对称量化（有 zero point），需额外累积 `sum(input_quant)` 用于 bias 修正

**Dispatcher**（`MetalConvolution1x1.mm`）：deferred dequant 条件 = area=1 (decode) + supportSimdGroupReduce。

**性能**：Qwen3-0.6B Q4, Mac M4：标准 GEMV decode ~140 tok/s → deferred dequant ~180 tok/s（**+28%**）。

### 1.1.2 双 Simdgroup GEMV + ushort4 向量加载（g4m1_2sg）

**问题**：单 simdgroup GEMV 的 occupancy 受限于寄存器压力和 simdgroup 数量；weight 读取粒度 `uchar4`（4 bytes）未充分利用 burst。

**优化**：
1. **双 simdgroup 并行**：一个 threadgroup 内 2 个 simdgroup 分别处理不同 OC 范围，input 通过 threadgroup memory 共享，TG 数减半
2. **ushort4 向量加载**：weight 用 `ushort4`（8 bytes）一次读取，load 指令数减半

```metal
kernel void conv1x1_gemv_g8_deferred_sg2(
    ...
    uint sgid [[simdgroup_index_in_threadgroup]],  // 0 or 1
    uint lid [[thread_index_in_simdgroup]], ...) {
    int oc_start = gid * 16 + sgid * 8;
    threadgroup half shared_input[IC_CHUNK];
    if (sgid == 0) { /* 协作加载 input */ }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 各 simdgroup 独立 GEMV，simd_sum reduction 在 simdgroup 内完成
}
```

**注意**：`ushort4` 读取需 weight buffer 8 字节对齐；双 simdgroup 要求 OC ≥ 16；小 OC 层仍走单 simdgroup kernel。

### 1.1.3 Pre-scaling Nibble Extraction（约 +5%，叠加在 deferred dequant 上）

标准解包的 `>> 4` / `& 0xF` / `- 8` 各占 ALU 指令。**Pre-scaling trick**：host 端 pack weight 时预乘系数，nibble 提取用乘法（MAD）替代 shift，同时隐式完成 zero point 减法——mask-only 解包，位权补偿预乘进输入。

### 1.1.4 GEMV 带宽效率画像与路线结论（counter profiler 定量，M4 Pro）

同一 kernel `g4m1_2sg` 在不同 dispatch 权重下的带宽（decode）：

| 场景 | 每 dispatch 权重 | 实测带宽 | 效率 |
|---|---|---|---|
| 0.6B 独立 GEMV（q/k/v/o/down） | ~0.8MB | **85GB/s** | 35% |
| 4B 独立 GEMV | ~5.9MB | 173GB/s | 71% |
| 4B gate_up fused | 28.8MB | 206GB/s | 84% |
| 4B lm_head g16 | 225MB | 226GB/s | 92%（到顶） |

→ **小权重 GEMV 是 latency-bound 不是 kernel-bound**：正确解法是减少 dispatch 数 / 增大单 dispatch 体量（导出期合并），不是继续调 kernel。

**路线关闭结论**：g4m1_2sg 的 lane/TG 配比微调已 **4 次证伪**（WIDE_MIDDLE / 4SG / unroll / split-K，见 §三）；M5 上 middle_step / VEC2 / VEC4 变体同样中性偏负。剩余 GEMV 空间需要权重预取 / 更大 OC tile + split-K 级别的**结构性改动**（高风险）。lm_head g16 已 182-226GB/s 接近带宽上限，headroom 小。

## 1.2 GEMM（prefill）

### 1.2.1 Fused Q4/Q8 GEMM（in-kernel 解包）+ M64 tile（tensor-API 设备）

`b71528f0d` 落地，`9ea642eed` 收敛开关。

- 机制：tensor-API 设备（M5+）prefill 量化 conv 在 GEMM kernel 内解包反量化，省 dequant 预处理 dispatch + mTempWeight 分配（~4× 权重体积带宽往返）；M64 tile（Q4+area≥128 自动）再省一半跨 TG 权重读冗余
- 实测：M64 tile M5 Qwen3-4B pp512 **+5.9%** / pp2048 **+6.8%**（greedy 前 20 token 逐字一致）
- 开关：`MNN_METAL_DISABLE_FUSED_Q4_GEMM=1` 回退

### 1.2.2 M64 sg_matrix GEMM 移植到 M4（EXP10，收益低于预期）

`conv1x1_fused_q4_gemm_stage_m64` 的计算主体在 `#ifdef USE_METAL_TENSOR_OPS` 内，M4 sg_matrix 移植 = 从零写新 kernel（寄存器×2、threadgroup mem 翻倍、全新 index math）。

- 实测（M4）：全场景 **+0.8~1.7%**（0.6B pp512/2048 +1.0/1.1%，4B outdeq +0.8~1.7%）；3 模型 greedy 一致；388/388 单测。**远低于路线图 +10% 预期——M4 GEMM 瓶颈不在权重重复读**
- 处置：kernel 保留，`MNN_METAL_GEMM_M64=1` 门控默认关（M3 待验证）
- 定量背景：0.6B pp3312 的 outdeq_gemm ≈ 5.3 TFLOPS（fp16 峰 ~7.5），**prefill GEMM 已在 ~70% 算力峰值**，M64 上限本就 ~+10% e2e

### 1.2.3 In-shader dequant 4M 阈值改面积相关（EXP11，已合入默认生效）

EXP10 副产物发现：4B 的 in-shader 4M 阈值在长 prompt 失效，outer-dequant 路径 pp1024 +3.4% / pp2048 +5.3%。改为 `area<512` 才走 in-shader（一行启发式，无新 env）：

- 4B pp2048 **+5.2%** / pp1024 +3.4% / 2B pp2048 +3.0%；pp256/512 无回归；峰值内存 +2MB；greedy 一致；388/388
- ⚠️ **M3 验证为 merge blocking**（4M 启发式有 M3 回退前科）；8B 待导出补测
- 另：M4 上 in-shader wquant GEMM 与 outer-dequant 实测仅差 0.1%——两条路径都未到硬件上限

### 1.2.4 iOS 26.5 Metal4 tensor API 探测修复（`acc1afaab`）

- 问题：MPP `matmul2d` 要求 M/N 至少一个 16 倍数、静态 K 16 倍数；探测 kernel 描述符错误导致探测失败 → tensor API 整体禁用；另有 legacy 16x16x8 路径（静态 K=8）探测通过但运行时反复编译失败反而回退
- 实测：修复后 iPhone 17 Pro Qwen3.5-2B prefill **953 → 1884 tok/s（+97%）**
- 教训已录 `skills/ios-llm-bench/SKILL.md`

## 1.3 Attention Kernel

### 1.3.1 Causal 三角 QK dispatch + 有界 softmax/AV（CAUSAL_TRI / CAUSAL_BOUND，默认开）

`9948c74e1` → `78ae7bc55` → `f28510967`，分支 `feature/metal-causal-tri-v2`。**M4/M5 上 prefill 最大单项收益**。

**机制**：causal mask 下三角假设下，上三角区域在 mTempQK/mTempSoftMax 中**完全不写不读**，省 QK 写 + softmax 读 + softmax 写各 O(seq²/2) 带宽：
1. `CAUSAL_TRI`（prefill_qk）：host 只 dispatch 因果对角线下的梯形 tile（pp512 tile 数 -48%），kernel 内二次方程反解线性 tile id → (slq, slk)；interior tile 跳过全部 per-element mask 读取/分支（三区域分解）
2. `CAUSAL_BOUND`（softmax_plane/_sg + prefill_qkv）：softmax 每行只归约/写出 causally-valid 前缀 + 24 元素零 pad（覆盖 prefill_qkv 的 8 对齐 tile 读界）；prefill_qkv 的 av_k_upper 截断同步激活（此前 qkvPrefillKeys 从不带 mask 宏，是死代码）

**实测**（M4 Pro，交替配对 rep=3，`-fa 0`）：

| 指标 | Baseline | 优化后 | Δ |
|---|---|---|---|
| 0.6B pp512 | 4879.4 | **5088.1** | **+4.3%** |
| 0.6B pp2048 | 3689.0 | **4346.9** | **+17.8%** |
| 4B pp512 | 686.1 | 695.6 | +1.4% |
| 4B pp2048 | 610.3 | **649.2** | **+6.4%** |
| 4B tg128 | 75.1 | 75.6 | ~0%（decode 无回归 ✓）|

M5（tensor-API 路径扩展增量）：0.6B pp2048 **+38.9%**，累计 +51.5% vs master。CAUSAL_BOUND 单项：pp2048 +26%、pp512 +12%、4B pp2048 +8.2%。

- 正确性：FA off + greedy 256 token 对拍，tri on/off 逐字一致
- 开关：`MNN_METAL_QK_CAUSAL_TRI=0` 回退（默认开；⚠️ **非 causal 模型必须 =0**）；条件门控 = simd-matrix 路径 + causal mask 宏 + kv>=q + 非 tensor-API + 非 FA
- 注意：decode 侧不存在等价优化——seq_q=1 时 1×kv 分数行 100% 因果有效，无三角可跳（`trivialFloatMask` 已把 mask 开销清零）

### 1.3.2 M4 级设备 FlashAttention 降级到三段路径（`472c76bd8`）

**发现**：优化后的三段路径（+causal-tri/bound）在 M4 Pro 上反超 FA，且差距随 seq 增长——causal-bound 省的是 O(seq²) 带宽而 FA kernel 未享受：

| 指标 | FA on | 三段+causal-tri v2 | 三段反超 |
|---|---|---|---|
| 0.6B pp512 | 4947.6 | 5088.1 | +2.8% |
| 0.6B pp2048 | 4077.6 | **4346.9** | **+6.6%** |
| 0.6B pp3312 | 3362 (FA kernel 414ms) | 3623 (QK+AV+softmax 合计 362ms) | +7.8% |

**处置**：M4 档默认走三段+causal-tri（pp2048 4088→4319，+5.7%，kv≤8192 生效）；FA 保留给长上下文（kv>8192，省 O(seq²) scratch 内存）/ head_dim∉{64,128,256} 之外场景兜底；`MNN_ENABLE_FLASH_ATTN_PREFILL=1` 可强制 FA。M5 同样默认 demote。M3 待验证。

### 1.3.3 Fused Prefill Flash-Attention（保留场景：长上下文 / 特殊 head_dim）

**问题**：标准三段 pipeline `prefill_qk` → `softmax_plane_sg` → `prefill_qkv` 通过 global memory 传递中间结果。Qwen3-0.6B pp2048 单层 mTempQK / mTempSoftMax 各 128 MiB，write+read ~512 MB/前向。

**方案**：融合 Q·K^T + online softmax + P·V 到一个 kernel，中间数据全部留在 threadgroup memory 和寄存器：
- **Q_TILE=16, KV_TILE=32, NSG=4**（128 线程）；Grid `(ceil(seq_q/16), num_head*batch, 1)`
- 每 simdgroup 拥有 2 行 Q 的 `M`（running max）/ `S`（running sum）寄存器
- 每 KV 块：QK → 在线 softmax → 同一段 P 做 PV → 累加到 `so`（O accumulator）

**Threadgroup memory 布局**（D=128 时 ~15 KB）：
- `sq[Q_TILE * HEAD_DIM]` half — Q 分块（cooperative load 一次）
- `sf[Q_TILE * KV_TILE]` float — QK 的 fp32 scratch
- `ss[Q_TILE * KV_TILE]` half — 归一化后的 P（half 存以便 half×half → float MMA）
- `so[Q_TILE * HEAD_DIM]` float — O accumulator，在线 rescale

**关键文件**：`MetalFlashAttnShader.hpp`（`gPrefillFlashAttn`）、`MetalAttention.mm/hpp`（门控 / pipeline 编译 / dispatch 分支）。

**实施要点**：
1. **Eligibility 门控**（onEncode）：config `attention_mode / 8 >= 1` 或 env `MNN_ENABLE_FLASH_ATTN_PREFILL=1`（=0 强制关，A/B 用）；且 supportSimdMatrix、非 mKvInDisk、head_dim ∈ {64,128,256}、group_size ∈ {1,2,4,8}、非 mShortSeq 且 seq ≥ 128。KV int8（mQuantKey/mQuantValue）通过 QUANT_K/QUANT_V 宏支持
2. **Pipeline keys**：`{"prefill_flash_attn", ftype, group_str, "HEAD_DIM_N"}` + 可选 `HAS_MASK` / `ATTENTION_C4` / `QUANT_K` / `QUANT_V`
3. **Dispatch**：grid `(ceil(seq/16), B*H, 1)`，threadgroup `(32,4,1)`；命中后 `continue` 跳过三段分支
4. **在线 softmax 数值稳定性**：`M_new = simd_max(fmax(M[j], s))`；`ms`/`vs` 对 `-INFINITY` 双短路是必需的（否则 `exp(-inf - -inf)` = NaN 从初始态或全 masked 行传播）
5. **KV int8**：D=256 下不能整 tile 反量化到 threadgroup（爆 32KB），每 k_step 分批 8×8 dequant；`k_scales`/`v_scales` 是 `device ftype*`（fp16）**不是** float，错声明必乱码；用 `threadgroup_barrier` 不是 `simdgroup_barrier`（8 lane 写、32 lane 读）

**避坑要点（踩过的坑）**：
1. **`ATTENTION_C4` 输出布局** — 最重要的坑。c4-head export 时 output 实际布局是 `[num_head*(head_dim/4), batch*seq_q, 4]`（NC4HW4-packed）。不区分则 token 输出**从第一步就乱码**且代码逻辑看着完全正确、地址全部合法。正确 epilogue：
   ```cpp
   #ifdef ATTENTION_C4
       int o_off = (h * (param.head_dim / 4) + (d / 4)) * 4 * param.batch * seq_q
                 + (b * seq_q + q_abs) * 4 + (d & 3);
   #else
       int o_off = ((b * seq_q + q_abs) * param.head_num + h) * param.head_dim + d;
   #endif
   ```
2. **不要用 threadgroup memory 中转 K/V**：初版怀疑 `simdgroup_load` 5 参 transpose flag 有 bug 而预排布局，正确但 pp2048 掉 45%。真 bug 是 ATTENTION_C4。经验：**先怀疑数据布局，再怀疑 Metal API**
3. **mixed-dtype MMA 只有 all-half 或 all-float**：QK 输出先写 fp32 `sf`，softmax 读 fp32、算完转 half 写 `ss` 供 PV MMA——两块 scratch 不能合并
4. **softmax→PV 之间的 `threadgroup_barrier` 不可少**：各 SG 只 rescale 自己 2 行 `so` 但 PV 读全部 8 行
5. **正确性验证必须 greedy sampling**（temperature=0, top_k=1）对拍前 40-60 token byte-identical，否则采样噪声掩盖数值差异

**性能数据**（M4 Pro，W4-block32 Q4，Metal fp16，vs 未优化三段路径）：

| Model | pp512 | pp2048 | tg128 |
|---|---:|---:|---:|
| Qwen3-0.6B | +2.5% | **+11.7%** | noise |
| Qwen3-4B | +2.1% | **+4.7%** | noise |
| Qwen3-8B | +3.1% | **+9.1%** | — |

正确性：Qwen3-0.6B/4B/8B × 256/512/1024.txt 共 9/9 组合 greedy 前 30 token byte-identical；Qwen3.5（head_dim=256）经 QUANT_K/V 支持 FA + KV int8。

**参数调优实验记录**：

| 变体 | 0.6B pp2048 | 0.6B pp512 | 结论 |
|---|---|---|---|
| Q_TILE=8, KV_TILE=32（初版）| +5.2% | +0.8% | 起点 |
| **Q_TILE=16, KV_TILE=32** | **+11.7%** | **+2.5%** | ✅ 采用 |
| Q_TILE=16, KV_TILE=64 | +3.6% | **−6.4%** ⚠ | ❌ 回退 |
| Q_TILE=32 | — | — | 跳过：threadgroup mem ~30KB 逼近 32KB 上限 |

教训：**减少 K read 冗余（Q_TILE↑）是长 prompt 最有效杠杆**（Grid.x 减半换 K 读半减）；KV_TILE↑ 反而差（累加器+每 iter V 读的开销抵消）；whole-tile causal early-exit 已达下三角 50% 理论上界，无需显式 block classifier。

**已探明的收益/风险边界**（不建议盲改）：F=2 多头融合（threadgroup mem 15→30KB，occupancy 减半，净收益不确定）；去循环末 barrier（4B pp512 -14%，barrier 有带宽调度作用，全保留）；`so` 显式清零必须保留（threadgroup 初值可能 NaN）；NSG 4→8 无收益。

### 1.3.4 Split-KV Decode Attention（长上下文，已默认开）

`ca68e4692`（M5 探索）→ `a93c1dfc7` 转正；EXP07 M4 标定；EXP09 KV int8。

- **机制**：flash-decoding 式——长 kv 时单 workgroup 串行扫 KV 是并行度瓶颈；KV 分段到多 workgroup 各算 online-softmax 部分结果（llama.cpp `flash_attn_ext_vec` 式单 pass QK+online softmax+AV），再跨 workgroup reduce
- **实测**：M4（EXP07）kv4K decode 0.6B **+19%** / 4B +5.5% / 2B +1~3%；M5 kv3600 +2.6% / kv5472 +4.3%，随 kv 增长扩大；kv 小时持平偏负（融合 kernel 已优）
- **KV int8 支持**（EXP09）：量化 KV decode 0.6B kv2K **+16.6%** / 4B kv4K **+15.6%**（fp16 同点仅 +5.5%——int8 带宽减半假设证实）。V 反量化折叠进 softmax 权重（s_vs 预乘 scale + vb bias 归约），AV 内层零 dequant 数学；fp16 路径 byte-identical
- **开关**：默认开（kv≥3072 触发，保守档）；`MNN_METAL_DECODE_SPLITKV=0` 关闭，`=N` 覆盖阈值；交叉点 M4 ~1.2-1.5k / M5 ~3072
- **踩坑**：① onEncode 内路径 flag 判定必须放在 `handleKVAllocMemory()` 之前，否则首个 decode step 临时缓冲未分配 → setTensor(null) SIGSEGV；② reduce kernel 必须 128 线程（32 线程版占用率不足吃掉收益）；③ nwg 启发式 div256 是 M4/M5 共同甜蜜点（EXP08 div128/512 均 -2~5%，配比微调路线闭合）

### 1.3.5 Fused Decode Attention GQA 扩展（group_size 2-8）

原 `decode_qk_softmax` fused kernel 只支持 group_size=1。扩展为模板化 group_size（`MetalAttentionShader.hpp` 编译时宏 + `MetalAttention.mm` dispatcher 按 `num_heads/num_kv_heads` 选 kernel），避免 Q/K 显式 repeat_kv 拷贝。对 GQA 模型（Qwen3 g=2, Llama3 g=4）**decode attention 提速 10-20%**。

## 1.4 其他 Kernel

### 1.4.1 RMSNorm 小 Batch 优化（`MetalLayerNorm.mm`）

Decode 时 batch=1，默认 kernel 选择倾向大 batch tile，launch overhead 反而大。`batch <= 4 && hidden_size <= 4096` 时用单 threadgroup 处理整个 norm。Decode RMSNorm 提速 ~5%，链路 ~1%。

---

# 二、系统架构层级优化

## 2.1 Dispatch 融合（运行时）

### 2.1.1 Gate/Up 双投影融合（leader/follower 单 dispatch，默认开）

MLP gate/up 两个同形状 GEMV 共享输入（RMSNorm 输出），合成一次 dispatch，每层省 1 次。

- **机制**：从 `MetalBinary` 的 MUL_SILU 输入关系发现 Gate/Up 配对；Leader（gate）在 `gid.z=2` 维度双路 dispatch，Follower（up）onEncode 直接返回
- **关键文件**：`MetalBinary.mm`（配对发现）、`MetalBackend.mm/hpp`（`registerConv1x1ForOutput`）、`MetalConvolution1x1.mm/hpp`（`setupGateUpFusion`、leader/follower 调度）、`ConvSimdGroupShader.hpp`（`GATE_UP_FUSED` 宏）
- **Buffer 绑定**：index 0 input（共享）/ 1 gate output / 2 const / 3-5 gate weight/bias/dequant / 6 up output / 7-9 up weight/bias/dequant
- **实测**：约 **+1%**（M5 A/B：238.2 vs 关闭 236.4）；`MNN_DISABLE_GATE_UP_FUSION=1` 关闭
- ⚠️ 对照：同思路的 **QKV triple runtime fusion 是 -36% 负收益已删除**（见 §三）；merged 导出普及后本机制也应淘汰（路线图 #9）

### 2.1.2 LN Fusion（RMSNorm 融进 GEMV kernel，默认开）

`87fb545fe`（含 sole-consumer 规则）。

- **机制**：post-attn RMSNorm 由下游 Conv1x1 GEMV in-kernel 计算（读 hidden+residual、写 residual 和、归一化输入），LayerNorm dispatch **57 → 29 次/token**
- **实测**：GPU 时间省 ~170us/token；e2e 约 **+1%**（M5 A/B：238.2 vs 236.1）；greedy 逐字一致；`MNN_METAL_DISABLE_LN_FUSION=1` 关闭
- **踩坑**：① profile 模式下 follower 的空 command buffer 仍计数计时，"calls 数"不能判断融合是否生效，要看 avg 时间或 debug 打印；② `matchLNFusions`/`matchQKVFusions` 的 backend 指针与注册 backend 可能不同（create/execute backend 分离），debug 必须打印 `this` 指针对账

## 2.2 Pipeline / 同步开销（decode 管线深化）

### 2.2.1 onResizeBegin per-backend fence（替代全局 GPU drain，默认开）

发现 `9356700fb` → 落地 `7f186691a`。

- **根因链**（MNN_RESIZE_TRACE 探针定位）：steady decode 的大图输入不脏不重排；每 token 强制 resize 的是 **1-op 的 logits `StridedSlice` 子模块**（`mUseContentInputs=true`）。代价不在自身而在 `MetalBackend::onResizeBegin` 的**无条件全队列 wait 排空**——每 token 中段把主图 GPU 工作与 lm_head 强制串行，这就是 60% GPU 利用率、~1.2ms/token "resize" 段的真正机制
- **修复**：改为 `waitOwnInflight()` 只等本 backend 自己最后一次 commit（`mLastOwnCommandBuffer`）。语义正确性：allocator reset 只需本 backend 自己的在飞工作完成
- **实测**：0.6B decode **+4% 稳定**（greedy 逐字一致；fence 与 `none` 等价即收益全额保留）；4B tg128 持平；pp512 微升
- **开关**：默认 fence；`MNN_METAL_RESIZE_WAIT=global` 回退，`=none` 跳过全部（实验）
- **方法论教训**：本机 tg128 存在热态双峰（~280 段 vs ~320 段），跨时段 A/B 完全污染——先前 NOWAIT 的 "+14%" 实为跨热态高估，真实 +4%

### 2.2.2 Content-cache：内容依赖子模块的 resize 缓存（Core 层，全后端受益，默认开）

`1ef0ace93`。

- **机制**：logits-slice 子模块的 shape 依赖控制张量内容（`logits_index`，decode 期间恒为 -1），每 token 全量 resize。`StaticModule::_resize` 对 content-for-shape 输入（整型控制张量）做**内容字节比对缓存**——内容未变则跳过 resize（每 token 一次 4 字节 memcmp）。浮点输入（embeds/mask）不参与比对，>4KB 或非 host 可读输入自动回退 always-resize
- **实测**：resize 段 **317ms → 6ms**（re-encode 131→6 次，仅剩真实 prefill↔decode 转换点）；tg128 +1%（fence 已吃掉大部分 wall 收益，本项把 CPU 侧 resize 成本结构性归零）；4B 77.3（新高）；386/386 单测；`MNN_LLM_CONTENT_RESIZE_ALWAYS=1` 回退
- **决策记录（为何不做导出端 shape-static 根治）**：`logits_index` 是有意的单图多模态设计（all-logits=0 / last-logit=-1 / spec-decode 变长）；根治要拆图或双输出，而 hidden-states 切片在 lm_head 之前正是为省 all-token lm_head 计算，双输出自相矛盾。content-cache 后残余价值 ≈ 0（16us/call × 6 次合法转换点），**不单独立项**；若 MUL_SILU_PACKED 导出改版落地可作搭车项

### 2.2.3 队内 H2D 上传（staging ring + queue-ordered blit，EXP02，默认开）

`757610526`（同事合入，rebase 引入）。

- **机制**：每 token 输入上传从"drain 全 GPU + 直写"改为 staging ring 槽位（命令缓冲租约复用）+ 队内 blit，队列顺序保证安全，per-token drain 归零
- **实测**：0.6B decode **+5.2%**（4 轮配对）、3.5-2B +1.6%、4B +1%，pp512 无回归，388/388 单测；`MNN_METAL_H2D_QUEUED=0` 回退

### 2.2.4 Decode 瓶颈定性画像（0.6B/M4，指导后续投入方向）

关键探针数据（详细过程见 §四工具）：

| 探针 | 结果 | 结论 |
|---|---|---|
| 双实例并发（早期）| 单实例 308.9，双实例合计 519.8 = 1.68× | 单流 decode GPU 利用率仅 ~60% |
| 双实例复测（EXP05，H2D 合入后）| 1.41×，剩余空泡 ~29% 在 kernel 间隙 | CPU 阻断已消 |
| CPU trace 生产模式 | op encode 仅 0.92us/op（0.16ms/token, 5%）；wait[copyD2H] 2.7ms/token 为剩余 wait 全部 | encode 便宜，ICB 复用死路；logits 回读是语义必需（sampling 依赖）|
| M5 对照 | 双实例 1.31×，空泡 ~24% | M5 更偏 GPU-bound，kernel 优化 e2e 兑现打 7 折 |

**结论**：CPU 阻断（resize drain / H2D drain）消除后，**<5us 级 GPU 节省不再兑现为 wall**（EXP04、commit N=50 busy -42ms wall 不动均证实）。wall 收益只剩两类：**CPU-GPU 串行化消除**（EXP02 类）与**大体量 GPU 节省**（导出期合并类）。剩余杠杆：GPU argmax/async sampling（被 `mExecutor=CPU` llm.cpp:1240 阻塞，Express 动态 op 跑 CPU，_ArgMax 无法消费 device logits，需 executor 改造）、投机多 token、GEMV 结构性带宽。
4B 及以上不受管线约束（GEMV 占 67%、GPU busy 逼近 wall；Sync 9.5%），GPU 优化仍直接兑现——**优化项要按模型档分别评估**。

## 2.3 图优化 / 导出侧

### 2.3.1 RoPE Fusion（inv_freq 直接传入，单 op）

标准 RoPE 需要 position_ids → inv_freq → cos/sin → 乘加多个小 op，各自 launch。融合为单个 op 直接接收 `position_ids + inv_freq`：

- **关键文件**：`schema/default/MNN.fbs`（RoPE 增加 `hasInvFreq`）、`MetalRope.mm`、`CPURoPE.cpp`（对齐）、`transformers/llm/export/utils/custom_op.py`（FusedRoPE 导出）、`utils/transformers.py`（inv_freq 传递）
- 配套 **RemoveDeadShapeOp** pass（`tools/converter/source/optimizer/postconvert/RemoveDeadShapeOp.cpp`）：清除 fusion 后无消费者的 Shape/Gather/Unsqueeze 死代码 op

### 2.3.2 导出期 QKV/GateUp 权重合并（P0 方向，实验已验证、schema 待评审）

**动机**：运行时 kernel 分段 fusion（QKV -36%）已证伪；正确做法是导出/图层权重 concat（Q/K/V → `[hidden, q_oc+k_oc+v_oc]` 单 conv，gate/up → `[hidden, 2*ffn]`，llama.cpp/vLLM 标准做法）——零 kernel 分支，自动受益全部现有 GEMV/GEMM 优化，CPU/OpenCL 同步受益。

**实验结果**（`MNN_EXPORT_MERGE_QKV=1`，M4 Pro，0.6B）：

| 指标 | 基线 | 合并（naive split 消费）| 结论 |
|---|---|---|---|
| decode conv GPU 时间 | ~2.0ms/token | **~1.62ms/token（-19%）** | ✅ 大 dispatch 带宽效率假设证实 |
| Raster（split 拷贝）| ~0 | +0.66ms/token（16.4% GPU）| ❌ naive Split 吃掉全部收益 |
| decode e2e | 268 tok/s | 247 tok/s（-8%）| 净回退；**零拷贝消费是硬前提** |

后续 packed-QKV 完整链路实测 decode **+6.2%**，但按需求整体移除（`515f077247`）；重做时直接用下方"零拷贝路径设计"。

**零拷贝路径设计**（重做时直接复用）：
- 新增 `MUL_SILU_PACKED` Binary 模式：gate/up 合并 conv 输出单 tensor `[2*ffn]`（ffn%4==0 时 NC4HW4 两半连续），Binary 同一 buffer 两偏移读取；Q4 分块量化按 OC 维 concat 权重无障碍
- QKV 合并还需 `FusedRoPE(q,k)` / `FusedAttention(q,k,v)` 支持通道偏移读取（C4 下 oc 段 4 对齐 ✓）
- 范围：schema enum 追加 + converter fuse pass + CPU/Metal 双后端 + 导出开关——跨层变更需单独评审
- 导出侧现状：q/k/v 为独立 `nn.Linear`（transformers.py:232-262）；原生融合模型（qkv_proj）当前反而被拆开（:152-181），需加导出开关

**⚠️ 跨后端合规前提（revert `7abf70c0f` 的根本原因，重做前必须解决）**：这是**模型格式变更**，不是 Metal 单点优化。MNN 原则"任意模型任意后端正确推理"靠 ① Geometry 分解兜底 ② CPU 逐 op 回退两个机制，而当时的垂直切片两者都没走通：
- `GeometryBinary.cpp` 里 MUL_SILU_PACKED 是直通（pass-through）非分解——OpenCL/Vulkan/CUDA 只能 CPU 回退，每层一次 device↔host 往返，GPU 后端性能灾难
- packed-QKV 通道偏移消费只实现在 Metal RoPE/Attention（`8bbebb710`），其他后端遇 packed 布局图不正确/不可跑
- 合规最低要求：MUL_SILU_PACKED 需 geometry 分解兜底（slice+SiLU+Mul，非支持后端退化为带 Raster 的正确路径）；packed-QKV 需**所有**实现 ATTENTION/RoPE 的后端支持通道偏移，或 geometry 层为不支持后端插显式 slice
- 产品约束"不适用于所有后端的改变模型的优化"即源于此；立项前先确认该约束是否解除

**⚠️ Converter 匹配塌方陷阱**（引入新图形状必踩）：`fuseHiddenStateC4Regions` 是 all-or-nothing 链式匹配——单个 matcher 因新图形（packed 单 conv）失配导致整个 hidden-state C4 区域融合放弃（Convert 2→114），converter 无告警、模型正确但慢 6%。铁律：① 新图形状必须审计**所有**下游 matcher（`matchMlpFromPostLayerNorm` / `matchHiddenBlockFromAttention` 等）；② all-or-nothing 优化必须带匹配率诊断（已加 `matched N / M attention blocks` 警告）。phase-2 QKV 合并将再次触发 `matchHiddenBlockFromAttention` 三 conv 假设，适配必做。

### 2.3.3 LLM 导出兼容性修复：inv_freq RoPE 的 unsqueeze 错误

模型显式指定 `head_dim` 且 `head_dim ≠ hidden_size / num_heads` 时（如 Qwen3-0.6B head_dim=128, hidden=1024, heads=16），cos/sin 维度变换错误：

```python
# 错误：unsqueeze(2).unsqueeze(1) 将 [seq, dim] → [seq, 1, dim, 1]，dim 落在 num_heads 位置
cos = torch.cat(..., dim=-1).unsqueeze(2).unsqueeze(1)
# 正确：[seq, dim] → [1, seq, 1, dim]，与 query_states [bsz, seq, heads, head_dim] 正确广播
cos = torch.cat(..., dim=-1).unsqueeze(0).unsqueeze(2)
```

关键文件：`transformers/llm/export/utils/transformers.py`。

---

# 三、无效 / 负收益实验记录（避免重试，为后续优化留档）

## 3.1 总表

| 实验 | 结果 | 处置 / 复用价值 |
|---|---|---|
| 运行时 QKV triple fusion（kernel gid.x 分段）| 0.6B tg128 **-36%** | 整条路径删除；正解是导出期合并（§2.3.2）|
| Decode QK+softmax+AV 全融合（FUSED_AV）| **-8.7%** | 回退；见 3.2 |
| causal-tri v1（fill kernel 方案）| pp512 ~0% | 被 v2（bounded softmax）取代；fill 写带宽 = 原 skip-tile 写带宽 |
| FA tileFullValid fast-path | -0.6~-1.1%（噪声）| revert；`in_bounds` uniform 分支硬件早已优化，真杠杆在 partial-tile 8x8 sub-block skip（复杂，ROI 低）|
| FA KV_TILE=64 / 去循环末 barrier / NSG=8 | pp512 -6.4% / 4B pp512 -14% / 无收益 | 见 §1.3.3 边界记录 |
| Async decode fire pattern + `mAsync` | profile ON +28% **全是测量伪影**，生产 0 收益 | revert；教训录 §4.2 |
| ICB / encode 复用 | encode 仅 0.92us/op（5%），假设证伪 | 不做；CPU trace 数据留档 |
| Session_Resize_Fix 全图缓存（`MNN_LLM_RESIZE_CACHE`）| greedy 发散且无收益 | 删除；`Pipeline::fixResizeCache` 加 Attention 硬排除保护；被 content-cache 取代 |
| EXP01 commit 粒度扫描（`MNN_METAL_COMMIT_NUM`）| 默认 N=10 已最优（N=2 -7%，N=999 -6%）| env 保留供设备标定 |
| EXP03 GEMV split-K 双 SG 协作 | -2.6% | 回退；**g4m1_2sg 配比微调第 4 次证伪（WIDE_MIDDLE/4SG/unroll/split-K），路线关闭** |
| EXP04 KV append 折入 decode_qk_softmax | greedy 一致但 wall ~0%（2.9us GPU 省下落入空泡）| 回退；思路对 4B+/iPhone（管线空泡更小）可能仍有价值 |
| EXP08 split-KV nwg 启发式微调（div128/512）| 均 -2~5% | 回退；div256 是 M4/M5 共同甜蜜点 |
| lm_head 4SG / G16_OC4 变体 | e2e 持平（4SG stddev 7× 恶化；OC4 kernel -4.8% 但 42us 落入空泡）| 删除，结论存档 |
| M5 decode GEMV 变体（middle_step / VEC2 / VEC4）| 中性偏负（VEC4 -1.4%）| M4 的 lane 划分调优在 M5 依然成立 |
| Decode 因果 mask 优化 | 数学上不存在（seq_q=1 无三角可跳）| `trivialFloatMask` 已清零 mask 开销 |
| MUL_SILU_PACKED / packed-QKV 导出链路 | packed-QKV decode **+6.2%**（正收益！）但因跨后端合规缺口 + 产品约束整体移除（非性能证伪） | 重做方案与合规前提见 §2.3.2 |
| greedy readMap 快路径 | ~0%（sample 由 GPU 同步主导）| 保留（省 600KB/tok 拷贝，零风险）|

## 3.2 关键负结果详解（含根因，防止换皮重试）

**QKV triple fusion / FUSED_AV 同类根因**：在 Apple GPU 上，**dispatch 合并省的启动开销 < 并行度/occupancy 损失**，尤其小模型 grid 本来就薄。FUSED_AV 把 AV 折进 `decode_qk_softmax`（grid = B×kv_heads = 0.6B 仅 8 个 TG）后，head_dim 维并行度从 `heads×head_dim` 线程网格坍缩到 8 TG × 4 SG。正确性通过（greedy 256 token 一致）但 297.8 → 271.8 tok/s。若重试需把 grid 扩为 (kv_heads × head_dim_block)——结构大改非增量。

**Async decode fire pattern（M5，最重要的测量教训）**：counter profiler 的每 op encoder pass sample buffer attachment 让 CPU encode 从 0.92us/op 涨到 4-20us/op（≈20×），**profile ON 环境人为制造 CPU 阻断 gap**（Raster→Raster 191 次 × 648us）。据此做的 fire 重排（sample → forwardVec(fire) → tokenizer_decode）在 profile ON 下 +28.2%，**production build 5-rep 交替中位 235.4 vs 235.1 = 0 收益**。生产环境 CPU encode sub-us、GPU busy ≈ wall，无 idle 可回收。原路线图 "wait 期间 CPU 闲置 ~1.4ms/token" 是 profile ON 特有。**此路无解，除非 speculative decoding 打破 sample→forward 串行依赖**。

**Session_Resize_Fix 为何死路**：92% command 可缓存但 wall 零收益——它只跳过本来就便宜的 shape 重算（~5us/call），贵的 allocMemory 簿记和 onResizeBegin GPU 排空照旧；且 greedy 发散（存在 onResize 烘焙值依赖状态的 op）。正确方向是 fence（§2.2.1）+ content-cache（§2.2.2）。

---

# 四、基础设施与测量方法论

## 4.1 Profiling / 诊断工具

### 4.1.1 Per-op GPU Profiling（`MNN_METAL_OP_PROFILE`，counter-sample 模式）

```bash
cmake .. -DMNN_METAL=ON -DMNN_METAL_OP_PROFILE=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
make -j8 llm_demo
```

现默认 **MTLCounterSampleBuffer** 模式（`MetalBackend.mm`，`887a1871a`）：
- 每 op 一个 compute encoder（`MTLComputePassDescriptor.sampleBufferAttachments` stage 边界采 GPU timestamp），command buffer 保持正常 commit 节奏——**绝对数字真实**
- tick→ns 两点校准（`MetalGpuTickScale`）；sample buffer 池全局化；单 command buffer 超 512 encoder 时 seal 续接
- 融合 follower 的空 encoder 用 `profileDropCurrentSample()` 主动丢弃（否则出现"LayerNorm 全部在 dispatch"伪影）；子阶段拆分走 `profileNextSubpass(subtag)`
- `MNN_METAL_OP_PROFILE_LEGACY=1` 回退旧模式（每 op 一 command buffer，绝对数字失真但相对排序可靠）

测量开销对比（0.6B）：legacy prefill 3556 / decode 206；counter **3869 / 255**；生产 ~4900 / ~294。

使用注意：第一次 inference 有 pipeline 编译开销，跳过前几个 token 再看数据；**profile ON 的 op 间 gap 被放大 ≈20×，不能作为优化目标**（见 4.2）。

### 4.1.2 时间轴 CSV + 甘特图分析（`tools/script/metal_profile_gantt.py`）

```bash
export MNN_METAL_OP_PROFILE_TIMELINE=/tmp/dec.csv
build/llm_demo config.json prompt.txt 100   # 需 -DMNN_METAL_OP_PROFILE=ON
tools/script/metal_profile_gantt.py /tmp/dec.csv
```

输出 GPU busy/idle 总账、8 档 gap size 分布、Top-N gap transitions、Top-N 单个最大 gap。（`0b1808ca5`/`dfe569084`）

### 4.1.3 CPU 侧 trace（编译宏 `-DMNN_SESSION_CPU_TRACE`）

`bd1ed1ce4`/`93409e9d9`，编译宏门控（生产 build 不编译，零开销）；开启后运行即统计、退出时打印，无需环境变量。输出：
- Session 级：resize / encode / malloc / run 计时
- Metal 级：op encode / commit / wait 三段拆分；wait 站点归因（resizeFence / copyD2H / copyH2D / onSync）
- GPU busy/gap 统计（completion handler）

resize drain、H2D drain 两个最大 decode 杠杆均由此工具发现。

### 4.1.4 其他

| 工具 | 出处 | 用途 |
|---|---|---|
| iOS 一键真机 bench（`ios_llm_bench.sh`）| `acc1afaab` | framework→打包→装机→定长 bench 全自动，分支对比 |
| env 开关注册表（[`env-registry.md`](./env-registry.md)）| `1cd9d39dd` 等 | 20→13 个 Metal env 集中登记：性能路径/融合 dispatch/profiling 三类，默认值/打开效果/定型状态/命名规范 |
| 双实例并发探针 | 无代码 | 同 GPU 跑两个 tg128，合计/单实例比值 = GPU 空泡定量（1.68× → 1.41× 演进见 §2.2.4）|

## 4.2 测量方法论铁律（血泪教训汇总）

1. **正确性 oracle 先于性能**：greedy sampling（temperature=0, top_k=1）对拍前 N token byte-identical 是黄金标准；采样噪声会掩盖数值差异
2. **交替配对 + 热态分段**：本机 tg128 存在热态双峰（~280 vs ~320 段），跨时段 A/B 完全污染（fence 的 "+14%" 实为跨热态高估，真实 +4%）；decode 对照必须交替配对且观察分段
3. **首轮数据不可信**：EXP07 2B 首轮 -12.7% 为冷启动伪影（rep2-4 正向）；本机噪声下限 ~±5%（M5 首轮测出 "V1 -41%" 是热/负载假象）
4. **profile ON 制造的 gap 不是优化目标**：counter profiler 绝对 GPU 时间可信，但 op 间 gap 放大 ≈20×；甘特图 idle 分析要在 profile ON/OFF 两种 build 下交叉验证——仅 profile ON 才见的 idle 是伪影；对生产 gap 的追问用 profile OFF 侧信道（双实例探针、wall vs CPU 侧 forward time 差值）
5. **profile 模式下 "calls 数" 不能判断融合是否生效**（follower 空 buffer 仍计数），看 avg 时间或 debug 打印
6. **GPU 时间省了 ≠ e2e 快了**：小模型 decode 的 <5us 级 GPU 节省落入空泡；收益必须按 wall 而非 GPU busy 评估，且按模型档（0.6B 管线约束 vs 4B+ GPU-bound）分别评估
7. **跨设备不可互推**：M3/M4/M5/iPhone A 系列需分别标定（4M 启发式有 M3 回退前科；M4 split-KV 交叉点 1.2k vs M5 3072）
8. **先怀疑数据布局，再怀疑 Metal API**（ATTENTION_C4 教训）
9. **all-or-nothing 图优化必须带匹配率诊断**（converter 匹配塌方 -6% 静默回归教训）

---

# 五、前瞻路线图（2026-07-23 修订 + 批次更新）

## 高优先级（定量依据的确定性收益）

| # | 方向 | 依据与预期 | 规模 |
|---|---|---|---|
| 1 | **Phase-2 QKV packed**（RoPE/Attention 通道偏移消费 + `matchHiddenBlockFromAttention` 单 conv 适配）| conv -19% 实验实锤；packed-QKV 曾实测 decode +6.2%；fence+content-cache 已消 CPU 空泡，GPU 节省可兑现；matcher 适配必做。⚠️ **前提**：跨后端合规（§2.3.2）+ 产品约束解除，否则挂起 | 中大 |
| 2 | ~~M64 sg_matrix GEMM 移植（M4 档）~~ | **已完成（EXP10）**：仅 +0.8~1.7%，远低于预期，M4 GEMM 瓶颈不在权重重复读；kernel 保留默认关 | — |
| 3 | **M3 Pro 验证**：causal-tri、FA 门控、fence、EXP11 面积阈值（merge blocking）| 全部启发式仅 M4 Pro 标定；M3 有 4M 启发式回退前科 | 小（纯测试）|

## 中优先级（瓶颈已定位，有阻塞或需设计）

| # | 方向 | 状态 |
|---|---|---|
| 4 | GPU argmax / async sampling | 4B decode Sync 9.5%；被 `mExecutor=CPU`（llm.cpp:1240）阻塞，需 executor 改造或图级 argmax |
| 5 | ~~排空尾部重叠 / async fire~~ | **搁置**：生产 0 收益（profile 伪影，§3.2）；除非 speculative decoding |
| 6 | MUL_SILU_PACKED schema 转正 | 垂直切片正收益已验证（非证伪）；⚠️ 与 #1 同前提：跨后端合规（geometry 分解兜底 + 各后端 Execution，§2.3.2）+ 产品约束解除 + schema/private 评审；搭车项：logits 切片模式化导出 |
| 7 | FA 补课 | tileFullValid 已证伪；剩余：partial-tile QK 8x8 sub-block skip（复杂）、扩 head_dim 96/fp32、kv>8192 段重标定；M4/M5 已 demote 到三段，ROI 有限 |
| 8 | GEMV 带宽新思路 | 微调路线 4 次证伪已关闭；需权重预取 / 更大 OC tile + split-K 级结构改动；4B 173→245GB/s 有空间 |
| 9 | 投机多 token decode | 打破 sample→forward 串行的唯一出路（§3.2 async 结论）|

## 低优先级（架构清理与基础设施）

| # | 方向 | 状态 |
|---|---|---|
| 10 | 淘汰 GateUp leader/follower 运行时配对机制 | 等 merged 导出普及 |
| 11 | env 开关注册表 | ✅ 完成（`env-registry.md`）|
| 12 | gantt 分析工具化 | ✅ 部分完成；待做：交替配对 + 热态分段检测脚本 |
| 13 | iPhone A 系列 / M5 跨设备标定 | Mac 数字不可互推 |

---

## 附：关联文档

- env 开关速查：[`env-registry.md`](./env-registry.md)
- 构建/测试/基线：[`build-and-test.md`](./build-and-test.md)
