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

### Decode 批次二（分支 feature/llm-gpu-sampling，2026-07-27/28，M4 Pro Qwen3-0.6B Q4）

7 个提交串行叠加（每项 greedy byte-identical 对拍 + 交替配对 A/B）：

| 提交 | 优化 | tg256@p12 | tg128@p512 | tg128@p2048 |
|---|---|---|---|---|
| `7b2c8bfcd8` | 设备端采样（ArgMax/TopKV2，§2.2.5）| — | — | — |
| `ca8496a648` | commit cadence 自动调优（§2.2.6，已移除）| — | — | — |
| `94a73ab19a` | encode replay 基建（§2.2.7）| — | +0.5% | — |
| `405beb8aa4` | attention replay（§2.2.7）| +1.4% | ~0% | +2.0% |
| `1b9f7ee8cf` | QKV 三投影融合 dispatch（§2.1.3）| +2.7% | +1.6% | ~0% |
| `a48d0d4910` | 输入 LN 融进 QKV（§2.1.3）| +1.9% | +1.7% | — |
| `db13a99f4f` | Split-K decode GEMV（§1.1.5）| +3.9% | +3.8% | +3.3% |
| **批次末端绝对值** | | **364.9** | **335.9** | **222.4** |

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

**路线结论（2026-07-28 修订）**：g4m1_2sg 的 lane/TG 配比微调曾 4 次证伪（WIDE_MIDDLE / 4SG / unroll / EXP03 split-K，见 §三），但 **SPLIT_K_2（§1.1.5）第 5 次尝试成功**——关键差异是保留 pre-scaling 内循环、行内 K 流对半拆给 2 个 simdgroup（在途读加倍），而非改 lane 划分。M5 上 middle_step / VEC2 / VEC4 变体中性偏负。lm_head g16 已 182-226GB/s 接近带宽上限，headroom 小。

### 1.1.5 Split-K Decode GEMV（SPLIT_K_2，+3.3~3.9% e2e，默认开）

`db13a99f4f`。

- **问题**：2sg GEMV 每行由 1 个 simdgroup 串行流式读整行，小投影（0.5-1.8MB 的 o/down/qkv）只有 88-137GB/s，而 87MB lm_head 达 252GB/s（92% 峰值）——小矩阵行数少、每行在途读不足，latency-limited
- **机制**：`SPLIT_K_2` 保留 kernel 原有 pre-scaling 内循环，改为 4 simdgroup/threadgroup——每行的 quant block 对半拆给 2 个 simdgroup，各算半段部分和，经 threadgroup memory 合并，行内在途读加倍
- **门控**：普通 area==1 decode GEMV（Q4/Q8 + oc%8==0 + 偶数 quant block 数）；gate/up / QKV / LN 融合管线——**SPLIT_K_2 扩展已证伪**（EXP12，§3.1：p512 -3.3% / p2048 -1.8%），融合管线的正确解是 §1.1.6 ROW_2 双行双流（行内 ILP，无 barrier）
- **实测**（M4 Pro 0.6B）：tg256@p12 +3.9%、tg128@p512 +3.8%、tg128@p2048 +3.3%；greedy byte-identical
- **反例存档**：先路由到现有 g8 kernel 的方案 e2e **-5%**——g8 的 nibble-unpack 内循环慢于 2sg 的 pre-scaling trick，勿回退到该方向
- **开关**：`MNN_METAL_GEMV_SPLITK=0` 回退

### 1.1.6 双行双流融合 GEMV（ROW_2，融合 dispatch 专属，auto 开）

EXP12 证伪后（融合 dispatch 上加 simdgroup 数 -3%），换思路：**加行内 ILP 而非加 simdgroup**——把 attention 内核 s0/s1 双流技巧搬到 GEMV。

- **机制**：`ROW_2` 编译变体（仅 gate_up/QKV/LN 融合管线；plain 路径继续走 SPLIT_K_2）——每 simdgroup 同时处理 2 个相邻 output slice，双 raw_dot 累加流共享同一次 input 读（LN 前导也共享），**无 barrier、无额外 simdgroup**；grid.x 减半（每 TG 4 slice）。第二行越界时别名到第一行（安全读，结果丢弃），QKV 各投影尾部由 per-row guard 处理
- **实测**（M4 Pro 0.6B，冷机复测 2 轮交替 A/B）：p12 tg256 **+2.8%**（363.8→373.9）/ p512 **+2.4%**（337→345.3）/ p1024 **+2.3%**（301→308）/ p2048 **+1.2%**（256→259）——全 kv 段正收益（对比 SPLIT_K_2 融合版全负）；节流态首测 +3.9/+2.3/+1.1 与冷机一致
- **门控**：non-tensor-API 设备（M4/M5 双端标定完成后 `MNN_METAL_GEMV_ROW2` 三态覆盖开关已删除，设备门控即唯一逻辑）；三处 setup（gate_up/QKV/LN）独立解析同一公式，保证 pipeline 宏与 grid 一致
- **正确性**：热稳定窗口 greedy **byte-identical**（0.6B p1024 on/off 各 3 次同哈希 + p2048 + Qwen3.5-2B）；开发中一度误判 DIFFERS——实为深度热节流下基线自身非确定（§4.2 铁律 10）
- **为什么 ROW_2 赢而 SPLIT_K_2 输**：融合 kernel 的寄存器/占用余量吃不起翻倍的 simdgroup + barrier；双流方案在同一线程内加 ILP，占用不变，还顺带把 input/LN 读减半

### 1.1.7 ⚠️ 短序列 GEMV 路径（area 2..16）从未优化——multi-token / 投机解码的前置瓶颈（2026-07-29 实测）

**背景**：§4.3 结论四/五 把单 token 路径的全部候选逐一证伪后，唯一剩下的数量级杠杆是"一次前向多算几个 token"（权重只读一次摊薄到 B 个 token）。本节量化该路径当前的效率——**结论：空间很大，且瓶颈明确是并行度而非算法**。

**e2e 实测**（M4 Pro，0.6B b64，`llm_bench -p B -n 1 -rep 6`，prefill 的 `area` 即 B，正是投机解码会走的路径）：

| B | prefill tok/s | 每次前向 wall | vs B=1 | **理想** | 等效 decode 加速 |
|---:|---:|---:|---:|---:|---:|
| 1 | 349.1 | 2865 us | 1.00 | 1.00 | 1.00× |
| 2 | 490.4 | 4079 us | **1.42** | ~1.0 | 1.40× |
| 4 | 901.1 | 4439 us | **1.55** | ~1.0 | 2.58× |
| 8 | 1248.4 | 6408 us | 2.24 | ~1.0 | 3.58× |
| 16 | 1685.3 | 9494 us | 3.31 | ~1.0 | 4.83× |

decode GEMV 是**权重带宽 bound**，B 个 token 权重只读一次 ⇒ 理想 `cost(B) ≈ cost(1)`。实测 B=2 就要 1.42×、B=4 要 1.55×，**没吃到应有的摊薄**。注意 **B=2 反常地比 B=4 更差**（1.42× 才换 2 token，而 1.55× 能换 4 token）。

**归因（`build_prof` counter profiler，us/call × 每前向调用数；绝对值失真见 §4.2 铁律 4，此处只用相对量）**：

| B | 走的 kernel | 每前向 GEMV | vs B=1 |
|---:|---|---:|---:|
| 1 | `qkv_fused/gate_up_fused/splitk2_gemv_g4m1_2sg`（融合 + SPLIT_K_2）| 2229 us | 1.00 |
| 2 | `conv1x1_gemv_g4m2_wquant_sg` | 3939 us | **1.77** |
| 4 | `conv1x1_gemv_g4m4_wquant_sg` | 6419 us | **2.88** |

**根因（`MetalConvolution1x1.mm:616-644`）：短序列路径拿到的 simdgroup 并行度只有 decode 路径的一半，且完全没有融合。**

| | grid | threadgroup | SG/TG | 总 simdgroup |
|---|---|---|---|---|
| area==1（SPLIT_K_2）| `UP_DIV(oc,8)` | 128 线程 | 4 | **oc/2** |
| area 2..5（`g4mN`, piece=1）| `UP_DIV(oc,4)` | **32 线程** | **1** | **oc/4**（**减半**）|

- 这条路径**从未收到过单 token 路径的任何优化**：没有 SPLIT_K_2（K 拆分）、没有 ROW_2（双行双流）、也拿不到 QKV/GateUp/LN 融合——`mIs2sgDecode = true` 与 `registerConv1x1ForOutput/ForQKV` **只在 `area == 1` 分支设置**，所以 B≥2 时每层是 7 个独立投影 dispatch（q/k/v/o/gate/up/down）而非 4 个融合 dispatch。
- 与 §2.2.4.1 的诊断自洽：小 GEMV 只到峰值 41-52%，是**并行度/occupancy 受限**；把 simdgroup 数砍半 ⇒ 时间近乎翻倍，正好解释 B=2 的 1.77×。
- ⇒ **可动项明确**：把 SPLIT_K_2 / 多 SG 并行度（以及后续的融合）移植到 `conv1x1_gemv_g4mN`。目标是把 `cost(2)` 从 1.77× 压到 ~1.1×。GEMV 占 decode ~70%，若达标，B=2 前向 wall 从 4079us → ~2900us，per-token 从 2040us → ~1450us。
- ⚠️ **前置依赖已就绪**：`transformers/llm/engine/src/speculative_decoding/` 已有 `lookahead` / `ngram` / `tokentree` / `eagle` / `mtp`——**n-gram lookahead 不需要 draft 模型**。实际收益 = 本节的摊薄曲线 × 接受率，立项时需实测接受率。


## 1.2 GEMM（prefill）

### 1.2.1 Fused Q4/Q8 GEMM（in-kernel 解包）+ M64 tile（tensor-API 设备）

`b71528f0d` 落地，`9ea642eed` 收敛开关。

- 机制：tensor-API 设备（M5+）prefill 量化 conv 在 GEMM kernel 内解包反量化，省 dequant 预处理 dispatch + mTempWeight 分配（~4× 权重体积带宽往返）；M64 tile（Q4+area≥128 自动）再省一半跨 TG 权重读冗余
- 实测：M64 tile M5 Qwen3-4B pp512 **+5.9%** / pp2048 **+6.8%**（greedy 前 20 token 逐字一致）
- 开关：`MNN_METAL_DISABLE_FUSED_Q4_GEMM=1` 回退

### 1.2.2 M64 sg_matrix GEMM 移植到 M4（EXP10，收益低于预期）

`conv1x1_fused_q4_gemm_stage_m64` 的计算主体在 `#ifdef USE_METAL_TENSOR_OPS` 内，M4 sg_matrix 移植 = 从零写新 kernel（寄存器×2、threadgroup mem 翻倍、全新 index math）。

- 实测（M4）：全场景 **+0.8~1.7%**（0.6B pp512/2048 +1.0/1.1%，4B outdeq +0.8~1.7%）；3 模型 greedy 一致；388/388 单测。**远低于路线图 +10% 预期——M4 GEMM 瓶颈不在权重重复读**
- 处置：kernel 保留，`MNN_METAL_W4W8_OUTER_DEQUANT_GEMM_M64_SGMATRIX=1` 门控**默认关（定型，2026-07-31）**。M3 Pro 标定（配对 rep5 双向，0.6B）：pp512 **-1.4~-1.5%** / pp1024 中性 / pp2048 +0.8%，短 prompt 回归否决默认开；greedy byte-identical
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
- **开关**：默认开，阈值按设备 auto（`MetalEnv.decodeSplitKvThresh=0` → 使用处解析：tensor-API 设备 3072 / 其余 1536；**auto 再对 fused kernel kv 容量上限取 min**——group2:2048 / group4:1024 / group8:512，见下）；`MNN_METAL_DECODE_SPLITKV=0` 关闭，`=N` 覆盖阈值（显式值不做 clamp）；交叉点 M4 ~1.5k / M5 ~3072
- **M5 gap 段修复**（2026-07-28）：M5 crossover 3072 > fused cap 2048（GS2），kv∈[cap, 3072) 掉进三段 decode_qk 慢路径。auto 阈值 clamp 到 cap 后：0.6B p2048 tg128 **+4.2%**（151.8→157.8，交替配对 x2）/ 扫档段 p2560 +4.0%；4B（GS4，gap [1024,3072)）p1536 +1.9% / p2048 +1.7~4.5%。路由验证：新 auto 与显式 =2048 byte-identical。M4（GS2 1536<2048）行为不变，仅 GS4/GS8 模型在 M4 上也受益于 clamp
- **M5 QK_QSPLIT 标定**（2026-07-28）：强开 p1024 tg128 **-2~3%**（181.5/178.0 → 175.9/175.0，两对方向一致）——确认 auto gate 排除 tensor-API 设备正确，M5 不启用
- **M4 阈值扫档**（2026-07-28，Qwen3-0.6B p2048 tg128，rep3）：1024→254.3 / **1536→256.2** / 2048→253.0 / 3072（旧默认，kv2391 未触发）→221.3。1536 档 **+15.3%**，与 MLX（~267）差距从 -17% 缩到 -4%；p512（kv<1536 不触发）不受影响。routing 验证：auto 输出与显式 =1536 byte-identical
- **短 kv 反证**（2026-07-28）：强制在 kv~512 触发 split-KV（阈值 256/384/512）→ p512 decode 334→310（**-7%**，2 轮一致）。根因：0.6B group_size=2 → fused `decode_qk_softmax` grid 仅 8 TG（每 kv-head-group 一个），split-KV 把 kv 切 nwg=2 段后 grid 升到 16 TG，但多出的 reduce dispatch（16 TG）+ partial buffer 全局读写在短 kv 下开销 > 并行度收益。**结论**：阈值下限 1536 是对的，"提升 p512 attention 并行度" 这条路经 split-KV 不通；按 q-head 拆（而非 kv 拆）的 grid 结构后续做成了——见 §1.3.5 QK_QSPLIT（kv∈[512,1536) 档 +1.6~2.7%）
- **踩坑**：① onEncode 内路径 flag 判定必须放在 `handleKVAllocMemory()` 之前，否则首个 decode step 临时缓冲未分配 → setTensor(null) SIGSEGV；② reduce kernel 必须 128 线程（32 线程版占用率不足吃掉收益）；③ nwg 启发式 div256 是 M4/M5 共同甜蜜点（EXP08 div128/512 均 -2~5%，配比微调路线闭合）

### 1.3.5 Q-head-split Fused Decode QK+Softmax（QK_QSPLIT，kv∈[512,1536) 档，auto 开）

背景：0.6B group_size=2 时 fused `decode_qk_softmax` grid 仅 8 TG（每 kv-head-group 一个），GPU 核心吃不满；split-KV 只覆盖 kv≥1536（短 kv 反证 -7%，见 §1.3.4）。

- **机制**：`QK_QSPLIT` 编译变体（仅 GROUP_SIZE==2）——grid.z=group_size，每个 q-head 独占一个 TG（8→16 TG），threadgroup 内存减半（单 scores 流）；配合**半宽 threadgroup**（localSize=kv/2 向上取 32 对齐，总线程数与不拆分持平）。代价：K 每 kv-group 读 2 次、失去 s0/s1 双流 ILP
- **实测**（M4 Pro 0.6B tg128）：p1024 **+2.7%**（294→300）/ p768 +1.6~2.4%（312→318）/ p512 中性 / p12 ~-1%；p2048 走 split-KV 不受影响。greedy：auto vs off 在 0.6B（natural/p1024/p2048）+ Qwen3.5-2B p1024 全 byte-identical
- **门控**：auto = non-tensor-API 设备 + kv≥512 + group_size==2 + `mDecodeQkSoftmax`（M4 正 / M5 负双端标定完成后 `MNN_METAL_QK_QSPLIT` 覆盖开关已删除，auto 即唯一逻辑）。决策在 `_computePathFlags`（每 token 重估），变体翻转纳入 `_pathSignature` bit25 → replay 正确失效重录
- **踩坑**：① threadgroup 宽度是命门——沿用 kv/6 窄公式（TG 数翻倍触发启发式换挡）时 **-5%**，必须用 kv/2 半宽公式；② 强制开时短 prompt 输出可能与基线有合法 fp 重排差异（reduction 顺序变），auto 门控下 kv<512 不触发、无此现象
- **后续**：M5 已标定为负（2026-07-28 强开 p1024 -2~3%，auto 排除正确，见 §1.3.4）；iPhone 未标定；group_size=4/8 的泛化未做（generic kernel 数组索引已有 15% 编译器劣化前科，见 §1.3.6 注）

### 1.3.6 Fused Decode Attention GQA 扩展（group_size 2-8）

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
- ⚠️ 对照：早期 **gid.x 分段式 QKV triple fusion 是 -36% 负收益已删除**（见 §三）；gid.z 选路的新设计已成功落地（§2.1.3）；merged 导出普及后本机制也应淘汰（路线图 #9）

### 2.1.2 LN Fusion（RMSNorm 融进 GEMV kernel，默认开）

`87fb545fe`（含 sole-consumer 规则）。

- **机制**：post-attn RMSNorm 由下游 Conv1x1 GEMV in-kernel 计算（读 hidden+residual、写 residual 和、归一化输入），LayerNorm dispatch **57 → 29 次/token**
- **实测**：GPU 时间省 ~170us/token；e2e 约 **+1%**（M5 A/B：238.2 vs 236.1）；greedy 逐字一致；`MNN_METAL_DISABLE_LN_FUSION=1` 关闭
- **踩坑**：① profile 模式下 follower 的空 command buffer 仍计数计时，"calls 数"不能判断融合是否生效，要看 avg 时间或 debug 打印；② `matchLNFusions`/`matchQKVFusions` 的 backend 指针与注册 backend 可能不同（create/execute backend 分离），debug 必须打印 `this` 指针对账

### 2.1.3 QKV 三投影融合 dispatch + 输入 LN 融合（decode，默认开）

`1b9f7ee8cf`（QKV）+ `a48d0d4910`（LN 融进 QKV）。每 decode GEMV 有 ~3.4us 固定 launch/ramp 成本（§1.1.4 画像），融合 q/k/v 三投影每层省 2 次 conv launch，再融入 input RMSNorm 省 1 次 LayerNorm dispatch。

- **配对发现**（`matchQKVFusions`，onResizeEnd）：某输入 tensor 恰有 3 个 decode-GEMV Conv1x1 消费者即为 q/k/v 模式；执行序第一个当 leader，其余两个不 encode。QKV 匹配先于 LN 匹配跑，让 leader 身份对 LN 匹配可见
- **kernel**（`QKV_FUSED` 变体 of `conv1x1_gemv_g4m1_2sg_wquant_sg`）：`gid.z` 选投影；与 GATE_UP_FUSED 不同，三投影输出通道不同——follower 经 seg buffer 携带自己的 output_slice + scale_coef，grid.x 按最大投影取
- **LN 融合**（`LN_FUSED + QKV_FUSED`）：镜像 gate/up leader 的做法，input RMSNorm 在融合 QKV kernel 内联计算，single-writer residual guard 覆盖 grid.z 维度
- ⚠️ **关键坑——follower 输出必须 re-home 到 STATIC 内存**：融合 dispatch 在 leader 的较早执行位置就写出 k/v，而调度在 leader 与 k/v 消费者之间的动态池中间量（q/k-norm cast、RoPE、attention scratch）可能复用其区域——表现为 KV 增长触发池重排后 decode 乱码
- **实测**（M4 Pro 0.6B）：QKV 融合 tg256@p12 +2.7% / tg128@p512 +1.6%（p2048 中性，attention-bound）；LN 融合叠加再 +1.9% / +1.7%；均 greedy byte-identical
- **开关**：`MNN_METAL_DISABLE_QKV_FUSION=1` 关（LN 部分受 `MNN_METAL_DISABLE_LN_FUSION` 控制）

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

**结论**：CPU 阻断（resize drain / H2D drain）消除后，**<5us 级 GPU 节省不再兑现为 wall**（EXP04、commit N=50 busy -42ms wall 不动均证实）。wall 收益只剩两类：**CPU-GPU 串行化消除**（EXP02 类）与**大体量 GPU 节省**（导出期合并类）。剩余杠杆：GPU argmax/async sampling（✅ 已落地，§2.2.5）、投机多 token、GEMV 结构性带宽。
4B 及以上不受管线约束（GEMV 占 67%、GPU busy 逼近 wall；Sync 9.5%），GPU 优化仍直接兑现——**优化项要按模型档分别评估**。

#### 2.2.4.1 精确 decode 画像（2026-07-28 counter profiler 实测，替代上表的估算）

`-DMNN_METAL_OP_PROFILE=ON` 单独 build（`build_prof/`，勿覆盖生产 build），p512 取稳态 60 个 forward 平均：

| 指标 | 实测 |
|---|---|
| **GPU dispatch / token** | **266**（层内 9 × 28 + 层外 ~28）|
| **GPU busy / token** | **2950 us** |
| 生产 wall / token（同期 341-348 t/s）| **2874-2933 us** |

**⚠️ 关键修正：GPU busy(2950us) ≈ 生产 wall(2874-2933us) ⇒ 生产 decode 基本 100% GPU-bound，没有可回收的空泡。** 上表"单流 GPU 利用率仅 ~60%"/"空泡 ~29%"的解读**是错的**：双实例 1.41× 不代表有 29% 空闲时间，而是**单实例 kernel 填不满 GPU（occupancy 不足），两实例共同调度才喂满机器**——是 occupancy 效应，不是 idle-gap 效应。（profile 内测得的 17% "idle" 是 §4.2 铁律 4 的 profile 伪影。）
这一条统一解释了本轮全部零收益：EXP14 砍 28 个 dispatch 仅 +0.7%、EXP19 去掉 per-token 同步 ≈0、EXP16/17 kernel 微调中性——**dispatch 数与 CPU 侧都不在关键路径上**。

decode GPU 时间去向（2950us/token）：

| op | 调用/token | us/token | 占比 |
|---|---:|---:|---:|
| gate_up 融合 GEMV | 28 | 624 | 21.2% |
| o_proj + down_proj GEMV | 56 | 579 | 19.6% |
| qkv 融合 GEMV | 28 | 502 | 17.0% |
| lm_head g16 | 1 | 352 | 11.9% |
| **GEMV 小计** | 113 | **2057** | **70%** |
| Attention qk_short + av + copy | 84 | 644 | 22% |
| RoPE | 28 | 127 | 4.3% |
| BinaryOp / LayerNorm / Raster / Cast / Unary | ~41 | ~120 | 4% |

**换算成带宽——这才是真正的剩余 headroom**（M4 Pro 峰值 ~273 GB/s）：

| dispatch | 权重量 | 实测带宽 | 峰值占比 |
|---|---|---:|---:|
| qkv 融合（17.9us/call）| 2.0 MB | 112 GB/s | 41% |
| o/down（10.3us/call）| ~1.3 MB | 125 GB/s | 46% |
| gate_up 融合（22.3us/call）| 3.1 MB | 141 GB/s | 52% |
| **lm_head（352us/call）** | **77.8 MB** | **221 GB/s** | **81%** |

小 GEMV 只到 41-52% 峰值，lm_head 到 81%——差别纯粹是**单 dispatch 体量**（2-3MB 撑不起 ramp-up 与足够在途读）。若小 GEMV 能达到 lm_head 的 221 GB/s，GEMV 从 2057us → ~1210us，decode 从 341 → **~479 t/s**。SPLIT_K_2 + ROW_2 已把它从 85 推到 112-141 GB/s，**剩余 ~60% 空间是 occupancy / 在途读问题**（非布局，EXP17 已证连续度无效；非图结构，见 §2.3.4）。


### 2.2.5 设备端采样（ArgMax / TopKV2，缩减 decode 回读）

`7b2c8bfcd8`（`transformers/llm/engine/src/sampler.cpp`）。此前 wait[copyD2H] 2.7ms/token 全部来自 logits 回读（§2.2.4 CPU trace）。

- **greedy**：`Express::_ArgMax(logits, -1)` 设备端算 argmax，跨设备边界只回读 4 字节 index，替代整份 fp32 logits（Qwen3 vocab ~600KB/token）。MetalArgMax 的 first-max tie-break 与原 CPU 循环一致
- **mixed + topK 前置**：`_TopKV2` 设备端取 top-k values/indices，后续 pipeline 步骤在 k 大小子集上跑（`SamplerState.is_subset`）。**精确等价前提**：topK 是第一个有效过滤步——要求 `logit_bias`/`banned_tokens` 为空，且 topK 在 mixedSamplers 首位（或首位是 no-op penalty：所有 penalty 系数为默认值）
- 不满足前提自动回退整份回读，语义不变

### 2.2.6 Commit cadence 自动调优（decode）——已移除

`ca8496a648` 曾在 `llm.cpp` load 末尾对 Metal 跑 `tuning(OP_ENCODER_NUMBER, {10, 20, 40, 80})`。**实际从未生效**：调用点位于 `mContext->status = RUNNING` 之前，`tuning()` 开头的 `CHECK_LLM_RUNNING` 判定 NOT_LOADED(-1) 直接 return，只留下一条 `[Error]: LLM in error state. Status: -1` 日志。且 llm_bench/llm_demo 等调用方本就在 load 后显式调 `tuning()`（候选集更全），故直接删除。commit cadence 仍由调用方 `tuning()` 或 `MNN_METAL_COMMIT_NUM=N` 决定（EXP01 标定用）。

### 2.2.7 Encode Replay（稳定 shape 前向录制重放，默认开）

`94a73ab19a`（基建）+ `405beb8aa4`（attention 接入）。`source/backend/metal/MetalReplay.hpp/mm`。

**基建机制**：MetalExecution 的输入/输出设备地址在连续调用间保持一致时，录制其 encode（pipeline / buffer 绑定 / dispatch grid），后续调用直接重放捕获的命令列表，跳过 onEncode 的全部 CPU 逻辑。

- **安全模型**：每次重放前所有 tensor-backed 绑定对 tensor 当前 buffer+offset 重新校验（`metalReplayValidate`）；任何不匹配丢弃录制、退回正常 encode 并可重录（KV-cache 扩容与 allocator 重排由此兜住）
- **豁免**：encode 依赖 per-token CPU 状态的 op 经 `canRecordEncode()` 排除——linear attention（状态重置）仍豁免；attention 已通过 onReplayUpdate 接入（见下）
- **防抖**：连续 8 次重放失败的录制被 ban，防止系统性 bail 的 hook 反复重录
- 编译期：`MNN_METAL_OP_PROFILE` 下禁用（subpass encoder 切换不可建模）

**Attention replay**（`405beb8aa4`）：decode attention 的 encode 是最重分支路径（路径决策 + shader key + branchy dispatch），改为稳定 token 上重放 + per-token 补丁：

- 决策提取为 `_computePathFlags()`，参数写入拆为 `_writeCopyParam/_writeQKVParam/_writeSoftmaxParam`，onEncode 与 `onReplayUpdate` hook 共用
- `_pathSignature()` 指纹化所有选 kernel 变体/事件布局的结构 flag（含 kv≤128 的 SHORT_KV_128 变体、kv 相关的 mQkvSimdReduce 翻转）；变化即 bail 到正常 encode + 重录
- `onReplayUpdate` 重写 copy/QKV/softmax 参数 buffer，补丁录制事件中 kv 相关的 grid/bytes（splitkv workgroup 数、fused qk_softmax local size、short-seq qk grid 深度），然后 pastLength 恰好前进一次
- ⚠️ **KV-cache 安全坑**：KV 扩容销毁旧 cache tensor，录制绑定里留下悬垂 tensor 指针——onReplayUpdate 在 metalReplayValidate 之前先比较 K/V tensor **指针身份（不解引用）**；scale buffer 同理
- **实测**：基建 p512 +0.5%（价值在基建本身）；attention replay p12 +1.4% / p2048 +2.0%；0.6B greedy 238 token byte-identical（失效仅发生在 kv=65/129 翻转 + 64-chunk KV 扩容，0 ban）
- **开关**：`MNN_METAL_DISABLE_REPLAY=1` 回退；`MNN_METAL_REPLAY_DEBUG=1` 看 record/ban/invalidate 日志

## 2.3 图优化 / 导出侧

### 2.3.1 RoPE Fusion（inv_freq 直接传入，单 op）

标准 RoPE 需要 position_ids → inv_freq → cos/sin → 乘加多个小 op，各自 launch。融合为单个 op 直接接收 `position_ids + inv_freq`：

- **关键文件**：`schema/default/MNN.fbs`（RoPE 增加 `hasInvFreq`）、`MetalRope.mm`、`CPURoPE.cpp`（对齐）、`transformers/llm/export/utils/custom_op.py`（FusedRoPE 导出）、`utils/transformers.py`（inv_freq 传递）
- 配套 **RemoveDeadShapeOp** pass（`tools/converter/source/optimizer/postconvert/RemoveDeadShapeOp.cpp`）：清除 fusion 后无消费者的 Shape/Gather/Unsqueeze 死代码 op

### 2.3.2 导出期 QKV/GateUp 权重合并（❌ 已闭合，headroom 已被运行时融合吃掉）

> **2026-07-28 结论（EXP15，§3.1）**：本节的 +6.2% / conv -19% 数据均产生于**运行时 QKV dispatch 融合（§2.1.3）落地之前**。二者抢同一份收益（3 次小 dispatch 的启动开销）。重做实测：隔离后权重合并仅 +1.1%，而现有融合 dispatch（≈339）本就快于 packed conv（≈331）；带 slice raster 为 **-4.1%**，完美 elision 的天花板也只是持平。**本方向已闭合，勿再以下方历史数据立项。** 以下内容仅作历史归档与跨后端合规经验保留。


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

### 2.3.4 融合遗留死代码清理（RemoveDeadShapeOp 放宽，图 op 数 -65%）

**发现（2026-07-28）**：Qwen3-0.6B 导出图 1116 个 op，从声明输出（`logits`/`hidden_states`）做可达性分析，**731 个（65.5%）不可达 = 死代码**，live 仅 385。死代码是 **transformer 融合的残留**：FusedRoPE/FusedAttention 把 reshape 的形状逻辑与 q_norm/k_norm 权重吸收进 op 属性后，原先构造 reshape 目标形状的子图（`Shape→Gather→算术→Unsqueeze→Concat`）失去消费者。每层 ~24 个：Unsqueeze×10、BinaryOp×5、Concat×3、Const×2（q/k_norm 权重）、StridedSlice×2、Squeeze×2、GatherV2×2。

**为什么原 pass 没清掉**：`RemoveDeadShapeOp` 的可达性分析本来就是对的，但白名单只有 `{Shape, Rank, Size}`，**死掉的类型一个都不在里面**（作者出于"这些 op 在别的模型里也参与数据计算"的顾虑刻意保守）。

**修法**：放宽白名单到纯形状/索引算术（+ Unsqueeze/Squeeze/Concat/StridedSlice/GatherV2/BinaryOp/Const），并**双重收窄作用域**：
1. `net->subgraphs.empty()` —— 子图（While/If）体可能消费外层张量而在 `oplists` 里没有边，可达性会漏判；
2. **图内必须存在 RoPE/Attention 融合 op** —— 只对"确实被我们的融合搞出孤儿"的图生效，其他模型族保持逐字节原行为。这条很重要：MNN 的通用 DCE 把无消费者张量当隐式网络输出，有些流程靠名字取中间张量，不能删。
   `Input`/`Extra` 永不入白名单（图输入即使不可达也必须保留；Extra 可能带运行时元数据）。

**实测**：`removed 84 → 815 dead shape ops`，图 **1116 → 385 op**，`llm.mnn` **301KB → 201KB（-33%）**。

**⚠️ 收益定性（别误解）**：**decode 性能 0 变化**——这些 op 从来没进 GPU（counter profiler 实测层内只有 9 个 dispatch，op 列表里根本没有 Unsqueeze/Concat/Gather/StridedSlice/Squeeze），且 decode 已 100% GPU-bound（§2.2.4.1）。3 对交替实测 decode ref ~324.8 vs dce ~320.7、prefill ~4730 vs ~4761，**均在噪声内**。真实价值是：模型体积、加载/resize CPU 开销、converter matcher 的走图面积更小、以及形状 op 会真正执行或引发同步的**非 Metal 后端**。
**正确性**：Metal（natural/p1024/p2048）+ CPU 后端 greedy 全部 **byte-identical**（与仅 DCE 不同的同管线导出对比）；`run_test.out` **388/388**。

关键文件：`tools/converter/source/optimizer/postconvert/RemoveDeadShapeOp.cpp`。

---

# 三、无效 / 负收益实验记录（避免重试，为后续优化留档）

## 3.1 总表

| 实验 | 结果 | 处置 / 复用价值 |
|---|---|---|
| 运行时 QKV triple fusion（kernel gid.x 分段）| 0.6B tg128 **-36%** | 整条路径删除；后由 gid.z 选路 + seg buffer 的新设计翻案成功（§2.1.3，+2.7%）；导出期合并仍是终局（§2.3.2）|
| Decode QK+softmax+AV 全融合（FUSED_AV）| **-8.7%** | 回退；见 3.2 |
| causal-tri v1（fill kernel 方案）| pp512 ~0% | 被 v2（bounded softmax）取代；fill 写带宽 = 原 skip-tile 写带宽 |
| FA tileFullValid fast-path | -0.6~-1.1%（噪声）| revert；`in_bounds` uniform 分支硬件早已优化，真杠杆在 partial-tile 8x8 sub-block skip（复杂，ROI 低）|
| FA KV_TILE=64 / 去循环末 barrier / NSG=8 | pp512 -6.4% / 4B pp512 -14% / 无收益 | 见 §1.3.3 边界记录 |
| Async decode fire pattern + `mAsync` | profile ON +28% **全是测量伪影**，生产 0 收益 | revert；教训录 §4.2 |
| ICB / encode 复用 | encode 仅 0.92us/op（5%），假设证伪 | ICB 不做；但录制重放思路以 MetalReplay 形态翻案（§2.2.7，p12 +1.4% / p2048 +2.0%——收益来自 attention 重编码逻辑消除而非普通 op encode）|
| Session_Resize_Fix 全图缓存（`MNN_LLM_RESIZE_CACHE`）| greedy 发散且无收益 | 删除；`Pipeline::fixResizeCache` 加 Attention 硬排除保护；被 content-cache 取代 |
| EXP01 commit 粒度扫描（`MNN_METAL_COMMIT_NUM`）| 默认 N=10 已最优（N=2 -7%，N=999 -6%）| env 保留供设备标定；后改为 load 时 tuning 自动选 {10,20,40,80}（§2.2.6）|
| EXP03 GEMV split-K 双 SG 协作 | -2.6% | 回退；后由 SPLIT_K_2（保留 pre-scaling 内循环 + 4sg/tg）翻案成功（§1.1.5，+3.3~3.9%）；g8 kernel 路由方案 -5% 仍是死路 |
| EXP04 KV append 折入 decode_qk_softmax | greedy 一致但 wall ~0%（2.9us GPU 省下落入空泡）| 回退；思路对 4B+/iPhone（管线空泡更小）可能仍有价值 |
| EXP08 split-KV nwg 启发式微调（div128/512）| 均 -2~5% | 回退；div256 是 M4/M5 共同甜蜜点 |
| lm_head 4SG / G16_OC4 变体 | e2e 持平（4SG stddev 7× 恶化；OC4 kernel -4.8% 但 42us 落入空泡）| 删除，结论存档 |
| M5 decode GEMV 变体（middle_step / VEC2 / VEC4）| 中性偏负（VEC4 -1.4%）| M4 的 lane 划分调优在 M5 依然成立 |
| Decode 因果 mask 优化 | 数学上不存在（seq_q=1 无三角可跳）| `trivialFloatMask` 已清零 mask 开销 |
| MUL_SILU_PACKED / packed-QKV 导出链路 | packed-QKV decode **+6.2%**（正收益！）但因跨后端合规缺口 + 产品约束整体移除（非性能证伪） | 重做方案与合规前提见 §2.3.2 |
| greedy readMap 快路径 | ~0%（sample 由 GPU 同步主导）| 保留（省 600KB/tok 拷贝，零风险）|
| EXP12 SPLIT_K_2 扩展到 fused QKV/gate-up/LN dispatch（128 线程 fused pipeline）| p512 tg128 **-3.3%** / p2048 **-1.8%**（新旧 binary 3 轮交替：335.5→324.5 / 253.2→248.7）| 回退（未入库）；fused dispatch 保持 64 线程。疑似主因：LN 前导每 sg 全量读 input+residual，4sg/tg 时冗余翻倍（QKV z=3 每 slice 都跑一遍）；另 128 线程 tg 在小 grid 上调度粒度变粗。若重试：LN 前导改为仅 sk_half==0 的 2 个 sg 计算 + tg 广播 inv_rms（多一次 barrier），可先把 LN 冗余压回 2sg 水平再测 |
| EXP13 ROW_4 四行流融合 GEMV（ROW_2 的 4 流延伸，2026-07-28）| p512 **-1~-3.5%**（346.5→343.8/334.1）；GATE_UP+LN+ROW_4 最重宏组合还出现确定性 knife-edge token 翻转（去掉 LN 或 gate_up 任一宏即 byte-identical——疑似寄存器压力下编译器重结合 LN 重算链）| 回退（未入库）；**ROW_2 是行数甜蜜点**：4 流寄存器压力（4×raw_dot + 8×scale/bias + 4 权重流）超过在途读收益，勿再加行数；GEMV 带宽下一步转权重预取/布局，或接受现状 |
| EXP14 RoPE(+qknorm) 折入 attention copy dispatch（路线图 #15，2026-07-28）| 完整实现并验证通过（greedy byte-identical：0.6B natural/p512/p1024/p1500/p2048 + Qwen3.5-2B partial-rope cut=64；388/388 单测），但收益仅 p512 **+0.7%** / p1024 +0.8% / p2048 +0.3%（3 轮×rep3 交替配对）——远低于 2-3% 预期，**按 ROI 决策回退**（非正确性证伪，commit `0c425cb7f5` 已 reset） | 回退。可复用结论：① 每 token 省 28 次 4.5us 级小 dispatch 在致密 decode 管线只兑现 ≈+0.5-0.8% wall——再次佐证 §2.2.4 "小 GPU 节省不兑现为 wall"，此类 dispatch 省除类优化上限就在 1% 档；② **Q 路折进 `decode_qk_softmax` 前导（tg staging + barrier，pair/QSPLIT/generic 三变体）p512 -1.5%**——QSPLIT 每 q-head TG 重复 staging + 热路径 kernel 前导串行化，勿重试该形态；正确形态是折进 copy dispatch（Q 写入被 skip 的 RoPE 输出 tensor，fused/split-KV/plain 三条 decode 路径零改动消费，fold 条件为 resize 期常量）；③ copy 内 4 heads/128 线程 TG 打包比 32 线程/头小 TG 差 ~0.8%（latency 型小 kernel 宽 TG 无益）；④ byte-identical 可达成：逐字复刻 rope kernel USE_SG 算术（half 载入/norm float 提升回写 half/旋转全 half 的类型序、`i=tiisg;i+=32` lane 映射、simd_sum 归约）实测 bit 级一致；⑤ 工程骨架（仿 LN fusion 的 registry claim + raw q/k/cos/sin STATIC re-home + `_pathSignature` bit26 + claimed RoPE `canRecordEncode()=false`）已验证可行，若未来 dispatch 成本变贵（如 iPhone）可按此重做 |
| EXP15 导出期 QKV 权重合并（路线图 #1/#6，2026-07-28）| **实现成功但前提证伪**：converter pass 完整落地（28/28 block 合并，Metal greedy **byte-identical**，CPU 输出连贯），但实测 p512 decode：① 关掉运行时 QKV dispatch 融合对照（隔离权重合并本身价值）baseline 3-dispatch ≈328 → merged ≈331，仅 **+1.1%**；② 开着现有融合（默认）baseline ≈339 → merged ≈325，**-4.1%**。即 **§2.1.3 的 gid.z 融合 dispatch（≈339）本身就比 packed conv（≈331）更快**。完美消除 slice raster 的天花板估算 ≈338 ≈ 与现有 baseline 持平 | 放弃（commit `85aea37383`/`6a42c512db` 已 reset）。**核心结论：历史 +6.2% 是在运行时 QKV dispatch 融合落地之前测的，两者抢同一份收益（3 次小 dispatch 的启动开销），§2.1.3 + ROW_2 已把这块吃掉，导出期权重合并现在只剩 ~+1% 且被 slice 成本反超。** 其他可复用结论：① Slice 是 **geometry 分解**（`GeometrySlice.cpp:117`）成 Raster 拷贝的，Metal 侧没有 Slice Execution 可置空——"slice-elision" 得改后-geometry 的内存分配，风险远高于预期；② packed conv 的 quant 权重 concat 方法已验证：`ConvolutionCommon::load` 解码 → 4bit 解包 → 按 OC 拼 buffer/alpha → `IDSTEncoder::encode`（保 scaleBit/asymmetric/originBits/aMin）**inline 写回 byte-identical 无损**；③ NC4HW4 下通道对齐区间恒为连续字节段（C4 group 是外层维），别名可行；④ writeFb 在 `optimizeLevel=1`(expectPasses 早退) 路径**不 re-externalize**（needExternalWeight=saveExternalData），inline 权重会胀进 .mnn；⑤ gate/up 合并同理已被 §2.1.1 GateUp 融合 + ROW_2 吃掉，无需再试 |
| EXP16 GEMV 权重软件流水预取（路线图 #8 的"权重预取"半，2026-07-28）| **中性证伪**：depth-1 software pipelining（把下一轮 weight+input 读提到本轮 nibble-mask FMA 链之前，`W_PREFETCH` 编译变体，算术完全不变故 greedy **byte-identical**）——M4 Pro 0.6B rep3：p512 **-0.1%**（341.0→340.6）/ p1024 **-0.3%**（290.5→289.7）/ p12 噪声内。| 回退（未入库）。**根因：Metal 编译器本来就已对这个紧凑 4 行循环体做了软件流水**（与文件内既有 unroll 注释"compiler already schedules this loop optimally"一致）——per-thread load-to-use 延迟不是瓶颈，瓶颈是**跨 lane 的 memory-level parallelism**，而那正是 SPLIT_K_2（K 半段拆 2 sg）与 ROW_2（双行双流）已经吃掉的。推论：**同一线程内加深在途读（prefetch / unroll / 加行数 ROW_4）这条轴已彻底关闭**（三次证伪）。#8 只剩"权重布局重排"一条：现布局 `[oc][z]` 每 ushort4=(4OC×4IC)/8B，0.6B 下 middle_step=4 → 每 simd 指令是 8 段各 32B 的离散访问（未打满 128B cache line）；真正的 layout 重排需同时改 host weightTransform 与 kernel 索引（踩 `kernel-basics` 陷阱 E 字节序风险），规模中等 |
| EXP17 GEMV 权重布局重排 / coalescing（路线图 #8 的"布局重排"半，2026-07-28）| **反向证伪 + 结构性阻塞，方向关闭**。用被删掉的 `WIDE_MIDDLE` 旋钮做零成本代理实验（middle_step=block 而非 block/4：32 lane 覆盖完整 128B cache line，而非 8 段离散 32B）——这正是 layout 重排要买的性质。实测 M4 Pro 0.6B rep3：p512 **-3.7%**（340.2→327.5）/ p1024 **-3.0%**（293.0→284.1）。**把 coalescing 改好反而更慢。** | 关闭（探针已回退）。**根因（重要）：GEMV 的限制因素不是 per-instruction 连续度，而是跨 lane 的独立在途读流数量（MLP）。** 现在的"看起来不连续"的窄划分是**故意更优**的：middle_step=4/outer_step=8 让每个 lane 走 **8 个不同 quant block** → 8 条独立累加/读流；WIDE_MIDDLE 塌成 2 条流。这与 SPLIT_K_2（K 半段拆 2 sg）、ROW_2（双行双流）成功的原因完全同源——**加独立流有效，加连续度/加单线程深度无效**。另有结构性阻塞：量化 `wt` buffer 被 ~15 个 kernel 共用（全部 prefill GEMM 变体、`conv1x1_w_dequant`、g8/g16/g4mx GEMV、2sg GEMV），按 GEMV lane 序重排会打烂所有 GEMM 的 `simdgroup_matrix` 布局；唯一替代是存第二份重排权重（0.6B 335MB→670MB，LLM 不可接受）。**结论：路线图 #8 两条轴（单线程深度 EXP16 / 连续度 EXP17）均已关闭，SPLIT_K_2 + ROW_2 已是该 kernel 的终点** |
| EXP14-M5 RoPE(+qk-norm) 折入 decode attention（路线图 #15，2026-07-28，M5 独立标定）| 实现完整且 greedy byte-identical（p12/p1k/p2k + 自拍 x4，replay 开启态）；但 **M5 强开 tg128 -1.4%**（p512 227.1→223.8 / p2048 159.7→157.7，交替配对）——RoPE dispatch 节省（~126us/token）被顺序 Q staging 串进 qk_softmax 关键路径 + copy kernel 变重抵消 | 回退（未入库）；⚠️ **并发双 simdgroup staging 三种写法（else-if / 指针选择 / 单数组分片）均间歇性输出翻转**（barrier 在位；M5 32-SG 大 TG 下复现，qsplit 单 SG 与顺序形式稳定）——重试并行 staging 前先在小 TG 上界定该现象；若重做：K 折进 `pastkv_copy`（per-kv-head simdgroup）+ Q 折进 qk_softmax/splitkv 前导 + graph claim registry + raw q/k+cos/sin re-home STATIC 的整体设计已验证可行 |

> **EXP17 补充：在线重排（online repack）提案的评估（2026-07-28）**
> 提案：磁盘只存一份 GEMV-lane 序权重，prefill 时在线转成 GEMM 布局，兼顾 decode/prefill 且不发两份权重。分开看"磁盘存储" vs "运行时内存/带宽"两个维度是对的（EXP17 主表把它们混在一起了：`WIDE_MIDDLE` 同时改了连续度↑ 和 独立读流数↓8→2，-3.7% 主要由读流数下降主导，并未干净地测"保持 8 流、只提升每流连续度"的理想 repack）。但三坎叠加使其在 0.6B 不划算：
> - **坎1 每 prefill 都要跑**：LLM 非"prefill 一次后长 decode"，每轮对话 + 长上下文/多模态都反复 re-prefill；在线重排 = 每次 prefill 全量读 335MB + 写 335MB，prefill 本就带宽 bound，几乎必然吃穿 GEMM 收益。
> - **坎2 没省运行时内存**：常驻重排 buffer→又是 670MB；临时 scratch→prefill 内存尖峰 + 坎1 带宽；load 时转一次→只省磁盘/发包，运行时仍 670MB。只有"每-prefill 临时 scratch"能省内存，而它最贵。
> - **坎3（最致命）decode GEMV 不是布局/带宽 bound**：§1.1.4 profiler 实测 0.6B 小投影 85GB/s（35%，latency-bound），布局重排优化的是带宽效率而非约束；且 decode 现读 `wt+uz*input_slice` 对每 output slice 本就行连续、EXP17 又证当前划分对 GEMV 已局部最优——**"对 GEMV 更好的目标布局"很可能不存在**，共享布局没拖累 decode，在线重排在解一个不成立的问题。
> - **4B/8B 翻案条件**：§1.1.4 显示 4B GEMV 已 173GB/s（71%，接近带宽 bound），那里"更好布局"才可能真有收益、共享约束才是真瓶颈；但 4B 非 MLX-gap 战场，且每-prefill 重排成本随模型变大。若将来专门优化 4B+ decode 且 profiler 确认其 bandwidth-bound，可重启并优先"load 时转一次 + 接受 670MB"或"离线双布局导出"，勿走每-prefill 在线转。
> - **未写代码即定案依据**：钉死用 §2.2.4 双实例探针 / counter profiler 复核"latency vs bandwidth bound"即可，无需写 host weightTransform + 15 kernel 索引改动。

> **EXP18 kv 中段 decode attention 路径重标定核查（路线图 #14，2026-07-28）**
> 目标：p1024（kv~1199，对 MLX -8.1%，最弱档）疑似 attention 路径选错。**结论：三档路径选择均已是当前最优，无重标定收益**（热稳定窗口内 clean A/B）：
> - **p1024（kv1199，QK_QSPLIT 档）**：default(qsplit) 292-294 > qsplit_off 288 > splitkv@1024 288-292 > splitkv@768 285。auto 已最优。
> - **p1500（kv1493，阈值 1536 下走 qsplit）**：自拍稳定 277/275/278/281；clean 交替 default 276-279 稳 vs splitkv@1280 261/278/278/235 抖且 ≤ default。**降 split-KV 阈值到 [1024,1536) 无收益**（此前"splitkv 稳赢"是拿 default 的冷启动 rep 对比的伪信号）。
> - **p2048（kv2391，split-KV active，nwg=div256）**：自拍 ~240 稳；split-KV 关掉回退 ~203（-15%，印证 split-KV 本身的 +15.3%）；nwg=div256 已由 EXP08 扫定（div128/512 均 -2~5%）。
> - **根因**：decode t/s 由 GEMV 主导（60-80%），attention 在 kv1199 仅约 20%；而 GEMV 已到 kernel 极限（EXP16/17）。**p1024 的 -8% 是结构性的（GEMV 极限 + 已调优的 attention 片），不是路径 misconfig**。唯一未扫的 sliver 是 split-KV 的 nwg 细分（需加 env 旋钮），但 EXP08 已用 div128/512 夹逼出 div256 内部最优，ROI 极低。
> - **推论**：0.6B/M4 上的 runtime 调参空间已探尽（#8 kernel + #14 attention 均已闭合），**缩小对 MLX 的 decode 差距只剩结构性一条：#9 投机解码**（打破 sample→forward 串行）。
> - ⚠️ **复现性说明（2026-07-29 补）**：上面的 A/B 是用 `MNN_METAL_QK_QSPLIT=0` / `MNN_METAL_DECODE_SPLITKV=N` 做的，而 `MNN_METAL_QK_QSPLIT` 已在 `94a03c9682`（"Retire calibrated env switches"）随 `MNN_METAL_GEMV_ROW2` 一并退役（标定完成后转为 auto-only）。**结论不变**（qsplit auto 门控逻辑未改），但那组对照**不能再用 env 复现**；若需重测须临时改回 `_computePathFlags` 里的 `mQkQsplit` 赋值。
> - ⚠️ **与 `096c6c4f36`（split-KV 阈值对 fused kv cap 取 min）的交互**：该 clamp 条件为 `fusedKvCap < threshold`，M4/0.6B 是 `2048 < 1536` = false ⇒ **M4 阈值仍为 1536，本条 M4 测量与结论不受影响**（clamp 只改 M5 的 3072→2048）。

> **EXP19 GPU-resident decode loop 上限探针（路线图 #19，2026-07-28）——方向关闭，并推翻 §4.3 结论二**
> 动机：借鉴 MLX lazy eval，把采样 token 留在 device、embedding 走图内 gather，消除每 token 的 GPU→CPU→GPU 往返。按"最便宜优先"分两步验证，**两步都是零收益，故未做昂贵的 Phase 3**：
> - **Phase 1（去掉每 token 全量 validation readMap）**：`llm.cpp:577-583` 原本无条件 `readMap<float>()` 全部输出（含 ~600KB logits）。改为只校验 `getTensor()` 非空 → greedy **byte-identical**（0.6B natural/p1024 + Qwen3.5-2B，与同 N 基线逐字一致）。但热稳定窗口内 6 对交替配对 **paired ratio 中位 0.996（-0.4%，6 对中 5 对 <1.0）** → **无收益，已回退**。根因：Apple 统一内存下 `Module::onForward` 本身已同步，之后的 `readMap` 只是 mapping，并非额外同步点。
> - **Phase 2（上限探针：跳过 `sample()` 喂常量 token，彻底去掉采样同步 + 值依赖，GPU 工作量不变）**：p512 paired ratio 中位 **1.0011（+0.1%）**、p1024 中位 **1.0032（+0.3%）**（前后自拍 bracket 确认窗口稳定：p512 340-345、p1024 291-296）。→ **per-token 采样同步的成本 ≈ 0，#19 的天花板 ≈ 0**，Phase 3（设备端 token + tied-lm_head packed 权重 gather kernel）**不做**。
> - **⚠️ 推翻 §4.3 结论二（我此前写进本文档的假设）**：曾以"双实例 ~29% 空泡 + EXP14 时间轴大空泡在 token 边界"论证"每 token CPU 同步是关键差异"。Phase 2 直接证否——**空泡不是采样同步造成的**。MNN decode 本就 GPU-bound、CPU 已被完全隐藏（与 §3.2 async-fire 零收益、§2.2.4 "CPU 阻断已消" 一致），剩余 ~29% 空泡属 **GPU 内部**（kernel 启动/occupancy 间隙），CPU 侧任何重构都动不了。
> - **方法论教训（重要）**：**"结构上存在同步点" ≠ "该同步点是瓶颈"**。本次差点为一个 ≈0 收益的方向去写新 Metal kernel + 改 decode 主循环；救回来的是"先用 10 行探针量上限"这条纪律（同 EXP17 用 WIDE_MIDDLE 代理）。**任何"消除同步/重排管线"类提案，先做常量化/短路探针量天花板，再决定是否投入。**
> - 附带确认：`tie_embeddings` 模型的 embedding 表就是已常驻 device 的 lm_head 权重（`llm_config.json` weight_offset/4bit/block64），设备端 gather **无需额外内存**——若将来 CPU 真成为瓶颈（如 iPhone 或更小模型更高 t/s），这条路的工程前提是现成的。

> **EXP20 "提高小 GEMV 带宽" 四轴批量探针（2026-07-28，M4 Pro / 0.6B）——四轴全部证伪，方向关闭**
> 背景：§2.2.4.1 精测显示 decode 中 GEMV 占 ~70%，小投影（0.5-1.8MB）实测 112-141 GB/s，而 87MB lm_head 达 221 GB/s、峰值 ~273 GB/s。据此列出四条候选并按成本从低到高逐条打，**四条全部中性或负收益，均已回退**（工作区干净，无 commit）：
>
> | 轴 | 做法 | 正确性 | p512 交替配对 paired ratio | 结论 |
> |---|---|---|---|---|
> | #1 QKV grid 去 padding | `QKV_FLATGRID`：把 QKV 融合 dispatch 的矩形 grid `(maxGridX, h, 3)`（Q/K/V 行数不等 ⇒ ~33% TG 空转）换成 1-D 无 padding grid，shader 内由 `gid.x` 反解 `qkv_proj`/`qkv_gx` | greedy byte-identical（0.6B nat/p1024/p2048 + 2B） | .9995/.9983/1.0094/1.0060/.9977/1.0034 → 中位 **+0.1%** | **中性**。空转 TG 在读任何内存前就 `return`，Apple 调度器退掉空 TG 近乎免费 ⇒ **112 vs 125 GB/s 的差距不是 grid padding** |
> | #2 ROW_2 × SPLIT_K_2 组合 | 不直接写 4sg 双行×K拆分的复杂 kernel，先用**判别性廉价探针**：临时 env 让 plain decode 路径走 `ROW_2`（64 线程，grid `UP_DIV(oc,4)`）对比其现役 `SPLIT_K_2` | greedy byte-identical | .9481/.9584/.9421/.9521/.9535/.9327 → **-4.5~-5%** | **不做组合**。同 dispatch 下 K-split 明显优于双行 ⇒ 组合只会把配置推向更差的一侧（且叠加 ROW_2 的寄存器压力），与 ROW_4/EXP12 同一面墙 |
> | #3 SPLIT_K_4 | 1 行 × 4 段 K（4 sg / 128 线程，grid `UP_DIV(oc,4)`，`sk_partial4[4]` + tg 归约）；host 门控 `oc%4==0 && blockSize%4==0` | greedy byte-identical（0.6B nat/p1024/p2048 + 2B） | .9895/.9761/.9726/.9705/.9613/.9780 → 中位 **-2.7%** | **回退**。K 拆分 2→4 时每段只剩 `block_size/4` 个 block，per-sg 的独立流反而变少，且多一次 4 路 tg 归约 + TG 数翻倍。**"加独立流" 的收益在 2 路就饱和** |
> | #4 quant_block 64→128 | 重新导出（`--quant_block 128 --lm_quant_block 128 --hqq`），把 scale/bias 流量从权重字节的 11.1% 降到 5.9%（`llm.mnn.weight` 335.7MB→317.1MB，**-5.5% 字节**） | 不同模型，非 byte-identical（预期） | .9646/.9758/.9825/.9801/.9756/.9807 → 中位 **-2.0%** | **回退（保留 b64）**。⚠️ **少读 5.5% 字节反而慢 2%**：`block_size` 16→8 使每 simdgroup 的独立 block 流减半。补充探针：b128 + `WIDE_MIDDLE` 仅 .9992/1.0146/1.0049（+0.5%），**证明不是 lane 划分可救的**。且粗块本身掉量化质量，双输 |
>
> **统一规律（本批次最重要产出，与 EXP12/16/17 完全一致）**：这个 GEMV kernel 上**只有"增加跨 lane 独立在途读流"这一条轴曾经奏效**（SPLIT_K_2 的 2 段 K、ROW_2 的双行），且**在 2 路就饱和**；而
> - 加单线程深度（EXP16 prefetch / unroll / ROW_4）、
> - 加访存连续度（EXP17 WIDE_MIDDLE 代理）、
> - 减 dispatch/TG 数（EXP14、本批 #1）、
> - 去 CPU 同步（EXP19）、
> - **减少要读的字节总数（本批 #4）**
>
> 五类全部 ≤±1% 或负收益。⇒ **小 GEMV 的 112-141 GB/s 不是"带宽没打满"，而是矩阵太小、每次 dispatch ~6.6us 固定开销 + occupancy 上不去的结构性下限**；`SPLIT_K_2 + ROW_2` 就是该 kernel 的终点。**路线图 #8（GEMV 带宽）自此三轴（深度/连续度/字节数）全部关闭。**
> 方法论复用：#2 是本批最省的一次决策——用一个 env 开关的现成变体做**判别性探针**，替代了一个上百行的新 kernel（同 EXP17 用 WIDE_MIDDLE 代理、EXP19 用常量 token 探针）。**任何"两个已有优化组合起来会更好"的提案，先分别 A/B 出二者在同一 dispatch 下的强弱，弱的一侧不必组合。**

> **EXP22 MLX dispatch 模型探针：并发编码器 / untracked hazard / commit 粒度（2026-07-29，M4 Pro / 0.6B）——朴素移植证伪，但测出 +50% 的调度上限**
> 背景：精读 MLX 0.26.1 源码发现其 Metal 提交模型与 MNN 有三点结构差异：① **所有 compute encoder 无条件 `MTLDispatchTypeConcurrent`**；② **所有 buffer 无条件 `HazardTrackingModeUntracked`**，依赖由 MLX 自己仅在真 RAW 处插一条 `memoryBarrier(BarrierScopeBuffers)`；③ command buffer 按 **50 op / 50MB 双阈值** commit（MNN 每 10 op）。用 env `MNN_METAL_ENABLE_CONCURRENT_ENCODER`（三档探针，默认 0 零影响）+ 现成 `MNN_METAL_COMMIT_NUM` 逐项检验（llm_bench -n 64 -rep 3，每配置清缓存+自暖）：
>
> | 探针 | 正确性 | p512 | p1024 | 结论 |
> |---|---|---:|---:|---|
> | 基线（串行 encoder，tracked）| — | 346-350 | 309-311 | — |
> | mode 1：并发 + untracked + **无 barrier**（上限探针，结果错误）| 预期错误 | **534（+53%）** | **468（+50%）** | **调度上限巨大**：串行 dispatch 的逐 kernel drain + 跨 command buffer 串行占 decode 墙钟 ~1/3。但该上限含"依赖 kernel 尾/头非法重叠"，正确执行不可全额兑现 |
> | mode 2：并发 + **每 dispatch 前显式 barrier**（proxy 注入，byte-identical）| ✓ | 335（**-3%**）| 299（**-3%**）| **显式 barrier ≈ 串行 drain 还略贵**。decode 每层 9 个 dispatch 几乎全是串行依赖链 ⇒ 依赖感知的 barrier 省略最多免掉每层 2-3 个 barrier，落在 -3% 与 +53% 之间靠近 -3% 的一端，**不值得做依赖分析基建** |
> | mode 3：untracked + 串行 encoder（byte-identical）| ✓ | 348（0%）| — | 驱动 hazard tracking 本身在 decode 稳态成本 ≈0 |
> | commit 粒度 20/30/50（对齐 MLX 的 50）| ✓ | 337-348（0%）| — | EXP01 结论扩展：10 已最优，20/30/50 全中性 |
>
> **结论**：MLX 的并发 dispatch 模型对 MNN 的 decode（每层强串行链）**没有可兑现收益**——它服务的是 MLX 自己 566 个细粒度 kernel 里的无依赖对；MNN 已把这些融合成 266 个 kernel，融合本身就是"并发"的静态化。mode 1 的 +50% 上限属于**非法重叠**（同 EXP19 的常量 token 探针性质），唯一能合法吃到它的路径是 **speculative/multi-token：多 token 前向天然打破 token 间串行**（again 指向 #9）。探针旋钮保留（默认 0），供 iPhone/M5 或 MoE 多分支图（真有无依赖 kernel 对）重测。
> 方法论：本次全程遵守"先上限探针再投入"（EXP19 纪律）——mode 1 一小时测出上限，mode 2 一小时证明兑现率≈0，避免了写整套依赖跟踪/静态 barrier 烘焙的大工程。

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
10. **greedy 对拍只在热稳定窗口内有效**（2026-07-28 实锤）：长时间连续压测把机器打入深度热节流后，**未改动的基线自身**在 p1024 连续 5 次运行产出 4 种不同输出（关掉全部优化开关、回退到前一日代码均复现；同日上午同一对比 byte-identical）。疑因 GPU 降频改变 kernel 内 fp16 亚正规/舍入行为或调度序，使临界 token 翻转。**处置**：对拍 DIFFERS 时先跑「基线 vs 基线」自拍；若自拍也不稳定，等热态恢复或换短 prompt——不要在节流态下给新 kernel 定罪/放行
11. **慢模型的配对内热漂移能伪造 ±2% 级"回归"，正反序各测一轮才可归因**（2026-07-29 实锤）：Qwen3.5-2B decode（~87 t/s，单次 run 时间是 0.6B 的数倍）在"base 先跑"配对下两对一致测出 -2.2%，被当成真回归立案；反序（新配置先跑）后符号翻转为 +0.3~+1.5%。而该 env 在 2B 上因 head_dim 门**结构性惰性**（不编译不 dispatch），物理上不可能影响 decode。**处置**：给慢模型定 ±3% 以内的差异时，必须正反序各测；更强的排除法是先确认改动在该配置下是否结构性生效（门条件、replay event 数）——结构惰性 + 有"差异" = 必是测量伪影
    - **⚠️ 2026-07-30 加强版：偏置可达 ±25%，不止 ±3%。** 用 `-n 1000`（单配置运行数分钟）测 0.6B p4096 时，同一对比在"新配置先跑"下是 **+24~26%**、在"基线先跑"下是 **-11~-15%**，**符号翻转**；两序平均才是 +5.8%。更危险的是它会伪造出结构上不可能的结论：纯 decode 路径的开关（`MNN_METAL_DECODE_SDPA`）在单序下测出 **prefill -45%**。**凡单配置运行时间 >1 分钟（长 `-n`、大模型、长 prompt），单序数据一律不得作为结论**
12. **`llm_demo` 单次运行不可用于新 kernel 的性能测量**（2026-07-29 实锤）：Metal pipeline 的 JIT 编译发生在首次 forward，会被整体计入 prefill 段。同一份 tensor-API kernel，`llm_demo` 单次测出 2172 tok/s（判"慢 2.9×"），`llm_bench -rep 5` 预热后 7488（实为 +4%），**差 3.4×**；unroll pragma 实验被误读成"寄存器溢出崩 17×"（实际只是编译变慢）。tensor-API kernel 编译尤其慢。**处置**：性能一律 `llm_bench -rep≥5`；`llm_demo` 只用于正确性对拍
    - **相关：`llm_bench -rep≥5 -n≥1000` 的 4-5min 持续负载会在 MacBook Air 上触发热节流累积**（2026-07-30 实锤）：同一 FA-NAX 优化，rep=5 n=1000 双向平均 +5.8%，rep=1 独立进程双向 6 对回到 **+17.7%**——差异全部来自 rep 循环的 GPU 持续负载。对结论性数据，长 n 协议须用独立进程或 rep=1；rep=5 n=128 总时长 <30s 不受此影响
13. **测量前必须断言二进制新鲜度；配对/反序验证覆盖不了"测的不是这个二进制"**（2026-07-30 实锤）：整轮 MNN-vs-MLX 对比跑在隔了一天的 `libMNN.dylib` 上（缺 `decode_sdpa` / `probe_coop_input` / `prefill_flash_attn_nax`），0.6B decode 偏低 6~8%。据此立案的"-9% 回归"经**正序 3 轮（0.9475/0.8952/0.9134）+ 反序 3 轮（0.9051/0.8891/0.8994）双向复现**、并二分两个 commit 后仍"稳定成立"——因为那个慢产物真实且稳定，只不过它不是 HEAD；重跑 `cmake .. && make` 后 ratio 立刻回到 0.9998/0.9997/1.0001。**铁律 2/11 只能排除热漂移，对装置错误零防护力。** 该产物还连带伪造出"b64 对 decode 只有 +0.2~3.2%"（真值 +4.1~9.2%）。**处置**：① 每轮测量前 `ls -l build/libMNN.dylib` + `strings build/libMNN.dylib | grep <本次新符号>`，找不到即全轮作废；② 疑似回归的第一步是这个断言，不是建 worktree 二分；③ 新增源文件后必须重跑 `cmake ..`（GLOB 的文件列表不会自动刷新，症状是链接期 `Undefined symbols` 且 dylib 已被删）；④ 留意工具输出格式变化这类"装置不对"的旁证——HEAD 的 `llm_bench -pg` 只打合并吞吐，那个 WIP 产物打两个数字。详见 `build-and-test.md` Step 0.5 与案例 3

## 4.3 同类框架对标（MLX / llama.cpp，2026-07-28 调研）

对标目的：本轮 4 个路线图项（#1/#6/#8/#14/#15）全部以"已最优/证伪"收尾后，需外部参照确认"是我们的 kernel 不行"还是"这条路本来就到顶了"。

**结论一：MLX 的 decode GEMV kernel 与我们架构几乎相同 → 我们的 kernel 已到位，#8 的关闭是对的。**
`mlx/backend/metal/kernels/quantized.h` 的 `qmv` / `qmv_fast`：

| 维度 | MLX qmv | 我们的 `conv1x1_gemv_g4m1_2sg_wquant_sg` |
|---|---|---|
| simdgroup / threadgroup | 2 | 2（legacy）/ 4（SPLIT_K_2）|
| 每 simdgroup 输出行数 | `results_per_simdgroup = 4` | 4（一个 output_slice）|
| 反量化 | deferred，融进 `qdot` | deferred（§1.1.1）|
| nibble 处理 | **激活按 2 的幂预缩放以补偿位移** | **完全相同的 pre-scaling trick**（§1.1.3）|
| 预取 / 双缓冲 | **无**，device→register 直流 | 无（EXP16 证实加了也中性）|

→ MLX 独立收敛到与我们相同的三个技巧，且**同样不做预取**，是 EXP16/EXP17 结论的第三方印证。剩余 4-8% 差距不在 GEMV kernel 内。

**结论二（⚠️ 已被 EXP19 实测推翻，保留原文以记录推理链）**：~~真正的架构差异是"每 token 不做 CPU 同步"~~
MLX 用 **lazy evaluation**：采样出的 token 是**未求值的 on-device array**，只有 `.item()` / print / 拿它做控制流才触发同步；官方文档明确警告"用标量数组做控制流会触发求值"，并建议在外层循环才 eval。因此 MLX 能在**不等 token N 的值回到 CPU** 的前提下就把 token N+1 的整个前向入队。
我们相反（`llm.cpp:667`）：`forwardVec(const std::vector<int>& input_ids)` 收 **host int** → `embedding(input_ids)` 上传 → `sample(logits)` 设备端 ArgMax 但**结果回读成 host int**。即使 §2.2.5 把回读压到 4 字节，**每 token 仍是一次硬 GPU→CPU→GPU 往返**。
当时的"交叉印证"（双实例 ~29% 空泡 + EXP14 时间轴大空泡在 token 边界）**是误读**——见 §3.1 EXP19：把采样同步与值依赖**完全去掉**（常量 token 探针）实测 p512 **+0.1%** / p1024 **+0.3%**，即**该同步成本≈0**。⇒ token 边界的空泡不是采样同步造成的，MNN decode 本就 GPU-bound、CPU 已被完全隐藏（与 §3.2 async-fire 零收益、§2.2.4 "CPU 阻断已消" 一致）。**教训：不要把"结构上存在同步点"直接当成"该同步点是瓶颈"，必须先用探针量上限。**

**结论四（2026-07-29 实测）：MLX 每 token 发 ~566 个 kernel，是我们 266 的 2.1 倍——"减少每层 op 数"这条候选杠杆被推翻。**

两种独立方法互证（Qwen3-0.6B-4bit，28 层，mlx 0.26.1 / mlx-lm 0.25.2，本机 M4 Pro）：

| 方法 | 做法 | 结果 |
|---|---|---|
| A 图计数 | 构造一个 decode step 的**惰性图**（prefill 后 `mx.eval` 落地 cache，再 `model(y, cache)` 不 eval），`mx.export_to_dot` 导出后按 label 统计 | 849 个 primitive，其中**发 kernel 的 566**：QuantizedMatmul 197 / RMSNorm 113 / RoPE 56 / SliceUpdate 56 / Add 56 / SDPA 28 / CompiledSigmoidMultiply 28 / Multiply 28 / Gather 3 / AffineQuantize 1；**纯元数据 0 kernel 的 283**：Reshape 112 / Transpose 112 / Slice 56 / Squeeze 3 |
| B 运行时差分 | `MLX_MAX_OPS_PER_BUFFER=1`（`device.h` 的 `buffer_ops`，规则是 `buffer_ops > max` 才 commit ⇒ 2 op/buffer），用 Instruments `Metal System Trace` 的 `metal-application-command-buffer-submissions` 表按 process 过滤计数，取 **N=15 与 N=5 的差 / 10**（差分法消除加载与预热常量） | 288.2 buffer/token × 2 = **576 op/token**；M=4 交叉核对 106.2 × 5 = 531（含 size-based flush 故略低）⇒ 区间 **530-580**，与方法 A 的 566 一致 |

**对比与推论**：

| | 每 token kernel 数 | 每层 | 每 kernel 平均 wall |
|---|---:|---:|---:|
| MNN Metal（§2.2.4.1）| **266** | 9 | 11.1 us |
| MLX | **~566** | ~20 | ~4.9 us（按本机 MLX 363.7 t/s = 2749us/token）|

- ⇒ **MLX 发的 kernel 是我们的 2.1 倍，却仍快 4-6%**。"MNN 图碎、op 多、要靠算子合并追平"这条假设**证伪**。我们的 QKV/GateUp/LN 融合已经把每层压到 9 个 dispatch，而 MLX 的 7 个投影**根本不融合**（q/k/v/o/gate/up/down 各一个 QuantizedMatmul）。
- ⇒ **也证伪"dispatch 固定开销是主成本"**：若每 dispatch 真花 ~6.6us，MLX 566 次就该慢得多。
- ⇒ **附带印证 §2.3.4**：MLX 每 token 有 283 个 Reshape/Transpose/Slice 是**零 kernel 的 view**；我们 decode 层内已经没有形状 op（counter profiler 只有 9 个 dispatch）——**这一项双方都已干净，不是差距来源**。

**结论五（2026-07-29 实测，替代上一版的"每投影 9.8us 估算"）：MLX 单 kernel GEMV 并不比我们快，反而略慢——差距不在 kernel 效率，在"并发重叠"。**
用 `MLX_MAX_OPS_PER_BUFFER=1`（每 buffer 1 op、强制串行、无重叠）录 30 步 decode 的 `metal-gpu-intervals`，按 duration 直方图分桶——每个 interval 是该 kernel 的**独立 GPU active 时间**：

| | 每 token GEMV GPU | 每投影（196 层投影）| lm_head |
|---|---:|---:|---:|
| **MLX**（本次 Instruments 实测，串行态）| 2604 us | **11.4 us** | 369 us |
| **MNN**（§2.2.4.1 counter profiler）| 2057 us | **8.7 us** | 352 us |

- 5-30us 桶恰好 5948 个 interval ≈ 196 投影/token × 31，锚定可靠；200-400us 桶 34 个 = 每次前向 1 个 lm_head，双重锚定。
- ⇒ **MLX 的 `qmv` 单发成本 ≥ 我们的 GEMV**（11.4 vs 8.7us/投影，虽跨工具口径有别但方向明确）。**这用计时数据坐实了结论一**（此前只是读源码判架构相同）——我们的 kernel 不是瓶颈。
- ⇒ 那 MLX 的 e2e 优势从哪来？**清洁实测（我自写的 per-step `mx.eval` 循环）MLX 仅 278 t/s < MNN 341**；MLX 只有走 `mlx_lm.stream_generate`（`async_eval` 流水 + 不每 token 同步）才到 363。即 **MLX 的优势 = 把 token N+1 的 CPU 编码/调度与 token N 的 GPU 执行重叠**，不是 GPU 算得快。而 **EXP19 已证 MNN 的 CPU 早被隐藏（GPU-bound）**——我们本就在做 async_eval 等价的事。
- ⇒ **决定性推论：单 token 路径上没有可打的单点了。** kernel 效率我们不输（结论一/五）、op 数不是问题（结论四）、CPU 同步 ≈0（EXP19）、图结构已干净（§2.3.4）。**剩余那点差距是把 196 个串行投影摊平后的噪声级余量，唯一的数量级杠杆是"一次前向多算几个 token"（multi-token / speculative，路线图 #9）——权重只读一次即可摊薄到 B 个 token。**

**结论六（2026-07-29 全实测，无估算；并**证伪**了上一版列为次要候选的 concurrent encoder）**

测法（避开"按 duration 猜归属"的老问题）：直接用 MLX Python API 按**真实 shape、真实权重**跑各算子，一次 eval 内发出该算子每 token 的真实条数；投影部分**遍历全部 196 个真实投影**（247.73 MB 权重，远超 cache，访存模式与真实前向一致），并用 **MLX 的零 kernel slice view 串成依赖链**得到 serial 成本、或各自独立输入得到 concurrent 成本。

**(a) CPU 侧拆分（MLX，kv≈512，40 次中位）**

| 项 | MLX | MNN |
|---|---:|---:|
| Python 图构建 | 444.8 us | — |
| encode + submit | **1201.9 us** | ~245 us（0.92us/op × 266，§3.2）|
| GPU wait | 1841.4 us | — |
| 每 token 同步态总计 | 3491.8 us（286.4 t/s）| — |
| **流水态（`async_eval`）** | **2712.2 us（368.7 t/s）** | 2874-2933 us（341-348 t/s）|

⇒ **MLX 每 token 的 CPU 成本 1646.7 us（566 kernel × 2.9us/kernel），是 MNN 的 6.7 倍**；MLX 必须靠 `async_eval` 流水才能把它藏起来（同步态只有 286 t/s，**低于 MNN 的 341**）。**MNN 在 CPU 侧本来就大幅领先且已完全隐藏（EXP19），这条没有任何可赚的。** 两者流水/GPU-bound 后：MLX GPU ≈ 2712us vs MNN ≈ 2900us，**e2e 差距 100% 落在 GPU 上**。

**(b) 196 个真实层投影（247.73 MB 权重）——serial vs concurrent**

| 配置 | us | GB/s | vs serial |
|---|---:|---:|---:|
| MLX serial（依赖链）| 2287 / 2188 | **108-113** | 1.00× |
| **MNN（qkv+gate_up 融合 + SPLIT_K_2，serial encoder）** | **1705** | **145.3** | **1.28-1.34×** |
| MLX concurrent G=2 | 1877 | 132.0 | 1.17× |
| MLX concurrent G=3 | 1927 | 128.6 | 1.14× |
| MLX concurrent G=4 / 7 / 14 / 28 | 2201 / 2124 / 2136 / 1943 | 112 / 117 / 116 / 128 | 0.99-1.13× |
| MLX concurrent G=98 | 1686 | 146.9 | 1.30× |
| MLX concurrent G=196 | 1381 | **179.4** | 1.58× |

**两条硬结论：**
1. **MNN 的 GEMV 路径（kernel + 融合）实测比 MLX 快 1.28-1.34×**（145.3 vs 108-113 GB/s，同权重同访存模式）。注意 MNN 这 1705us 取自 counter profiler（绝对值偏大，§4.2 铁律 4），真值只会更低 ⇒ 该结论只会更强。**至此"我们 kernel 不行"被彻底排除。**
2. **❌ concurrent dispatch 证伪、关闭**：decode 单层的依赖链只提供 **G=2~3** 的并发度（不融合时 q/k/v 三路、gate/up 两路），而该档 MLX 只有 **132 / 128.6 GB/s，仍低于 MNN 已有的 145.3**。要超过 MNN 得 G≥98——**单 token decode 结构上不可能有 98 个独立投影**。⇒ 把 MNN 改成 concurrent encoder **拿不到收益**，无需再做"故意不加 barrier"探针。

**(c) MLX 各算子的干净 GPU 成本（`build / async_eval(encode) / eval(GPU wait)` 三段拆分，真实 shape）**

⚠️ **踩坑记录**：第一版把 `mx.async_eval(o[-1])` 只传最后一个输出 → 对独立算子列表只覆盖 1 个 op，其余 encode 全被算进 GPU 列，得到虚高的 2.4-5.7us/op。必须 `async_eval(*o)` 传全部输出。

| 算子组 | build us | encode us | GPU wait us | GPU us/op |
|---|---:|---:|---:|---:|
| QuantizedMatmul x196（全并发）| 85.5 | 439.0 | 799.9 | 4.08 |
| lm_head x1 | 1.3 | 16.2 | 450.2 | 450.2 |
| RMSNorm x113 | 38.8 | 139.8 | 97.6 | **0.86** |
| RoPE x56 | 44.5 | 82.9 | 110.0 | **1.97** |
| SDPA x28（kv512）| 17.1 | 46.2 | 284.3 | **10.15** |
| Add x56 | 22.7 | 68.6 | 112.5 | **2.01** |
| SwiGLU x28 | 28.2 | 61.3 | 115.3 | 4.12 |

⚠️ **更正上一版的错误论断**：我曾写"MLX 的 6% 优势来自把 ~280 个小 kernel 重叠掉"——**该结论是从对账倒推的，测干净后不成立**。小算子有实实在在的 GPU 成本，并没有被"藏起来"。且 encode 与 GPU 执行本身重叠，故 "GPU wait" 列是下界、`encode+GPU wait` 是上界，**无法干净分解 MLX 的 2712us**：各组上界之和 ≈3755us > 2712us，说明真实前向确有重叠，但**三种效应（q/k/v 与 gate/up 的真实并发、per-group 固定开销、encode/GPU 重叠）混在一起，本次测量无法定量归因**。教训：跨口径求和对账只能用来发现矛盾，不能用来下结论。

**两边非 GEMV 总量其实接近**：MNN 891us（attention 644 + RoPE 127 + 其他 120，§2.2.4.1）vs MLX 约 720-1120us（区间源于上面的上下界）。⇒ **"MNN 小算子更贵"也不成立。**

**⛔ 已作废（2026-07-29 第二轮重测）：本小节的 MLX 数字口径错误，「decode attention 是最大单点差距」不成立。**

干净口径重测 MLX SDPA（分离 Python 建图计时 + 轮换 28 份不同 k/v + best of 3）：kv512 **16.31** / kv576 **16.36** / kv640 **16.57** / kv1024 23.28 / kv2048 41.97 us/op —— 而非下表的 10.10 / 10.45 / 11.09。工作集从 4.7MB 扫到 132MB 全平（无缓存驻留效应），`sets=1` 反而 41.65us。**修正后 kv≈576：MNN ≈17.0us/层 vs MLX 16.36us/层 ⇒ 基本持平（1.04×）；attention 块 624 vs 545 us/token（1.15×，非 1.94×）。** 仅 kv1024 档 MNN 尚慢 ~16%。

⇒ 由此推导的「追平 attention 可 +13% 反超 MLX」全部作废；路线图 #20 已永久关闭。第二轮还实现并实测了 4 种单 kernel 融合形态（含 MLX 逐行同构版）全部 **e2e -15~18%**，并逐一证伪 score 全局往返（只占 1.5% 访存）、K 合并度、并行度、内存池扰动、GQA 冗余 V 读五个归因。**完整数据与新增测量纪律见 [`plan-fused-decode-attention.md`](./plan-fused-decode-attention.md) §Phase 5。**

**⚠️ 已复测确认：decode attention 是 MNN 目前最大的单点差距（2026-07-29 apples-to-apples）**

复测口径：kv 严格对齐（MNN p512 decode 期 kv 512→640，均值 ~576；MLX 取 kv=576）。MLX 侧用**串行依赖链**（`q_{n+1} = SDPA(q_n,k,v)`，N=280）保证零重叠，得干净 serial GPU 成本；kv 写入用**真实 `mlx_lm.models.cache.KVCache.update_and_fetch`**（非我先前那个会整块拷贝 2MB 的错误写法）。MNN 侧用 `build_prof` counter profiler（总量比生产 wall 高约 7%，下表未折扣，折扣后结论不变）。

| | MNN | MLX | 倍数 |
|---|---:|---:|---:|
| attention 数学（MNN `qk_short` 11.159 + `av` 11.149 = 22.31us/层；MLX `SDPA` 10.45us/层）| **625 us** | **293 us** | **2.13×** |
| kv cache 写入（MNN `copy` 4.05us/层 ×28；MLX `KVCache` 1.56us/op ×56）| 113 us | 87 us | 1.30× |
| **合计 / token** | **738 us** | **380 us** | **1.94×** |

MLX SDPA 随 kv 线性：kv512 → 10.10us、kv576 → 10.45us、kv640 → 11.09us/op（换算 283/293/311 us/token）。

**换成带宽——这才是根因**：K+V 共 2×8×576×128×2B = **2.36 MB**。
- MLX 单个融合 SDPA 读完 2.36MB 用 10.45us ⇒ **226 GB/s = 峰值(273) 的 83%**
- MNN 拆成两个 kernel，`qk_short` 只读 K(1.18MB)、`av` 只读 V(1.18MB)，各 11.16us ⇒ **各只有 106 GB/s = 39%**

⇒ 与 §2.2.4.1 小 GEMV 的诊断**完全同源**：**单个 kernel 的访存量太小、喂不满 GPU**。MLX 把 QK+softmax+AV 融成一个 kernel，一次性有 2.36MB 在途 ⇒ 拿到 83% 峰值；我们拆成两个 1.18MB 的 kernel ⇒ 各 39%。

**收益上限（量化）**：若 MNN attention 块追平 MLX（738 → 380us），decode GPU 2950 → 2592us ⇒ **341 → ~386 t/s（+13%），反超 MLX 的 368.7**。**这是目前已识别的单点最大收益。**

**廉价路线已排除**：把现有 split-KV decode attention 的阈值降到覆盖 kv~576（`MNN_METAL_DECODE_SPLITKV=384`）实测 p512 交替配对 **.9283/.9141/.9234/.9336 → -7%**。现有 split-KV 是为长 kv 调的（§1.3.4，阈值 1536），在中短 kv 反而更差。⇒ **必须走"真正的融合单 kernel"，没有旋钮捷径。**

⚠️ **前科与正确形态**：§3.2 `FUSED_AV`（把 AV 折进 `decode_qk_softmax`）实测 297.8 → 271.8 t/s（**-8.7%**），根因是 grid 从 `heads×head_dim` 坍缩到 **8 个 TG**——即"融合对了但并行度丢了"。**正确形态必须同时满足两点：① 一个 kernel 内读完 K+V（拿满在途访存量）；② grid 不塌（参考 `kv_heads × head_dim_block`，或像 MLX 的 2-pass sdpa_vector 那样在 kv 维再切分以补并行度）。** 另注意本条**不与 EXP18 冲突**：EXP18 只验证了 kv1199/1493/2391 三档的**路径选择**已最优，从未与 MLX 比较**绝对成本**，且 kv~576 这一档从未测过 split-KV。

**⇒ 对路线图的净影响**：#8（kernel）、#19（CPU 同步）、concurrent encoder 三条全部关闭；**唯一提高 GEMV 带宽的办法只剩"把单个 kernel 做大"，即 multi-token 批处理**——G=196 的 179.4 GB/s 证明硬件还有 **1.58×** 余量，但单 token decode 结构上取不到，**批处理能取**。计划见 `plan-multitoken-gemv.md`。

**结论三：llama.cpp 的 Metal graph reuse ≈ 我们的 MetalReplay（§2.2.7），此项已对齐，无可借鉴增量。**

⚠️ 数字口径警告：arXiv 2601.19139 报 Qwen3-0.6B MLX 525.5 / llama.cpp 281.5 tok/s，而**本机同热态实测 MLX 仅 363-396**（见 `mlx-comparison.md`）——硬件/配置不同，**勿把 525 当作我们的目标线**。

---

# 五、前瞻路线图（2026-07-28 修订：decode 批次二后）

## 高优先级（定量依据的确定性收益）

| # | 方向 | 依据与预期 | 规模 |
|---|---|---|---|
| 1 | ~~Phase-2 QKV packed（导出期权重合并）~~ | **已实施并放弃（EXP15，2026-07-28，§3.1）**：converter pass 做通（28/28 合并、Metal byte-identical），但 **+6.2% 的前提已失效**——那是运行时 QKV dispatch 融合（§2.1.3）落地前测的，两者抢同一份 3-dispatch 开销。实测隔离后权重合并仅 +1.1%，而现有融合 dispatch（≈339）本就快于 packed conv（≈331）；带 slice raster 实测 -4.1%，完美 elision 天花板≈持平。**结论：此路已无 headroom，勿再以 +6.2% 立项** | 已闭合 |
| 2 | ~~M64 sg_matrix GEMM 移植（M4 档）~~ | **已完成（EXP10）**：仅 +0.8~1.7%，远低于预期，M4 GEMM 瓶颈不在权重重复读；kernel 保留默认关 | — |
| 3 | **M3 Pro 验证**：causal-tri、FA 门控、fence、EXP11 面积阈值（merge blocking）| 全部启发式仅 M4 Pro 标定；M3 有 4M 启发式回退前科 | 小（纯测试）|

## 中优先级（瓶颈已定位，有阻塞或需设计）

| # | 方向 | 状态 |
|---|---|---|
| 4 | ~~GPU argmax / async sampling~~ | **已完成（`7b2c8bfcd8`，§2.2.5）**：greedy 走设备端 `_ArgMax`（回读 600KB→4B），mixed+topK 前置走 `_TopKV2` 子集回读 |
| 5 | ~~排空尾部重叠 / async fire~~ | **搁置**：生产 0 收益（profile 伪影，§3.2）；除非 speculative decoding |
| 6 | ~~MUL_SILU_PACKED schema 转正（gate/up 合并）~~ | **已闭合（EXP15 连带，2026-07-28，§3.1）**：与 #1 同根因——gate/up 的 dispatch 合并收益已被 §2.1.1 GateUp leader/follower 融合 + §1.1.6 ROW_2 吃掉，导出期权重合并无剩余 headroom，不值得付跨后端合规（geometry 分解兜底 + 各后端 Execution）+ schema 评审的代价 |
| 20 | ~~融合 decode attention（QK+softmax+AV 单 kernel）~~ | **⛔ 永久关闭（2026-07-29 第二轮）：前提被证伪，MNN 与 MLX 的 decode attention 基本持平。** 干净口径重测 MLX SDPA（分离 Python 建图 + 轮换 28 份 k/v + best of 3）：kv576 **16.36us/op**，不是原记录的 10.45；工作集 4.7→132MB 全平（无缓存效应）。修正后 kv≈576 **MNN ≈17.0 vs MLX 16.36 us/层（1.04×）**，attention 块 624 vs 545 us/token（1.15×，非 1.94×）；仅 kv1024 档 MNN 尚慢 ~16%。<br>第二轮按"不设停损"实现并实测 4 种融合形态（splitkv 单 pass 特化 / NSG 4-32 / per-q-head grid / **MLX 逐行同构 wide** / 两相位 fuseav 含 K,V 组内共享），**全部 e2e -15~18%**，且孤立 kernel 更快而 e2e 更慢（wide 15.6 vs 基线 18.3us/层）——**结构性原因：融合形态的 TG 数上限只有 `B×kv_head`(8)/`B×head`(16)，比现役 AV 的 ~2048 TG 少一个数量级，这个并行度损失远大于省下的 score 往返（仅占 1.5% 访存）**。<br>五个归因逐一证伪：score 全局往返（1.5%）、K 合并度（合并读无收益，lane 顺序吃完整行本就 100% 利用 cache line）、并行度、内存池扰动、GQA 冗余 V 读（已被 cache 吸收）。**数据 + 新增测量纪律见 `plan-fused-decode-attention.md` §Phase 5。重启前必须先复现该节的 MLX 口径测量。** |
| 7 | FA 补课 | tileFullValid 已证伪；剩余：partial-tile QK 8x8 sub-block skip（复杂）、扩 head_dim 96/fp32、kv>8192 段重标定；M4/M5 已 demote 到三段，ROI 有限 |
| 8 | ~~GEMV 带宽新思路~~ | **已彻底闭合（EXP16+EXP17+EXP20，2026-07-28，§3.1）**：SPLIT_K_2（§1.1.5 +3.8%）+ ROW_2（§1.1.6 p512 +2.4%）回收两档后，**其余全部轴均证伪**——① 单线程加深在途读（ROW_4/unroll/prefetch）中性偏负；② 连续度/布局重排（WIDE_MIDDLE 代理）**-3~-3.7%**；③ **EXP20 四轴**：grid 去 padding 中性（空 TG 近乎免费）、plain-ROW_2 **-4.5%**（故不做 ROW_2×SPLIT_K_2 组合）、SPLIT_K_4 **-2.7%**（独立流收益 2 路即饱和）、**quant_block 128 少读 5.5% 字节反而 -2.0%**（字节数不是约束）。**该 kernel 已到终点**，112-141 GB/s 是小矩阵 + ~6.6us/dispatch 固定开销 + occupancy 的结构性下限，除非换设备（4B/iPhone 需另标定） |
| 9 | **multi-token / 投机 decode（唯一剩余的数量级方向，2026-07-29 升为最高优先）** | **前提已实测确认（§1.1.7）**：decode GEMV 是权重带宽 bound，B 个 token 权重只读一次 ⇒ 理想 `cost(B)≈cost(1)`。M4/0.6B 实测摊薄曲线 B=2/4/8/16 → 每前向 1.42×/1.55×/2.24×/3.31×（等效 decode 加速 1.40×/2.58×/3.58×/4.83×），**已经有数量级收益，且离理想还差很远**。<br>⚠️ **第一步不是算法而是 Metal kernel 补齐**：短序列路径 `conv1x1_gemv_g4mN`（`MetalConvolution1x1.mm:616-644`）只有 **1 SG/TG、32 线程**，总 simdgroup 是 decode 路径（SPLIT_K_2，4 SG/TG）的**一半**，且拿不到 QKV/GateUp/LN 融合（`mIs2sgDecode` 只在 `area==1` 置位）⇒ B=2 的 GEMV 要 1.77×。把 SPLIT_K_2/多 SG/融合移植过去，目标 `cost(2)` 1.77×→~1.1×。<br>✅ 上层已就绪：`src/speculative_decoding/{lookahead,ngram,tokentree,eagle,mtp}`，**n-gram lookahead 无需 draft 模型**。实际收益 = 摊薄曲线 × 接受率（需实测）。<br>⚠️ EXP19 后修正的机理：收益来自**单次前向摊薄多 token 的 GPU 工作**，不是"消除 sample→forward 串行"（该同步成本实测 ≈0） |
| 19 | ~~GPU-resident decode loop（借鉴 MLX lazy eval）~~ | **已探针关闭（EXP19，2026-07-28，§3.1）**：Phase 1（去掉每 token 全量 logits validation readMap，byte-identical）实测 **-0.4%**；Phase 2 上限探针（跳过 sample 喂常量 token，彻底去掉采样同步+值依赖）实测 p512 **+0.1%** / p1024 **+0.3%** → **天花板 ≈0，未做 Phase 3**。**并推翻 §4.3 结论二**：token 边界空泡不是采样同步造成的，MNN decode 本就 GPU-bound。工程前提（tied lm_head 即 embedding 表、已常驻 device）已确认，留待 CPU 真成瓶颈的设备 | 已闭合 |
| 14 | ~~长 kv decode attention（p2048 档主瓶颈）~~ | **已完成 + 复核闭合**：split-KV 阈值放宽（2026-07-28，M4 auto 1536，p2048 tg128 221→255 +15.3%）；**M5 gap 段 clamp 已完成（同日）**：auto 阈值对 fused cap 取 min，M5 0.6B p2048 +4.2% / 4B +1.7~4.5%（§1.3.4）；M5 QK_QSPLIT 标定为负（-2~3%），auto gate 维持排除 tensor-API；**EXP18 复核（§3.1）**确认 p1024/p1500/p2048 三档路径选择均已最优、无进一步重标定收益（降阈值伪信号已排除，nwg=div256 已 EXP08 扫定）。剩余仅 KV int8 (+16%，依赖导出配置) + nwg 细分 sliver（ROI 极低） |
| 15 | ~~RoPE(+qknorm) 折入 attention 侧~~ | **双机证伪，已闭合（EXP14 M4 / EXP14-M5，2026-07-28，§3.1）**：M4 完整实现（copy-dispatch 折叠形态 + registry claim）greedy byte-identical + 388/388 全过，但仅 p512 +0.7% / p1024 +0.8% / p2048 +0.3%，按 ROI 回退；M5 独立实现（K 折 `pastkv_copy` / Q 折 qk_softmax 前导）强开 tg128 **-1.4%**。原 2-3% 预期证伪：**dispatch 省除类优化在 Mac decode 的上限就在 1% 档**。工程骨架与避坑（Q 折 decode_qk_softmax 前导 -1.5% 勿重试；M5 并发 staging 间歇翻转）存 EXP14/EXP14-M5，iPhone 等 dispatch 更贵的设备可按此重做 | 已闭合 |

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
- vs MLX 跨框架对比（M5, Qwen3 0.6B/4B, 4bit b64）：[`mlx-comparison.md`](./mlx-comparison.md)
