# feature/metal-speed 分支性能报告

- 分支:`feature/metal-speed`(相对 `master`,共 27 个提交)
- 主线:Apple Metal 后端 LLM 推理加速 —— decode GEMV 融合束、split-K 实验、SDPA 校准、
  设备端采样、LinearAttention scan 优化、decode_splitkv 对齐 MLX sdpa_vector + V 布局翻转
- 常用测试协议:`llm_bench` 配对/交替 A/B(热漂移可制造虚假收益,禁止顺序单向对比),
  greedy 输出逐字节对拍后再谈性能。方法学详见本目录各 sub-doc 的方法论章节
  (`kernel-dev-and-optimize.md` §2.5、`graph-fusion.md` §9、`runtime-scheduling.md` §8)。

## 提交总表

按时间正序。"—" 表示纯 bugfix/文档提交,无独立性能数据。

| # | Commit | 优化方法(一句话) | 性能收益 |
|---|---|---|---|
| 1 | `b7296676e5` | decode-GEMV 融合束:SwiGLU 折进 gate/up GEMV 尾段、QKV 打包 grid、LN 前序双 SG 拆分、host 预算 middle_step | M4 Pro 0.6B tg128 **+2.0%**(340.9→347.7,4/4 同向;单项未拆分) |
| 2 | `64d9efd865` | tensor-API 设备上以 ROW_2 强制编译启用 GATE_UP_SILU 折叠 | M4 Pro 0.6B tg128 **+1.9%**(3/3 同向) |
| 3 | `4b19f2e62e` | LinearAttention 递推状态留寄存器 + chunk64 前缀和/前代并行化 | 2B pp512 **+11.6%**、pp2048 **+10.4%**;0.8B pp2048 **+23.6%**;decode 持平 |
| 4 | `7cc45d6ad6` | Converter 无条件融合 LinearAttention 投影(正确性前提) | —(正确性;qkv-off 配置此前输出乱码) |
| 5 | `efa05cc2ac` | M5 decode SDPA 默认 NSG 8→16 | iPad M5 2B kv4096 **+2.14%**(4/4);Mac M5 kv4096 **+3.95%** / kv8192 **+4.86%** |
| 6 | `6b85a08fa1` | grouped-SDPA(GS_LOCAL=GROUP_SIZE)长 kv 复评(结论归档,代码回退) | M5 0.6B kv16384 **+2.53%**(3/3),kv4096 +0.59%(噪声)——未采纳 |
| 7 | `5436a18c81` | 记录 NSG 校准过程(文档) | — |
| 8 | `7b21806431` | LN_SPLIT_SG 笔记 + barrier/提前退出 UB 陷阱(文档) | —(机制收益计入 #1) |
| 9 | `4bb11e33fe` | 退役 4 个默认开的融合实验 kill-switch(收敛为默认行为) | —(greedy 逐字节一致验证) |
| 10 | `894414f95a` | SPLIT_K_SHUFFLE decode GEMV 变体(单 SG 内收拢 K 半区,免 tg-mem/barrier) | M4 Pro 0.6B tg128 291.8 vs legacy 286.6(**+1.8%**),但 −1.0% 于 SPLIT_K_2 ⇒ 未成默认 |
| 11 | `6b479f5e49` | 2sg GEMV 量化解包向量化 + LN 加载去重(重构) | 未单独记录(引入后被 #12 修复的转置乘积 bug) |
| 12 | `5fc13f01f3` | causal-mask 检测笔记(文档) | — |
| 13 | `8f98e59e62` | 修复 2sg GEMV W8 转置 dequant(正确性) | — |
| 14 | `4d53509d78` | lm_head split-K 实验(开关默认关) | =1:kernel −96% 但 e2e 仅 **+1.3%**(尾部是 lm_head→采样同步瓶颈);=2:持平偏负 ⇒ 方向证伪 |
| 15 | `ed68dec073` | 归档 lm_head split-K 证伪 + W8 转置乘积陷阱(文档) | — |
| 16 | `295e53cad0` | greedy ArgMax 从 Express expr 换 NEON 首最大值循环(expr 实际跑在 CPU executor) | M4 Pro ~330us/token → 12us/token;整 token 周期 **+5.3% wall** |
| 17 | `ccc33b537a` | 重写采样章节文档(记录 #16 数据 + decode 计时盲区) | — |
| 18 | `f18606645e` | Conv1x1 const buffer 跨 resize 保稳(encode-replay 正确性) | — |
| 19 | `40f16a2874` | rebase 修复:缺失 #endif、row2 声明(正确性) | — |
| 20 | `e8be9b13f8` | 修复 SPLIT_K_SHUFFLE+W16 lane 拆分跳半数 weight 块(正确性) | — |
| 21 | `a452188ecc` | 归档编译期 BLOCK_SLICES 证伪(文档) | —(7 轮配对 −0.24%,中性 ⇒ 证伪) |
| 22 | `f7b5d19447` | 修复 rebase 丢失的 ROW_2 键/宏(正确性) | —(曾致 decode 乱码 + 幻影 +13% 假收益) |
| 23 | `e15afadbeb` | decode_splitkv 重写为 MLX sdpa_vector 形态;V cache 全局翻转为行主序(镜像 K),适配全部 7 个读 V kernel | 使能 #24;长 KV 4266 上 decode 188.8 vs base 178.9(**+5.5%**,Mac M 系) |
| 24 | `47a3f473c0` | decode SDPA 自动阈值从 3072/1536(+group cap)降为 128 | M5 qwen3-0.6b **-pg 全区间 +2.5%~+6.8%**;Qwen3.5-2B 持平(混合架构稀释) |
| 25 | `dea5d07d59` | decode SDPA 自动阈值再降到 2(短 kv sweep:splitkv 从 kv≥2 全胜 fallback) | M4 Pro `-n128` 0.6B **+16.6%** / 4B **+11.9%**;greedy byte-identical |
| 26 | `ce9b57a327` | decode SDPA 默认 NSG 统一 32(对齐 MLX sdpa_vector 固定 1024 线程),取代 #5 的设备分档 nsg16 | Mac M5 0.6B p2048 decode **+6.8%**(179.3 vs 167.9)、p512 +2.1%;3/3 同向;此前 iPad M5 nsg16 结论早于 V 行主序重写,复扫不再成立 |
| 27 | `df2eeb6e81` | lm_head G16_SPLIT_K 默认开(推翻 #14 早先单模型证伪) | 全模型 4×3 轮配对(M5):0.6B **+2.9%** / 0.8B **+1.0%** / 4B +0.5% / 2B −0.4%(噪声),无回归 |

## 各提交详情

### b7296676e5 — decode-GEMV 融合束(四项)

1. **SwiGLU 折进 gate/up GEMV 尾段**:gate/up 两路 GEMV 输出直接在 epilogue 做
   `silu(gate)*up`,省掉每层一次 MUL_SILU dispatch 和 gate/up 结果的显存往返。
2. **QKV packed fused grid**:矩形 grid 下大量 threadgroup 早退空跑
   (Qwen3.5 线性层 67% TG 闲置),打包成一维紧凑 grid。
3. **LN 前序拆分到两个 simdgroup**:2sg GEMV 中 input+residual 的读取量减半。
4. **host 预算 GEMV_MIDDLE_STEP**:SPLIT_K_2 的中段步长变编译期常量。

收益(b+c+d 打包测得,单项未隔离,存在"归因债"):M4 Pro 0.6B tg128
340.9→347.7 tok/s(+2.0%,4/4 同向)。QKV packed grid 单项:M4 Pro Qwen3.5-2B
tg128 +0.7%(噪声内偏正)。

### 64d9efd865 — tensor-API 设备启用 GATE_UP_SILU 折叠

同一 SwiGLU 折叠在 M5 级(tensor-API)设备上因 pipeline 编译路径差异未能自动命中,
用 ROW_2 强制编译该融合 pipeline(grid 不变,z 2→1)后启用。
M4 Pro 0.6B tg128 配对 +1.9%(3/3 同向)。

### 4b19f2e62e — LinearAttention scan 优化(Qwen3.5 gated delta rule)

- 递推状态从两次 device 往返改为寄存器驻留(每 timestep 一次 load/write);
- 新增 `sg_v2` dk==64 特化;
- chunk64 前缀和(Hillis-Steele scan)与前代求解并行到全部 simdgroup;
- MetalGatedRMSNorm 改 2 simdgroup/threadgroup。

背景:LinearAttention 占 0.8B p2360 prefill GPU 时间的 32.6%。
收益:2B pp512 +11.6%、pp2048 +10.4%;0.8B pp2048 +23.6%;decode 持平。

### efa05cc2ac — SDPA NSG 校准 nsg8→nsg16(M5)

单 workgroup 融合 decode SDPA 的 simdgroup 数是占用率/调度开销的折中,需按设备档
实测校准。早期 sweep 只对比了 8 vs 32 得出"M5 偏 nsg8",补测 16 后翻案:
iPad M5 kv4096 +2.14%(4/4 配对),Mac M5 2B kv4096 +3.95%、kv8192 +4.86%。
M4 级(非 tensor-API)维持 nsg32。可用 `MNN_METAL_DECODE_SDPA_NSG` 覆盖。

### 894414f95a — SPLIT_K_SHUFFLE GEMV 变体

K 维两半在同一 simdgroup 内累加后 shuffle 收拢,不用 threadgroup 内存与 barrier,
threadgroup 缩到 64 线程。M4 Pro 0.6B W4 tg128 双向配对 rep5:
legacy 286.6 / SPLIT_K_2 294.6 / shuffle 291.8 tok/s。结论:收益来自翻倍在途
lane 而非免 barrier,=1(SPLIT_K_2)保持默认,变体留作开关(`MNN_METAL_GEMV_SPLITK=2`)。

### 4d53509d78 — lm_head split-K 实验(证伪记录)

`MNN_METAL_ENABLE_LMHEAD_SPLITK=1` 把 lm_head 路由到 SPLIT_K_2:kernel 计时
2310us→87us(−96%,profiler 口径失真),但 e2e decode 仅 +1.3%(100.7→102.0,
10 轮交替)——decode 尾部瓶颈是 lm_head→采样的 GPU→CPU 同步,不是 kernel 本身。
=2(G16_SPLIT_K)kernel 与 e2e 均持平偏负。方向关闭,开关保留默认关。

### 295e53cad0 — greedy ArgMax 去 Express 化

greedy 采样的 ArgMax 走 Express `_ArgMax`,实际从不在 GPU 上执行(CPU executor),
每 token ~330us。改为 NEON 两遍首最大值循环后 ~12us/token;整 token 周期
10265→9746us(+5.3% wall),greedy 逐字节一致。附带教训:llm 的 `decode_us`
计时在 `sample()` 之后启动,decode 速度指标对采样开销天然失明。

### e15afadbeb — decode_splitkv 对齐 MLX + V 布局翻转

- kernel 重写为 MLX `sdpa_vector` 形态:Q 驻寄存器、逐 token 交错流式
  (`i = sgitg; i < kv; i += NSG`)、score 立即用同 token 的 V 行更新 O、
  MLX 式转置跨 SG 归并;
- V cache 从转置 `[kvHead, headDim, maxKv]` 全局翻转为行主序
  `[maxKv, batch, kvHead*headDim]`(镜像 K),7 个读 V kernel 全部适配
  (copy、decode_qkv、decode_qkv_c2、prefill_qkv 标量/simd/tensor、
  prefill_flash_attn、prefill_flash_attn_nax);
- 磁盘前缀缓存 `.v`→`.v2` 使旧布局文件失效;QUANT_K/V/DYNAMIC int8 保留。

关键 bug:归并块沿用旧 lane↔d 映射(`d = dd*32+lane`),新形态下应为
`d = lane*DPT+dd`,长 prompt 后首 token 乱码,修复后与 legacy 路径 greedy 完全一致。
性能(Mac M 系,qwen3-0.6b):长 KV(4266→4778)decode 178.9→188.8 tok/s(+5.5%)。

### 47a3f473c0 — SDPA 阈值降至 128

V 翻转后,回退路径(decode_qkv/qk_softmax)的 V 读退化为逐 token 跨步标量读,
中 KV 段 −7%~−14%;而新 splitkv 的行内向量读天然匹配新布局。把自动阈值从
3072/1536(+group-based fusedKvCap)统一降为 128 后:
M5 qwen3-0.6b `-pg 128/512/1024/2048,128` 全区间 +2.5%~+6.8%;
Qwen3.5-2B(6/24 层 full-attn,权重带宽瓶颈)持平——splitkv 相对同布局回退
仍快 +9%,恰好对冲布局翻转成本。

## 分支整体(历史对照,来自前序战役报告,非本分支 24 提交直接测得)

优化战役起点 `46ceea0ab` → 融合战役 HEAD(M5,Q4,pp512/tg128 双向配对):

| 模型 | prefill | decode |
|---|---|---|
| Qwen3-0.6B(28 层全 attn) | +173% | +18.9% |
| Qwen3.5-0.8B(全 attn) | +58.3% | +12.7% |
| Qwen3.5-2B(混合 6/24 full attn) | +107% | +5.3% |
| Qwen3-4B(全 attn,GQA=4) | +194% | +3.2% |

decode 增幅随模型增大递减:本轮优化主要削减每 token 固定开销与 dispatch 数,
模型越大越被权重带宽支配;混合架构额外稀释 attention 侧优化。

## 未采纳方向的证伪记录

| 方向 | 结果 | 证据 |
|---|---|---|
| GEMV 编译期 BLOCK_SLICES | 中性,证伪 | M4 Pro 0.6B W8 tg128 7 轮配对 −0.24%;W16 真正收益在 16B 宽 load |
| lm_head G16_SPLIT_K(=2) | **df2eeb6e8 默认开**(推翻本行早先结论) | 早先单模型配对持平偏负;2026-08-20 全模型 4×3 轮配对(M5)0.6B +2.9% / 0.8B +1.0% / 4B +0.5% / 2B −0.4%,无回归 |
| SPLIT_K_SHUFFLE 作为默认 | 不敌 SPLIT_K_2 | −1.0%,收益来源是在途 lane 翻倍而非免 barrier |
| grouped-SDPA(GS_LOCAL=GROUP_SIZE) | 仅超长 kv 微增,回退 | kv16384 +2.53%,kv4096 噪声;代码回退、结论归档 |
| GEMV_MLX lane-per-row(对齐 MLX qmv 形态) | 中性,证伪 | 2026-08-20 0.6B/4B × p512/p2048 配对 decode:−0.7%~+0.2% 全噪声内;SPLIT_K_2 的 tg 归约开销本就不大,形态不是瓶颈;代码回退 |
| RoPE in-kernel freq(消除 /rotary 链 14 dispatch/token) | decode 零收益,证伪 | 2026-08-21 0.6B 重导 `--rope_in_kernel` 配对 3 轮:decode p512 +0.3% / p2048 −1.1%(噪声);rotary 链张量极小(1 token),GPU 耗时藏在大 kernel 间隙——印证"削减小 op dispatch 不兑现 decode"(仅 prefill +0.8~1.0%);正确性已验证 byte-identical;代码全回退 |
