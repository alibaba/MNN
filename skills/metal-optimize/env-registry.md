# MNN Metal 环境变量注册表

> **目的**：集中登记全部 Metal 后端环境变量 —— 默认值 / 语义 / 提交出处 / 适用设备 / 定型状态，避免"某开关默认怎么走" / "哪个 env 是关哪个是开"的坑。
>
> **单一入口**：所有开关在 `source/backend/metal/MetalEnv.hpp` 统一声明与解析（2026-07-24 工程债清理），后端代码禁止散落 `getenv`。新增开关必须同时改 MetalEnv.hpp 与本表。
>
> **详解文档**：开关背后的优化机制与实测数据见 [`kernel-dev-and-optimize.md`](./kernel-dev-and-optimize.md)（kernel 层）、[`graph-fusion.md`](./graph-fusion.md)（融合）、[`runtime-scheduling.md`](./runtime-scheduling.md)（调度）。
>
> **形态约定**：`1` = 显式开；`0` = 显式关；未设 = 走默认（每项在"默认"列注明）。语义不一是历史遗留，新增开关请遵循 `MNN_METAL_ENABLE_*` = 默认关、`MNN_METAL_DISABLE_*` = 默认开的命名。

## 性能路径开关

| Env 变量 | 默认 | 打开效果 (=1) | 关闭 (=0 或 unset) | 生效条件 | 定型状态 |
|---|---|---|---|---|---|
| `MNN_ENABLE_FLASH_ATTN_PREFILL` | 由 config `attention_mode/8` 决定 | 强制启用 FA prefill | =0 强制走三段路径；unset 由 config 决定 | prefill + fp16 + head_dim∈{64,128,256} | 定型 A/B 用；M4/M5 默认已 demote 到三段。**causal 假设现由数据驱动**（`mCausalLayout`）：真实 mask 张量 ⇒ FA 自动不启用，无需手动关；非 causal 模型无需再设任何 env（2026-07-31 起，见已删除表 `MNN_METAL_QK_CAUSAL_TRI`）|
| `MNN_METAL_PREFILL_INSHADER_DEQUANT_SGMATRIX` | 自动（>4M 且 area<512 走 in-shader，EXP11） | =1 强制 in-shader dequant | =0 强制 outer-dequant + fp GEMM | non-tensor-API 设备 + area>1 + Q4/Q8 | 定型 A/B 用；area<512 边界仅 M4 标定（4B pp2048 +5.3%），**M3 验证 blocking**。**2026-07-31 由 `MNN_METAL_PREFILL_INSHADER_DEQUANT` 改名**（补 SGMATRIX 标志：仅非 tensor-API 设备生效，M5+ 恒走 outer-dequant + tensor API；旧名已失效）|
| `MNN_METAL_RESIZE_WAIT` | **`local`**（per-backend fence，隐式）| `global` = 旧全局 wait；`none` = 完全跳过（实验，不安全）| 默认 fence 已生效 | onResizeBegin 每次 | 定型；`local` 转正（`7f186691a`）|
| `MNN_METAL_PREFILL_FA_TENSORAPI` | **默认开（unset=开，2026-07-30 M4 Pro 定型）** | **=1 tensor-API 版 `prefill_flash_attn_nax`（默认，唯一变体）**：`matmul2d(16,32,16,tb=true)` + input cooperative tensor + 零 threadgroup memory，S/O 全寄存器、score 不落全局 | =0 显式关（回退三段/legacy FA）| prefill + fp16 + head_dim∈{64,128} + causal（ADD mask 或 kv_cache）+ fp16 KV；需 seq≥64 且 `isSupportTensorCoopInput()` | **sentinel/耦合语义（2026-07-30）**：unset→-1，默认解析为开，但**耦合 `MNN_METAL_QK_CAUSAL_TRI=0`**——非 causal 模型设该开关即同时关掉 FA2 nax（其 kernel 硬编码 causal、无独立 opt-out），一个 env 覆盖所有 causal-假设 prefill 路径；显式 `MNN_METAL_PREFILL_FA_TENSORAPI=1` 绕过耦合（A/B 用）。**M4 Pro 验证（2026-07-30）**：M4 在 `noAICoreDevice` 名单，`isSupportTensorCoopInput()=false`→nax 结构性休眠，默认开为**零影响 no-op**（prefill 持平，噪声内），run_test 388/388；故默认开只对 M5+ 生效，对 M4/M3/iPhone 无风险。**结果（2026-07-29 M5 冷机配对 rep5）**：0.6B pp512 **+4.2%** / pp1024 **+5.5%** / pp2048 **+9.0%** / pp4096 **+17%**（随 seq 单调增大）；4B +1.2~3.7%；2B（head_dim=256，被门挡掉，env 惰性）中性——初测 decode -2.2% 已排除为配对顺序热漂移（反序 +0.3~1.5%，p32 对照噪声级）。greedy 三模型 byte-identical，run_test 388/388。⚠️ 测新 kernel 性能必须用 `llm_bench -rep≥5`：`llm_demo` 单次会把 JIT 编译计入 prefill，同一份 kernel 测出 2172 vs 7488（差 3.4×）。**长 decode 收窄机制已查清（2026-07-30）**：rep=5 n=1000 观测的 +5.8% 是 rep 循环累积热漂移，非内在机制——rep=1 独立进程双向 6 对实测 n=1000 仍为 **+17.7%**，与 n=128 的 +19.3% 一致；MacBook Air 无风扇下 rep=5 n=1000 单次 rep ~50s、总时长 4-5min 触发热节流累积。已并入 [`kernel-dev-and-optimize.md`](./kernel-dev-and-optimize.md) §2.0 的测量注意事项。simdgroup 版（旧 `MNN_METAL_PREFILL_FA2=1`）已删除（2026-07-30），判负数据见文末已删除表；**2026-07-31 改名 `MNN_METAL_PREFILL_FA2` → `MNN_METAL_PREFILL_FA_TENSORAPI` 并把取值从 =2 简化为 =1（旧名与 =2 写法均已失效）** |
| `MNN_METAL_DECODE_SDPA` | **auto 开（默认，2026-07-30 M4 Pro 定型）** | unset→1（auto，device 分档 kv 阈值：tensor-API 3072 / 其余 1536，clamp 到 fused cap）；=N>1 显式 kv 阈值 | =0 显式关（回退 fused decode_qk_softmax(kv≤cap) / 三段 decode_qk(kv>cap)）| decode seq=1 + kv cache + trivial/无 mask + head_dim%32==0 + kv≥阈值 | **split-KV 已删除（2026-07-30）**：SDPA 成为 kv≥阈值**唯一**融合 decode 路径，原 `MNN_METAL_DECODE_SPLITKV`/`_NWG` 及 `decode_splitkv_reduce` kernel 全删（见文末已删除表）。SDPA 全设备默认开（nsg 才是设备相关项，见下行）；=0 回退 fused/三段，M4 p4096 实测 default 196 vs =0(三段) 156.5 = **+25%**、p6144 158 vs 124 = **+28%**、8B p2048 42.3 vs 40.6 = **+4%**（decode 被权重 GEMV 稀释）。**M4 Pro 验证（2026-07-30 配对 rep5）**：默认(nsg32)=255.8 vs =0(legacy)=240.8 → **+6.3%**；p4096 **+7.5%**；default 输出与 legacy greedy 200tok **byte-identical**（fp16/跨阈值 1366→1666），run_test 388/388。**实验（2026-07-29 M5）**：单 pass 融合 SDPA（`SDPA_SINGLE_PASS`，nwg=1、无 reduce、score 不落全局、kernel 直写输出）。0.6B p2048 e2e **+5.2~6.4%**（3 对干净配对，base bracket ±0.3%）；p1024 以下阈值外结构性零影响（replay event 数已验证）；短中 kv 强开为负（p1024 -0.5~-4.8%）故阈值门控。**跨规模冷机复测**：4B p2048 **+2.5~3.4%**；Qwen3.5-2B（混合架构，仅 6/24 层 full attention）-0.1~-0.2% 噪声级中性、无回退。正确性：0.6B/4B/2B × kvq8 × replay × 跨阈值 200tok 全 byte-identical，run_test 388/388 |
| `MNN_METAL_DECODE_SDPA_NSG` | **0 = 设备分档（默认，2026-07-30）** | `=4/8/16/32` 显式设融合 kernel 的 simdgroup 数 | 0/其他值 = 设备分档：tensor-API/M5→8、非 tensor-API/M4-class→32 | 仅 SDPA 路径 | **设备分档在 `MetalAttention.mm` 用 `isSupportTensorApi()` 解析**（MetalEnv 无设备信息）。**M5 e2e 标定**：p2048 nsg8 +6.2% > nsg16 +4.8% > nsg32 +3.3%（e2e 随 TG 变宽单调恶化）。**M4 Pro 标定（2026-07-30 配对 rep5，与 M5 相反）**：p2048 nsg8 **-3.5%** / nsg16 +3.8% / **nsg32 +6.0%**（p4096 nsg32 +7.5%）；nsg4 崩到 192（-20%）。⇒ 非 tensor-API 分档取 32。M1/M2/M3/iPhone 未标定，继承 M4 非 tensor 分支（nsg32），待上机重标。⚠️ profile build 与 e2e 常相反（§4.2 铁律 4 伪影），标定只认 e2e 配对 |
| `MNN_METAL_W4W8_OUTER_DEQUANT_GEMM_TENSORAPI` | **关**（fused in-kernel unpack 默认启用）| =1 走 outer-dequant + fp GEMM（A/B 基线 / 紧急回滚）| unset/=0 走 fused | tensor-API 设备 + W2/W3/W4/W8 + area≥64（prefill）；**W3 额外要求 conv 权重 ≥4M 元素**（2026-08-03 M5 标定：0.6B W3 fused pp2048 -3.6% 判负、4B W3 +11~24%，小模型路由回 outer）；非 M5 设备本开关为 no-op | 定型；fused 路径含 M64 tile 自动策略（Q4 + area≥128，M5 pp512 +5.9% / pp2048 +6.8%）。M5 W2/W3 标定（2026-08-03）：W2 +6~44% 全正，W3 4B +11~24%，**2026-07-31 由 `MNN_METAL_DISABLE_FUSED_Q4_GEMM` 改名**（正向表述 + W4W8 位宽 + TENSORAPI 标志，旧名已失效）|
| `MNN_METAL_FUSED_Q4_KSPLIT` | **auto**（unset）| =1 强制开 | =0 强制关（单趟 GEMM，不走 reduce）| fused-Q4 stage kernel（tensor-API + Q4 + 非 M64 tile）+ `KVMeta::spec_block > 0`；auto 门 = `area<=32 && UP_DIV(oc,64)<=48 && blockSize>=4` | K 按 `gid.z` 四分写 fp32 partial，第二趟 `conv1x1_fused_q4_ksplit_reduce` 求和 + bias + activation，中间 `memoryBarrierWithScope:MTLBarrierScopeBuffers`。投机 verify 形状输出 tile 太少、TG 数喂不满带宽，拆 K 补并行度；lm_head 这类 TG 本来就多的被门挡在外。**改了累加顺序 ⇒ 输出非 bit-identical**（实测绝大多数 prompt 输出逐字一致，个别在近平局的 greedy argmax 上出现语义等价的分叉，接受长度不降），评审需明示 |
| `MNN_METAL_FUSED_Q4_M8` | **auto**（unset）| =1 无效（等同 unset）| =0 强制关（退回 M32 tile）| 同上，且 K-split 未启用 + `1 < area <= 8` | `conv1x1_fused_q4_gemm_stage_m8`，matmul2d descriptor 直接按 M=8，避免 8 行塞进 M32 tile 只用四分之一占用。**不能强制开**：kernel 对 area>8 结果错误，故 =1 不绕过 area 门 |
| `MNN_METAL_FUSED_Q4_KSPLIT_M8` | **auto**（unset）| =1 无效（等同 unset）| =0 保持 M32 tile | K-split 已启用 + `area <= 8` | 把 M8 tile 叠在 K-split 上（`conv1x1_fused_q4_gemm_stage_ksplit_m8`）。同样不能强制开 |
| `MNN_METAL_GEMV_SPLITK` | **开** | =1 显式启用（与 unset 默认相同） | =0 恢复旧 2sg GEMV（64 线程，每行 1 个 simdgroup 流式读整行） | decode GEMV (area==1) + Q4/Q8 + oc%8==0 + 偶数 quant block 数 | 实验（2026-07-28）：`SPLIT_K_2` = 同一 2sg kernel 4 simdgroup/tg（每行 2 个 K 半段 + tg 归约），行内在途读加倍；M4 Pro Qwen3-0.6B tg 12/512/2048 +3.9%/+3.8%/+3.3%。注意先走 g8 kernel 的方案（nibble-unpack 内循环）反而 -5%，勿回退到那个方向 |
| `MNN_METAL_H2D_QUEUED` | **开** | =1 显式启用（与 unset 默认相同） | =0 恢复旧的逐 token drain + 直写上传路径 | decode 每 token 输入上传 | 定型（`757610526`，0.6B decode +5.2%）；`0` 用于回滚 |
| `MNN_METAL_COMMIT_NUM` | 0（走 runtime hint `OP_ENCODER_NUMBER_FOR_COMMIT`，llm.cpp tuning 决定）| `=N` 覆盖每 commit 的 op encoder 数 | unset 走 hint | Metal 每次 commit 判断 | 设备标定用（EXP01：默认 N=10 已最优，N=2 -7%，N=999 -6%；**EXP22 补扫 20/30/50 全中性**——MLX 的 50 op 批次对 MNN 无增益）|
| `MNN_METAL_DISABLE_REPLAY` | **开 replay**（unset/=0 时启用）| =1 关掉 encode replay（稳定 shape 下录制 op encode 命令列表、后续 token 直接重放，见 `MetalReplay.hpp`）| unset/=0 走 replay | MetalExecution::onExecute；attention 经 onReplayUpdate hook 接入（per-token 参数/grid 补丁 + KV 指针身份校验）；linear-attention decode 已接入（seqLen==1 + resize-generation guard），prefill 仍豁免 | 实验（`94a73ab19a` + `405beb8aa4`：0.6B p12 +1.4% / p2048 +2.0%）；`MNN_METAL_REPLAY_DEBUG=1` 看 record/replay/invalidate 日志。2026-07-31 起按值判断（仅 =1 关闭）|
| `MNN_METAL_REPLAY_DEBUG` | 关 | =1 打印每个 op 的 record/ban/invalidate 事件 | unset/=0 静默 | 配合 replay 使用 | 诊断用 |

## 融合 / dispatch 开关

| Env 变量 | 默认 | 打开效果 (=1) | 生效条件 | 定型状态 |
|---|---|---|---|---|
| `MNN_METAL_DISABLE_LN_FUSION` | **折入投影**（两阶段路径关；unset/=0）| =1 启用两阶段回退：独立 binary AddRMSNorm 物化 `residual_out + normalized`，再由已融合的 QKV/GateUp 投影读取 `normalized` | 带 `has_ln` 的 `FusedLinear` decode；prefill 本来就是独立 dispatch | 默认保持折入投影。Qwen3.5-2B 的旧 4-SG 两阶段 A/B：Mac decode **+4.84%～+5.94%**；iPad M5 **-0.33%～-0.37%**，无收益。当前独立宽行 AddRMSNorm 已自动使用 **8-SG**（旧 `MNN_METAL_LAYERNORM_SIMDS` 已删除），精确幅度需按当前版本重测；设备方向结论保留。该开关不影响独立的 `GatedRMSNorm` op。 |
| `MNN_METAL_DISABLE_GATE_UP_FUSION` | **开融合**（unset/=0 时启用）| 关掉 Gate/Up leader/follower fusion | Metal 后端；MUL_SILU 匹配 | 定型 A/B。2026-07-31 起按值判断（仅 =1 关闭）；**同日由 `MNN_DISABLE_GATE_UP_FUSION` 改名补上 METAL 前缀，旧名已失效****2026-08-05 阶段二**：gate/up 配对已从 `MetalBinary::onResize` 的反查移入 `MetalFusedProj::setupFusion()`（按导出成员顺序），`mOutputToConv1x1` 已删除；本开关现在控制该函数内的 gate/up 分支。**未带 `--fuse_gate_up_proj` 导出的旧模型不再有此融合**。 |
| `MNN_METAL_DISABLE_QKV_FUSION` | **开融合**（unset/=0 时启用）| 关掉 Q/K/V leader/follower fusion，恢复独立 GEMV dispatch | decode GEMV + 同一输入恰有 **3 或 4** 个 conv1x1 消费者（attention q/k/v，或 Qwen3.5 linear-attention 层的 qkv/z/b/a，2026-08-03 泛化 `QKV_FUSED_P4`：grid.z=4、buffers 15-18、seg 6 floats）；input-LN 会进一步融进 3 **或 4** 投影 leader（受 `MNN_METAL_DISABLE_LN_FUSION` 控制）。LN×P4 曾因分配器把 `out_q` 与 LN residual 输入分配到同址而逐次非确定，2026-08-03 已根因修复（别名检测 + STATIC re-home，见 [`graph-fusion.md`](./graph-fusion.md) §4.3），硬门控已移除 | 实验（2026-07-28）：M4 Pro Qwen3-0.6B tg 12/512 QKV 融合 +2.7%/+1.6%，叠加 LN 融合再 +1.9%/+1.7%；（2026-08-03）Qwen3.5 上 3 路匹配从未命中（linear 层 4 消费者、全 miss），P4 泛化后 24/24 层命中，0.8B tg128@p512 **+2.7%**（256.5 vs 249.7，双向配对）/ 2B +0.6%；GEMV dispatch 144→96/token。k/v(/w) 输出 re-home 到 STATIC 内存避免提前写被动态池复用覆写（详见 MetalConvolution1x1::setupQKVFusion 注释）。2026-07-31 起按值判断（仅 =1 关闭）**2026-08-05 语义变更**：融合决策已上移到导出图。`matchQKVFusions`/`matchLNFusions` 两个运行时匹配器连同注册表（`mInputToConv1x1Group`/`mLayernormMap`）已删除，改由 `MetalFusedProj::setupFusion()` 按导出的 `FusedLinearParam` 成员顺序确定性建立（backend 在 `onResizeEnd` 的 allocator `compute()` 之后回调）。本开关现在控制该函数内的对应分支。**未带 `--fuse_qkv_proj`/`--fuse_gate_up_proj`/`--fuse_ln_proj` 导出的旧模型在 Metal 上不再享有这两项融合**（实测 Qwen3.5-0.8B decode dispatch 246→365），需重新导出。 |
| `MNN_METAL_ENABLE_QKV_COMPACT_GRID` | **关**（unset/=0 使用旧矩形网格） | =1 保留 QKV/P4、LN 和 W16 融合，并把各投影 TG 在 `grid.x` 上前缀和拼接、`grid.z=1` | 已命中 QKV/P4 fused decode GEMV；不需要重导模型 | 默认关闭，按设备显式启用。2026-08-11 Mac M5 Pro：Qwen3.5-2B pp512→tg128 配对中位数 **+1.51%**，pp2048→tg128 **+2.00%**，6/6 对同向；iPad M5 端到端 decode **无收益**。greedy 64 token 与旧网格一致。 |

## Profiling / 诊断

| Env 变量 | 默认 | 打开效果 (=1 / =<path>) | 用途 | 依赖 |
|---|---|---|---|---|
| `MNN_METAL_OP_PROFILE_TIMELINE=/path.csv` | 关 | 把每 op GPU (t0, t1) 时间戳 dump 到 CSV，给 `tools/script/metal_profile_gantt.py` 消费 | 甘特图 / GPU idle gap 归因 | 需要 `-DMNN_METAL_OP_PROFILE=ON` |
| `MNN_METAL_OP_PROFILE_LEGACY` | 关 | 回退到旧的每 op 一 command buffer 的 profile 模式（绝对数字失真但简单）| 兼容不支持 stage-boundary sampling 的设备 | 需要 `-DMNN_METAL_OP_PROFILE=ON` |

> CPU 侧 trace（Session 级 resize/malloc/run + Metal 级 encode/commit/wait 计时汇总，退出时打印）已从 env 开关改为**编译宏** `-DMNN_SESSION_CPU_TRACE`（`source/core/Session.cpp` + `source/backend/metal/`），生产 build 不编译、零开销。详见 `source/core/Session.cpp` 与 `source/backend/metal/` 中的 `MNN_SESSION_CPU_TRACE` 分支。

## Profile 环境警告（重要）

**`-DMNN_METAL_OP_PROFILE=ON` + `MNN_METAL_OP_PROFILE_TIMELINE` 数据只用于诊断，不能作为优化目标**（`7cd053d7f` / 2026-07-23 的 async fire 负面结果教训）：

- Counter sample buffer attachment 让 CPU op encode 从 ~0.92us/op 涨到 4-20us/op（20×）
- 制造出的 GPU idle 是**测量伪影**，生产环境（profile OFF）decode 已经 GPU-bound
- 任何基于 profile ON gantt 的假设收益，**必须** 用 production build 3-rep 交替配对 A/B 交叉验证

## 已淘汰 / 已删除

| Env | 状态 | 说明 |
|---|---|---|
| ~~`MNN_METAL_DISABLE_QKV_COMPACT_GRID`~~ | 已改名并改为默认关（2026-08-11） | iPad M5 端到端 decode 无收益，因此 portable 默认回到旧矩形网格；Mac M5 Pro 可用 `MNN_METAL_ENABLE_QKV_COMPACT_GRID=1` 显式启用，保留 pp512 +1.51% / pp2048 +2.00% 的实测收益。 |
| ~~`MNN_METAL_QK_CAUSAL_TRI`~~ | 已删除（2026-07-31）| causal 判定改为**数据驱动**（`MetalAttention.mm` `_computePathFlags` 派生 `mCausalLayout`）：真实 mask 张量 ⇒ 非 causal ⇒ 逐元素 honor、关掉 causal-tri/bound/FA-v1/faNax；标量哨兵/无 mask + kvcache ⇒ causal。配套 `llm.cpp` 对 metal 后端也发 shape 空、值 0 的标量哨兵 mask（与 cpu/hexagon 同约定，:414+:1520）。原开关是"非 causal 模型必须设 =0 否则静默乱码"的正确性陷阱，且盖不住 FA-v1（无 opt-out）——数据驱动后两者都自动正确，env 无存在价值。causal-tri/bound **优化本身保留**，仅改由 mCausalLayout 驱动。历史 A/B 数据（p1024/p2048/p4096 全矩形 0.756/0.657/0.586×）见 git 历史 |
| ~~`MNN_METAL_DECODE_SPLITKV_NWG`~~ | 已删除（2026-07-30）| split-KV 专属 workgroup 数探针，随 split-KV 路径一并删除。并行度假设早已证伪（单调 nwg 无法转正，见 git 历史）|
| ~~`MNN_METAL_PREFILL_FA2`~~（含 `=1` 变体）| 已改名/取值删除（2026-07-30 删 =1，2026-07-31 整体改名为 `MNN_METAL_PREFILL_FA_TENSORAPI`，取值 =2→=1）| `prefill_flash_attn_v2`（8x8 simdgroup_matrix 融合 prefill）kernel + `mFa2Prefill` 路径整体删除，开关本身以新名保留（=0/=1）。**判负存档（2026-07-29 M5）**：数值全对但 p512/p2048/p4096 = 0.90/0.76/0.68×，二次系数 0.067 = 三段的 1.9 倍；根因是 M5 上 8x8 simdgroup MMA 对该 shape 只有 matmul2d 一半 FLOP 效率（四种载入策略 760-920us/层 vs 三段 357）。nax（=2）已定型默认开，v2 无重启价值；=1 现解析为 0（关）。详见 git 历史 |
| ~~`MNN_METAL_DECODE_SDPA_QSPLIT`~~ | 已删除（2026-07-30）| SDPA grid 变体收敛：默认的每 q head 一个 TG（qsplit=1，对齐 MLX sdpa_vector）转正为唯一形态，`GS_LOCAL` 定死 1；=0（每 kv head 一个 TG、GROUP_SIZE 头共享 K 读）已判更差（p1024 e2e -4.8% vs -3.6%），分支连同宏删除 |
| ~~`MNN_METAL_DECODE_SDPA_COALESCED`~~ | 已删除（2026-07-30）| 单 pass SDPA 的 QK 合并读变体（simdgroup↔kv 行，256B 连续 K 读）。e2e p2048 co8 +6.0% ≈ leg8 +6.2%，合并读不兑现（EXP17 模式第三次复现），默认关从未转正，shader 分支连同 env 删除。**勿再重试 K 读合并方向**（kernel 级快 / e2e 平的伪收益模式）|
| ~~`MNN_METAL_W4W8_OUTER_DEQUANT_GEMM_M64_SGMATRIX`~~（含前名 `MNN_METAL_GEMM_M64`）| 已删除（2026-07-31）| M64 tile 转为 **arch-gen 设备分档自动策略**（`MetalBackend.mm` 解析 `MTLDevice.architecture.name`，`applegpu_g<gen><size>`：gen≥16 且非 p 档（M4/M4 Pro/M4 Max Mac 及 M4 iPad）→ 默认走 `conv1x1_gemm_64x64_split_k_sg`；gen≤15（M1/M2/M3）与 phone p 档 → 32x64；macOS<14/iOS<17 无 architecture API → 保守关）。依据：M4 Pro 配对 rep5×2 pp2048 **+1.1~+2.4%** / pp512 无回归；M3 Pro pp512 **-1.4~-1.5%** 否决全局开。family API 区分不了 M3/M4（同 Apple9），故用 architecture.name（MLX 同款做法）。greedy byte-identical、run_test 与基线一致。旧 env 及更早的 `MNN_METAL_GEMM_M64` 均失效 |
| ~~`MNN_METAL_ENABLE_CONCURRENT_ENCODER`~~ | 已删除（2026-07-31）| EXP22 探针（MLX 并发 dispatch 模型移植：mode1 并发 encoder+untracked 无 barrier 上限 / mode2 并发+每 dispatch barrier 代理 / mode3 仅 untracked），连同 `MNNMetalBarrierEncoderProxy`、`MTLDispatchTypeConcurrent` 分支、`HazardTrackingModeUntracked` 分支全删。**判负存档（EXP22，M4）**：mode1 上限 p512 +53%（结果错误不可兑现）；mode2 **-3%**（显式 barrier ≥ 串行 drain，decode 每层强串行链无可省 barrier）；mode3 0%（driver hazard tracking 成本≈0）⇒ 朴素移植证伪，单 token decode 无可兑现收益。原保留理由（MoE 多分支图 / iPhone 重测）撤销——重启时从 git 历史恢复即可，判负数据以本行为准 |
| ~~`MNN_METAL_CPU_TRACE`~~ | 已改为编译宏 | 运行时 env 开关删除，改为编译宏 `-DMNN_SESSION_CPU_TRACE`（见上方 Profiling 一节说明）|
| ~~`MNN_ENABLE_QKV_FUSION`~~ / ~~`MNN_DISABLE_QKV_FUSION`~~ | 已删除（2026-07-24 工程债清理）| 运行时 QKV triple fusion 实测 tg128 -36%（0.6B）/+1%（4B），整条路径（matchQKVFusions + QKV_FUSED shader）连同开关删除 |
| ~~`MNN_METAL_LMHEAD_4SG`~~ | 已删除（2026-07-24）| lm_head 4SG 变体 e2e 持平且 stddev 7× 恶化，G16_4SG shader 分支删除；结论存 git 历史 |
| ~~`MNN_METAL_LMHEAD_VARIANT`~~ | 已删除（2026-07-24）| G16_OC4 变体 kernel -4.8% 但 e2e 持平（M5 管线空泡吸收），shader 分支删除 |
| ~~`MNN_METAL_FUSED_Q4_STAGE`~~ / ~~`MNN_METAL_FUSED_Q4_M_TILE`~~ | 已删除（2026-07-24）| 开发期 bisect 脚手架（stage 1-3 stub 模式 + conv1x1_dequant_only_q4 kernel 一并删除）；收敛为 `MNN_METAL_W4W8_OUTER_DEQUANT_GEMM_TENSORAPI` 单 kill-switch，M64 tile 转为自动策略 |
| ~~`MNN_METAL_FORCE_TENSOR_API`~~ | 已删除（2026-07-24）| tensor-API 设备（M5+）prefill Q4/Q8 现无条件走 tensor API 路径（sg_matrix A/B 实测 -48%，开关无日常价值）|
| ~~`MNN_LLM_RESIZE_CACHE`~~ | 已删除（2026-07-24） | 被 `1ef0ace93` content-cache 取代；实验路径已知 greedy 发散，代码已从 llm.cpp 移除 |
| ~~`MNN_EXPORT_MERGE_QKV`~~ | 已删除（2026-07-24，整条链路） | 导出开关（transformers.py qkv/gate-up 合并）+ 消费端（converter `fuseQkvPackedC4`、RoPE/Attention 形状放行、Metal `kBaseOffset`/`v_channel_offset`）全部删除。历史数据（M4 Pro decode +6.2%）与判负结论见 [`graph-fusion.md`](./graph-fusion.md) §7.2 |
| ~~`MNN_METAL_RESIZE_NOWAIT`~~ | 已淘汰 | 被 `MNN_METAL_RESIZE_WAIT=local` (默认) / `=global` / `=none` 三值统一取代 |
| ~~`MNN_METAL_QK_QSPLIT`~~ | 已删除（2026-07-28 收敛）| M4 正（p1024 +2.7%）/ M5 负（p1024 -2~3%）双端标定完成，auto gate（non-tensor-API + kv≥512 + group_size==2）定型为唯一逻辑，三态覆盖开关删除（env 治理：标定完成的开关及时收敛）|
| ~~`MNN_METAL_GEMV_ROW2`~~ | 已删除（2026-07-28 收敛）| 历史上 M4 正（p12 +3.9% / p512 +2.3%）、M5 中性偏负，曾收敛为 non-tensor-API 设备门控；2026-08-07 为跨设备融合 W16 测试取消该门控，GateUp/QKV/LN 融合 Decode 在 M4/M5 均固定走 non-ROW_2。 |
| ~~`MNN_METAL_LAYERNORM_SIMDS`~~ | 已删除（2026-08-11 收敛）| 形状分档已定型为唯一逻辑：宽行 decode（`outside==1 && channelUnit>=128`）固定 8sg，其余保持 legacy（单输入 1sg / binary 2sg）。探针覆盖开关删除，优化代码保留。M5 Pro 三模型 decode +2.4%~3.3%，prefill 中性。 |
| ~~`MNN_METAL_GEMV_BLOCK64_W16`~~ / ~~`_LMHEAD`~~ / ~~`_FUSED`~~ | 已删除（2026-08-11 收敛）| Q4 block 32/64/128/256 的 W16 specialization 已定型并固定用于 standalone、g16 lm_head、QKV/GateUp/LN-folded fused decode GEMV；三层 A/B/回退开关删除，优化代码保留。 |

## 命名规范建议（后续新开关）

1. 优先按"默认行为反义"命名：默认开的加 `MNN_..._DISABLE_...`，默认关的加 `MNN_..._ENABLE_...`
2. 三值开关（`local` / `global` / `none`）用 named string 而非 0/1
3. 每个新增 env **必须**：
   - 在 `source/backend/metal/MetalEnv.hpp` 加字段（禁止散落 `getenv`）
   - 在本文件加一行
   - 在 commit message 里注明默认值和 A/B 用法
