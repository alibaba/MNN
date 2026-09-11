# MNN Metal 环境变量注册表

> **目的**：登记 Metal 后端**当前存在**的环境变量 —— 只写**名称 / 默认值 / 功能 / 生效条件**四项。
>
> **单一入口**：所有开关在 `source/backend/metal/MetalEnv.hpp` 统一声明与解析，后端代码禁止散落 `getenv`。新增开关必须同时改 `MetalEnv.hpp` 与本表；删除开关必须同时删这里的行。
>
> ## ⛔ 本文件的硬性约束
>
> 1. **禁止写任何性能数字**：百分比、ms、GB/s、TFLOPS、tok/s、配对结果、"+x%/−x%"、A/B 矩阵、扫点表，一律不许出现。
> 2. **禁止写实验叙事**：转正 / 证伪 / 翻案 / 推翻 / 标定过程 / 机制推断 / 教训 / 陷阱 / 待办，一律不写。
> 3. **禁止写溯源信息**：commit hash、日期、设备型号、模型名、"某某人某天改的"，一律不写。
> 4. **禁止写已不存在的开关**：改名后的旧名、已删除的开关、被回退的实验开关，直接删行，不留删除线残骸。
> 5. **"功能"列只描述这个开关做什么、以及它在哪条代码路径上生效**，不评价好坏。
>
> 判断某个开关值不值得开、以及它在某台机器上的实测表现，属于实验记录，不进本文件（见 [`SKILL.md`](./SKILL.md) 的记录规则）。
>
> **形态约定**：`1` = 显式开，`0` = 显式关，未设 = 走"默认"列。历史开关语义不一致；新增开关请遵循 `MNN_METAL_ENABLE_*` = 默认关、`MNN_METAL_DISABLE_*` = 默认开。

## Attention — prefill

| Env 变量 | 默认 | 功能 | 生效条件 |
|---|---|---|---|
| `MNN_ENABLE_FLASH_ATTN_PREFILL` | unset = 由 config `attention_mode/8` 决定 | `1` 强制启用 legacy FA prefill，`0` 强制走三段路径 | prefill + fp16 + head_dim ∈ {64,128,256} |
| `MNN_METAL_PREFILL_FA_SG` | unset = auto（高带宽非 tensor 档的 32q/8kv/head_dim128 shape seq≥512；其他 eligible 非 tensor-coop shape seq≥1024） | `prefill_flash_attn_sg`：simdgroup-matrix 版融合 prefill，S/O 驻 simdgroup fragment，不物化 QK 中间张量。`1` 强制开，`0` 强制走三段 | prefill + fp16 + causal + head_dim ∈ {64,128} + fp16 KV + seq≥64 + 支持 simdgroup matrix |
| `MNN_METAL_FASG_Q_REG` | 开（`0` 关） | 把 `HEAD_DIM/8` 个 Q fragment 在 kv 循环外一次载入寄存器，不再每个 kv tile 从 smem 重发 | 仅 `prefill_flash_attn_sg` |
| `MNN_METAL_FASG_NSG8` | 开（`0` 回退 4 simdgroup） | 每 threadgroup 8 个 simdgroup（256 线程、q tile 64 行），减少每层重读 device K/V 的次数 | 仅 `prefill_flash_attn_sg` |
| `MNN_METAL_FASG_LOADBATCH` | 4 | `=N`：消费任何 fragment 前先批量发射 N 个 `simdgroup_load`，让 load 延迟被前面的 MMA 覆盖；N 同时是 PV 侧 V-fragment 批宽。`0` 关闭批量发射 | 仅 `prefill_flash_attn_sg`；host 侧要求 `N | HEAD_DIM/8`，否则降为 0 |
| `MNN_METAL_PREFILL_FA_TENSORAPI` | unset = 由数据驱动的 `mCausalLayout` 决定（标准 causal mask ⇒ 开） | `prefill_flash_attn_tc`：Metal tensor API 版融合 prefill，S/O 全寄存器，score 不落全局内存。`1` 显式开，`0` 显式关 | prefill + fp16 + head_dim ∈ {64,128} + causal + fp16 KV + seq≥64 + `isSupportTensorCoopInput()`（M5+）；kernel 硬编码 causal |
| `MNN_METAL_DISABLE_FATC_Q_REG` | 关（即 Q 驻寄存器） | `1` 回退成每个 `(kv tile, head_dim frag)` 都从 device 重新 gather Q；默认把 q-tile 载入寄存器并预乘 `scale*log2e` | 仅 `prefill_flash_attn_tc` |
| `MNN_METAL_FATC_QK_K32` | 开（`0` 回 K=16） | QK matmul2d 的 K 维 16→32，head_dim 累加调用次数减半 | 仅 `prefill_flash_attn_tc` + `HEAD_DIM%32==0` |
| `MNN_METAL_FATC_O_CT` | 开（`0` 关） | online-softmax 的 O 累加器常驻 destination cooperative tensor，省掉每次 PV 在 CT 与标量数组之间的往返 | 仅 `prefill_flash_attn_tc` |
| `MNN_METAL_FATC_Q_CT` | 开（`0` 关） | QK 的 left input cooperative tensor 提到 kv 循环外只填一次，不再每 tile 重建重填 | 仅 `prefill_flash_attn_tc`；要求 `FATC_Q_REG` 与 `FATC_QK_K32` 都开 |
| `MNN_METAL_FATC_PV_K32` | 开（`0` 回 K=16） | PV matmul2d 的 K 维 16→32，一次调用吃掉整个 `FATC_BK` kv tile | 仅 `prefill_flash_attn_tc`；要求 `FATC_O_CT` 开 |
| `MNN_METAL_FATC_KV_DEV_TENSOR` | 开（`0` 关） | K/V 以 `tensor<device ftype, ...>` strided 句柄就地喂给 matmul2d，由硬件自己发操作数 load，不再逐 lane 手填 right input CT | 仅 `prefill_flash_attn_tc`；要求 `FATC_O_CT` 开 |
| `MNN_METAL_FATC_QREV` | 开（`0` 回正序） | q-tile → `threadgroup.x` 的映射反序，把 causal 下 kv 循环最长的 tile 排在最前 | 仅 `prefill_flash_attn_tc`；纯索引置换，输出 bit-identical |
| `MNN_METAL_FATC_DSPLIT` | unset = auto（head_dim 256 取 2，其余取 1） | `=N` 把 head_dim 切 N 份放到 dispatch grid 的 z 维，常驻 O destination CT 的寄存器占用除以 N，代价是 QK 按 z 冗余重算。`1` 强制不切 | 仅 `prefill_flash_attn_tc`；要求 `FATC_O_CT`，且每片须为 32 宽 O tile 的整数倍（否则 host 回落到 1） |
| `MNN_METAL_ATTN_QSPLIT` | unset = 0（按显存预算推导片数） | `=N`（2 的幂）强制把 q 序列切成 N 片，三段路径的 `mTempQK`/`mTempSoftMax` 峰值除以 N；各片 q 行相互独立，无额外算术开销 | 三段 prefill 路径 |
| `MNN_METAL_ATTN_QSPLIT_MB` | 128（MB） | auto 策略给每片 scratch 的预算上限，`MNN_METAL_ATTN_QSPLIT` 未显式设时由它推导片数 | 同上 |
| `MNN_METAL_PREFILL_INSHADER_DEQUANT_SGMATRIX` | unset = auto（按权重规模与 area 阈值） | `1` 强制 in-shader dequant，`0` 强制 outer-dequant + fp GEMM | 非 tensor-API 设备 + area>1 + Q4/Q8（M5+ 恒走 outer-dequant + tensor API，本开关无效） |

## Attention — decode

| Env 变量 | 默认 | 功能 | 生效条件 |
|---|---|---|---|
| `MNN_METAL_DECODE_SDPA` | unset = 1（auto，按设备分档的 kv 阈值，clamp 到 fused cap） | 单趟融合 decode attention（向量化 SDPA 形态）：ntg 固定 1、无 reduce dispatch、kernel 直写输出。`0` 显式关（回退 fused `decode_qk_softmax` / 三段 `decode_qk`）；`N>1` 显式覆盖 kv 阈值 | decode seq=1 + kv cache + trivial/无 mask + `head_dim%32==0` + kv ≥ 阈值 |
| `MNN_METAL_DECODE_SDPA_NSG` | unset = 0（设备能力档 × `batch*head_num/qh` 自动分档） | `=4/8/16/32` 显式设融合 decode kernel 每 threadgroup 的 simdgroup 数；auto：tensor 档固定 32，高带宽非 tensor 档用 product 512，标准非 tensor 档用 product 256，并按能力档处理 short-KV cap | 仅 SDPA 路径。解析必须排在 `mSdpaQhPerTg` 之后；高带宽档的高寄存器压力角落 `kv<256 && TG<16 && qh*head_dim>256` cap 到 16，标准档 `kv<512 && TG<16` cap 到 16 |
| `MNN_METAL_DECODE_SDPA_QH_PER_TG` | unset = 0（auto，按 GQA `group_size` 解析） | `=1/2/4/8` 设每个 threadgroup 承担的 q head 数：1 = 每 q head 一个 TG（各自重读共享 KV 行），>1 = 一次取 K/V 行供多个 q head 用，以 threadgroup 数换 KV 读请求数 | 仅 SDPA 路径；必须整除 `group_size`，否则忽略 |
| `MNN_METAL_DECODE_SDPA_NTG` | unset = 0（auto，实际默认走单趟） | `=N`（2 的幂，≤64）把 kv 扫描切到 grid.x 的 N 个 threadgroup：pass 1 各写 `(S, m)` + 未归一化 O partial，pass 2 `decode_splitkv_reduce` 合并归一化。`1` 显式单趟 | 仅 SDPA 路径。ntg 在 resize 时定型、不随 kv 变，以保证 encode-replay 的 grid 与 kv 无关；partial buffer 须按 float 字节数申请 |

## LayerNorm

| Env 变量 | 默认 | 功能 | 生效条件 |
|---|---|---|---|
| `MNN_METAL_LN_TOKENS_PER_TG` | 8 | `=N` 让 N 个 token 共用一个 threadgroup（N 个 simdgroup 各归约一行，互不通信）；`1` 回退一 token 一 threadgroup 的 `layernorm_c4_rms_sg`。归约始终在 simdgroup 内，本开关只在 threadgroup 数与 threadgroup 宽度之间取舍 | C4 + RMSNorm + simdgroup reduce + `outside>=256`（prefill；decode `outside==1` 恒走旧路径） |
| `MNN_METAL_LN_SUM_REREAD` | 1 | binary（add + RMSNorm）C4 kernel 的 normalize 趟需要再次拿到求和行：默认回读本线程刚写进 `out0` 的和（一次 DRAM 读）；`0` 改为重新相加 `in0+in1`（两次读） | binary C4 RMSNorm kernel |
| `MNN_METAL_LN_TOKEN_LANE` | 0 | `1` 让 binary C4 prefill kernel 的 simdgroup lane 映射到 token 而非 channel：C4 激活是 channel-major，lane-per-token 使 32 lane 读连续 float4，行归约变为 lane 私有、去掉 `simd_sum` | binary C4 RMSNorm prefill kernel；需 `outside>=32` 填满一个 simdgroup |
| `MNN_METAL_LN_STAGE` | unset = auto（仅 hidden ≤ 1024 开） | 折叠 LN 的 prologue 已为算 `sq_sum` 读了整条输入，`LN_STAGE` 顺手把 `(in+res)*gamma` 存进 threadgroup 内存，body 不再重读 `in` / `ln_residual_in` / `gamma`。三态：`1` 强制开（仍受 8KB threadgroup 上限约束），`0` 强制关，unset 走 auto。auto 的 hidden ≤ 1024 是经验门限，不是机理推导。注意代价/收益比**不**随融合成员的悬殊程度变化：packed grid 下每个 threadgroup 恒产出 `quadsPerTG` 个 output quad，staging 代价恒为 `lnStageQuads/quadsPerTG` 次写/output quad | `LN_FUSED` decode GEMV + QKV wide split-K（或 gate/up 且 `MNN_METAL_GATEUP_LN_STAGE` 开）+ `hidden*4 <= 8192` |
| `MNN_METAL_GATEUP_LN_STAGE` | 开（`0` 关） | 允许 gate/up 融合组也用 `LN_STAGE`。默认开但实际很少生效：gate/up 的 `GEMV_2OCQUAD_PER_SG` 形状 body 只重读 2 遍，收益薄，且还要过 `MNN_METAL_LN_STAGE` 的 auto 门 | 同上，gate/up leader |
| `MNN_METAL_LN_STAGE_FP32` | 开（`0` 关） | staged 向量按 `float4` 而非 `ftype4`（fp16 下 `half4`）存储，避免 staging 引入额外的半精度往返 | `LN_STAGE` 已生效时 |

## Decode GEMV（量化权重）

| Env 变量 | 默认 | 功能 | 生效条件 |
|---|---|---|---|
| `MNN_METAL_GEMV_W16` | 开（`0` 关） | `GEMV_QBLOCK_W16` 特化：16 字节 `uint4` 权重读、编译期 quads-per-block、per-lane 连续 pair run。`0` 整支退回通用 `ushort4` body（8B 加载 + 运行期 `quadsPerBlock` + `min()` 尾判） | area==1 + W4 + `ic_4 % blockCount == 0` + `quadsPerBlock ∈ {8,16,32,64}`。一处门控（`MetalConvolution1x1::onResize`）覆盖 plain / split-K 2sg GEMV、QKV / GateUp / LN 三条融合管线；g16 lm_head 没有 W16 body；融合 QKV 的 wide split-K 路由以 W16 为前置条件，`=0` 时该路由自动退到窄形或 dual stream |
| `MNN_METAL_GEMV_W16_MID` | unset = 0（auto，`chooseQ4W16LanesPerBlock`） | `=1/2/4/8` 强制 `GEMV_QBLOCK_W16_MID`，即每个 quant block 由几个 lane 分摊 | W4 + W16 decode GEMV（三档 mid 一致，含 split-K 的 `skMid`）；对 g16 lm_head 分支无效 |
| `MNN_METAL_QKV_SPLITK_MAX_BLOCKS` | 16 | `=N` 把融合 QKV decode GEMV 窄形 split-K（2 quad/TG、4 simdgroup/128 线程）路由的适用范围限制到 quant block 数 ≤ N（设 0 或非法值回落 16） | 融合 QKV decode GEMV 的窄形 split-K 分支（优先级低于 wide 与 dual stream；非 tensor-API 设备走 dual stream、W16 wide 形态均不受此门限制），追加 quant block 数 ≤ N |
| `MNN_METAL_QKV_MERGE` | unset = auto（仅 3 成员 W4 组合并） | 三态：把融合组各成员的权重 / dequant scale 拷进同一块 allocation，shader 以一个基址 + 成员偏移寻址，dispatch 打开的并发 DRAM 流数从 N 降到 1。`1` 连 4 成员组也合并，`0` 从不合并（每成员各自一条权重 + scale 流）。这是把「融合 shader body」与「N 条并发流」分开定价的开关 | 融合 QKV / linear_in decode GEMV，且各成员量化位宽与 quant block 形状一致 |
| `MNN_METAL_QKV_MERGED_OUT` | unset = auto（开） | 三态：把融合组各成员的 output tensor 铺进同一块 STATIC allocation（成员 tensor alias 进 holder），shader 用 buffer(1) 一个基址 + 编译期成员偏移写回，取代 N 个分支选择的 output 指针。`1` 强制开，`0` 强制关（回退到每成员各自 STATIC output + 分支选指针）。注意：任何发生在 `setupFusion` 之后的 `onAcquireBuffer` 都会丢掉这个 alias —— 见 `MetalBackend::onResizeEnd` 里 linear-attention gate fold 必须先于 `setupFusion` 的顺序约束 | 融合 QKV / linear_in decode GEMV（`QKV_MERGED_OUT`） |
| `MNN_METAL_QKV_PACKED_GRID` | unset = auto（仅成员 outputChannel 不等时用 packed） | 三态：`1` 恒用 packed（扁平 1D grid，成员基址由 `qkv_seg` 给出，不再发只会 early-return 的 threadgroup），`0` 恒用矩形 grid `(x, y, numProj)`。两种形态在任何组上都正确 —— packed 基址是无条件填的；这是把 grid 维度与 shader body 分开定价的开关 | 融合 QKV / linear_in decode GEMV |
| `MNN_METAL_ENABLE_LMHEAD_SPLITK` | unset = auto（decode 下 `blockCount < 32 && oc > 200000` 走 g16 内部切 K，其余路由到 split-K 2sg kernel；area≠1 恒走 plain g16） | 超大 oc 的 lm_head decode GEMV 的 K-split 形态：`2` = 在 g16 kernel 内部切 K（`G16_SPLIT_K`：128 线程/tg，每两个 simdgroup 结对拆同一 dual-row 的 K 半段 + threadgroup memory 归约，grid 不变）；`1` = 离开 g16、改路由到 split-K 2sg kernel（quant block 数 ≥ 32 时退到 legacy 2sg）；`0` = legacy g16 | decode + area==1 + oc>16384；`1` 需 `oc%8==0` + quant block 数为偶数；`2` 需 `oc%16==0` + quant block 数为偶数 |

## 量化 GEMM（prefill）

| Env 变量 | 默认 | 功能 | 生效条件 |
|---|---|---|---|
| `MNN_METAL_W4W8_OUTER_DEQUANT_GEMM_TENSORAPI` | 关（即走 fused in-kernel unpack） | `1` 改走 outer-dequant + fp GEMM（A/B 基线 / 紧急回滚） | tensor-API 设备 + W2/W3/W4/W8 + area≥64；非 tensor-API 设备上为 no-op |
| `MNN_METAL_FUSED_Q4_KSPLIT` | unset = auto | K 按 `gid.z` 四分写 fp32 partial，第二趟 `conv1x1_fused_q4_ksplit_reduce` 求和 + bias + activation。`1` 强制开，`0` 强制关（单趟 GEMM）。⚠️ 改变累加顺序 ⇒ 输出非 bit-identical | fused-Q4 stage kernel（tensor-API + Q4 + 非 M64 tile）+ `KVMeta::spec_block > 0`；auto 门 = `area<=32 && UP_DIV(oc,64)<=48 && blockSize>=4` |
| `MNN_METAL_FUSED_Q4_M8` | unset = auto | `conv1x1_fused_q4_gemm_stage_m8`：matmul2d descriptor 直接按 M=8，避免 8 行塞进 M32 tile 只用四分之一占用。`0` 强制关（退回 M32 tile）；**`1` 无效**（kernel 对 area>8 结果错误，不绕过 area 门） | 同上，且 K-split 未启用 + `1 < area <= 8` |
| `MNN_METAL_FUSED_Q4_KSPLIT_M8` | unset = auto | 把 M8 tile 叠在 K-split 上（`conv1x1_fused_q4_gemm_stage_ksplit_m8`）。`0` 保持 M32 tile；`1` 同样无效 | K-split 已启用 + `area <= 8` |
| `MNN_METAL_FUSED_Q4_M64` | unset = auto（`area>=64` 走 M64） | `1` 强制 M64 tile，`0` 强制回 M32 tile | tensor-API 设备 + fused-quant GEMM prefill。⚠️ tri-state 门必须写 `>= 0` / `!= -1`，不能写 `== 0`（`envTriState` unset 返回 0） |
| `MNN_METAL_FUSED_Q4_SMEM_PAD` | 开（`0` 不 pad） | 把 M64 fused-Q4 GEMM 两个 staged operand 的行 stride 从 32 pad 到 40 ftype，打散 threadgroup memory bank conflict（实现上给 matmul2d 传显式 stride）。⚠️ pad 后 stride 必须仍是 8 ftype 的倍数，否则 operand 行不再 16 字节对齐、matmul2d 在 fp16 下出错数 | tensor-API + fused-quant M64 GEMM prefill |
| `MNN_METAL_FUSED_Q4_UNORM` | 开（`0` 回标量解包） | 用 `unpack_unorm4x8` 一次解 4 个 nibble，替代逐位 shift/mask | tensor-API + fused-Q4 GEMM |
| `MNN_METAL_FUSED_Q4_SILU_MUL` | 开（`0` 保留独立 dispatch） | 把 gate/up 组的 `MUL_SILU` 折进 up projection 的 M64 epilogue：up 直接读 gate tile 并存 `up * silu(gate)`，省掉一次全张量写 + 一次全张量读 | fused-Q4 M64 tile + `area >= 256` |
| `MNN_METAL_FUSED_Q4_GATEUP_DUAL` | 开（`0` 保持两次 dispatch） | 一次 dispatch 同时累加 gate 与 up 的 N=32 tile 并在寄存器里做 silu-mul，gate 张量完全不落地 | 同 `FUSED_Q4_SILU_MUL`，叠在其上 |

## 融合

| Env 变量 | 默认 | 功能 | 生效条件 |
|---|---|---|---|
| `MNN_METAL_DISABLE_LN_FUSION` | 关（即 AddRMSNorm 折入投影） | `1` 启用两阶段回退：独立 binary AddRMSNorm 物化 `residual_out + normalized`，再由已融合的 QKV/GateUp 投影读取 `normalized` | 带 `has_ln` 的 `FusedLinear` decode（prefill 本来就是独立 dispatch）。不影响独立的 `GatedRMSNorm` op |
| `MNN_METAL_DISABLE_QKV_FUSION` | 关（即开融合） | `1` 关掉 Q/K/V leader/follower 融合，恢复独立 GEMV dispatch | decode GEMV + 同一输入恰有 3 或 4 个 conv1x1 消费者（attention q/k/v，或 linear-attention 层的 qkv/z/b/a，`QKV_FUSED_P4`：grid.z=4）。融合关系由 `MetalFusedProj::setupFusion()` 按导出的 `FusedLinearParam` 成员顺序建立 ⇒ 未带 `--fuse_qkv_proj` / `--fuse_ln_proj` 导出的模型无此融合 |
| `MNN_METAL_DISABLE_GATE_UP_FUSION` | 关（即开融合） | `1` 关掉 Gate/Up leader/follower 融合 | 同上（`setupFusion()` 内的 gate/up 分支）⇒ 未带 `--fuse_gate_up_proj` 导出的模型无此融合 |
| `MNN_METAL_DISABLE_ROPE_X_CACHE` | 关（即开缓存） | 融合 RoPE kernel 的 `USE_SG` 路径原本把 q/k **整条读两遍**：一遍算 RMSNorm 的 `square_sum`，一遍做旋转。`step=32` 下 lane `t` 覆盖 `{t, t+32, ...}`，而旋转要的 `i` 与 `i+ropeHalfD` 在 `ropeHalfD % 32 == 0` 时落在同一 lane ⇒ 第一遍读到的值留在寄存器里即可复用，无需 shuffle。`1` 恢复重读（rollback + A/B）。`ROPE_D` / `ROPE_HALF_D` 随之烘成 shader 常量（否则循环上界是 runtime 值、不展开，寄存器数组会溢到 thread-local 内存），故二者进 pipeline cache key | `MNN_SUPPORT_TRANSFORMER_FUSE` 的 `MetalRope`；需 `q_norm`/`k_norm` 存在且 `supportSimdGroupReduce()`（即 `USE_SG` 路径），且 `headDim % 32 == 0`、`ropeHalfD % 32 == 0`、`headDim/32 <= 8`（寄存器数组上限，即 `headDim <= 256`）。不满足则走原重读路径 |
| `MNN_METAL_ROPE_TILE` | unset = auto（`outerSize` 到 launch-bound 门限以上开，decode 与极小 prefill 走标量路径） | `rope_kernel_tile`：token 分块的融合 RoPE kernel。C4 输入里同一 plane 内 token 相邻，标量 kernel 一 lane 一个 channel slice ⇒ 每条 simd load 拆成每 lane 一次请求；本 kernel 第一趟让 lane 铺 `ROPE_TILE_T` 个连续 token，每个 plane group 只发一次连续读，把 tile 暂存进 threadgroup memory，第二趟切回"一 lane 一个输出 quad"做 `simd_sum` 归约与旋转（输出是稠密 `[token][head][dim]`，该映射下写本来就是连续的）。`1` 只要 shape 满足就强制开，`0` 不编译这条 pipeline | `MNN_SUPPORT_TRANSFORMER_FUSE` 的 `MetalRope`；需 `USE_SG` 路径 + `ropeHalfD > 0` + `headDim % 4 == 0` + `ropeHalfD % 4 == 0` + `(headDim/4) % (32/tileT) == 0` + `headDim <= 512`。标量与分块两条 pipeline 都在构造函数里编译，`onResize` 按 `outerSize` 选一条；未编译成功时自动回落标量 |
| `MNN_METAL_ROPE_TILE_T` | unset = 8 | `=N` 设分块 kernel 每个 simdgroup 承担的 token 数，也就是一次连续读的宽度；N 同时决定 threadgroup staging 的大小，故是"请求数 vs occupancy"的取舍旋钮 | 仅 `rope_kernel_tile`；需 N 整除 32 且 `(headDim/4) % (32/N) == 0`，否则忽略、回落默认值 |

## 调度 / 运行时

| Env 变量 | 默认 | 功能 | 生效条件 |
|---|---|---|---|
| `MNN_METAL_RESIZE_WAIT` | `local`（per-backend fence） | `global` = 旧的全局 drain；`none` = 完全跳过（实验，不安全） | `onResizeBegin` 每次 |
| `MNN_METAL_H2D_QUEUED` | 开（`0` 关） | 输入上传走 queued 路径；`0` 恢复旧的逐 token drain + 直写 | decode 每 token 输入上传 |
| `MNN_METAL_SPIN_WAIT` | 开（`0` 关） | `MetalBackend::wait()` 自旋轮询 `buffer.status`（`sched_yield`）而非阻塞 `waitUntilCompleted`，约 20ms 封顶后仍回落阻塞。代价是等待期间占满一个核 | 每次 `MetalBackend::wait()` |
| `MNN_METAL_COMMIT_NUM` | 0（走 Metal 常量 30；若显式设过 `OP_ENCODER_NUMBER_FOR_COMMIT` 则该 hint 优先） | `=N` 覆盖每次 commit 的 op encoder 数 | `isCmdBufferCommit()` 每次判断 |
| `MNN_METAL_DISABLE_REPLAY` | 关（即开 replay） | `1` 关掉 encode replay（稳定 shape 下录制 op encode 命令列表、后续 token 直接重放，见 `MetalReplay.hpp`） | `MetalExecution::onExecute`；attention 经 `onReplayUpdate` hook 接入（per-token 参数/grid 补丁 + KV 指针身份校验），linear-attention decode 已接入（seqLen==1 + resize-generation guard），prefill 豁免 |

## Profiling / 诊断

| Env 变量 | 默认 | 功能 | 依赖 |
|---|---|---|---|
| `MNN_METAL_OP_PROFILE_TIMELINE=<path.csv>` | 关（unset/空 ⇒ nullptr） | 把每 op 的 GPU `(t0, t1)` 时间戳 dump 到 CSV，供 `tools/script/metal_profile_gantt.py` 消费 | `-DMNN_METAL_OP_PROFILE=ON` |
| `MNN_METAL_OP_PROFILE_LEGACY` | 关（`1` 开） | 回退到旧的"每 op 一个 command buffer"profile 模式 | `-DMNN_METAL_OP_PROFILE=ON` |
| `MNN_METAL_REPLAY_DEBUG` | 关（`1` 开） | 打印每个 op 的 record / ban / invalidate 事件 | 配合 replay 使用 |
| `MNN_METAL_PIPELINE_INFO` | 关（`1` 开） | 每条源码编译出的 pipeline 打印 kernel 名、`maxTotalThreadsPerThreadgroup` / `threadExecutionWidth` / static threadgroup memory，以及它被特化时的全部 preprocessor 宏。用来直接确认某个 shape 实际路由到了哪个 shader 变体，而不是从 host 侧门控反推 | 无（production build 亦可用） |

> CPU 侧 trace（Session 级 resize/malloc/run + Metal 级 encode/commit/wait 计时汇总，退出时打印）不是 env 开关，而是**编译宏** `-DMNN_SESSION_CPU_TRACE`（`source/core/Session.cpp` + `source/backend/metal/`），生产 build 不编译、零开销。

> ⚠️ **profile 数据只能用于诊断，不能作为优化目标**：counter sample buffer attachment 会把 CPU op encode 开销放大一个数量级，由此"制造"出的 GPU idle 是测量伪影。任何基于 profile-ON gantt 的假设，必须用 production build 交替配对 A/B 复核。

## 新增开关的规范

1. 优先按"默认行为反义"命名：默认开的用 `MNN_METAL_DISABLE_*`，默认关的用 `MNN_METAL_ENABLE_*`。
2. 多值开关用 named string（如 `local` / `global` / `none`）而非 0/1/2。
3. 每个新增 env **必须**：
   - 在 `source/backend/metal/MetalEnv.hpp` 加字段（禁止散落 `getenv`）；
   - 在本文件加一行（只写名称 / 默认值 / 功能 / 生效条件）；
   - 在 commit message 里写默认值与 A/B 用法。
4. 开关**删除或改名时，本文件的对应行必须同步删除**，不保留旧名与删除线。
