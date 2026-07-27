# MNN Metal 环境变量注册表

> **目的**：集中登记全部 Metal 后端环境变量 —— 默认值 / 语义 / 提交出处 / 适用设备 / 定型状态，避免"某开关默认怎么走" / "哪个 env 是关哪个是开"的坑。
>
> **单一入口**：所有开关在 `source/backend/metal/MetalEnv.hpp` 统一声明与解析（2026-07-24 工程债清理），后端代码禁止散落 `getenv`。新增开关必须同时改 MetalEnv.hpp 与本表。
>
> **详解文档**：开关背后的优化机制、实测数据与正交性分析见 [`perf-playbook.md`](./perf-playbook.md)。
>
> **形态约定**：`1` = 显式开；`0` = 显式关；未设 = 走默认（每项在"默认"列注明）。语义不一是历史遗留，新增开关请遵循 `MNN_METAL_ENABLE_*` = 默认关、`MNN_METAL_DISABLE_*` = 默认开的命名。

## 性能路径开关

| Env 变量 | 默认 | 打开效果 (=1) | 关闭 (=0 或 unset) | 生效条件 | 定型状态 |
|---|---|---|---|---|---|
| `MNN_METAL_QK_CAUSAL_TRI` | **开** | (无效，本身就默认开) | 关掉 `mQkCausalTri` 和 `mCausalBound` → 三段 QK 走矩形 dispatch + softmax/AV 不做因果边界优化 | prefill + causal mask + `!mFlashAttnPrefill` | ✅ 定型（`9948c74e1` + `78ae7bc55` + `f28510967`）。⚠️ **正确性警告**：默认开启依赖 "mask 是 causal lower-triangular" 的隐式假设。**加载 SWA (Mistral v0.1, Gemma-2, Ministral) / prefix LM / bidirectional / encoder 类模型必须显式设 =0**，否则静默乱码。详见 `general-debug/SKILL.md` §7 + `metal-optimize/build-and-test.md` § "Attention causal 假设" |
| `MNN_ENABLE_FLASH_ATTN_PREFILL` | 由 config `attention_mode/8` 决定 | 强制启用 FA prefill | =0 强制走三段路径；unset 由 config 决定 | prefill + fp16 + head_dim∈{64,128,256} | 定型 A/B 用；M4/M5 默认已 demote 到三段。⚠️ **FA kernel 也硬编码 causal 假设**且**无 opt-out**：非 causal 模型必须用 `=0` 关 FA，同时用 `MNN_METAL_QK_CAUSAL_TRI=0` 关三段 causal-tri |
| `MNN_METAL_PREFILL_INSHADER_DEQUANT` | 自动（>4M 且 area<512 走 in-shader，EXP11） | =1 强制 in-shader dequant | =0 强制 outer-dequant + fp GEMM | non-tensor-API 设备 + area>1 + Q4/Q8 | 定型 A/B 用；area<512 边界仅 M4 标定（4B pp2048 +5.3%），**M3 验证 blocking** |
| `MNN_METAL_RESIZE_WAIT` | **`local`**（per-backend fence，隐式）| `global` = 旧全局 wait；`none` = 完全跳过（实验，不安全）| 默认 fence 已生效 | onResizeBegin 每次 | 定型；`local` 转正（`7f186691a`）|
| `MNN_METAL_DECODE_SPLITKV` | **开（阈值 3072）** | `=N`(N>1) 覆盖 kv 阈值；`=1` 等价默认 3072 | =0 关闭 | decode seq=1 + fp16 或 int8 量化 KV（EXP09）+ trivial/无 mask + head_dim%32==0 | 定型默认开（EXP07 M4 标定：kv4K 0.6B +19% / 4B +5.5% / 2B +1~3%；交叉点 M4 ~1.2k / M5 3072，阈值取保守档 3072，iPhone/M3 标定后可放宽）|
| `MNN_METAL_DISABLE_FUSED_Q4_GEMM` | **关**（fused 默认启用）| =1 回退 outer-dequant + fp GEMM（A/B 基线 / 紧急回滚）| unset 走 fused | tensor-API 设备 + Q4/Q8 + area≥64（prefill）| 定型；fused 路径含 M64 tile 自动策略（Q4 + area≥128，M5 pp512 +5.9% / pp2048 +6.8%）|
| `MNN_METAL_GEMM_M64` | **关** | =1 outer-dequant prefill 走 `conv1x1_gemm_64x64_split_k_sg`（sg_matrix M64 tile，权重重复读减半） | unset/=0 走 32x64 | non-tensor-API 设备 + outer-dequant + area≥128 | 实验（EXP10）：M4 +0.8~1.7% 全场景正收益、3 模型 greedy 一致；待 M3 验证后评估默认开 |
| `MNN_METAL_H2D_QUEUED` | **开** | (无效，默认开) | =0 恢复旧的逐 token drain + 直写上传路径 | decode 每 token 输入上传 | 定型（`757610526`，0.6B decode +5.2%）；`0` 用于回滚 |
| `MNN_METAL_COMMIT_NUM` | 0（走 runtime hint `OP_ENCODER_NUMBER_FOR_COMMIT`，llm.cpp tuning 决定）| `=N` 覆盖每 commit 的 op encoder 数 | unset 走 hint | Metal 每次 commit 判断 | 设备标定用（EXP01：默认 N=10 已最优，N=2 -7%，N=999 -6%）|

## 融合 / dispatch 开关

| Env 变量 | 默认 | 打开效果 (=1) | 生效条件 | 定型状态 |
|---|---|---|---|---|
| `MNN_METAL_DISABLE_LN_FUSION` | **开融合**（unset 时启用）| 关掉 LayerNorm fusion，恢复独立 dispatch | Metal 后端；merged proj 场景 | 定型 A/B；日常不动 |
| `MNN_DISABLE_GATE_UP_FUSION` | **开融合**（unset 时启用）| 关掉 Gate/Up leader/follower fusion | Metal 后端；MUL_SILU 匹配 | 定型 A/B（历史命名，无 METAL 前缀）|

## Profiling / 诊断

| Env 变量 | 默认 | 打开效果 (=1 / =<path>) | 用途 | 依赖 |
|---|---|---|---|---|
| `MNN_METAL_OP_PROFILE_TIMELINE=/path.csv` | 关 | 把每 op GPU (t0, t1) 时间戳 dump 到 CSV，给 `tools/script/metal_profile_gantt.py` 消费 | 甘特图 / GPU idle gap 归因 | 需要 `-DMNN_METAL_OP_PROFILE=ON` |
| `MNN_METAL_OP_PROFILE_LEGACY` | 关 | 回退到旧的每 op 一 command buffer 的 profile 模式（绝对数字失真但简单）| 兼容不支持 stage-boundary sampling 的设备 | 需要 `-DMNN_METAL_OP_PROFILE=ON` |

> CPU 侧 trace（Session 级 resize/malloc/run + Metal 级 encode/commit/wait 计时汇总，退出时打印）已从 env 开关改为**编译宏** `-DMNN_SESSION_CPU_TRACE`（`source/core/Session.cpp` + `source/backend/metal/`），生产 build 不编译、零开销。详见 [`perf-playbook.md`](./perf-playbook.md) §4.1.3。

## Profile 环境警告（重要）

**`-DMNN_METAL_OP_PROFILE=ON` + `MNN_METAL_OP_PROFILE_TIMELINE` 数据只用于诊断，不能作为优化目标**（`7cd053d7f` / 2026-07-23 的 async fire 负面结果教训）：

- Counter sample buffer attachment 让 CPU op encode 从 ~0.92us/op 涨到 4-20us/op（20×）
- 制造出的 GPU idle 是**测量伪影**，生产环境（profile OFF）decode 已经 GPU-bound
- 任何基于 profile ON gantt 的假设收益，**必须** 用 production build 3-rep 交替配对 A/B 交叉验证

## 已淘汰 / 已删除

| Env | 状态 | 说明 |
|---|---|---|
| ~~`MNN_METAL_CPU_TRACE`~~ | 已改为编译宏 | 运行时 env 开关删除，改为编译宏 `-DMNN_SESSION_CPU_TRACE`（见上方 Profiling 一节说明）|
| ~~`MNN_ENABLE_QKV_FUSION`~~ / ~~`MNN_DISABLE_QKV_FUSION`~~ | 已删除（2026-07-24 工程债清理）| 运行时 QKV triple fusion 实测 tg128 -36%（0.6B）/+1%（4B），整条路径（matchQKVFusions + QKV_FUSED shader）连同开关删除 |
| ~~`MNN_METAL_LMHEAD_4SG`~~ | 已删除（2026-07-24）| lm_head 4SG 变体 e2e 持平且 stddev 7× 恶化，G16_4SG shader 分支删除；结论存 `perf-playbook.md` |
| ~~`MNN_METAL_LMHEAD_VARIANT`~~ | 已删除（2026-07-24）| G16_OC4 变体 kernel -4.8% 但 e2e 持平（M5 管线空泡吸收），shader 分支删除 |
| ~~`MNN_METAL_FUSED_Q4_STAGE`~~ / ~~`MNN_METAL_FUSED_Q4_M_TILE`~~ | 已删除（2026-07-24）| 开发期 bisect 脚手架（stage 1-3 stub 模式 + conv1x1_dequant_only_q4 kernel 一并删除）；收敛为 `MNN_METAL_DISABLE_FUSED_Q4_GEMM` 单 kill-switch，M64 tile 转为自动策略 |
| ~~`MNN_METAL_FORCE_TENSOR_API`~~ | 已删除（2026-07-24）| tensor-API 设备（M5+）prefill Q4/Q8 现无条件走 tensor API 路径（sg_matrix A/B 实测 -48%，开关无日常价值）|
| ~~`MNN_LLM_RESIZE_CACHE`~~ | 已删除（2026-07-24） | 被 `1ef0ace93` content-cache 取代；实验路径已知 greedy 发散，代码已从 llm.cpp 移除 |
| ~~`MNN_EXPORT_MERGE_QKV`~~ | 已删除（2026-07-24，整条链路） | 导出开关（transformers.py qkv/gate-up 合并）+ 消费端（converter `fuseQkvPackedC4`、RoPE/Attention 形状放行、Metal `kBaseOffset`/`v_channel_offset`）全部删除。历史数据（M4 Pro decode +6.2%）与重做方案见 [`perf-playbook.md`](./perf-playbook.md) §2.3.2 |
| ~~`MNN_METAL_RESIZE_NOWAIT`~~ | 已淘汰 | 被 `MNN_METAL_RESIZE_WAIT=local` (默认) / `=global` / `=none` 三值统一取代 |

## 命名规范建议（后续新开关）

1. 优先按"默认行为反义"命名：默认开的加 `MNN_..._DISABLE_...`，默认关的加 `MNN_..._ENABLE_...`
2. 三值开关（`local` / `global` / `none`）用 named string 而非 0/1
3. 每个新增 env **必须**：
   - 在 `source/backend/metal/MetalEnv.hpp` 加字段（禁止散落 `getenv`）
   - 在本文件加一行
   - 在 commit message 里注明默认值和 A/B 用法
