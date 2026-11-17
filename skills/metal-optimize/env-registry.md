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
| `MNN_METAL_FASG_LOADBATCH` | 4 | `=N`：消费任何 fragment 前先批量发射 N 个 `simdgroup_load`，让 load 延迟被前面的 MMA 覆盖；N 同时是 PV 侧 V-fragment 批宽。`0` 关闭批量发射 | 仅 `prefill_flash_attn_sg`；host 侧要求 `N | HEAD_DIM/8`，否则降为 0 |
| `MNN_METAL_ATTN_QSPLIT_MB` | 128（MB） | auto 策略给每片 scratch 的预算上限，三段路径的 `mTempQK`/`mTempSoftMax` 峰值按它推导 q 序列切分数 | 三段 prefill 路径 |

## LayerNorm

| Env 变量 | 默认 | 功能 | 生效条件 |
|---|---|---|---|
| `MNN_METAL_LN_TOKENS_PER_TG` | 8 | `=N` 让 N 个 token 共用一个 threadgroup（N 个 simdgroup 各归约一行，互不通信）；`1` 回退一 token 一 threadgroup 的 `layernorm_c4_rms_sg`。归约始终在 simdgroup 内，本开关只在 threadgroup 数与 threadgroup 宽度之间取舍 | C4 + RMSNorm + simdgroup reduce + `outside>=256`（prefill；decode `outside==1` 恒走旧路径） |

## Decode GEMV（量化权重）

| Env 变量 | 默认 | 功能 | 生效条件 |
|---|---|---|---|
| `MNN_METAL_QKV_SPLITK_MAX_BLOCKS` | 16 | `=N` 把融合 QKV decode GEMV plain split-K（2 quad/TG、4 simdgroup/128 线程）路由的适用范围限制到 quant block 数 ≤ N（设 0 或非法值回落 16） | 融合 QKV decode GEMV 的 plain split-K（`useOc8Sg4K2`）分支；oc16Sg8K2 与双流体（`oc16Sg2K1`）形态不受此门限制 |

## Prefill GEMM（量化权重）

| Env 变量 | 默认 | 功能 | 生效条件 |
|---|---|---|---|
| `MNN_METAL_FUSED_Q4_GATEUP_DUAL_MIN_AREA` | 128 | `=N` 覆盖启用 gate/up 双累加器（`FQ4_GATEUP_DUAL`，N 拆成两个 32 宽 tile、gate 张量完全不落地）与 `FQ4_SILU_MUL` 折叠的最小 prefill area；非法值（<64）回落 128 | `conv1x1_fused_q4_gemm_stage_m64`，**仅 Q4**（W8 走同一 kernel 但不启用 dual/silu 折叠），且 owner 提供了 dual peer / gate 张量 |

## 调度 / 运行时

| Env 变量 | 默认 | 功能 | 生效条件 |
|---|---|---|---|
| `MNN_METAL_RESIZE_WAIT` | `local`（per-backend fence） | `global` = 旧的全局 drain；`none` = 完全跳过（实验，不安全） | `onResizeBegin` 每次 |
| `MNN_METAL_H2D_QUEUED` | 开（`0` 关） | 输入上传走 queued 路径；`0` 恢复旧的逐 token drain + 直写 | decode 每 token 输入上传 |
| `MNN_METAL_SPIN_WAIT` | 开（`0` 关） | `MetalBackend::wait()` 自旋轮询 `buffer.status`（`sched_yield`）而非阻塞 `waitUntilCompleted`，约 20ms 封顶后仍回落阻塞。代价是等待期间占满一个核 | 每次 `MetalBackend::wait()` |
| `MNN_METAL_QOS_BOOST` | 开（`0` 关） | 前向窗口（`onExecuteBegin`→`onExecuteEnd`）内把调用线程 QoS 提升到 `USER_INTERACTIVE`，窗口结束恢复原值；防止编码/提交循环被调度器放到能效核导致 per-token encode 延迟膨胀、command buffer 间出现 GPU 气泡。线程已 ≥ 该 QoS 时跳过（嵌套 backend / 宿主自管 QoS 不受影响） | 每次前向 |
| `MNN_METAL_DISABLE_REPLAY` | 关（即开 replay） | `1` 关掉 encode replay（稳定 shape 下录制 op encode 命令列表、后续 token 直接重放，见 `MetalReplay.hpp`） | `MetalExecution::onExecute`；attention 经 `onReplayUpdate` hook 接入（per-token 参数/grid 补丁 + KV 指针身份校验），linear-attention decode 已接入（seqLen==1 + resize-generation guard），prefill 豁免 |

> commit 节奏（每次 commit 的 op encoder 数）由 `Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT` hint 或设备标定决定，不是 env 开关。

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
