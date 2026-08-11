---
name: metal-optimize
description: MNN Metal 后端 op/kernel 开发与优化入口。索引四份 sub-doc：kernel 开发规范与优化知识库（命名/写法/GEMV/GEMM/attention）、算子融合全链路（导出图→converter→Metal 单 dispatch）、运行时调度（fence/content-cache/H2D/replay）、构建测试基线、env 开关注册表。根据当前任务选择性阅读对应 sub-doc。
---

# MNN Metal 优化 Skill（索引）

> **触发**：新增或修改 Metal kernel / shader / dispatcher；LLM decode/prefill 性能优化；
> 算子融合（导出侧声明 + 后端单 dispatch）；per-op profiling 定位瓶颈；跑 Metal LLM 测试或对拍。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

## 使用方式

**先按任务类型定位到具体 sub-doc**，本文件只做索引和路由，不承载技术内容。

## Sub-doc 结构

| 文件 | 何时阅读 | 内容 |
|---|---|---|
| **[`kernel-dev-and-optimize.md`](./kernel-dev-and-optimize.md)** | 写或改任何 Metal kernel **之前**都先读第一部分；做 kernel 层性能优化时读第二部分 | **第一部分 开发规范**：核心原则、dispatcher 结构、shader 组织与 kernel 命名约定、编译期宏与 pipeline 缓存 key 的四处同步、Execution 骨架与注册、6 个通用陷阱、packed weight 设计、修改流程、正确性验证、tensor API cooperative tensor 布局、Apple GPU 杠杆。<br>**第二部分 优化知识库**：优化总纲与 decode 画像、GEMV（deferred dequant / 双 SG / pre-scaling / SPLIT_K / ROW_2 / W2W3 / 短序列缺口）、GEMM（fused Q4 / M64 / in-shader 阈值）、Attention（causal-tri / FA / tensor-API nax / 单 pass SDPA / QSPLIT / GQA / 路由速查）、其他 kernel（RMSNorm 小 batch / LinearAttention scan 演进 / gated norm） |
| **[`graph-fusion.md`](./graph-fusion.md)** | 要让多个算子合成一次 dispatch；改 `FusedLinear` / `GatedRMSNorm`；排查"融合没命中 / 融合后输出错" | 融合全链路：Python 导出期声明分组 → converter LN 吸收 → geometry 兜底拆分 → Metal `setupFusion` 装配 leader/follower。含内存别名铁律与 STATIC re-home、链式门控依赖、`GatedRMSNorm` 独立链路、已删除的后端图匹配历史、排查清单 |
| **[`runtime-scheduling.md`](./runtime-scheduling.md)** | 怀疑 decode 有 CPU 阻塞 / GPU 空泡；改 resize 时机、commit 节奏、H2D、Encode Replay | per-backend fence、content-cache、队内 H2D 上传、设备端采样（ArgMax/TopKV2）、commit cadence、Encode Replay（安全模型 / attention 与 LinearAttention 接入 / KV 悬垂指针坑）、调度类改动的验证套路 |
| **[`build-and-test.md`](./build-and-test.md)** | 改完代码要 build / 跑测试 / 对拍 / 查文件索引 | cmake 编译命令、模型导出命令、性能测试命令、CPU/Metal 对拍、性能基线数据、全文件索引 |
| **[`env-registry.md`](./env-registry.md)** | 查 / 新增 Metal 相关环境变量开关 | env 集中登记：性能路径 / 融合 / profiling 三类；默认值、打开效果、定型状态；命名规范；profile ON 数据不能作为优化目标的警告 |

## 快速任务→sub-doc 索引

| 想做的事 | 优先读 |
|---|---|
| 新加一个 Metal op / kernel（该叫什么名、写在哪、怎么注册）| `kernel-dev-and-optimize.md` §1.3 / §1.5 |
| 加一个编译期变体宏 | `kernel-dev-and-optimize.md` §1.4（**四处必须同步**）|
| 新加 quant bit / 改 dispatcher 路径 | `kernel-dev-and-optimize.md` §1.2 / §1.6 / §1.7 |
| 想知道 Metal 的坑（宏 alias / weight byte order / getDequantScale coef）| `kernel-dev-and-optimize.md` §1.6 |
| Apple GPU 优化杠杆选择（sg_matrix / sg_reduce / tensor API）| `kernel-dev-and-optimize.md` §1.11 / §1.10 |
| **决定要不要投入某个优化方向（先看这条）** | `kernel-dev-and-optimize.md` §2.0——先自测 GPU busy vs wall；若已 GPU-bound 且 occupancy 受限，<5us 级 GPU 节省不兑现为 wall |
| GEMV 优化（decode 主战场）| `kernel-dev-and-optimize.md` §2.1 |
| GEMM / prefill 优化 | `kernel-dev-and-optimize.md` §2.2 |
| Attention 优化 / 想知道当前走哪条路径 | `kernel-dev-and-optimize.md` §2.3（§2.3.8 是路由速查）|
| LinearAttention（Qwen3.5 gated delta rule）| `kernel-dev-and-optimize.md` §2.4.2 / §2.4.3 |
| 算子融合：为什么没命中 / 融合后输出错 | `graph-fusion.md` §8 排查清单 |
| 新模型结构要加融合 | `graph-fusion.md` §1（导出侧声明）+ §4（后端装配）|
| 融合后输出逐次不同 | `graph-fusion.md` §4.3 内存别名 |
| decode 每 token 的 CPU 阻塞 / 同步开销 | `runtime-scheduling.md` |
| Encode Replay 相关（新 op 要不要接入 / 为什么被 ban）| `runtime-scheduling.md` §6 |
| cmake 编译选项 / 模型导出命令 / 找哪个文件负责什么 | `build-and-test.md` |
| 查某个 env 开关的默认值和语义 | `env-registry.md` |

## 通用原则速览（细节见 `kernel-dev-and-optimize.md`）

1. **shader 是嵌入的 C++ 字符串**（`R"metal(...)metal";` 在 `*Shader.hpp` 里），不是独立 `.metal` 文件；公共头靠**字符串拼接**共享。改完直接 make。
2. **变体用 `preprocessorMacros`**，不用 function constants。加宏必须同步改四处：shader `#ifdef`、pipeline 缓存 key、宏字典、`onResize` 的 grid/threadgroup。
3. **dispatcher 要先摸清**：一个 op 常有多条 kernel，扩之前先决定支持哪几条 + 让其他路径显式 fallback。
4. **Apple GPU ≠ Android**：M3/M4/M5 之间都不能互推，更不能推 Vulkan/OpenCL。设备分档看 `architecture.name` / `isSupportTensorApi()`。
5. **正确性 oracle 先于性能**：fp32（`precision: high`）bit-identical 是最强证据；fp16 greedy 对拍是次强。**token 级一致不等于 bit 级一致**。
6. **融合必查内存别名**：把前驱折进后继时，前驱的输入可能已被分配器复用为后继的输出。
7. **A/B 必须交替配对**：热态漂移能造出 3 倍虚假收益；profile build 的绝对数字是伪影。

## 相关 Skills

- `skills/bugfix/` — 内存别名 / 生命周期错误排查（Metal 后端共用同一套方法论）
- `skills/general-debug/` — 正确性 bug / 回归诊断
- `skills/opencl-optimize/`、`skills/vulkan-optimize/`、`skills/arm-cpu-optimize/` — 其他后端
- `skills/support-new-llm/` — 新增 LLM 模型的完整流程
- `skills/test-ci/` — 单测 / 回归测试
