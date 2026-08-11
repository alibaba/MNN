---
name: metal-optimize
description: MNN Metal 后端 op/kernel 优化与扩展入口。索引四份 sub-doc：kernel 修改方法论（新加 op / quant bit / dispatcher / weight pack）、LLM 优化知识库（kernel 层 GEMM/GEMV/attention + 系统架构层 + 失败实验档案 + 测量基建）、构建-测试-性能基线、env 开关注册表。根据当前任务选择性阅读对应 sub-doc。
---

# MNN Metal 优化 Skill（索引）

> **触发**：修改或优化 Metal 端 kernel（conv/gemm/gemv/attention 等），新加算子，调 dispatcher 或 weight pack；LLM decode/prefill 端到端性能优化；per-op profiling 定位瓶颈；跑 Metal LLM 性能测试或对拍。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

## 使用方式

**先按任务类型定位到具体的 sub-doc**，本文件只做索引和路由，不承载技术内容。

## Sub-doc 结构

| 文件 | 何时阅读 | 内容 |
|---|---|---|
| **[`kernel-basics.md`](./kernel-basics.md)** | 修改任何 Metal shader / kernel / dispatcher **之前**都先读 | Metal 特有的核心原则、入口定位、5 个通用陷阱（宏 alias / dispatcher 漏路径 / weightTransform 多签名 / getDequantScale coef / tile byte 顺序）、packed weight 设计、shader 修改流程、Apple GPU 性能观察、通用正确性验证 |
| **[`perf-playbook.md`](./perf-playbook.md)** | 做 LLM decode / prefill 端到端性能优化，或参考具体优化的实现细节 / 查某方向是否已试过 | 优化知识库：**一、Kernel 层**（GEMV deferred dequant / 双 SG / pre-scaling / 带宽画像；GEMM fused-Q4 / M64 / 面积阈值；attention causal-tri / FA / split-KV）；**二、系统架构层**（dispatch 融合 / fence / content-cache / H2D 队列 / 图与导出侧）；**三、无效与负收益实验档案**（避免重试）；**四、profiling 基建与测量方法论**。含性能数据、调优实验、避坑记录、前瞻路线图 |
| **[`build-and-test.md`](./build-and-test.md)** | 改完代码要 build / 跑测试 / 对拍 / 查文件索引 | cmake 编译命令、模型导出命令、性能测试命令、CPU/Metal 对拍与 FA A/B 对拍、性能基线数据、全文件索引 |
| **[`env-registry.md`](./env-registry.md)** | 查/新增 Metal 相关环境变量开关 | 14 个 env 集中登记：性能路径 / 融合 dispatch / profiling 三类；默认值 / 打开效果 / 定型状态；命名规范；profile ON 数据不能作为优化目标的警告 |
| **[`mlx-comparison.md`](./mlx-comparison.md)** | 需要 MNN vs MLX 差距数据，或优化合入后更新对比 | 同机同热态交替配对的 MNN/MLX decode+prefill 对比：测量方法（模型/命令/协议）、当前差距表、历史记录（每次优化后追加）、差距归因 |
| **[`plan-multitoken-gemv.md`](./plan-multitoken-gemv.md)** | **接手 decode 性能优化时先读这份**——当前唯一在推进的方向 | 短序列（area 2..16）GEMV 路径补齐的执行计划（路线图 #9 第一步）：已测摊薄曲线与根因、爆炸半径界定、4 个 Phase 与各自判据、验证矩阵、回退与红线。单 token 路径的候选已全部证伪，勿再从那边立项 |

## 快速任务→sub-doc 索引

| 想做的事 | 优先读 |
|---|---|
| **继续 / 接手 decode 性能优化（先看这条）** | ① `plan-multitoken-gemv.md`（路线图 #9，**当前主推方向**，摊薄曲线实测 B=4 已 2.58×）；② `plan-fused-decode-attention.md`（路线图 #20，**「追 MLX」命题已永久关闭**：干净口径重测 MLX SDPA kv576 = 16.36us/op（非 10.45），**MNN ≈17.0us/层 与 MLX 本就持平**，不要再基于「MLX 融合 kernel 更快」立项，重启须先复现该文档 §5.3 的 MLX 口径测量。**但另一命题成立**：单 pass 融合 SDPA `MNN_METAL_DECODE_SDPA`（该文档 §7）在 kv≥split-KV 阈值档**替代自家 3-dispatch split-KV** e2e +5~6%（M5 0.6B p2048，4B +2.5~3.4%），短中 kv 仍判负维持现状——这是对自家路径的改进，与 MLX 无关）。单 token GEMV/CPU 同步/concurrent encoder 等候选**已全部证伪**（`perf-playbook.md` §4.3 结论四/五/六），勿从那边重新立项 |
| 新加 quant bit / 新 op / 改 dispatcher 路径 | `kernel-basics.md`（陷阱 A/B/E、packed weight 设计、Shader 修改流程） |
| 想知道 Metal 的坑（`#ifdef` alias / weight byte order / getDequantScale coef）| `kernel-basics.md` § 通用陷阱 |
| Apple GPU 优化杠杆选择（sg_matrix / sg_reduce / g4mN）| `kernel-basics.md` § Apple GPU 性能观察 |
| Metal LLM 端到端 profile 定位瓶颈 | `perf-playbook.md` §4.1 |
| GEMV 优化（Q4 deferred dequant / 双 SG / pre-scaling / 带宽画像）| `perf-playbook.md` §1.1 |
| GEMM 优化（fused-Q4 / M64 tile / in-shader 阈值）| `perf-playbook.md` §1.2 |
| Attention 优化（causal-tri / flash-attn / split-KV / GQA 扩展）| `perf-playbook.md` §1.3 |
| Dispatch 融合（Gate/Up、LN fusion）与管线同步（fence / content-cache / H2D）| `perf-playbook.md` §2.1–2.2 |
| 图优化 / 导出侧（RoPE fusion、QKV/GateUp 权重合并、导出修复）| `perf-playbook.md` §2.3 |
| 想做的优化是否已试过 / 有无负收益前科 | `perf-playbook.md` §三 |
| 测量方法论（交替配对、热态分段、profile 伪影）| `perf-playbook.md` §4.2 |
| cmake 编译选项 / 模型导出命令 | `build-and-test.md` § 编译 / 模型导出 |
| CPU vs Metal 对拍 / FA A/B 对拍 | `build-and-test.md` § 正确性验证 |
| 找哪个文件负责什么 | `build-and-test.md` § 文件索引 |
| Qwen3-0.6B 性能基线数字 | `build-and-test.md` § 性能基线 |
| MNN vs MLX 差距 / 优化后更新对比 | `mlx-comparison.md` |

## 通用原则速览（细节见 `kernel-basics.md`）

1. **shader 是嵌入的 C++ 字符串**（`R"metal(...)metal";` 在 `*.hpp` 里），不是独立 `.metal` 文件。改完直接 make。
2. **dispatcher 要先摸清**：一个 op 常有多条 kernel，扩之前先决定支持哪几条 + 让其他路径显式 fallback。
3. **Apple GPU ≠ Android**：M3/M4 数字不代表 iPhone A 系列，更不能推 Vulkan/OpenCL。
4. **正确性 oracle 先于性能**：CPU / temperature=0 greedy 对拍前 N token 是黄金标准。
5. **常见坑**：宏 alias 让 `#ifdef` 多分支同时为真、dispatcher 漏路径（lm_head g16）、weight tile byte 顺序反了、getDequantScale coef 双折叠。任何数值 bug 优先查这几条。

## 相关 Skills

- `skills/bugfix/` — 内存别名 / 生命周期错误排查（Metal 后端也共用同一套方法论）
- `skills/opencl-optimize/`、`skills/vulkan-optimize/`、`skills/arm-cpu-optimize/` — 其他后端的相似技巧
- `skills/support-new-llm/` — 新增 LLM 模型的完整流程
- `skills/test-ci/` — 单测 / 回归测试
