---
name: metal-optimize
description: MNN Metal 后端 op/kernel 优化与扩展入口。索引三份 sub-doc：kernel 修改方法论（新加 op / quant bit / dispatcher / weight pack）、LLM decode/prefill 端到端优化案例（11 步）、构建-测试-性能基线。根据当前任务选择性阅读对应 sub-doc。
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
| **[`llm-optimizations.md`](./llm-optimizations.md)** | 做 LLM decode / prefill 端到端性能优化，或参考具体优化的实现细节 | 优化总纲 + 路线图，11 个具体 Step（P0 profiling / P1 Q4 GEMV deferred dequant / 双 SG + ushort4 / prefill flash-attn / P2 pre-scaling / GQA fused attn / RMSNorm / Gate-Up fusion / QKV fusion / P3 RoPE fusion / RemoveDeadShapeOp / LLM 导出兼容性修复）。含性能数据、参数调优实验、避坑记录 |
| **[`build-and-test.md`](./build-and-test.md)** | 改完代码要 build / 跑测试 / 对拍 / 查文件索引 | cmake 编译命令、模型导出命令、性能测试命令、CPU/Metal 对拍与 FA A/B 对拍、性能基线数据、全文件索引 |

## 快速任务→sub-doc 索引

| 想做的事 | 优先读 |
|---|---|
| 新加 quant bit / 新 op / 改 dispatcher 路径 | `kernel-basics.md`（陷阱 A/B/E、packed weight 设计、Shader 修改流程） |
| 想知道 Metal 的坑（`#ifdef` alias / weight byte order / getDequantScale coef）| `kernel-basics.md` § 通用陷阱 |
| Apple GPU 优化杠杆选择（sg_matrix / sg_reduce / g4mN）| `kernel-basics.md` § Apple GPU 性能观察 |
| Metal LLM 端到端 profile 定位瓶颈 | `llm-optimizations.md` Step 1 |
| GEMV 优化（Q4 deferred dequant / 双 SG / pre-scaling）| `llm-optimizations.md` Step 2–4 |
| Fused attention / GQA 扩展 | `llm-optimizations.md` Step 5 |
| Fused prefill flash-attention（长 prompt、内存受限、KV int8）| `llm-optimizations.md` Step 11 |
| Gate/Up 或 QKV projection fusion（减少 dispatch）| `llm-optimizations.md` Step 8/9 |
| RoPE fusion / RemoveDeadShapeOp | `llm-optimizations.md` Step 7 |
| LLM 导出 RoPE unsqueeze / head_dim 匹配问题 | `llm-optimizations.md` Step 10 |
| cmake 编译选项 / 模型导出命令 | `build-and-test.md` § 编译 / 模型导出 |
| CPU vs Metal 对拍 / FA A/B 对拍 | `build-and-test.md` § 正确性验证 |
| 找哪个文件负责什么 | `build-and-test.md` § 文件索引 |
| Qwen3-0.6B 性能基线数字 | `build-and-test.md` § 性能基线 |

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
