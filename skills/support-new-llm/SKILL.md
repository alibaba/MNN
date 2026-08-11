---
name: support-new-llm
description: 为 MNN 框架添加新的 LLM 模型支持。支持从 HuggingFace/ModelScope 下载模型，分析架构，添加映射，Hook 对齐测试，导出 MNN 模型。采用 TDD 模式，分 6 步执行，每步有独立测试标准。
---

# MNN LLM 新模型支持 SKILL

> **触发条件**：当用户请求支持/添加/适配一个新的 LLM 模型时触发。常见表述包括："支持xxx模型"、"添加xxx模型支持"、"适配xxx"、"导出xxx模型"等。

## 概述

本 SKILL 指导 AI Agent 为 MNN 框架添加新的 LLM 模型支持。整个流程采用 **TDD（测试驱动）模式**，每一步都有明确的测试标准，**必须通过当前步骤的测试后才能进入下一步**。

### 核心思路

MNN 的模型导出本质上是**对照 HuggingFace transformers 库中原始模型的实现代码**，将其计算逻辑映射到 MNN 的统一框架中。核心步骤是：

1. 读懂 HF 模型的 `config.json` 和 `modeling_*.py`
2. 在 `model_mapper.py` 中注册字段映射
3. 用 Python `--test` 验证映射正确性
4. 导出 MNN 模型并用 C++ 引擎验证

### 注意事项

> **🚨 严禁将输出错误归因于"量化精度不够"**：4bit 量化的 0.5B 小模型都能正确输出。如果 C++ 输出完全不对（如图片识别不出、输出乱码），**一定是实现细节没有与 HF 对齐**，必须逐步 dump 数据对比定位，不要靠猜。

> **🚨 测试标准要有定力**：每步的通过标准是明确的（如"C++ 能正确描述图片内容"），不能因为"差不多能跑"就跳过。"能感知到一些信号但描述不准确"不等于通过，必须达到与 HF 模型相当的输出质量才算完成。

> **严禁访问以下目录**：`schema/private/` 和 `source/internal/`，包含内部私有代码，**不得读取、修改或引用**。

> **禁止猜测**：如果不确定某个字段名或路径，必须通过工具读取实际文件确认。

---

## 核心文件清单

| 文件路径 | 作用 | 修改频率 |
|---------|------|---------:|
| `transformers/llm/export/utils/model_mapper.py` | 模型字段映射 | **几乎每个新模型** |
| `transformers/llm/export/utils/model.py` | 统一模型类 `LlmModel` | 偶尔 |
| `transformers/llm/export/utils/transformers.py` | Attention/Decoder/Rotary 等组件 | 新架构时 |
| `transformers/llm/export/utils/config.py` | 模型配置类 `LlmConfig` | 偶尔 |
| `transformers/llm/export/utils/vision.py` | Vision Encoder 实现 | 视觉模型 |
| `transformers/llm/export/utils/audio.py` | Audio Encoder 实现 | 音频模型 |
| `transformers/llm/export/utils/custom_op.py` | 自定义算子导出 | 新算子时 |
| `transformers/llm/export/llmexport.py` | 导出主流程入口 | 偶尔 |

---

## 分步流程总览

整个流程分为 **6 个步骤**，每个步骤都有独立的文档和测试标准：

```
┌──────────────────────────────────────────────────────────┐
│  步骤 1: 下载、理解与测试模型 (step1-analyze.md)           │
│  输入: 模型链接(HF/ModelScope)或本地路径                   │
│  输出: 模型下载到本地 + transformers 推理成功               │
│       + 架构分析完成 + Tier 判定                           │
│  测试: test_origin.py 输出正确 + 5 个架构问题已回答        │
├──────────────────────────────────────────────────────────┤
│  步骤 2: 添加映射 (step2-mapping.md)                      │
│  输入: 步骤1的差异清单                                     │
│  输出: model_mapper.py 中的新映射                          │
│  测试: LlmModel.from_pretrained 加载不报错                 │
├──────────────────────────────────────────────────────────┤
│  步骤 3: Hook 对齐测试 (step3-test-python.md)             │
│  输入: 步骤1的原始模型结果 + 步骤2的 LlmModel              │
│  输出: Hook 中间结果数值对齐                               │
│  测试: 5 个检查点(embed/layer0/lastlayer/norm/logits)一致  │
├──────────────────────────────────────────────────────────┤
│  步骤 4: 导出与 C++ 测试 (step4-export.md)                │
│  输入: 步骤3通过                                           │
│  输出: MNN 模型文件                                        │
│  测试: C++ llm_demo 输出正确                               │
├──────────────────────────────────────────────────────────┤
│  步骤 5: 视觉/音频支持 (step5-multimodal.md)              │
│  输入: 仅 Tier 4/5/6 需要                                  │
│  输出: vision.py 或 audio.py 中的新子类                    │
│  测试: 多模态推理测试通过                                   │
├──────────────────────────────────────────────────────────┤
│  步骤 6: 特殊架构支持 (step6-new-architecture.md)         │
│  输入: 仅 Tier 6 需要                                      │
│  输出: 新算子 + C++ 实现                                   │
│  测试: 全链路测试通过                                       │
└──────────────────────────────────────────────────────────┘
```

### 步骤选择指南

**不是所有步骤都需要执行。** 根据步骤 1 判定的 Tier，选择需要执行的步骤：

| Tier | 需要执行的步骤 | 说明 |
|------|--------------|------|
| Tier 1 (纯文本 Llama-like) | 1 → 2 → 3 → 4 | 最简单，仅需映射 |
| Tier 2 (轻微架构差异) | 1 → 2 → 3 → 4 | 可能需要修改 transformers.py |
| Tier 3 (MoE 模型) | 1 → 2 → 3 → 4 | 需要 mlp/expert 映射 + routing 实现，参见 `common-pitfalls.md` 第 9 节 |
| Tier 4 (音频模型) | 1 → 2 → 3 → 5 → 4 | 需要 audio.py |
| Tier 5 (视觉模型) | 1 → 2 → 3 → 5 → 4 | 需要 vision.py |
| Tier 6 (全新架构) | 1 → 2 → 6 → 3 → 4 | 需要新算子（如叠加 Tier 4/5 则加入 step5） |

---

## Tier 判定速查

根据 `config.json` 中的字段快速判定 Tier：

```
config.json 中是否有 num_experts 或 num_local_experts?
├─ 是 → Tier 3 (MoE)
└─ 否 → 继续

config.json 中是否有 vision_config?
├─ 是 → Tier 5 (视觉)
└─ 否 → 继续

config.json 中是否有 audio_config?
├─ 是 → Tier 4 (音频)
└─ 否 → 继续

layer_types 中是否有非 Attention 层（如 conv / mamba / rwkv）?
├─ 是 → Tier 6 (混合架构，如 lfm2 的 conv + full_attention)
└─ 否 → 继续

modeling_*.py 中是否有全新的 Attention 类型（非标准 SDPA）?
├─ 是 → Tier 6 (新架构，如 qwen3_5 的 gated_delta_rule)
└─ 否 → 继续

是否有额外 LayerNorm / scale_depth / scale_emb / 没有 post_attention_layernorm?
├─ 是 → Tier 2 (轻微差异)
└─ 否 → Tier 1 (标准 Llama-like)
```

> **注意**：Tier 可以叠加。例如 MoE + 视觉 = Tier 3+5，需要同时执行对应步骤。

### Tier 叠加时的执行顺序

当模型跨多个 Tier 时，按以下原则确定步骤顺序：

| 叠加 | 执行顺序 | 说明 |
|------|---------|------|
| Tier 3+6 (MoE + 新架构) | 1 → 2 → 6 → 3 → 4 | 先支持新架构（如 short_conv），再加 MoE routing |
| Tier 3+5 (MoE + 视觉) | 1 → 2 → 3 → 5 → 4 | MoE 是 text-only，视觉单独加 |
| Tier 5+6 (视觉 + 新架构) | 1 → 2 → 6 → 5 → 3 → 4 | 先支持新架构，再加视觉 |

**原则**：Tier 6（新架构）最先实现，因为它影响基础层结构；Tier 3（MoE）次之，因为它在 Tier 6 层结构之上；Tier 4/5（多模态）最后，因为它与层结构独立。

---

## 已支持模型速查表

收到新模型请求时，先查 `config.json` 的 `model_type`，在下表中搜索。如果已存在则**无需修改**。

| model_type | Tier | 类型 |
|-----------|------|------|
| `llama`, `qwen2`, `internlm`, `mobilellm` | 1 | 文本 |
| `baichuan` | 1 | 文本 (fused QKV) |
| `qwen` | 1 | 文本 (Qwen1) |
| `qwen3` | 1 | 文本 (q/k norm) |
| `chatglm`, `chatglm2` | 1 | 文本 (特殊 RoPE) |
| `phi-msft`, `phi` | 1 | 文本 |
| `gemma2` | 2 | 文本 (额外 LayerNorm) |
| `gemma3_text` | 2 | 文本 |
| `minicpm` | 2 | 文本 (scale_depth) |
| `gpt_oss` | 3 | MoE (topk → softmax) |
| `qwen3_moe` | 3 | MoE (softmax → topk) |
| `lfm2_moe` | 3+6 | MoE (sigmoid routing) + 混合架构 (short_conv) |
| `qwen2_audio` | 4 | 音频 |
| `funaudiochat` | 4 | 音频 |
| `lfm2_audio` | 4+6 | 音频 (FastConformer + MLP adapter) + 混合架构 |
| `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl` | 5 | 视觉 |
| `internvl_chat` | 5 | 视觉 |
| `gemma3` | 5 | 视觉 |
| `gemma4` | 5 | 视觉 (PLE + KV共享 + 双Rotary + patch-based vision) |
| `glm_ocr` | 5 | 视觉 (Gemma2残差 + interleaved M-RoPE) |
| `smolvlm` | 5 | 视觉 (SigLIP + Perceiver) |
| `idefics3` | 5 | 视觉 (SigLIP + Perceiver) |
| `lfm2_vl` | 5+6 | 视觉 (SigLIP2 NaFlex + pixel_unshuffle) + 混合架构 |
| `lfm2` | 6 | 混合架构 (short_conv + full_attention) |
| `qwen3_5` | 6 | 视觉+LinearAttn (gated_delta_rule) |

---

## 常见陷阱

**在开始之前，建议先浏览 `common-pitfalls.md`**，了解已知的常见问题和解决方案（RoPE 变体、dtype 级联、Jinja 限制、stop token、残差模式、MoE 支持要点、FakeLinear axis 陷阱、**do_map 静默失败与 rope_theta 间接存储**、非标准模型加载等）。

---

## 开始执行

**现在请打开 `skills/support-new-llm/step1-analyze.md`，开始步骤 1。**
