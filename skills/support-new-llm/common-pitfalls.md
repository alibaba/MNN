# 常见陷阱

> 本文档汇总了在支持新 LLM 模型过程中反复出现的常见问题和解决方案。建议在开始新模型适配前浏览一遍。

---

## 1. RoPE 旋转方式（half-half vs interleaved）

### 问题描述

不同模型的 `rotate_half` 实现方式不同，旋转方式错误会导致 step3 layer0 检查点 diff > 0.01，且难以从表面现象判断根因。

### 判断方法

搜索模型源码中 `rotate_half` 函数的实现：

- **half-half**（标准，大多数模型）：`x[..., : x.shape[-1] // 2]`（前半后半分割）
- **interleaved**（交错，如 ernie4_5, glm_ocr）：`x[..., 0::2]`（奇偶分割）

另一个线索：`apply_rotary_pos_emb` 中使用 `repeat_interleave` 通常意味着交错模式。

### 解决方案

在 `model_mapper.py` 的映射中或 `transformers.py` 的 Rotary 实现中，确保使用正确的旋转方式。MNN 的 `Rotary` 类需要根据模型选择对应的实现。

---

## 2. Vision embed_ dtype 级联问题

### 问题描述

在多模态模型的 `embed()` 方法中，如果对视觉嵌入调用 `.float()` 进行 dtype 转换，可能会级联影响到共享的 embedding 层（`embed_`），导致后续文本推理也受到影响。

### 解决方案

- 在替换嵌入时使用 `.type(input_embeds.dtype)` 而不是 `.float()`
- 确保 `embed_()` 的输出 dtype 不被意外修改

---

## 3. Jinja 模板兼容性限制

### 问题描述

HuggingFace 模型的 chat template 可能使用了 MNN minja parser 不支持的高级 Jinja 特性，导致 C++ 推理时出现 `stof` 异常或死循环。

### 排查方法

1. 检查模型 `tokenizer_config.json` 中的 `chat_template` 字段
2. 查看是否包含复杂的条件逻辑、过滤器或自定义函数

### 解决方案

在 `llmexport.py` 中为该模型覆盖一个简化的 Jinja 模板，仅保留基本的 role/content 拼接逻辑。

---

## 4. Stop Token 配置

### 问题描述

C++ 推理输出不停止（无限重复某个 token 序列），通常是因为模型未生成标准 EOS token，且缺少额外的 stop token 配置。

### 解决方案

在 `tokenizer.py` 中 `MNNTokenizer.__init__` 方法里，为该模型添加额外的 stop token：

```python
if model_type == 'glm_ocr':
    user_ids = self.tokenizer.encode('<|user|>', add_special_tokens=False)
    if len(user_ids) == 1:
        self.stop_ids.append(user_ids[0])
```

常见的额外 stop token 包括：`<|user|>`、`<|im_end|>`、`<|endoftext|>` 等角色标记。

---

## 5. 多模态模型加载类注册

### 问题描述

多模态模型通常不能用 `AutoModelForCausalLM` 加载，需要使用对应的 `XxxForConditionalGeneration` 类。

### 解决方案

在 `model.py` 的 `MODEL_CLASS_MAPPING` 中添加映射：

```python
MODEL_CLASS_MAPPING = {
    'new_model_type': 'NewModelForConditionalGeneration',
}
```

---

## 6. Decoder 残差模式（Standard / Gemma2 / Phi / MiniCPM）

### 问题描述

不同模型的 Decoder 层使用不同的残差连接模式，这影响了 LayerNorm 的数量和映射方式。

### 残差模式速查表

| 模式 | Decoder 中 LayerNorm 数量 | 典型模型 | 需要额外映射字段 |
|------|-------------------------|---------|----------------|
| **Standard** | 2 (`input_layernorm` + `post_attention_layernorm`) | Llama, Qwen2, Qwen3 | 无 |
| **Gemma2 (4-norm)** | 4 (input + post_self_attn + post_attn + post_mlp) | Gemma2, glm_ocr | `pre_feedforward_layernorm`, `post_feedforward_layernorm` |
| **Phi (并行)** | 各异 | Phi | 特殊分支 |
| **MiniCPM (缩放)** | 2 + scale_depth | MiniCPM | config 中添加 `scale_depth` |

### Gemma2 模式映射注意事项

Gemma2 风格的 4-norm 模式中，MNN 统一键名与 HF 实际属性名的对应关系可能反直觉：

```python
new_decoder = {
    'self_attn': 'self_attn',
    'mlp': 'mlp',
    'input_layernorm': 'input_layernorm',
    'post_attention_layernorm': 'post_self_attn_layernorm',       # MNN的post_attn → HF的post_self_attn
    'pre_feedforward_layernorm': 'post_attention_layernorm',      # MNN的pre_ff → HF的post_attn
    'post_feedforward_layernorm': 'post_mlp_layernorm'            # MNN的post_ff → HF的post_mlp
}
```

---

## 7. config.py 新字段注册

### 问题描述

模型引入了 `LlmConfig.__init__` 中未定义的配置字段（如 LFM2 的 `conv_L_cache`）。虽然 `ModelMapper.do_map` 能通过 `setattr` 动态设置，但缺少默认值会导致其他模型访问该字段时报 `AttributeError`。

### 判断方法

检查 config 映射中的目标字段是否已在 `LlmConfig.__init__` 中定义：

```python
# 已有的字段（无需添加）：
# hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads,
# head_dim, rope_theta, rope_ratio, sliding_window, layer_types,
# attention_type, tie_word_embeddings, conv_L_cache
```

### 解决方案

在 `config.py` 的 `LlmConfig.__init__` 中添加带默认值的新字段：

```python
self.new_field = kwargs.pop("new_field", default_value)
```

> 默认值应该是"无效"值（如 `0`、`None`、`[]`），使得不具备该字段的模型行为不变。

---

## 8. 混合架构的 layer_types 映射

### 问题描述

混合架构模型（如 LFM2）的 `layer_types` 包含不同类型的层（如 `["conv", "conv", "full_attention", ...]`）。同一个 decoder 映射需要同时包含 `self_attn`（用于 attention 层）和 `linear_attn`（用于 conv 层），`ModelMapper.do_map` 会将不存在的属性设为 `None`。

### 关键机制

```python
# decoder 映射同时包含两种层类型的入口：
decoder = {
    'self_attn': 'self_attn',     # attention 层有此属性，conv 层为 None
    'linear_attn': 'conv',         # conv 层有此属性，attention 层为 None
    'mlp': 'feed_forward',
    # ...
}
```

`Decoder.__init__` 中的判断逻辑会自动处理：
- `self_attn is not None` → 创建标准 `Attention`
- `linear_attn is not None` → 通过 `create_linear_attention` 工厂创建对应的变体

### 注意事项

- 两个属性是互斥的（一层只会有一个非 None），不会冲突
- `Decoder.forward` 通过 `self.layer_type`（`'full_attention'` 或 `'linear_attention'`）决定 forward 路径
- `linear_attention` 类型的层不使用 `rotary_pos_emb` 和 `attention_mask`（跳过传入）

---

## 9. MoE 模型支持要点

> **Tier 3 并不简单。** MoE 模型涉及 expert 存储拆分、routing 算法差异、dense/MoE 层混合等问题，远比"添加 mlp 映射"复杂。

### 9.1 MoE 整体架构

MoE 模型在 MNN 中的导出和执行涉及以下组件：

```
Python 导出侧                          C++ 推理侧
┌─────────────────────┐               ┌─────────────────────┐
│ transformers.py     │               │ MoEModule.cpp       │
│   Mlp.forward():    │               │   onForward():      │
│     gate(x)→routing │    MoE op     │     decode: 逐expert│
│     topk→gather→    │ ────────────> │     prefill: 按token │
│     normalize       │  (custom op)  │     分发到expert    │
│     custom_moe(x,   │               │                     │
│       rw, experts)  │               │   mExperts[]:       │
│                     │               │     每个expert是独立│
│ custom_op.py        │               │     的子Module      │
│   MoEOp / MoEModule │               │                     │
│                     │               │   compute submodule:│
│ mnn_converter.py    │               │     weighted sum    │
│   expert 权重拆分    │               └─────────────────────┘
└─────────────────────┘
```

### 9.2 Routing 算法类型

不同模型使用不同的 routing 算法，对应 `transformers.py` 中 `Mlp.forward()` 的 `moe_type` 分支：

| moe_type | Routing 算法 | 典型模型 | 关键特征 |
|----------|-------------|---------|---------|
| `default` | softmax → topk → normalize | Mixtral, Qwen3-MoE | `F.softmax(logits) → topk` |
| `gpt_oss` | topk → softmax | GPT-OSS | `topk(logits) → F.softmax` |
| `lfm2_moe` | sigmoid + bias → topk → gather → normalize | LFM2-MoE | `sigmoid(logits) + expert_bias → topk → gather → normalize` |

**关键区别**：`lfm2_moe` 使用 **sigmoid**（而非 softmax），并且有 `expert_bias`、`norm_topk_prob`、`routed_scaling_factor` 等额外参数。

### 9.3 Expert 权重存储

HF 模型中 expert 权重通常以 3D tensor 存储（如 `gate_up_proj [num_experts, 2*intermediate, hidden]`），但 MNN 需要拆分为独立的 expert 子图。

**拆分在 `mnn_converter.py` 中完成**：`convert_expert()` 方法将 3D expert 权重沿 axis=0 切片，每个 expert 导出为独立的 subgraph（命名格式 `/expert/{layer_id}_{expert_id}`）。

**需要在 `model_mapper.py` 中添加 expert 映射**：

```python
mlp = {
    'gate': 'gate',                    # routing gate linear
    'experts': 'experts',              # expert 集合
}
expert = {
    'gate_up_proj': 'gate_up_proj',    # 或按模型实际命名
    'down_proj': 'down_proj',
}
```

**重要**：如果 HF 模型的 expert 使用独立的 `gate_proj` 和 `up_proj`（而非 fused `gate_up_proj`），需要使用 `Qwen3Expert` 类在 `transformers.py` 中将它们 concat 为 `gate_up_proj`。检查 HF 模型源码中 expert 的实际属性名。

### 9.4 Dense 层与 MoE 层混合

部分 MoE 模型有 `num_dense_layers` 配置（如 LFM2-MoE 前 2 层使用 dense MLP，后 22 层使用 MoE）。这需要在 `Decoder.__init__` 和 `Mlp.__init__` 中根据 `layer_id` 判断该层是否为 MoE：

```python
# 在 Decoder.__init__ 中：
is_moe_layer = (layer_id >= config.num_dense_layers) if hasattr(config, 'num_dense_layers') else True

# Mlp 初始化时需要据此决定：
# - MoE 层：初始化 gate、experts、routing 参数
# - Dense 层：初始化标准的 gate_proj/up_proj/down_proj
```

**注意**：dense 层和 MoE 层的 MLP 子模块命名可能不同（如 LFM2-MoE 的 dense 层用 `w1/w3/w2`，MoE 层用 `gate_up_proj/down_proj`），需要在 mapper 中同时提供两种映射。

### 9.5 MoE 的 config 和 mapper 字段

MoE 模型通常需要以下额外配置字段：

```python
# config.py 中（需确保有默认值）：
self.num_experts = kwargs.pop("num_experts", 0)
self.num_experts_per_tok = kwargs.pop("num_experts_per_tok", 0)
self.num_dense_layers = kwargs.pop("num_dense_layers", 0)
self.norm_topk_prob = kwargs.pop("norm_topk_prob", False)
self.routed_scaling_factor = kwargs.pop("routed_scaling_factor", 1.0)

# model_mapper.py config 映射中：
config = {
    'num_experts': 'num_local_experts',        # 注意：HF 可能叫 num_local_experts
    'num_experts_per_tok': 'num_experts_per_tok',
    'num_dense_layers': 'num_dense_layers',
    # ...
}
```

---

## 10. FakeLinear 维度变换与 axis 参数

### 问题描述

这是一个**高危陷阱**，可能导致 C++ 推理输出完全错误（如全零、乱码或重复 token），且 Python `--test` 完全正常（因为 Python test path 不经过 FakeLinear 转换）。

### 根因

MNN 的 LLM 导出流程中，`torch.nn.Linear` 通过 FakeLinear 自定义算子导出，在 `mnn_converter.py` 中被替换为 Convolution。这个过程会**改变 tensor 维度**：

```
ONNX 图中 Linear 输出:     [seq_len, hidden_size]        (2D)
MNN 中 FakeLinear 转换后:   [1, seq_len, hidden_size]     (3D)

转换过程：
  pre_reshape  [seq, H] → [seq, H, 1, 1]
  pre_convert  NCHW → NC4HW4
  Convolution  [seq, H_in, 1, 1] → [seq, H_out, 1, 1]
  post_convert NC4HW4 → NCHW
  post_reshape [seq, H_out, 1, 1] → [1, seq, H_out]    ← 增加了 batch 维度！
```

**问题**：如果 Linear 输出后接了使用硬编码 axis 的 op（如 `torch.gather(x, dim=1, ...)`），ONNX 中 axis=1 指向 `hidden_size`，但 MNN 转换后 axis=1 变成了 `seq_len`。

### 实际案例

LFM2-MoE 的 routing 计算：

```python
# Python 中（2D tensor）：
router_logits = self.gate(hidden_states)        # [seq, 32]
routing_weights = router_logits.sigmoid()        # [seq, 32]
routing_weights = torch.gather(routing_weights, dim=1, index=selected_experts)  # axis=1 → 第32维 ✓

# MNN 转换后（3D tensor）：
# gate 输出经 FakeLinear 变为 [1, seq, 32]
# sigmoid 保持 [1, seq, 32]
# GatherElements axis=1 → 第seq维 ✗  （应该是 axis=2）
```

### 解决方案

**规则：在 `transformers.py` 的 ONNX 导出路径中，所有涉及 axis/dim 参数的 torch 操作应始终使用负数索引（`dim=-1`）而非正数硬编码。**

```python
# 错误 ✗（FakeLinear 增加维度后 axis 指向错误位置）：
torch.gather(x, dim=1, index=idx)
x.sum(dim=1, keepdim=True)

# 正确 ✓（负数索引不受维度增加影响）：
torch.gather(x, dim=-1, index=idx)
x.sum(dim=-1, keepdim=True)
```

MNN 的 GatherElements 几何实现正确处理了负数 axis（`if (axis < 0) axis = D + axis;`），TopKV2 也支持负数 axis。

### 排查方法

当 C++ 推理结果错误但 Python `--test` 正确时：

1. 用 `MNNDump2Json` 导出 MNN 模型图：`build/MNNDump2Json model.mnn model.json`
2. 搜索 `GatherElements`、`TopKV2`、`Reduction` 等使用 axis 的 op
3. 检查其 axis 输入 tensor 的值，确认在 3D shape 下仍指向正确维度
4. 对比 ONNX 图中对应 op 的 axis 值

---

## 11. C++ 推理结果错误的系统排查流程

### 问题描述

C++ 推理不崩溃，但输出无意义内容（乱码、重复 token、全同字符等），而 Python `--test` 输出正确。

### ⚠️ 关键原则：不要猜测是量化精度问题

**以下归因几乎总是错误的：** "int4/int8/fp16 量化导致精度不够所以输出错误"。实际经验表明，即使 0.5B 的小模型用 int4 量化也能正常工作。如果 C++ 输出和 Python 差异巨大（如完全说不出图片内容），**一定是实现细节没有对齐**，不是量化问题。

### 正确的排查流程

**必须按以下步骤逐一排查，不能跳步：**

```
步骤 1: 确认 Python 链路是否正确
   └─ 用 Python `--test` 或手动 forward 验证输出
   └─ 如果 Python 也不对，先修 Python 链路
   └─ ⚠️ Python 链路正确是所有后续排查的前提

步骤 2: 对比 Vision 模型的输入输出（多模态模型）
   └─ 把 Python 的 patches/position_ids dump 到文件
   └─ C++ 加载这些文件作为 vision encoder 输入
   └─ 对比 vision encoder 输出的 first few values 和 norm
   └─ 如果不一致 → 定位 patchify/resize 差异
   └─ 如果一致 → vision encoder 没问题，继续

步骤 3: 对比 LLM 的输入
   └─ Python dump 完整的 hidden_states（embedding merge 后）
   └─ C++ 加载 Python 的 hidden_states 直接传入 LLM
   └─ 如果 C++ 用 Python 输入仍然错误 → 问题在 ONNX 模型本身
   └─ 如果 C++ 用 Python 输入正确 → 问题在 embedding merge

步骤 4: 逐层对比
   └─ 导出只有 1 层的模型，对比 layer0 输出
   └─ 找到第一个 diff 显著的位置
```

### 多模态模型的特殊排查点

**当图片输入不被识别时：** 逐行阅读 HF 的 `XxxModel.forward()`，找出 vision tokens 在 embedding / attention mask / 其他 per-token 计算中是否有特殊预处理（如 token id 替换、特殊 mask 等），然后确认 MNN 侧是否完全对齐了这些处理。

---

## 12. VL 模型的 mapper 路径前缀

### 问题描述

VL（Vision-Language）模型的 `config.json` 通常是嵌套结构（`text_config` / `vision_config`），这导致 **config 映射和 model 映射都需要加前缀**。遗漏前缀会导致配置字段读不到（默认值 0/None）或权重加载不上（全零参数）。

### 两个前缀

| 映射类型 | 纯文本模型 | VL 模型 | 说明 |
|---------|-----------|---------|------|
| **config** | `'hidden_size': 'hidden_size'` | `'hidden_size': 'text_config.hidden_size'` | 文本配置嵌套在 `text_config` 下 |
| **model** | `'embed': 'model.embed_tokens'` | `'embed': 'model.language_model.embed_tokens'` | 文本模型嵌套在 `model.language_model` 下 |

### 确认方法

1. **config 前缀**：读模型 `config.json`，看 `hidden_size` 等字段在顶层还是嵌套在 `text_config` 中
2. **model 前缀**：用 `safetensors.safe_open` 列出权重 key 的前缀，如 `model.language_model.layers.0.self_attn.q_proj.weight` → blocks 应映射为 `model.language_model.layers`

### 常见 VL 模型路径模式

| 模型 | config 前缀 | embed 路径 | blocks 路径 | visual 路径 |
|------|-----------|-----------|------------|------------|
| gemma3 | `text_config.` | `language_model.model.embed_tokens` | `language_model.model.layers` | `vision_tower.vision_model` |
| lfm2_vl | `text_config.` | `model.language_model.embed_tokens` | `model.language_model.layers` | `model.vision_tower` |
| smolvlm | `text_config.` | `model.text_model.embed_tokens` | `model.text_model.layers` | `model.vision_model` |
| qwen2_vl | _(无前缀)_ | `model.embed_tokens` | `model.layers` | `model.visual` |

> **Qwen2-VL 是例外**：它的文本配置直接在顶层，不需要 `text_config.` 前缀。

### 易犯错误

- 从已有纯文本模型（如 `lfm2`）复制映射到 VL 变体（如 `lfm2_vl`）时，忘记给 config 加 `text_config.` 前缀
- 忘记给 model 路径加 `model.language_model.` 前缀，导致权重全部加载为 None
- decoder / attention / linear_attention 等子映射**不需要**加前缀（它们是相对于 block 的局部路径）

---

## 13. do_map 静默失败与 rope_theta 间接存储

`ModelMapper.do_map()` 在源属性不存在时**不会报错**，静默设为 `None`，post-processing 再用默认值覆盖。最常见的受害者是 **rope_theta**：部分模型（如 LFM2）将 `rope_theta` 存在 `rope_parameters` dict 中而非顶层，导致映射 `'rope_theta': 'rope_theta'` 静默失败，rope_theta 被错误回退为 10000。

**修复**：config 映射中同时添加 `'rope_parameters': 'rope_parameters'`。`Rotary.__init__` 已有代码从中提取 rope_theta。

**防御**：step2 测试中必须验证 `model.rotary.rope_theta` 与 HF config 一致。如果 step3 layer0 就出错但权重匹配，首先检查 rope_theta。

---

## 14. 非标准模型加载方式

### 问题描述

部分模型不使用标准的 HuggingFace `AutoModelForCausalLM` 加载方式，需要使用第三方包或自定义加载逻辑。

### 典型案例

| 模型 | 加载方式 | 包 |
|------|---------|-----|
| LFM2-Audio | `liquid_audio.LFM2AudioModel.from_pretrained()` | `liquid_audio` |
| FunAudioChat | `AutoModelForSeq2SeqLM` | `transformers` |

### 解决方案

在 `model.py` 的 `from_pretrained` 方法中添加特殊加载分支：

```python
elif model_type == 'lfm2_audio':
    from liquid_audio import LFM2AudioModel
    original_model = LFM2AudioModel.from_pretrained(
        Path(pretrained_model_name_or_path), device='cpu', dtype=torch.bfloat16
    )
```

### 注意事项

- 非标准加载的模型**权重路径可能不同**于标准 HF 模型，需要通过 `print(original_model)` 或 `state_dict().keys()` 确认实际路径
- 嵌套的 config 结构可能需要在 `config.py` 的 `from_pretrained` 中手动提取子配置
- 某些包的注意力实现默认使用 `flash_attention_2`，CPU 上需要手动切换为 `sdpa` 或 `eager`
