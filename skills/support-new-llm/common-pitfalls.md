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
