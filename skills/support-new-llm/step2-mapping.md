# 步骤 2：添加映射

> **目标**：在 `model_mapper.py` 中添加新模型的映射，使模型能正确加载。
>
> **前置条件**：步骤 1 已通过，5 个关键问题已有明确答案。

---

## 2.1 理解映射系统

MNN 使用 4 层映射将 HuggingFace 模型结构转换为统一接口：

| 映射键 | 作用 | 说明 |
|--------|------|------|
| `config` | HF config.json 字段 → LlmConfig 属性 | 把模型配置正确读入 |
| `model` | HF 模型权重路径 → LlmModel 属性 | 找到 embed/layers/norm/lm_head |
| `decoder` | HF Decoder 层子模块 → Decoder 属性 | 找到 attn/mlp/layernorm |
| `attention` | HF Attention 子模块 → Attention 属性 | 找到 q/k/v/o 投影层 |
| `mlp`（可选）| HF MoE 子模块 → Mlp 属性 | 仅 MoE 模型需要 |
| `linear_attention`（可选）| HF LinearAttn 子模块 → LinearAttention 属性 | 仅特殊架构需要 |

### default_map 的完整定义

```python
default_config = {
    'hidden_size': 'hidden_size',
    'head_dim': 'head_dim',
    'num_attention_heads': 'num_attention_heads',
    'num_hidden_layers': 'num_hidden_layers',
    'num_key_value_heads': 'num_key_value_heads',
    'rope_theta': 'rope_theta',
    'rope_scaling': 'rope_scaling',
    'max_position_embeddings': 'max_position_embeddings'
}
# ⚠️ 注意：如果模型的 rope_theta 不在顶层而是在 rope_parameters 中，
# 需要额外映射 'rope_parameters': 'rope_parameters'（参见 common-pitfalls.md 第 13 节）
default_model = {
    'lm': 'lm_head',
    'embed': 'model.embed_tokens',
    'blocks': 'model.layers',
    'final_layernorm': 'model.norm',
    'visual': 'visual'
}
default_decoder = {
    'self_attn': 'self_attn',
    'linear_attn': 'linear_attn',
    'mlp': 'mlp',
    'input_layernorm': 'input_layernorm',
    'post_attention_layernorm': 'post_attention_layernorm'
}
default_attention = {
    'qkv_proj': 'qkv_proj',
    'q_proj': 'q_proj',
    'k_proj': 'k_proj',
    'v_proj': 'v_proj',
    'o_proj': 'o_proj',
    'q_norm': 'q_norm',
    'k_norm': 'k_norm'
}
```

**映射规则**：左边是 MNN 统一的键名，右边是 HF 模型中实际的属性名。如果一致就直接复用 default；如果不同就自定义。

---

## 2.2 根据步骤 1 的结果选择修改方案

对照步骤 1 的 5 个问题答案，逐一检查：

### 检查 1：config 是否需要自定义？

**判断条件**：config.json 的字段是否直接在顶层？

- 字段直接在顶层（如 `hidden_size` → 直接读） → 使用 `self.default_config`
- 字段在 `text_config` 下（如 `text_config.hidden_size`） → **需要自定义 config**
- **⚠️ rope_theta 特殊检查**：确认 `rope_theta` 是否在顶层。如果不在顶层但存在 `rope_parameters` dict，必须额外映射 `'rope_parameters': 'rope_parameters'`（参见 `common-pitfalls.md` 第 13 节）

```python
# 自定义 config 示例：
new_config = {
    'hidden_size': 'text_config.hidden_size',
    'head_dim': 'text_config.head_dim',
    'num_attention_heads': 'text_config.num_attention_heads',
    'num_hidden_layers': 'text_config.num_hidden_layers',
    'num_key_value_heads': 'text_config.num_key_value_heads',
    'rope_theta': 'text_config.rope_theta',
    'rope_parameters': 'text_config.rope_parameters',  # ← 如果 rope_theta 在此 dict 中
    'rope_scaling': 'text_config.rope_scaling',
    'max_position_embeddings': 'text_config.max_position_embeddings'
}
```

### 检查 2：model 路径是否需要自定义？

**判断条件**：步骤 1 问题 1 的回答。

- 路径是标准的 `model.embed_tokens`, `model.layers`, `model.norm`, `lm_head` → 使用 `self.default_model`
- 路径不同 → **需要自定义 model**

```python
# 自定义 model 示例（嵌套在 language_model 下）：
new_model = {
    'lm': 'language_model.lm_head',
    'embed': 'language_model.model.embed_tokens',
    'blocks': 'language_model.model.layers',
    'final_layernorm': 'language_model.model.norm',
}
```

### 检查 3：attention 是否需要自定义？

**判断条件**：步骤 1 问题 2 的回答。

- 标准 `q_proj/k_proj/v_proj/o_proj` → 使用 `self.default_attention`
- Fused QKV（如 `c_attn`, `W_pack`） → 需要自定义，映射 `qkv_proj`
- O 投影名不同（如 `dense`, `c_proj`） → 需要自定义

```python
# 自定义 attention 示例（fused QKV）：
new_attention = {
    'qkv_proj': 'c_attn',      # fused QKV 层名
    'o_proj': 'c_proj'          # output 投影层名
}
```

### 检查 4：decoder 是否需要自定义？

**判断条件**：步骤 1 问题 3 和问题 4 的回答。

- 标准名称 → 使用 `self.default_decoder`
- 名称不同 → 需要自定义
- 有额外 LayerNorm → 需要自定义并添加额外字段
- 没有 `post_attention_layernorm` → 需要自定义（省略该字段）

先确定新模型属于哪种残差模式，参考 `common-pitfalls.md` 第 6 节的速查表和 Gemma2 映射示例。

### 检查 5：config.json 是否有非标准字段？

**判断条件**：config.json 中是否有 `LlmConfig.__init__` 尚未定义的字段，且该字段会通过 config 映射被使用？

`LlmConfig.__init__` 已定义的字段（有默认值，无需添加）：
```
hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads,
head_dim, rope_theta, rope_ratio, sliding_window, layer_types,
attention_type, tie_word_embeddings, conv_L_cache
```

- 所需字段已在上述列表中 → 无需修改
- 需要新字段（如某模型引入了新的配置参数） → **需要在 `config.py` 的 `LlmConfig.__init__` 中添加**

```python
# config.py 中添加新字段示例：
class LlmConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        # ... 已有字段 ...
        self.new_field = kwargs.pop("new_field", default_value)  # ← 添加
```

> **为什么需要这步？** `ModelMapper.do_map` 使用 `setattr` 设置值，即使不在 `__init__` 中定义也能工作。但添加默认值有两个好处：(1) 其他模型不会因缺少该属性而报 `AttributeError`；(2) 代码中可以用 `config.new_field` 直接访问而不需要 `hasattr` 保护。

### 检查 6：是否需要 mlp 映射？（MoE 模型）

**判断条件**：config.json 中是否有 `num_experts`。

- 没有 → 不需要 mlp 映射
- 有 → 需要添加 mlp 映射

```python
# MoE mlp 映射示例：
new_mlp = {
    'num_experts': 'num_experts',
    'top_k': 'top_k',
    'norm_topk_prob': 'norm_topk_prob',
    'gate': 'gate',
    'experts': 'experts'
}
# 如果有 shared_expert：
new_mlp['shared_expert'] = 'shared_expert'
new_mlp['shared_expert_gate'] = 'shared_expert_gate'
```

---

## 2.3 编写映射代码

在 `model_mapper.py` 中添加新模型的注册方法。

### 方案 A：完全匹配 default_map（Tier 1 最简单情况）

找到 `regist_llama` 方法，在其中追加一行：

```python
def regist_llama(self):
    llama_map = self.default_map
    self.regist('llama', llama_map)
    self.regist('qwen2', llama_map)
    # ... 其他已有模型
    self.regist('新的model_type', llama_map)  # ← 添加这一行
```

### 方案 B：需要自定义映射

创建新的注册方法，并在 `__init__` 中调用：

```python
# 1. 在 ModelMapper 类中添加新方法
def regist_new_model(self):
    # 按照 2.2 中的检查结果，组装映射
    new_map = {
        'config': self.default_config,      # 或自定义
        'model': self.default_model,        # 或自定义
        'decoder': self.default_decoder,    # 或自定义
        'attention': self.default_attention # 或自定义
    }
    self.regist('model_type值', new_map)

# 2. 在 __init__ 中调用这个方法
def __init__(self):
    # ... 已有的注册方法
    self.regist_new_model()  # ← 添加这一行
```

### 方案 C：多模态模型额外需要注册模型类

如果模型不能用 `AutoModelForCausalLM` 加载（多模态模型通常不行），需要在 `model.py` 的 `MODEL_CLASS_MAPPING` 中添加：

```python
# model.py 中的 get_model_class 方法
MODEL_CLASS_MAPPING = {
    # ... 已有映射
    'new_model_type': 'NewModelForConditionalGeneration',
}
```

---

## 步骤 2 测试标准

### 测试方法

执行以下命令测试模型是否能正确加载，**并验证关键 config 值**：

```bash
cd transformers/llm/export
python3 -c "
from utils.model import LlmModel
import argparse
args = argparse.Namespace(lora_path=None, lora_split=False, skip_weight=False, test=False, eagle_path=None)
model = LlmModel.from_pretrained('/path/to/model', args=args)
print('✅ 模型加载成功')
print(f'  hidden_size: {model.config.hidden_size}')
print(f'  num_layers: {model.config.num_hidden_layers}')
print(f'  num_heads: {model.config.num_attention_heads}')
print(f'  num_kv_heads: {model.config.num_key_value_heads}')
print(f'  head_dim: {model.config.head_dim}')
print(f'  blocks 数量: {len(model.blocks)}')
print(f'  embed 类型: {type(model.embed)}')
print(f'  lm 类型: {type(model.lm)}')

# ===== 关键：验证 config 值与 HF 原始 config 一致 =====
# rope_theta 是高频出错项，必须检查（参见 common-pitfalls.md 第 13 节）
print(f'  rope_theta (Rotary): {model.rotary.rope_theta}')

# 与原始 config 对比（手动确认与 config.json 中的值一致）
origin = model.config.origin_config
print(f'  [对比] origin rope_theta: {getattr(origin, \"rope_theta\", \"NOT_FOUND\")}')
if hasattr(origin, 'rope_parameters') and origin.rope_parameters:
    print(f'  [对比] origin rope_parameters: {origin.rope_parameters}')

# 验证 config 映射完整性：检查每个映射字段是否在源 config 中存在
print()
print('Config 映射检查:')
model_map = model.config.model_map
for dst, src in model_map.get('config', {}).items():
    val = origin
    for attr in src.split('.'):
        val = getattr(val, attr, None)
        if val is None:
            break
    status = '✅' if val is not None else '⚠️  None'
    print(f'  {status} {dst} <- {src} = {val if not isinstance(val, dict) else type(val).__name__}')
"
```

### 通过标准

- [ ] **不报 KeyError**：说明 config 映射正确
- [ ] **不报 AttributeError**：说明 model/decoder/attention 路径正确
- [ ] **hidden_size / num_layers / num_heads 值正确**：与 config.json 中一致
- [ ] **blocks 数量正确**：等于 num_hidden_layers
- [ ] **embed 和 lm 类型不是 None**
- [ ] **rope_theta 值正确**：与 HF config 中的值一致（**不是默认值 10000**，除非模型确实使用 10000）
- [ ] **Config 映射检查无 ⚠️ None 项**：所有映射的源字段都存在（如果有 None 项，需确认是否需要替换映射路径或添加间接映射如 `rope_parameters`）

### 常见错误与修复

| 错误 | 原因 | 修复 |
|------|------|------|
| `KeyError: 'hidden_size'` | config 字段在 text_config 下 | 自定义 config 映射，加上 `text_config.` 前缀 |
| `AttributeError: 'NoneType' has no attribute 'embed_tokens'` | model 路径错误 | 检查 embed/blocks/norm 的实际路径 |
| `blocks 数量为 0` | blocks 路径错误 | 检查 layers 的实际路径 |
| `embed 或 lm 为 None` | 路径不存在 | 检查实际的权重名称 |

### 失败处理

如果测试失败：
1. 阅读报错信息，定位是哪个映射出了问题
2. 使用工具查看 HF 模型的实际属性名
3. 修改映射后重新测试
4. **在修复问题之前，不要进入步骤 3**

---

## 下一步

**步骤 2 通过后，进入 `step3-test-python.md`（步骤 3：Python 推理测试）。**
