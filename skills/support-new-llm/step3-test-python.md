# 步骤 3：Hook 对齐测试

> **目标**：通过 hook 对比 transformers 原始模型和 llmexport 模型的中间结果，确保映射后的推理逻辑完全正确。
>
> **前置条件**：步骤 2 已通过（LlmModel 能正确加载），步骤 1 的 `test_origin.py` 已成功运行。

---

## 核心思路

仅看最终输出文本是否"合理"是不够的。本步骤通过 **hook 机制**在两个模型的关键位置截取中间结果，逐层对比，精确定位映射或实现中的错误。

**对比的两套模型**：
1. **原始 transformers 模型**：步骤 1 中加载的 `AutoModelForCausalLM`（标准答案）
2. **MNN LlmModel**：步骤 2 中映射转换后的 `LlmModel`（需要验证）

**对比的关键检查点**：
1. **Embedding 输出**：验证 embed_tokens 路径正确
2. **第 0 层 Decoder 输出**：验证 Attention + MLP + 残差正确
3. **最后一层 Decoder 输出**：验证所有层都正确
4. **Final LayerNorm 输出**：验证 norm 路径正确
5. **Logits / Top-1 Token**：验证 lm_head 正确

**对比两个阶段**：
- **Prefill**：输入整个 prompt（多 token），对应第一次 forward
- **First Decode**：输入上一步生成的 token（1 token），对应第二次 forward

---

## 3.1 编写原始模型 Hook 脚本

创建 `test_origin_hook.py`：

```python
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "模型路径"  # ← 替换为实际路径
prompt = "你好"
output_file = "/tmp/origin_hook_results.json"

# ===== 1. 加载模型 =====
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
)
model.eval()

# ===== 2. 构造输入 =====
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
try:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
except Exception:
    text = prompt

input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
print(f"输入 token 数: {input_ids.shape[1]}")
print(f"输入 token ids: {input_ids[0].tolist()}")

# ===== 3. 注册 Hook =====
hook_results = {"prefill": {}, "decode": {}}
hooks = []
phase = "prefill"  # 用于标记当前阶段

# 找到模型内部结构路径
# 注意: 不同模型路径不同，需要根据步骤 1 问题 1 的答案调整
# 以下以标准 Llama-like 结构为例:
#   model.model.embed_tokens
#   model.model.layers[i]
#   model.model.norm
#   model.lm_head

# Hook: embedding 输出
def hook_embed(module, input, output):
    data = output.detach().float()
    hook_results[phase]["embed_output_shape"] = list(data.shape)
    hook_results[phase]["embed_output_first5"] = data[0, 0, :5].tolist()
    hook_results[phase]["embed_output_last5"] = data[0, 0, -5:].tolist()

# Hook: decoder layer 0 输出
def hook_layer0(module, input, output):
    # 有些模型 decoder 层的输出是 tuple，取第一个
    data = output[0].detach().float() if isinstance(output, tuple) else output.detach().float()
    hook_results[phase]["layer0_output_shape"] = list(data.shape)
    hook_results[phase]["layer0_output_first5"] = data[0, -1, :5].tolist()
    hook_results[phase]["layer0_output_last5"] = data[0, -1, -5:].tolist()

# Hook: 最后一个 decoder layer 输出
def hook_last_layer(module, input, output):
    data = output[0].detach().float() if isinstance(output, tuple) else output.detach().float()
    hook_results[phase]["last_layer_output_first5"] = data[0, -1, :5].tolist()
    hook_results[phase]["last_layer_output_last5"] = data[0, -1, -5:].tolist()

# Hook: final layernorm 输出
def hook_final_norm(module, input, output):
    data = output.detach().float()
    hook_results[phase]["final_norm_output_first5"] = data[0, -1, :5].tolist()
    hook_results[phase]["final_norm_output_last5"] = data[0, -1, -5:].tolist()

# ===== 注册 hooks =====
# ⚠️  以下路径需要根据步骤 1 问题 1 的答案修改 ⚠️
# 标准 Llama-like 路径：
num_layers = model.config.num_hidden_layers
hooks.append(model.model.embed_tokens.register_forward_hook(hook_embed))
hooks.append(model.model.layers[0].register_forward_hook(hook_layer0))
hooks.append(model.model.layers[num_layers - 1].register_forward_hook(hook_last_layer))
hooks.append(model.model.norm.register_forward_hook(hook_final_norm))

# 如果是嵌套路径（如 language_model.model.xxx），需要修改为：
# hooks.append(model.language_model.model.embed_tokens.register_forward_hook(hook_embed))
# hooks.append(model.language_model.model.layers[0].register_forward_hook(hook_layer0))
# hooks.append(model.language_model.model.layers[-1].register_forward_hook(hook_last_layer))
# hooks.append(model.language_model.model.norm.register_forward_hook(hook_final_norm))

# ===== 4. Prefill 阶段 =====
phase = "prefill"
with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    logits = outputs.logits
    past_key_values = outputs.past_key_values

prefill_token = torch.argmax(logits[0, -1, :]).item()
hook_results["prefill"]["logits_last_pos_first5"] = logits[0, -1, :5].tolist()
hook_results["prefill"]["logits_last_pos_last5"] = logits[0, -1, -5:].tolist()
hook_results["prefill"]["top1_token_id"] = prefill_token
hook_results["prefill"]["top1_token_str"] = tokenizer.decode([prefill_token])
print(f"Prefill top1 token: {prefill_token} = '{tokenizer.decode([prefill_token])}'")

# ===== 5. First Decode 阶段 =====
phase = "decode"
decode_input_ids = torch.tensor([[prefill_token]])
with torch.no_grad():
    outputs = model(decode_input_ids, past_key_values=past_key_values, use_cache=True)
    logits = outputs.logits

decode_token = torch.argmax(logits[0, -1, :]).item()
hook_results["decode"]["logits_last_pos_first5"] = logits[0, -1, :5].tolist()
hook_results["decode"]["logits_last_pos_last5"] = logits[0, -1, -5:].tolist()
hook_results["decode"]["top1_token_id"] = decode_token
hook_results["decode"]["top1_token_str"] = tokenizer.decode([decode_token])
print(f"Decode top1 token: {decode_token} = '{tokenizer.decode([decode_token])}'")

# ===== 6. 保存结果 =====
for h in hooks:
    h.remove()

with open(output_file, "w") as f:
    json.dump(hook_results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到 {output_file}")
```

### 运行脚本

```bash
python3 test_origin_hook.py
```

### 脚本测试标准

- [ ] 脚本无报错运行完成
- [ ] `/tmp/origin_hook_results.json` 文件已生成
- [ ] `prefill.top1_token_id` 和 `decode.top1_token_id` 有合理的值

---

## 3.2 编写 LlmModel Hook 脚本

创建 `test_llmexport_hook.py`：

```python
import sys
sys.path.insert(0, '.')  # 确保能导入 utils
import torch
import json

# ===== 1. 加载 LlmModel =====
from utils.model import LlmModel
model_path = "模型路径"  # ← 替换为实际路径
prompt = "你好"
origin_file = "/tmp/origin_hook_results.json"

model = LlmModel.from_pretrained(model_path)
model.args = type('Args', (), {'test': True, 'eagle_path': None})()
model.eval()

# 加载原始模型的 hook 结果（标准答案）
with open(origin_file, "r") as f:
    origin_results = json.load(f)

# ===== 2. 构造输入（与原始模型相同） =====
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
try:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
except Exception:
    text = prompt

input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
print(f"输入 token 数: {input_ids.shape[1]}")

# ===== 3. 注册 Hook =====
llm_results = {"prefill": {}, "decode": {}}
hooks = []
phase = "prefill"

def hook_embed(module, input, output):
    data = output.detach().float()
    llm_results[phase]["embed_output_shape"] = list(data.shape)
    llm_results[phase]["embed_output_first5"] = data[0, 0, :5].tolist()
    llm_results[phase]["embed_output_last5"] = data[0, 0, -5:].tolist()

def hook_layer0(module, input, output):
    data = output.detach().float() if not isinstance(output, tuple) else output[0].detach().float()
    llm_results[phase]["layer0_output_shape"] = list(data.shape)
    llm_results[phase]["layer0_output_first5"] = data[0, -1, :5].tolist()
    llm_results[phase]["layer0_output_last5"] = data[0, -1, -5:].tolist()

def hook_last_layer(module, input, output):
    data = output.detach().float() if not isinstance(output, tuple) else output[0].detach().float()
    llm_results[phase]["last_layer_output_first5"] = data[0, -1, :5].tolist()
    llm_results[phase]["last_layer_output_last5"] = data[0, -1, -5:].tolist()

def hook_final_norm(module, input, output):
    data = output.detach().float()
    llm_results[phase]["final_norm_output_first5"] = data[0, -1, :5].tolist()
    llm_results[phase]["final_norm_output_last5"] = data[0, -1, -5:].tolist()

# LlmModel 的路径是固定的：
# model.embed       → Embedding
# model.blocks[0]   → 第 0 层 Decoder
# model.blocks[-1]  → 最后一层 Decoder
# model.final_layernorm → Final LayerNorm
hooks.append(model.embed.register_forward_hook(hook_embed))
hooks.append(model.blocks[0].register_forward_hook(hook_layer0))
hooks.append(model.blocks[-1].register_forward_hook(hook_last_layer))
hooks.append(model.final_layernorm.register_forward_hook(hook_final_norm))

# ===== 4. Prefill =====
phase = "prefill"
seq_len = input_ids.shape[1]

with torch.no_grad():
    attention_mask = model.get_attention_mask(seq_len, 0)
    position_ids = model.get_position_ids(seq_len, 0, input_ids)
    input_embeds = model.embedding(input_ids)
    logits, _, _ = model.forward(
        input_ids=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        logits_index=torch.tensor([-1], dtype=torch.int32)
    )

prefill_token = torch.argmax(logits[0, -1, :]).item()
llm_results["prefill"]["logits_last_pos_first5"] = logits[0, -1, :5].tolist()
llm_results["prefill"]["logits_last_pos_last5"] = logits[0, -1, -5:].tolist()
llm_results["prefill"]["top1_token_id"] = prefill_token
print(f"LlmModel Prefill top1 token: {prefill_token}")

# ===== 5. First Decode =====
phase = "decode"
decode_input_ids = torch.tensor([[prefill_token]])

with torch.no_grad():
    attention_mask = model.get_attention_mask(seq_len + 1, 1)
    position_ids = model.get_position_ids(seq_len + 1, 1, decode_input_ids)
    input_embeds = model.embedding(decode_input_ids)
    logits, _, _ = model.forward(
        input_ids=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        logits_index=torch.tensor([-1], dtype=torch.int32)
    )

decode_token = torch.argmax(logits[0, -1, :]).item()
llm_results["decode"]["logits_last_pos_first5"] = logits[0, -1, :5].tolist()
llm_results["decode"]["logits_last_pos_last5"] = logits[0, -1, -5:].tolist()
llm_results["decode"]["top1_token_id"] = decode_token
print(f"LlmModel Decode top1 token: {decode_token}")

# ===== 6. 对比结果 =====
for h in hooks:
    h.remove()

print("\n" + "=" * 60)
print("对比结果")
print("=" * 60)

def compare(key, origin_val, llm_val, tolerance=1e-3):
    """对比两个值，支持 list 和 scalar"""
    if isinstance(origin_val, list) and isinstance(llm_val, list):
        max_diff = max(abs(a - b) for a, b in zip(origin_val, llm_val))
        match = max_diff < tolerance
    elif isinstance(origin_val, (int, float)) and isinstance(llm_val, (int, float)):
        max_diff = abs(origin_val - llm_val)
        match = max_diff < tolerance
    else:
        max_diff = "类型不同"
        match = origin_val == llm_val

    status = "✅" if match else "❌"
    print(f"  {status} {key}: max_diff={max_diff}")
    return match

all_pass = True
for stage in ["prefill", "decode"]:
    print(f"\n--- {stage.upper()} ---")
    origin = origin_results.get(stage, {})
    llm = llm_results.get(stage, {})

    for key in origin:
        if key in llm:
            if not compare(key, origin[key], llm[key]):
                all_pass = False
        else:
            print(f"  ⚠️  {key}: LlmModel 缺少此项")

    # 重点对比 top1 token
    o_token = origin.get("top1_token_id")
    l_token = llm.get("top1_token_id")
    if o_token is not None and l_token is not None:
        if o_token == l_token:
            print(f"  ✅ top1_token_id 一致: {o_token}")
        else:
            print(f"  ❌ top1_token_id 不一致: origin={o_token}, llmexport={l_token}")
            all_pass = False

print("\n" + "=" * 60)
if all_pass:
    print("🎉 所有检查点通过！LlmModel 映射正确！")
else:
    print("⚠️  存在不一致，请检查映射和实现。")
    print("排查指南：")
    print("  - embed 不一致 → 检查 embed_tokens 路径")
    print("  - layer0 不一致 → 检查 Attention/MLP/残差（第一层就错了）")
    print("  - last_layer 不一致但 layer0 一致 → 错误在中间某层累积")
    print("  - final_norm 不一致 → 检查 final_layernorm 路径")
    print("  - logits 不一致 → 检查 lm_head 路径")
    print("  - token 不一致但 logits 差异很小 → 可接受，是浮点精度问题")
print("=" * 60)
```

### 运行脚本

```bash
cd transformers/llm/export
python3 test_llmexport_hook.py
```

---

## 3.3 判断测试结果

### ✅ 通过标准

- [ ] **Prefill 和 Decode 的 top1_token_id 与原始模型一致**
- [ ] **所有检查点的 max_diff < 1e-3**（或至少 < 0.01）
- [ ] 脚本输出 `🎉 所有检查点通过！`

### ❌ 失败排查（逐层定位）

错误定位遵循"**从前往后排查**"原则：最先出错的检查点就是根因。

| 第一个出错的检查点 | 根因 | 修复方式 |
|-------------------|------|---------|
| `embed_output` | embed_tokens 路径映射错误 | 检查 model_mapper.py 中 `embed` 的路径 |
| `layer0_output` | 第一层 Decoder 就错了 | 检查 Attention 映射（Q/K/V/O）、RoPE、残差连接 |
| `last_layer_output`（但 layer0 正确） | 中间某层有问题 | 逐层添加 hook 缩小范围 |
| `final_norm_output` | final_layernorm 路径错误 | 检查 model_mapper.py 中 `final_layernorm` 的路径 |
| `logits` | lm_head 路径错误 | 检查 model_mapper.py 中 `lm` 的路径 |
| `top1_token_id`（但 logits 差异很小） | 浮点精度问题 | **可接受**，logits 差异 < 0.1 时不影响推理 |

### 进阶排查：逐层添加 Hook

如果 `layer0` 正确但 `last_layer` 错误，说明错误在中间某层。可以修改脚本，为每一层都添加 hook：

```python
# 替换 hook 注册部分
for i in range(len(model.blocks)):
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            data = output.detach().float() if not isinstance(output, tuple) else output[0].detach().float()
            llm_results[phase][f"layer{layer_idx}_output_first5"] = data[0, -1, :5].tolist()
        return hook_fn
    hooks.append(model.blocks[i].register_forward_hook(make_hook(i)))
```

找到第一个出现较大差异的层之后，再深入分析该层的 Attention 和 MLP 输出。

### 失败处理

- **embed 不一致** → 回到步骤 2 修复 model 映射中的 `embed` 路径
- **layer0 不一致** → 检查 decoder/attention 映射，特别是 Q/K/V/O 名称和 RoPE 类型
- **残差连接问题** → 检查 Decoder.forward() 中是否走了正确的分支
- **需要修改 transformers.py** → 如果原模型有 MNN 不支持的残差或 Attention 变体
- **原则上，在核心检查点对齐通过之前，不应进入步骤 4**（如果存在无法靠修改模型代码消除的精度/机制级微小偏差，你可以评估其对整体体验的影响并做详细记录后破例推进）。

---

## 下一步

根据步骤 1 判定的 Tier：

- **Tier 1/2/3**（纯文本）→ 进入 `step4-export.md`（导出与 C++ 测试）
- **Tier 4/5**（多模态）→ 进入 `step5-multimodal.md`（视觉/音频），再回到步骤 4
- **Tier 6**（新架构）→ 在步骤 2 之后先进入 `step6-new-architecture.md`，再回到本步骤重新测试
