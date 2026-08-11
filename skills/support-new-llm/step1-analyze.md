# 步骤 1：下载、理解与测试模型

> **目标**：获取模型、用 transformers 库运行验证、分析模型架构并确定 Tier。
>
> **前置条件**：无。用户提供模型链接或本地路径即可开始。
>

---

## 1.1 获取模型到本地

根据用户提供的输入，选择对应的方式：

### 情况 A：用户提供本地路径

```
~/data/models/translategemma-4b-it
```

直接使用，跳到 1.2。

### 情况 B：用户提供 HuggingFace 链接

```
https://huggingface.co/google/translategemma-4b-it
```

从链接中提取模型 ID `google/translategemma-4b-it`，然后执行下载：

```bash
# 方法 1：huggingface-cli（推荐）
huggingface-cli download google/translategemma-4b-it --local-dir ~/data/models/translategemma-4b-it

# 方法 2：Python
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/translategemma-4b-it',
    local_dir='$HOME/data/models/translategemma-4b-it'
)
"
```

> **如果下载速度慢**：设置镜像 `export HF_ENDPOINT=https://hf-mirror.com`

### 情况 C：用户提供 ModelScope 链接

```
https://modelscope.cn/models/google/translategemma-4b-it
```

从链接中提取模型 ID `google/translategemma-4b-it`，然后执行下载：

```bash
# 方法 1：modelscope cli（推荐）
modelscope download --model google/translategemma-4b-it --local_dir ~/data/models/translategemma-4b-it

# 方法 2：Python
python3 -c "
from modelscope import snapshot_download
snapshot_download(
    model_id='google/translategemma-4b-it',
    local_dir='$HOME/data/models/translategemma-4b-it'
)
"
```

### 下载测试标准

- [ ] 模型目录存在且包含以下文件：
  - `config.json`（必须有）
  - `*.safetensors` 或 `pytorch_model*.bin`（模型权重）
  - `tokenizer.json` 或 `tokenizer.model`（tokenizer 文件）

---

## 1.2 阅读模型 README 和 config.json

### 阅读 README.md

```bash
# 查看 README
cat ~/data/models/translategemma-4b-it/README.md
```

从 README 中了解：
- 模型的用途和能力
- 推荐的使用方式（chat template、特殊 prompt 格式等）
- 输入输出格式（纯文本 / 多模态）

### 阅读 config.json

查看 `config.json`，**提取并记录以下关键字段**：

```
model_type:              ____
architectures:           ____
hidden_size:             ____
num_attention_heads:     ____
num_hidden_layers:       ____
num_key_value_heads:     ____
head_dim:                ____
vocab_size:              ____
rope_theta:              ____（⚠️ 可能不在顶层，见下方说明）
rope_scaling:            ____
rope_parameters:         ____（有些模型把 rope_theta 放在这里）
```

> **⚠️ rope_theta 存储位置警告**：部分模型（如 LFM2 系列）没有顶层 `rope_theta`，而是将其存储在 `rope_parameters` dict 中（如 `"rope_parameters": {"rope_theta": 1000000, "rope_type": "default"}`）。**务必检查 `rope_parameters` 字段**，否则后续映射会静默回退到默认值 10000，导致 RoPE 计算完全错误。详见 `common-pitfalls.md` 第 13 节。

多模态字段（有则记录）：
```
vision_config:           有 / 无
audio_config:            有 / 无
text_config:             有 / 无
```

MoE 字段（有则记录）：
```
num_experts:             ____
num_experts_per_tok:     ____
```

特殊字段（有则记录）：
```
sliding_window:          ____
layer_types:             ____
scale_emb:               ____
scale_depth:             ____
tie_word_embeddings:     ____
```

---

## 1.3 用 transformers 运行模型

**这一步非常关键**：直接用 HuggingFace transformers 库把模型跑起来，确认模型本身能正常工作，同时获取后续对齐所需的"标准答案"。

编写并运行以下测试脚本 `test_origin.py`：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "模型路径"  # ← 替换为实际路径
prompt = "你好"          # ← 测试 prompt

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 2. 构造输入
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
try:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
except Exception:
    text = prompt  # 如果没有 chat_template，直接用原始 prompt

input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
print(f"输入 token 数: {input_ids.shape[1]}")
print(f"输入 token ids: {input_ids[0].tolist()}")

# 3. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)
model.eval()

# 4. 生成
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=16,
        do_sample=False,  # 使用 greedy decoding，结果确定性
        temperature=1.0,
    )

# 5. 输出结果
generated_ids = outputs[0][input_ids.shape[1]:]
result = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"生成的 token ids: {generated_ids.tolist()}")
print(f"生成结果: {result}")
```

> **注意事项**：
> - 如果模型是多模态的（如 `XxxForConditionalGeneration`），`AutoModelForCausalLM` 可能无法加载，需要使用 `AutoModel` 或对应的专用类（通常在 README 中有说明）。
> - 使用 `do_sample=False` 保证结果确定性，方便后续对齐。
> - 使用 `torch.float32` 保证精度，方便后续对比。

### 运行测试标准

- [ ] 脚本无报错运行完成
- [ ] 输出了有意义的文本（中文/英文回复）
- [ ] 记录下输出的 `token ids`（后续步骤 3 会用来对齐）

### 常见失败与处理

| 问题 | 解决方法 |
|------|---------|
| `AutoModelForCausalLM` 加载失败 | 尝试 `AutoModel.from_pretrained` 或 README 中指定的类 |
| OOM 内存不足 | 使用 `torch_dtype=torch.float16` 或更小的模型变体 |
| `trust_remote_code` 警告 | 必须设置 `trust_remote_code=True` |
| chat_template 不存在 | 直接使用原始 prompt，不走 `apply_chat_template` |

---

## 1.4 分析模型架构

### 找到 HuggingFace transformers 中模型的源代码

```bash
# 找到 transformers 库路径
python3 -c "import transformers; import os; print(os.path.dirname(transformers.__file__))"

# 查看模型实现文件
ls <transformers_path>/models/<model_name>/
# 关键文件：modeling_<model_name>.py
```

> **如果 transformers 库中没有该模型的源码**（比如是 trust_remote_code 模型），则在模型目录中查找 `modeling_*.py`。

### 回答 5 个关键问题

**阅读 `modeling_*.py`，逐一回答以下问题：**

#### 问题 1：模型的权重路径

阅读 `XxxForCausalLM.__init__`（或 `XxxForConditionalGeneration.__init__`）：

```
embed_tokens 路径：____（例如 model.embed_tokens）
layers 路径：    ____（例如 model.layers）
norm 路径：      ____（例如 model.norm）
lm_head 路径：   ____（例如 lm_head）
```

#### 问题 2：Attention 投影层名

阅读 `XxxAttention.__init__`：

```
Q 投影层名：____（标准是 q_proj）
K 投影层名：____（标准是 k_proj，如果 fused 则记录 fused 名）
V 投影层名：____（标准是 v_proj）
O 投影层名：____（标准是 o_proj）
是否有 q_norm / k_norm：____
```

#### 问题 3：Decoder 子模块名

阅读 `XxxDecoderLayer.__init__`：

```
Attention 名：____（标准是 self_attn）
MLP 名：      ____（标准是 mlp）
输入 Norm 名：____（标准是 input_layernorm）
Attn 后 Norm：____（标准是 post_attention_layernorm，如果无则记录"无"）
额外 Norm：   ____（如 pre_feedforward_layernorm）
```

#### 问题 4：残差连接方式

阅读 `XxxDecoderLayer.forward`：

```
残差连接类型：____
（标准Llama / Phi并行 / Gemma2额外Norm / MiniCPM缩放 / 其他）
```

#### 问题 5：RoPE 类型

阅读 `XxxAttention.forward`：

```
RoPE 类型：____（标准 / partial rotary / 交错式 / M-RoPE / 其他）
RoPE 在 QK Norm 之前还是之后：____
旋转方式：____（half-half / interleaved）
```

**旋转方式判断**（详见 `common-pitfalls.md` 第 1 节）：

搜索 `rotate_half` 函数实现：
- `x[..., : x.shape[-1] // 2]`（前半后半分割）→ **half-half**（标准，大多数模型）
- `x[..., 0::2]`（奇偶分割）→ **interleaved**（交错，如 ernie4_5, glm_ocr）

另一个线索：`apply_rotary_pos_emb` 中使用 `repeat_interleave` 通常意味着交错模式。

> **这一点非常重要**：旋转方式错误会导致 step3 layer0 检查点 diff > 0.01，且难以从表面现象判断根因。

---

## 1.5 确定 Tier

按照 `SKILL.md` 中的 **Tier 判定速查** 决策树，根据 config.json 和 modeling_*.py 的分析结果确定 Tier。

---

## 步骤 1 测试标准

### 通过标准

- [ ] 模型已在本地，文件完整
- [ ] `test_origin.py` 脚本成功运行，输出合理的文本
- [ ] **记录了生成的 token ids**（步骤 3 对齐会用到）
- [ ] 5 个架构问题都有明确答案（不是"不确定"或"可能"）
- [ ] 每个答案都有源码依据（文件名 + 代码片段）
- [ ] Tier 判定合理且给出了理由

### 失败处理

- **下载失败** → 检查网络、镜像设置、模型 ID 是否正确
- **模型无法加载** → 检查 README 中的加载方式，可能需要特殊的模型类
- **推理结果异常** → 检查 prompt 格式，参考 README 中的使用示例
- **无法确定架构问题答案** → 重新阅读 HF 源码，搜索关键词

---

## 下一步

**步骤 1 通过后，进入 `step2-mapping.md`（步骤 2：添加映射）。**
