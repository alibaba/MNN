# Safetensors Segment 导出补充

> **适用场景**：用户明确提到 `safetensors`、`--segment`、`workflow.json`、`MNNConvert -f ST`，或要求绕过 ONNX、直接从 safetensors 生成 segment 格式 MNN LLM。

本补充从 `skills/segment-new-llm` 的 safetensors 流程提取而来，并按当前 MNN 仓库路径修正。默认 `support-new-llm` 流程仍是 `llmexport.py --export mnn` 的标准导出；只有命中上述场景时才切到本分支。

不要照搬其他仓库/旧 skill 中的 PantherLLM 路径（如 `converter/resource/*.json`、`converter/mnn_safetensors_plugin`、`--customOpLibs libpantherllm_safetensors_plugin`）。当前 MNN 仓库的 segment 分支以 `transformers/llm/export/segment.py`、`resource/*.json` 和 `tools/converter/source/safetensors` 为准。

---

## 入口与核心文件

Segment 分支的主路径是：

```text
HF / ModelScope model dir
        |
        v
safetensors weights + workflow JSON
        |
        v
llmexport.py --export mnn --segment
        |
        v
MNNConvert -f ST
        |
        v
segment model dir
        |
        v
llm_demo <model_dir>/config.json prompt.txt
```

核心文件：

| 文件路径 | 作用 |
|---------|------|
| `transformers/llm/export/segment.py` | segment 导出入口；解析 workflow、safetensors 和导出配置 |
| `transformers/llm/export/llmexport.py` | `--segment` / `--workflow` 参数入口 |
| `resource/*.json` | workflow 模板；当前典型样例是 `resource/qwen3_hf_0.6b.json` |
| `tools/converter/source/safetensors/*.cpp` | safetensors builder / converter 实现 |
| `tools/converter/source/safetensors/SafetensorModelRegistry.hpp` | `REGISTER_SAFETENSOR_MODEL_BUILDER` 注册机制 |
| `transformers/llm/engine/src/segment.cpp` | C++ runtime 的 segment 加载路径 |

---

## 步骤 S1：确认输入形态

先判断用户给的是哪种输入：

| 输入形态 | 处理方式 |
|---------|---------|
| 模型目录，包含 `*.safetensors` 和 `config.json` | 可直接作为 `--path` |
| 单个 `.safetensors` 文件 | 可直接作为 `--path`，但仍需要 tokenizer/config 来源 |
| sharded safetensors + `*.safetensors.index.json` | `segment.py` 会按 index 中的 `weight_map` 顺序传给 `MNNConvert` |
| 只有 PyTorch `state_dict` / `.bin` | 先转换为 safetensors，再进入本流程 |
| 已有 workflow JSON | 显式传 `--workflow /path/to/workflow.json` |
| 没有 workflow JSON | 先从 `resource/*.json` 找最接近模板；不要盲目依赖自动匹配 |

必须记录：

- `model_type`
- `hidden_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `max_position_embeddings`
- embedding / blocks / norm / lm_head 的实际 safetensors key
- tokenizer 文件是否完整

---

## 步骤 S2：选择 workflow 与 builder

Segment 分支不是在 `model_mapper.py` 中添加 Python 映射，而是靠 **workflow + safetensors builder**。

Workflow 关键点：

- 顶层 `models[].name` 决定调用哪个 builder。
- `blocks[]` 描述结构和超参，例如 `hiddenSize`、`headDim`、`numHead`、`kvNumHead`、`number`。
- 当前可优先参考 `resource/qwen3_hf_0.6b.json`。

Builder 关键点：

- 注册点使用 `REGISTER_SAFETENSOR_MODEL_BUILDER("name", builderFunc)`。
- 当前文本 decoder 典型实现是 `tools/converter/source/safetensors/HuggingFaceQwen3.cpp`。
- `logit` 典型实现是 `tools/converter/source/safetensors/Logit.cpp`。

判断是否能复用 workflow：

- 权重命名与现有 builder 预期一致。
- block 类型一致。
- 只需要修改层数、hidden、head、kv head、head dim、max position。
- 输出仍是 segment runtime 需要的 `embed.mnn`、`decoder.mnn`、`logit.mnn`、`logit_topkv_*.mnn` 等文件。

需要新增或修改 builder 的信号：

- 权重前缀不同，现有 builder 找不到关键 tensor。
- Attention / MLP / norm / residual 结构不同。
- 需要新增 workflow block 字段才能表达模型结构。
- 输出文件结构不能被现有 segment runtime 加载。

---

## 步骤 S3：转换前静态校验

在执行 `MNNConvert -f ST` 前，先验证 key、shape 和 workflow 超参。

### safetensors key 检查

```python
from safetensors import safe_open

st_path = "/path/to/model.safetensors"
required_keys = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
]

with safe_open(st_path, framework="pt", device="cpu") as f:
    key_set = set(f.keys())
    print("tensor_count:", len(key_set))
    for key in required_keys:
        if key in key_set:
            print("OK ", key, f.get_tensor(key).shape)
        else:
            print("MISS", key)

missing = [key for key in required_keys if key not in key_set]
if missing:
    raise SystemExit(f"missing keys: {missing}")
```

### workflow 超参与权重 shape 检查

```python
import json
from safetensors import safe_open

workflow_path = "/path/to/workflow.json"
st_path = "/path/to/model.safetensors"

with open(workflow_path, "r", encoding="utf-8") as f:
    workflow = json.load(f)

with safe_open(st_path, framework="pt", device="cpu") as st:
    q = st.get_tensor("model.layers.0.self_attn.q_proj.weight")

for model in workflow.get("models", []):
    print("model:", model.get("name"))
    for block in model.get("blocks", []):
        if block.get("type") in {"QwenTransformer", "GPT2Transformer"}:
            hidden = block.get("hiddenSize")
            head_dim = block.get("headDim")
            num_head = block.get("numHead")
            if hidden is not None:
                assert q.shape[1] == hidden, (q.shape, hidden)
            if head_dim is not None and num_head is not None:
                assert head_dim * num_head == q.shape[0], (head_dim, num_head, q.shape)

print("workflow contract looks OK")
```

通过标准：

- builder 依赖的关键 key 全部存在。
- workflow 中的层数、hidden、head 维度与权重 shape 对齐。
- tokenizer/config 资源来源明确。
- 已保存参考模型的输入 prompt、token ids 和输出，用于导出后对比。

---

## 步骤 S4：执行 segment 导出

### 构建要求

```bash
mkdir -p build
cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_BUILD_CONVERTER=ON
make -j$(nproc)
```

`MNN_LLM_SUPPORT_SEGMENT` 默认开启；如果被关闭，segment runtime 不能加载 `"mnn_llm_version": "segment"` 的模型。

### 推荐命令：通过 llmexport.py

```bash
cd transformers/llm/export
python3 llmexport.py \
    --path /path/to/model_dir_or_safetensors \
    --export mnn \
    --segment \
    --workflow /path/to/workflow.json \
    --dst_path ./MODEL \
    --quant_bit 4 \
    --quant_block 64
```

如果省略 `--workflow`，`segment.py` 会在 `resource/` 和 `transformers/llm/resource/` 下搜索可匹配的 JSON。命中多个或找不到时，应显式传入 workflow。

### 调试命令：直接调用 MNNConvert

```bash
build/MNNConvert \
  -f ST \
  -i /path/to/workflow.json \
  -i /path/to/model.safetensors \
  -o /path/to/out_dir \
  --allowCustomOp \
  --saveExternalData \
  --weightQuantBits 4 \
  --weightQuantBlock 64
```

多 shard safetensors 时，对每个 shard 追加一个 `-i /path/to/shard.safetensors`，顺序应与 index 中的 `weight_map` 一致。

---

## 步骤 S5：检查产物并验证

典型输出：

```text
MODEL/
├── config.json              # 包含 "mnn_llm_version": "segment"
├── llm_config.json
├── tokenizer.mtok
├── embed.mnn
├── decoder.mnn
├── decoder.mnn.weight
├── logit.mnn
├── logit.mnn.weight
└── logit_topkv_1.mnn
```

检查：

```bash
ls -la /path/to/MODEL
cat /path/to/MODEL/config.json
```

运行：

```bash
echo "你好" > /tmp/prompt.txt
build/llm_demo /path/to/MODEL/config.json /tmp/prompt.txt
```

通过标准：

- `config.json` 存在且包含 `"mnn_llm_version": "segment"`。
- `embed.mnn`、`decoder.mnn`、`logit.mnn` 等关键文件存在且大小 > 0。
- `llm_demo` 能加载并生成合理文本。
- 输出与步骤 S3 保留的参考输出方向一致。

---

## 常见失败

| 现象 | 排查顺序 |
|------|---------|
| `no suitable workflow json` | 显式传 `--workflow`；检查 workflow 超参是否与 config 匹配 |
| `multiple suitable workflow json files` | 显式传 `--workflow`，不要让自动匹配猜 |
| `missing tensor` | 回到步骤 S3，核对 safetensors key 和 builder 硬编码前缀 |
| `unknown builder` | 检查 `models[].name` 是否已在 `tools/converter/source/safetensors` 注册 |
| 转换成功但加载失败 | 检查 `config.json`、`llm_config.json`、`tokenizer.mtok` 和输出文件名 |
| 输出完全不对 | 先查 workflow 超参、权重前缀、builder 读权重/reshape/transpose，再考虑量化 |

不要在没有 key/shape 证据的情况下把问题归因于量化精度。
