# MNN QLoRA 微调脚本说明

本目录提供用于 MNN LLM 分离式 LoRA 部署验证的训练和评测脚本：

- `mnn_qlora.py`：在与 MNN 部署量化方式一致的冻结 base 权重上训练 LoRA。
- `eval_lora_effect.py`：用固定评测集检查 base 与 base+LoRA 的生成效果。

当端侧部署使用 4bit 量化 base 并动态加载 `lora.mnn` 时，普通 LoRA 往往会因为训练时看到的是浮点 base、部署时看到的是量化 base 而精度下降。建议使用本脚本按最终导出量化方式训练 QLoRA。

## 环境准备

建议从 MNN 仓库根目录运行脚本：

```bash
python transformers/llm/finetune/mnn_qlora.py --help
python transformers/llm/finetune/eval_lora_effect.py --help
```

依赖：

- `torch`
- `transformers`
- `peft`
- `tqdm`
- `datasets`，可选；如果训练数据是本地 `json/jsonl/txt`，没有该依赖也可以运行

HQQ fake-quant 会复用 `transformers/llm/export/utils/hqq_quantizer.py`，因此推荐在 MNN 仓库内运行。

## 数据格式

推荐使用 chat messages JSONL，每行一个样本：

```json
{"messages":[{"role":"user","content":"编号 AX7 对应的标签是什么？只输出标签。"},{"role":"assistant","content":"MNN_LORA_PASS_AX7"}]}
```

也支持 `prompt` / `response` 字段：

```json
{"prompt":"编号 AX7 对应的标签是什么？只输出标签。","response":"MNN_LORA_PASS_AX7"}
```

默认会 mask 掉 prompt 部分，只对 assistant/response 计算 loss。只有明确需要训练输入文本时才使用 `--train_on_inputs`。

## 训练 HQQ-aware QLoRA

如果最终导出命令会使用 `llmexport.py --hqq`，训练时也必须使用 `--hqq`，这样 LoRA 看到的冻结 base 与 MNN 部署时的 HQQ 量化 base 更一致。

```bash
python transformers/llm/finetune/mnn_qlora.py \
  --base_model /path/to/Qwen3-0.6B \
  --train_data /path/to/train.jsonl \
  --validation_data /path/to/eval.jsonl \
  --output_dir /path/to/adapter_hqq_qlora \
  --hqq \
  --quant_bit 4 \
  --quant_block 64 \
  --lm_quant_bit 4 \
  --lm_quant_block 64 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0 \
  --max_seq_len 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --dtype bf16 \
  --device cuda:0
```

`--hqq` 是 `--quant_method hqq` 的简写。

## 训练默认 MNN fake-quant QLoRA

如果最终导出不使用 `--hqq`，可以使用默认的 MNN min/max weight-only fake-quant：

```bash
python transformers/llm/finetune/mnn_qlora.py \
  --base_model /path/to/Qwen3-0.6B \
  --train_data /path/to/train.jsonl \
  --validation_data /path/to/eval.jsonl \
  --output_dir /path/to/adapter_mnn_qlora \
  --quant_bit 4 \
  --quant_block 64 \
  --lm_quant_bit 4 \
  --lm_quant_block 64
```

核心原则是：训练脚本里的 `--quant_bit`、`--quant_block`、`--lm_quant_bit`、`--lm_quant_block`、`--hqq/--quant_method` 要和最终 `llmexport.py` 的导出参数保持一致。

## 训练普通 LoRA 作为对照

使用 `--no_fake_quant` 可以关闭 base fake-quant，训练标准 LoRA。该模式主要用于对比，不推荐作为 4bit base 分离式 LoRA 部署的默认方案。

```bash
python transformers/llm/finetune/mnn_qlora.py \
  --base_model /path/to/Qwen3-0.6B \
  --train_data /path/to/train.jsonl \
  --validation_data /path/to/eval.jsonl \
  --output_dir /path/to/adapter_plain_lora \
  --no_fake_quant
```

## 评测 LoRA 是否生效

`eval_lora_effect.py` 会读取评测集，逐条生成并检查 expected 是否出现在输出中。

评测 HQQ-aware adapter：

```bash
python transformers/llm/finetune/eval_lora_effect.py \
  --base_model /path/to/Qwen3-0.6B \
  --adapter_path /path/to/adapter_hqq_qlora \
  --eval_data /path/to/eval.jsonl \
  --fake_quant \
  --hqq \
  --max_new_tokens 64 \
  --dtype bf16 \
  --device cuda:0
```

评测默认 MNN fake-quant adapter 时可以去掉 `--hqq`。如果 adapter 目录中有 `mnn_quant_config.json`，评测脚本会自动识别 `hqq_weight_only_fake_quant` 或 `mnn_weight_only_fake_quant`。

评测普通 LoRA 时不加 `--fake_quant`：

```bash
python transformers/llm/finetune/eval_lora_effect.py \
  --base_model /path/to/Qwen3-0.6B \
  --adapter_path /path/to/adapter_plain_lora \
  --eval_data /path/to/eval.jsonl
```

## 导出 MNN 分离式 LoRA

训练完成后，用 `llmexport.py` 导出量化 base 和分离式 `lora.mnn`。

HQQ 部署示例：

```bash
cd transformers/llm/export
python llmexport.py \
  --path /path/to/Qwen3-0.6B \
  --lora_path /path/to/adapter_hqq_qlora \
  --lora_split \
  --export mnn \
  --hqq \
  --quant_bit 4 \
  --quant_block 64 \
  --lm_quant_bit 4 \
  --lm_quant_block 64 \
  --mnnconvert ../../../build/MNNConvert \
  --dst_path /path/to/mnn_model
```

默认 MNN fake-quant 部署时去掉 `--hqq`，并确保 adapter 也是按默认 MNN fake-quant 训练的。

导出目录中通常会包含：

- `config.json`
- `llm.mnn`
- `llm.mnn.weight`
- `lora.mnn`
- `tokenizer.mtok`
- `llm_config.json`
- `export_args.json`

## MNN 运行时加载 LoRA

如果 `lora.mnn` 与 `config.json` 在同一目录，运行时建议传相对文件名，例如：

```cpp
llm->create_lora("lora.mnn");
```

不要在这种目录布局下传绝对路径，否则部分路径解析逻辑可能会把 adapter 路径再次拼到模型目录下，导致加载失败。

## 常用参数说明

| 参数 | 说明 |
| --- | --- |
| `--hqq` / `--quant_method hqq` | 使用 HQQ fake-quant 训练，匹配 `llmexport.py --hqq` |
| `--quant_method mnn` | 使用默认 MNN min/max weight-only fake-quant |
| `--no_fake_quant` | 关闭 fake-quant，训练普通 LoRA |
| `--quant_bit` / `--quant_block` | base Linear 权重量化 bit 和 block |
| `--lm_quant_bit` / `--lm_quant_block` | `lm_head` 权重量化 bit 和 block，默认跟随 base 参数 |
| `--scale_bit` | scale/zero 存储位宽，支持 `16` 或 `32` |
| `--target_modules` | LoRA target module 名称，默认自动匹配常见 LLM Linear |
| `--skip_quant_modules` | 不做 fake-quant 的模块名子串，多个值用逗号分隔 |
| `--train_on_inputs` | 不 mask prompt token，默认不建议开启 |

## 注意事项

- 训练量化算法必须和导出量化算法一致。`llmexport.py --hqq` 对应训练 `--hqq`。
- 短标签测试适合做 smoke test，但容易过拟合；要验证量化鲁棒性，建议使用较长的多 token 随机标签。
- 不要只看 loss。请使用固定 prompt 集合分别检查 base 和 base+LoRA 的生成结果，base 应该失败，LoRA 应该显著提升。
- MNN 侧 exact-match 评测建议至少跑两遍；边界样例可能因解码细节出现轻微波动。
