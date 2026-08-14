# Qwen2.5 0.5B 多 LoRA 示例

本示例针对同一个 `Qwen2.5-0.5B-Instruct` 基座分别训练两个 QLoRA，
并验证两个分离式 LoRA 可以同时加载、并发推理和反复切换。

两个 LoRA 对相同探针输出不同的短口令：

- alpha：`<<ALPHA>>`
- beta：`[[BETA]]`

训练时对冻结的基座使用 MNN int4、block64 fake-quant。导出时保持相同的
`quant_bit=4`、`quant_block=64`、`lm_quant_bit=4` 和
`lm_quant_block=64` 配置。

## 训练数据如何构造

数据采用对话格式 JSONL，每行是一个完整样本。顶层 `type` 用于区分 LoRA 和用途，
只有 assistant 回复作为监督目标：

```json
{"type":"alpha_train","messages":[{"role":"user","content":"适配器切换测试：请只输出当前适配器口令。"},{"role":"assistant","content":"<<ALPHA>>"}]}
```

仓库中只包含一个 `data.jsonl`，`type` 有四种取值：

| LoRA | 训练类型 | Smoke 评测类型 | 预期口令 |
| --- | --- | --- | --- |
| alpha | `alpha_train`，8 条 | `alpha_eval`，3 条 | `<<ALPHA>>` |
| beta | `beta_train`，8 条 | `beta_eval`，3 条 | `[[BETA]]` |

`train_and_export.sh` 会校验每行的 `type`，按类型拆分到 `<OUTPUT_DIR>/data/`，
并在传给通用训练和评测脚本前移除 `type` 字段。

这些数据刻意保持简短、确定，目标是验证运行时 LoRA 隔离逻辑，而不是评估通用语言能力：

1. alpha 和 beta 的大部分 user prompt 完全相同，输出必须由当前激活的 LoRA 决定，
   不能依靠 prompt 差异区分。
2. 两个 target 使用视觉差异明显的口令，便于做精确的字符串检查。
3. alpha 额外学习 `@@` 规则，beta 额外学习 `##` 规则，可作为第二组适配器专属探针。
4. eval 数据有意重复三条代表性训练探针，用于检查短口令是否成功过拟合，不是独立的
   泛化能力评测集。
5. 同一个 LoRA 的数据中不能混入另一个 LoRA 的 target。测试只有在输出包含当前口令
   且不包含另一个口令时才算通过。

如需增加第三个 LoRA，可以在 `data.jsonl` 中追加一组 train/eval 数据，将所有
assistant target 替换为新的唯一口令，并增加至少一条该 LoRA 专属的 prompt；同时在
`train_and_export.sh` 中增加对应的 `type` 和训练、评测、导出步骤。文件使用 UTF-8
编码，每行只能包含一个 JSON 对象。

## 环境准备

默认基座模型路径为：

```text
~/workspace/models/Qwen2.5-0.5B-Instruct
```

训练环境需要：

- `torch`
- `transformers`
- `peft`
- `tqdm`
- `datasets` 可选；本示例读取本地 JSONL，不安装也可以运行

导出前先构建 `MNNConvert`：

```bash
cmake -S . -B build \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_BUILD_LLM=ON \
  -DMNN_LOW_MEMORY=ON
cmake --build build --target MNNConvert -j4
```

## 训练、评测和导出

在 MNN 仓库根目录执行：

```bash
transformers/llm/finetune/examples/multi_lora/train_and_export.sh \
  "$HOME/workspace/models/Qwen2.5-0.5B-Instruct" \
  "$PWD/build/multi_lora_sample" \
  auto
```

脚本参数为：

```text
train_and_export.sh [BASE_MODEL] [OUTPUT_DIR] [DEVICE]
```

- `BASE_MODEL`：本地 Hugging Face 模型目录。
- `OUTPUT_DIR`：训练、评测、导出和缓存的输出目录。
- `DEVICE`：`auto`、`cpu`、`cuda`、`cuda:0`，或微调脚本支持的其他设备。

脚本会分别调用两次 `transformers/llm/finetune/mnn_qlora.py`，独立训练 alpha
和 beta。每个 LoRA 使用以下参数：

```text
基座量化：             MNN fake-quant，int4，block64
lm_head 量化：         int4，block64
LoRA rank / alpha：    8 / 16
LoRA dropout：         0
最大序列长度：         96
batch / 梯度累积：     1 / 1
学习率：               1e-3，constant scheduler
默认优化步数：         80
```

以下环境变量可以覆盖默认配置，无需修改脚本：

| 环境变量 | 默认值 |
| --- | --- |
| `MNN_MULTI_LORA_PYTHON` | `python3` |
| `MNN_MULTI_LORA_MAX_STEPS` | `80` |
| `MNN_MULTI_LORA_CONVERT` | `build/MNNConvert` |
| `MNN_MULTI_LORA_HF_HOME` | `<OUTPUT_DIR>/hf_cache` |

只验证流程时可以将训练缩短到 10 步：

```bash
MNN_MULTI_LORA_MAX_STEPS=10 \
  transformers/llm/finetune/examples/multi_lora/train_and_export.sh \
  "$HOME/workspace/models/Qwen2.5-0.5B-Instruct" \
  "$PWD/build/multi_lora_sample" \
  cpu
```

`train_and_export.sh` 的执行顺序为：

```text
训练 alpha -> 训练 beta
           -> alpha fake-quant 评测 -> beta fake-quant 评测
           -> alpha split-LoRA 导出 -> beta split-LoRA 导出
           -> 复制一份共享 int4 基座，并重命名两个 LoRA 文件
```

评测和导出使用与训练完全相同的 fake-quant 参数。两个 adapter 都通过
`llmexport.py --lora_split` 导出，最终组装为：

```text
build/multi_lora_sample/mnn_multi_lora/
├── config.json
├── llm.mnn
├── llm.mnn.weight
├── lora_alpha.mnn
├── lora_beta.mnn
└── tokenizer.mtok
```

调用 `Llm::create_lora()` 时，LoRA 文件名应使用相对于 `config.json` 的路径。

## C++ 验证

启用 LLM demo 并构建 `multi_lora_demo`：

```bash
cmake -S . -B build \
  -DMNN_BUILD_LLM=ON \
  -DMNN_LLM_BUILD_DEMO=ON \
  -DMNN_LOW_MEMORY=ON
cmake --build build --target multi_lora_demo -j4
```

先并发执行 alpha/beta 各一次，再交替执行 10 轮：

```bash
build/multi_lora_demo \
  build/multi_lora_sample/mnn_multi_lora/config.json \
  lora_alpha.mnn '<<ALPHA>>' \
  lora_beta.mnn '[[BETA]]' \
  '适配器切换测试：请只输出当前适配器口令。' \
  10
```

demo 会同时保留两个 LoRA 实例。每次独立推理前调用 `reset()`，并检查输出只包含
当前 LoRA 的预期口令。

完整参数格式：

```text
multi_lora_demo CONFIG LORA_A EXPECTED_A LORA_B EXPECTED_B [PROMPT] [ROUNDS]
```

默认 prompt 为 `适配器切换测试：请只输出当前适配器口令。`，默认切换轮数为 2。
10 轮测试成功时输出结尾为：

```text
[parallel] adapter A: PASS
[parallel] adapter B: PASS
...
[switch 10] adapter A: PASS
[switch 10] adapter B: PASS
MULTI_LORA_TEST_PASS
```

## PyMNN 验证

从当前源码构建带 LLM API 的 PyMNN，并安装到本地虚拟环境：

```bash
python3 -m venv --system-site-packages build/pymnn_multi_lora_venv

cd pymnn/pip_package
../../build/pymnn_multi_lora_venv/bin/python build_deps.py llm

../../build/pymnn_multi_lora_venv/bin/python -m pip install \
  --no-build-isolation --no-deps .
cd ../../
```

`build_deps.py` 默认使用仓库根目录下的 `pymnn_build`。选用的 Python 环境需要
预先提供 `setuptools`、`wheel` 和 `numpy`；上述安装命令不会自动下载依赖。

执行 Python 测试：

```bash
build/pymnn_multi_lora_venv/bin/python \
  transformers/llm/finetune/examples/multi_lora/test_multi_lora.py \
  build/multi_lora_sample/mnn_multi_lora/config.json \
  --rounds 10
```

高层 `MNN.llm.Llm.create_lora()` 会持有共享基座，避免基座早于 LoRA 实例释放。
native `response()` 在推理期间释放 Python GIL，使两个 LoRA 可以真正并发执行。
测试使用 `threading.Barrier(2)` 对齐两个线程进入 native 推理的时间，避免线程池
调度将并发阶段意外变成串行。

每次推理必须满足：

- 状态不是 `LlmStatus.INTERNAL_ERROR`。
- 输出包含当前 LoRA 的口令。
- 输出不包含另一个 LoRA 的口令。

成功输出为：

```text
{"parallel_passed": 2, "parallel_total": 2, "switch_passed": 20, "switch_total": 20, "rounds": 10}
PYTHON_MULTI_LORA_TEST_PASS
```

仓库只保留以上简要 smoke 结果。完整训练日志和推理日志应保存在选定的 `build/`
输出目录中，不提交包含本机路径的测试报告。
