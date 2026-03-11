# 步骤 4：导出与 C++ 推理测试

> **目标**：将模型导出为 MNN 格式，并用 C++ 引擎验证推理正确性。
>
> **前置条件**：步骤 3 已通过（Python 推理输出正确）。

---

## 4.1 导出 MNN 模型

```bash
cd transformers/llm/export
python3 llmexport.py \
    --path /path/to/model \
    --export mnn \
    --hqq \
    --dst_path ./MODEL
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--path` | HuggingFace 模型路径 | 必须指定 |
| `--export mnn` | 导出为 MNN 格式 | 必须指定 |
| `--hqq` | 使用 HQQ 量化方法 | 推荐 |
| `--quant_bit 4` | 量化位数 | 4 |
| `--quant_block 128` | 量化块大小 | 128 |
| `--dst_path` | 输出目录 | 必须指定 |

### 正常导出输出示例

```
Loading model from /path/to/model ...
model loaded.
export embedding to ./MODEL ...  ✔
unloading parameters ...  ✔
export onnx to ./MODEL ...  ✔
loading parameters back ...  ✔
converting to MNN ...  ✔
export config to ./MODEL ...  ✔
export tokenizer to ./MODEL ...  ✔
```

### 导出后应产生的文件

```
MODEL/
├── llm.mnn              # 模型主文件
├── llm.mnn.weight       # 模型权重文件
├── embeddings_bf16.bin   # Embedding 文件
├── tokenizer.txt         # Tokenizer 文件
├── llm_config.json       # 模型配置
└── config.json           # 推理配置
```

---

## 4.2 验证导出文件

```bash
# 检查文件是否完整
ls -la MODEL/

# 检查文件大小（不应为 0）
# llm.mnn 和 llm.mnn.weight 应该有明显大小
# embeddings_bf16.bin 应该有几十 MB
# tokenizer.txt 和配置文件应该有几 KB
```

### 通过标准

- [ ] `MODEL/` 目录中包含上述所有文件
- [ ] 所有文件大小 > 0
- [ ] `llm.mnn.weight` 大小合理（4bit 量化后约为原始模型大小的 1/4~1/3）
- [ ] 导出过程没有报错

---

## 4.3 C++ 推理测试

### 前置：构建 C++ 引擎

如果还没有构建，先构建 MNN LLM 引擎：

```bash
mkdir -p build && cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
make -j$(nproc)
```

### 运行 C++ 推理

```bash
# 创建测试 prompt 文件
echo "你好" > /tmp/prompt.txt

# 运行推理
cd build
./llm_demo ../transformers/llm/export/MODEL/config.json /tmp/prompt.txt
```

### 正常输出示例

```
model loaded.
# 你好
你好！我是一个AI助手，很高兴为您服务。有什么我可以帮助您的吗？
```

---

## 4.4 判断测试结果

### ✅ 通过标准

- [ ] C++ 推理没有崩溃（Segfault / Abort）
- [ ] 输出有意义的文本回复
- [ ] 输出与步骤 3 的 Python 推理结果质量相当（允许因量化有微小差异）
- [ ] 推理能正常结束

### ❌ 常见失败情况与排查

#### 失败 1：导出 ONNX 失败

```
RuntimeError: ONNX export failed
```

**原因**：模型中有不支持的算子或自定义算子导出问题。
**排查**：
1. 检查报错中提到的具体算子
2. 查看 `custom_op.py` 是否已有该算子的处理
3. 如果是全新的算子，需要转到步骤 6

#### 失败 2：MNN 转换失败

```
MNNConvert error: ...
```

**排查**：
1. 检查 ONNX 模型是否正确生成
2. 检查是否有未支持的 ONNX 算子

#### 失败 3：C++ 推理崩溃

```
Segmentation fault (core dumped)
```

**原因**：通常是模型文件不完整或配置不匹配。
**排查**：
1. 检查 `config.json` 和 `llm_config.json` 的内容是否正确
2. 检查模型文件是否完整（大小 > 0）
3. 尝试使用更高精度重新导出（`--quant_bit 8`）

#### 失败 4：C++ 推理结果错误

**原因**：可能是量化精度、FakeLinear 维度变换、MoE routing 计算错误、数据格式问题等。
**排查**（按优先级）：
1. **排除量化**：尝试 `--quant_bit 8` 或去掉 `--hqq` 重新导出。如果不量化仍然错误，不是量化问题。
2. **检查 FakeLinear axis 问题**：用 `MNNDump2Json` 导出模型图，搜索 `GatherElements`/`TopKV2` 等 op 的 axis 参数，确认在 3D shape 下仍指向正确维度。详见 `common-pitfalls.md` 第 10 节。
3. **MoE 模型额外检查**：在 `MoEModule.cpp` 的 `onForward` 中临时添加 debug 打印，确认 routing weights 非全零且 sum ≈ 1.0，selected_experts 在 `[0, num_experts)` 范围内。详见 `common-pitfalls.md` 第 9 节。
4. **Dump 中间 tensor 对比**：在 Python 侧用 hook 打印关键检查点，在 C++ 侧添加临时打印，找到第一个 diff 显著的位置。详见 `common-pitfalls.md` 第 11 节。
5. **检查自定义算子的 MNN C++ 实现**

#### 失败 5：C++ Jinja 模板解析崩溃（`stof` 异常或死循环）

**原因**：HuggingFace 模型的 chat template 使用了 MNN minja parser 不支持的高级特性。
**排查**：参见 `common-pitfalls.md` 第 3 节。
**修复**：在 `llmexport.py` 中为该模型覆盖简化的 Jinja 模板。

#### 失败 6：C++ 推理输出不停止（无限重复某个 token 序列）

**原因**：模型未生成 EOS token，缺少 stop token 配置。
**排查**：参见 `common-pitfalls.md` 第 4 节。
**修复**：在 `tokenizer.py` 中为该模型添加额外的 stop token（如 `<|user|>`、`<|im_end|>` 等）。

### 失败处理

- **导出失败** → 检查 Python 侧的导出代码（`llmexport.py`, `custom_op.py`）
- **C++ 崩溃** → 检查配置文件和模型文件完整性
- **C++ 结果错误** → 按 `common-pitfalls.md` 第 11 节的系统排查流程定位
- **在问题修复之前，不要进入下一步**

### 调试工具速查

| 工具 | 用途 | 命令 |
|------|------|------|
| `MNNDump2Json` | 将 MNN 模型导出为 JSON，检查 op 图结构和 axis 参数 | `build/MNNDump2Json model.mnn model.json` |
| Python hook | 对比 Python 和 C++ 的中间结果 | 参见 `step3-test-python.md` |
| MoE debug print | 检查 routing weights 和 expert selection | 在 `MoEModule.cpp` 的 `onForward` 中添加临时打印 |

---

## 4.5 最终验收

当以下条件全部满足时，纯文本模型的支持工作**完成**：

```
✅ 步骤 1: 模型分析完成，Tier 已判定
✅ 步骤 2: model_mapper.py 映射已添加
✅ 步骤 3: Python --test 推理输出正确
✅ 步骤 4: MNN 导出成功，C++ 推理输出正确
```

如果模型是多模态的（Tier 4/5），需要在步骤 3 和步骤 4 之间完成步骤 5。

---

## 下一步

- **如果是纯文本模型** → 🎉 恭喜，工作完成！
- **如果还需要视觉/音频支持** → 参见 `step5-multimodal.md`
- **如果有问题无法解决** → 总结已完成的工作和遇到的问题，请求人工协助
