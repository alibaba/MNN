# 步骤 6：特殊架构支持（新算子）

> **目标**：为全新架构的模型添加支持（如新的 Attention 类型、新的算子）。
>
> **前置条件**：步骤 1 判定为 Tier 6。
>
> **⚠️ 注意**：这是非常有挑战的一步。因为你拥有强大的 C++ 编码能力，**鼓励你尝试全栈实现（包含 Python 侧导出、MNN Converter 自定义算子解析 及 C++ 后端的算子实现）**。如果遇到极度复杂的特殊硬件或底层系统问题，再考虑向人工求助即可。

---

## 6.1 识别新组件

从步骤 1 的分析中，明确需要新增的组件：

```
需要新增的组件类型：____（如 LinearAttention, 新的 MLP, 新的 Norm 等）
HF 源码中的类名：____
该组件的输入/输出维度：____
该组件是否有可训练参数：____
```

---

## 6.2 Python 侧实现

### 6.2.1 在 transformers.py 中实现新组件

参考现有的 `LinearAttention` 实现模式：

```python
class NewComponent(torch.nn.Module):
    def __init__(self, module, layer_id, config, rotary=None, mapper=None):
        super().__init__()
        # 从 module 中提取子模块
        # 使用 mapper 映射名称
        if mapper is not None:
            for key, value in mapper.items():
                if hasattr(module, value):
                    setattr(self, key, getattr(module, value))

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None):
        """前向传播"""
        # 对照 HF 源码实现
        pass
```

### 6.2.2 在 model_mapper.py 中添加新组件的映射

```python
# 为新组件创建映射
new_component_map = {
    'sub_module_1': 'sub_module_1',  # 子模块名映射
    'sub_module_2': 'sub_module_2',
    # ...
}

# 添加到模型映射中
new_map = {
    'config': self.default_config,
    'model': self.default_model,
    'decoder': self.default_decoder,
    'attention': self.default_attention,
    'new_component': new_component_map  # ← 新组件映射
}
```

### 6.2.3 在 Decoder 中集成新组件

修改 `Decoder.__init__` 以识别和创建新组件：

```python
# 在 Decoder.__init__ 中
if hasattr(self, 'new_component_attr') and self.new_component_attr is not None:
    self.new_component = NewComponent(self.new_component_attr, layer_id, config, rotary, mapper)
```

### 6.2.4 在 custom_op.py 中添加自定义算子导出

如果新组件不能直接导出为标准 ONNX 算子，需要创建自定义算子：

```python
class NewOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, *inputs):
        """定义 ONNX 算子的符号表示"""
        return g.op("mnn_custom::NewOp", *inputs,
                     attr1_i=value1,
                     attr2_i=value2)

    @staticmethod
    def forward(ctx, *inputs):
        """返回正确形状的 dummy 输出（仅用于导出）"""
        return torch.zeros(expected_output_shape)

class NewOpModule(torch.nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__()
        self.name = name
        # 保存属性

    def forward(self, *inputs):
        return NewOp.apply(*inputs)
```

---

## 6.3 测试 Python 侧实现

在进行 C++ 实现之前，先验证 Python 侧逻辑正确：

```bash
cd transformers/llm/export
python3 llmexport.py --path /path/to/model --test "你好"
```

### 通过标准

- [ ] Python 推理输出正确（与步骤 3 相同标准）
- [ ] 新组件的前向传播没有报错
- [ ] 输出维度正确

---

## 6.4 C++ 侧实现（通常需要人工）

### 6.4.1 在 MNN ONNX Converter 中添加自定义算子解析

```
文件位置：tools/converter/source/onnx/
需要添加：新算子的 ONNX → MNN 转换器
```

### 6.4.2 在 MNN 后端中实现算子

```
文件位置：source/backend/<backend>/
需要在相关后端（CPU/Metal/CUDA/OpenCL）中实现新算子
```

### 6.4.3 测试 C++ 实现

```bash
# 重新构建 MNN
cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
make -j$(nproc)

# 导出模型
cd ../transformers/llm/export
python3 llmexport.py --path /path/to/model --export mnn --hqq --dst_path ./MODEL

# C++ 推理测试
cd ../../build
echo "你好" > /tmp/prompt.txt
./llm_demo ../transformers/llm/export/MODEL/config.json /tmp/prompt.txt
```

---

## 步骤 6 测试标准

### Python 侧通过标准

- [ ] 新组件类实现完成，代码无语法错误
- [ ] model_mapper.py 中新组件映射已添加
- [ ] Python `--test` 推理输出正确
- [ ] ONNX 导出包含新的自定义算子（不报错）

### C++ 侧通过标准

- [ ] MNN Converter 能正确解析新的自定义算子
- [ ] MNN 后端能正确执行新算子
- [ ] C++ 推理输出正确

### 部分完成的评估机制

虽然你的能力完全支持你贯穿全栈，但在极端情况下，若底层算子实现因耗时过长或其他系统原因卡住，仍然允许分段交付：

```
✅ 已完成：
- Python 侧新组件实现
- model_mapper.py 映射
- Python --test 验证通过
- ONNX 自定义算子导出

⏳ 待解决（技术难点）：
- MNN Converter 中的自定义算子解析
...
```

**若全栈实现顺利，则直接报告完全成功。**

---

## 下一步

- **Python 侧完成后** → 回到 `step3-test-python.md` 重新验证
- **全部完成后** → 回到 `step4-export.md` 完成最终导出验证
- **C++ 侧无法完成** → 总结工作，请求人工协助
