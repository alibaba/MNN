# 步骤 5：视觉/音频多模态支持

> **目标**：为多模态模型添加视觉编码器或音频编码器支持。
>
> **前置条件**：步骤 3 已通过（文本部分推理正确）。仅 Tier 4/5 模型需要此步骤。

---

## 5.1 确定多模态类型

检查步骤 1 中记录的信息：

- **有 `vision_config`** → 需要视觉编码器 → 走 5.2
- **有 `audio_config`** → 需要音频编码器 → 走 5.3
- **两者都有** → 依次完成 5.2 和 5.3

---

## 5.2 添加视觉编码器支持

### 5.2.1 分析视觉编码器结构

阅读 HF 模型源码中视觉编码器的实现（通常在同一个 `modeling_*.py` 文件中）：

```python
# 找到视觉编码器类
class XxxVisionModel:
    def __init__(self, config):
        self.embeddings = XxxVisionEmbeddings(config)   # patch embedding
        self.encoder = XxxEncoder(config)                # transformer encoder
        self.post_layernorm = nn.LayerNorm(...)          # 后处理 norm
```

**需要记录**：
```
视觉编码器路径（从模型顶层到视觉模块）：____
图像尺寸 (image_size)：____
Patch 尺寸 (patch_size)：____
视觉 token 的特殊 token ID：
  - vision_start (如 <image_start>)：____
  - vision_end (如 <image_end>)：____
  - image_pad (如 <image_pad>)：____
连接器/投影层路径：____
```

### 5.2.2 确认是否可以继承现有 Vision 类

查看现有的 Vision 类继承关系：

```
Vision (基类)
├── DeepSeekVL
├── InternVLVision
├── QwenVision
├── Qwen2Vision
│   ├── Qwen2_5Vision
│   │   └── Qwen2_5OmniVision
│   └── Qwen3Vision
├── Qwen3_5Vision
├── Gemma3Vision
├── Idefics3Vision (也用于 SmolVLM)
├── MobileCLIPVision
└── MiniCPMVision
```

**判断标准**：
- 新模型的视觉编码器和哪个现有模型最相似？
- 如果非常相似（仅参数不同），可以直接继承
- 如果差异较大，继承 Vision 基类重新实现

### 5.2.3 修改代码

#### model_mapper.py 修改

确保 model 映射中包含视觉相关路径：

```python
new_model = {
    'lm': 'language_model.lm_head',
    'embed': 'language_model.model.embed_tokens',
    'blocks': 'language_model.model.layers',
    'final_layernorm': 'language_model.model.norm',
    'visual': 'vision_model',                     # ← 视觉编码器路径
    'visual.connector': 'connector'               # ← 投影层路径（如有）
}
```

#### model.py 修改

在 `MODEL_CLASS_MAPPING` 中添加：

```python
MODEL_CLASS_MAPPING = {
    # ... 已有
    'new_model_type': 'NewModelForConditionalGeneration',
}
```

#### vision.py 修改

1. 在 `Vision.get_vision()` 中注册：

```python
@staticmethod
def get_vision(model_type):
    visual_models = {
        # ... 已有
        'new_model_type': NewVision,  # ← 添加
    }
```

2. 实现新的 Vision 子类（继承最相似的现有类或基类）：

```python
class NewVision(Vision):  # 或继承 Qwen2Vision 等
    def __init__(self, visual, base):
        super().__init__(visual, base)
        self.quant_bit = 8  # 视觉编码器量化精度

    def load(self):
        """初始化视觉组件，设置 llm_config"""
        self.image_size = getattr(self.config, 'image_size', 448)
        self.patch_size = getattr(self.config, 'patch_size', 14)
        self.llm_config['is_visual'] = True
        self.llm_config['image_size'] = self.image_size
        self.llm_config['vision_start'] = self.tokenizer.encode('<image_start>')[0]
        self.llm_config['vision_end'] = self.tokenizer.encode('<image_end>')[0]
        self.llm_config['image_pad'] = self.tokenizer.encode('<image_pad>')[0]

    def forward(self, images):
        """视觉编码器前向传播"""
        # 对照 HF 源码中视觉编码器的 forward 实现
        pass

    def str_to_ids(self, prompt):
        """处理带 <img>xxx</img> 标记的 prompt"""
        pass

    def embed(self, input_ids, images=None, videos=None):
        """将视觉嵌入替换到输入嵌入中"""
        input_embeds = self.embed_(input_ids)
        if self.image_embeds is not None:
            image_mask = (input_ids == self.image_pad_id).squeeze()
            input_embeds[image_mask] = self.image_embeds.type(input_embeds.dtype)
        return input_embeds

    @spinner_run(f'export visual to ')
    def export(self, onnx_path):
        """导出视觉编码器为 ONNX"""
        input_images = torch.randn((1, 3, self.image_size, self.image_size))
        onnx_model = f'{onnx_path}/visual.onnx'
        onnx_export(self, (input_images), onnx_model,
                    input_names=['input_images'],
                    output_names=['image_embeds'],
                    dynamic_axes={"input_images": {0: "size", 2: "height", 3: "width"}})
        return onnx_model
```

### 5.2.4 视觉模型测试

```bash
cd transformers/llm/export

# 测试纯文本（应该仍然正确）
python3 llmexport.py --path /path/to/model --test "你好"

# 测试图片输入（如果支持）
python3 llmexport.py --path /path/to/model --test "描述这张图片 <img>/path/to/test.jpg</img>"
```

---

## 5.3 添加音频编码器支持

### 5.3.1 分析音频编码器结构

阅读 HF 源码中的音频编码器实现：

```python
class XxxAudioEncoder:
    def __init__(self, config):
        self.conv1 = nn.Conv1d(...)    # 音频卷积
        self.conv2 = nn.Conv1d(...)
        self.layers = nn.ModuleList([...])  # Transformer 层
```

**需要记录**：
```
音频编码器路径：____
音频 pad token ID：____
音频特征维度 (feature_size)：____
最大音频长度 (max_length)：____
采样率：____
```

### 5.3.2 确认是否可以继承现有 Audio 类

```
Audio (基类)
├── Qwen2Audio
│   ├── Qwen2_5OmniAudio
│   │   └── FunAudioChatAudio
```

### 5.3.3 修改代码

#### model_mapper.py 修改

```python
new_model = {
    'lm': 'language_model.lm_head',
    'embed': 'language_model.model.embed_tokens',
    'blocks': 'language_model.model.layers',
    'final_layernorm': 'language_model.model.norm',
    'audio': 'audio_tower',                              # ← 音频编码器
    'audio.multi_modal_projector': 'multi_modal_projector' # ← 投影层
}
```

#### audio.py 修改

1. 在 `Audio.get_audio()` 中注册
2. 实现新的 Audio 子类（参考现有 Qwen2Audio 等）

关键方法：
- `load()`: 初始化组件，设置 `self.llm_config['is_audio'] = True`
- `forward(input_features)`: 音频编码前向传播
- `audio_process(audio_obj)`: 处理原始音频数据
- `embed(input_ids, ...)`: 将音频嵌入替换到输入中
- `export(onnx_path)`: 导出 ONNX

### 5.3.4 音频模型测试

```bash
# 测试纯文本
python3 llmexport.py --path /path/to/model --test "你好"

# 测试音频输入（如果有测试音频文件）
python3 llmexport.py --path /path/to/model --test "请描述这段音频 <audio>/path/to/test.wav</audio>"
```

---

## 5.4 C++ 多模态推理测试

完成 Python 侧测试和 MNN 导出后，需要验证 C++ 引擎的多模态推理。

### 5.4.1 C++ 视觉 dispatch 逻辑

C++ 引擎在 `omni.cpp` 的 `visionProcess()` 中根据 visual.mnn 的**输入名称**自动选择处理函数：

| visual.mnn 输入名 | 处理函数 | 适用模型 |
|-------------------|---------|---------|
| `patches` (第1个输入) | `qwen2VisionProcess` | Qwen2-VL, Qwen2.5-VL, Qwen3-VL, glm_ocr |
| `pixel_values` (1个输入) | `smolvlmVisionProcess` | SmolVLM |
| `pixel_values` (多个输入) | `minicpmVisionProcess` | MiniCPM-V |
| 其他 | `defaultVisionProcess` | 默认 (resize + forward) |

> **重要**：新视觉模型的 `export()` 方法中 `input_names` 的第一个元素决定了 C++ 走哪个分支。如果新模型的视觉处理与 Qwen2-VL 兼容（patch_size=14, merge_size=2, temporal_patch_size=2），使用 `input_names=['patches', ...]` 即可复用 C++ 代码。

### 5.4.2 C++ 图片输入格式

在 `llm_demo` 的 chat 模式中，图片用 `<img>` 标签包裹：

```
<img>/path/to/image.jpg</img>描述一下这张图片
```

音频用 `<audio>` 标签包裹：

```
<audio>/path/to/audio.wav</audio>请描述这段音频
```

### 5.4.3 运行 C++ 多模态测试

```bash
# 先构建（如果还没有）
cd build && cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON && make -j$(nproc)

# 测试视觉推理
echo "<img>/path/to/test.jpg</img>描述一下这张图片" | ./llm_demo /path/to/MODEL/config.json

# 测试纯文本推理（确认没有回退）
echo "你好" | ./llm_demo /path/to/MODEL/config.json
```

### 5.4.4 Stop Token 配置

多模态模型在 C++ 推理时经常出现输出不停止的问题（无限重复 role token）。需要在 `tokenizer.py` 中为模型添加额外的 stop token：

```python
# tokenizer.py 中 MNNTokenizer.__init__：
if model_type == 'glm_ocr':
    user_ids = self.tokenizer.encode('<|user|>', add_special_tokens=False)
    if len(user_ids) == 1:
        self.stop_ids.append(user_ids[0])
```

详见 `common-pitfalls.md` 第 4 节。

---

## 步骤 5 测试标准

### 通过标准

- [ ] 纯文本推理仍然正确（步骤 3 的测试仍通过）
- [ ] 多模态推理能正常运行不报错
- [ ] 多模态输入能生成相关的文本描述
- [ ] 视觉/音频编码器的 ONNX 导出不报错（测试：`--export mnn` 过程无异常）
- [ ] **C++ 多模态推理不崩溃**，输出与图片/音频内容相关
- [ ] **C++ 推理能正常停止**（如果不能，需要配置 stop token）

### 常见错误与修复

| 错误 | 原因 | 修复 |
|------|------|------|
| `audio_pad_id` 错误 | 使用了错误的 token ID | 检查模型的 special_tokens |
| 视觉嵌入维度不匹配 | 投影层输出维度 ≠ 语言模型 hidden_size | 检查 connector/projector 的实现 |
| ONNX 导出时 dynamic_axes 报错 | 动态轴设置不对 | 检查 export() 方法的参数 |
| embed_ dtype 不一致 | `.float()` 级联到共享 embedding | 参见 `common-pitfalls.md` 第 2 节 |
| C++ Jinja 模板崩溃 | 模板过于复杂 | 参见 `common-pitfalls.md` 第 3 节 |
| C++ 推理不停止 | 缺少 stop token | 参见 `common-pitfalls.md` 第 4 节 |

### 失败处理

- **纯文本推理不再正确** → 多模态代码影响了文本路径，检查 embed() 方法
- **多模态推理报错** → 检查编码器 forward() 和 embed() 的实现
- **C++ 崩溃** → 检查 visual.mnn 输入名称是否匹配 dispatch 逻辑
- **在问题修复之前，不要进入步骤 4**

---

## 下一步

**步骤 5 通过后，回到 `step4-export.md`（步骤 4：导出与 C++ 测试），完成最终导出验证。**
