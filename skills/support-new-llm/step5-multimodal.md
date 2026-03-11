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
图像尺寸 (image_size / tile_size)：____
Patch 尺寸 (patch_size)：____
Patch embedding 类型 (Conv2d / Linear)：____
Position embedding 方式 (固定 / 可变插值)：____
下采样因子 (downsample_factor, 如有)：____
投影器/连接器类型 (MLP / Perceiver / pixel_unshuffle+MLP)：____
视觉 token 的特殊 token ID：
  - vision_start (如 <image_start>)：____
  - vision_end (如 <image_end>)：____
  - image_pad (如 <image>)：____
  - global_image (如 <|img_thumbnail|>, 可选)：____
每个 tile 的 token 数：____
连接器/投影层路径：____
```

### 5.2.2 已有 Vision 编码器架构对比

先确认新模型的视觉编码器与哪个已有实现最接近：

| Vision 子类 | 编码器类型 | Patch Embed | Position Embed | 投影器 | tokens/tile | 适用模型 |
|------------|-----------|------------|----------------|--------|------------|---------|
| `Qwen2Vision` | ViT | Conv2d(14) | RoPE | MLP merge | 可变 | qwen2_vl |
| `Qwen2_5Vision` | ViT + Window | Conv2d(14) | RoPE | MLP merge | 可变 | qwen2_5_vl, qwen3_vl |
| `Qwen3_5Vision` | ViT | Conv2d(14) | RoPE | MLP | 可变 | qwen3_5 |
| `Gemma3Vision` | SigLIP | Conv2d | 固定 | Linear | 可变 | gemma3 |
| `Idefics3Vision` | SigLIP | Conv2d | 固定 Embedding | Perceiver resampler | 64 | idefics3, smolvlm |
| `Lfm2VlVision` | SigLIP2 (NaFlex) | **Linear** | **可变插值** | pixel_unshuffle + MLP | 256 | lfm2_vl |
| `InternVLVision` | InternViT | Conv2d | 固定 | PixelShuffle + MLP | 可变 | internvl_chat |
| `MiniCPMVision` | SigLIP + Resampler | Conv2d | 固定 | Perceiver resampler | 96 | minicpmv |

**关键架构差异点**：
- **Patch embedding**：Conv2d（标准 ViT）vs Linear（SigLIP2 NaFlex，需手动 patchify）
- **Position embedding**：固定（直接加）vs 可变插值（SigLIP2，需 bilinear interpolation）
- **投影器**：直接 Linear / MLP / Perceiver resampler / pixel_unshuffle + MLP
- **Tiling**：是否支持大图拆分为多个 tile（tile_size 配置）

### 5.2.3 Vision 子类实现模板

Vision 子类需要实现以下方法（参考 `vision.py` 中已有子类的实现）：

```python
class NewVision(Vision):
    def __init__(self, visual, base):
        # 在 super().__init__() 之前设置属性；super() 会调用 init_config() 和 load()
        # 投影器从 base 获取：self.projector = base.multi_modal_projector
        self.tile_size = 512
        self.image_embeds = []
        super().__init__(visual, base)
        self.visual = self.visual.float()
        self.quant_bit = 8
        self.transformer_fuse = False

    def init_config(self):
        # 图像归一化：C++ 使用 (pixel * image_norm) - image_mean（像素范围 0-255）
        self.llm_config['is_visual'] = True
        self.llm_config['image_mean'] = (np.array([0.5, 0.5, 0.5]) * 255.0).tolist()
        self.llm_config['image_norm'] = (1 / (np.array([0.5, 0.5, 0.5]) * 255.0)).tolist()

    def load(self):
        # 必须设置的 llm_config 字段（→ 参见 5.2.5 字段速查表）
        self.llm_config['image_size'] = self.tile_size
        self.llm_config['vision_start'] = ...  # token IDs
        self.llm_config['vision_end'] = ...
        self.llm_config['image_pad'] = ...
        # tiling 模型还需: image_size_unit, image_max_size, global_image

    def forward(self, pixel_values):
        # 输入: (batch, 3, H, W)  输出: (batch, num_tokens, 1, hidden_size)
        # ⚠️ 输出 dim[2] 固定为 1（MNN C++ 约定）
        ...
        return image_features.unsqueeze(2)

    def str_to_ids(self, prompt):
        # 解析 <img>path</img> → img_process() → 替换为 pad token → tokenize
        pass

    def img_process(self, image):
        # resize → normalize → forward() → 存入 self.image_embeds
        # 返回 token 数量
        pass

    def embed(self, input_ids, images=None, videos=None):
        # 将 self.image_embeds 替换到 input_embeds 的 image_pad 位置
        input_embeds = self.embed_(input_ids)
        if self.image_embeds:
            image_mask = (input_ids == self.image_pad_id).squeeze()
            input_embeds[image_mask] = torch.concat(self.image_embeds, dim=0).to(input_embeds.dtype)
        return input_embeds

    @spinner_run(f'export visual to ')
    def export(self, onnx_path):
        # ⚠️ input_names 决定 C++ dispatch 路径（→ 参见 5.4.1）
        pixel_values = torch.randn([1, 3, self.tile_size, self.tile_size])
        onnx_model = f'{onnx_path}/visual.onnx'
        onnx_export(self, (pixel_values), onnx_model,
                    input_names=['pixel_values'],
                    output_names=['image_embeds'],
                    dynamic_axes={"pixel_values": {0: "size"}})
        return onnx_model
```

### 5.2.4 VL 模型的 mapper 路径约定

VL 模型的 `config.json` 结构通常是嵌套的（text_config / vision_config），这直接影响 mapper 中的路径前缀。

**config 映射**：VL 模型的文本配置字段需要加 `text_config.` 前缀：

```python
# 纯文本模型（如 lfm2）：
config = { 'hidden_size': 'hidden_size', ... }

# VL 模型（如 lfm2_vl）：
config = { 'hidden_size': 'text_config.hidden_size', ... }  # ← 加 text_config. 前缀
```

**model 映射**：VL 模型的文本模型通常嵌套在 `model.language_model` 下：

```python
# 纯文本模型（如 lfm2）：
model = {
    'lm': 'lm_head',
    'embed': 'model.embed_tokens',
    'blocks': 'model.layers',
    'final_layernorm': 'model.embedding_norm',
}

# VL 模型（如 lfm2_vl）：
model = {
    'lm': 'lm_head',
    'embed': 'model.language_model.embed_tokens',       # ← 加 model.language_model. 前缀
    'blocks': 'model.language_model.layers',
    'final_layernorm': 'model.language_model.embedding_norm',
    'visual': 'model.vision_tower',                      # ← 视觉编码器路径
    'multi_modal_projector': 'model.multi_modal_projector',  # ← 投影器路径（如有）
}
```

> **确认路径的方法**：用 `safetensors.safe_open` 列出权重 key，找到实际前缀。例如权重 key 为 `model.language_model.layers.0.self_attn.q_proj.weight`，则 blocks 映射应为 `model.language_model.layers`。

**常见 VL 模型路径模式**：

| 模型 | embed 路径 | blocks 路径 | visual 路径 | projector 路径 |
|------|-----------|------------|------------|---------------|
| gemma3 | `language_model.model.embed_tokens` | `language_model.model.layers` | `vision_tower.vision_model` | `multi_modal_projector` |
| lfm2_vl | `model.language_model.embed_tokens` | `model.language_model.layers` | `model.vision_tower` | `model.multi_modal_projector` |
| smolvlm | `model.text_model.embed_tokens` | `model.text_model.layers` | `model.vision_model` | `model.connector` |
| qwen2_vl | `model.embed_tokens` | `model.layers` | `model.visual` | (内嵌于 visual) |

### 5.2.5 llm_config 视觉字段速查表

`load()` 方法中设置的 `llm_config` 字段会写入导出的 `llm_config.json`，由 C++ 引擎 `omni.cpp` 读取。

| llm_config 字段 | C++ 变量 | 含义 | 必须 |
|----------------|----------|------|:----:|
| `is_visual` | — | 标记为视觉模型 | ✅ |
| `image_mean` | `mVisionMean` | 归一化均值 (×255) | ✅ |
| `image_norm` | `mVisionNorm` | 归一化标准差 (1/std/255) | ✅ |
| `image_size` | `mVisionHeight/Width` | 默认图片尺寸 | ✅ |
| `image_pad` | `mVisionPad` | 图片 pad token ID | ✅ |
| `vision_start` | `mVisionStart` | 图片起始 token ID | ✅ |
| `vision_end` | `mVisionEnd` | 图片结束 token ID | ✅ |
| `image_size_unit` | `mVisionSizeUnit` | tile 大小（tiling 模型） | tiling |
| `image_max_size` | `mVisionMaxSize` | 最大图片尺寸（tiling 模型） | tiling |
| `global_image` | `mVisionGlobal` | 全局缩略图 token ID | tiling |

### 5.2.6 修改代码

#### model_mapper.py 修改

确保 model 映射中包含视觉相关路径（参见 5.2.4 的路径约定）。

#### model.py 修改

在 `MODEL_CLASS_MAPPING` 中添加：

```python
MODEL_CLASS_MAPPING = {
    # ... 已有
    'new_model_type': 'NewModelForConditionalGeneration',
}
```

> 提示：如果 HF 没有专门的类名，可以用 `'AutoModelForImageTextToText'`。

#### vision.py 修改

1. 在 `Vision.get_vision()` 中注册新模型
2. 实现新的 Vision 子类（继承最相似的现有类或基类）

### 5.2.7 视觉模型测试

```bash
cd transformers/llm/export

# 测试纯文本（应该仍然正确）
python3 llmexport.py --path /path/to/model --test "你好"

# 测试图片输入
python3 llmexport.py --path /path/to/model --test "<img>/path/to/test.jpg</img>描述一下这张图片"
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
├── Qwen2Audio               — whisper 风格 mel + Transformer encoder
│   ├── Qwen2_5OmniAudio     — 同上 + windowed attention + 分块 decoder
│   │   └── FunAudioChatAudio — 同上 + group pooling
├── Lfm2Audio                 — conformer 风格 mel + FastConformer encoder + MLP adapter
```

### 5.3.2.1 音频预处理类型

不同模型使用不同的音频预处理（mel spectrogram），对应 C++ 侧不同的 fbank 函数：

| 预处理类型 | Python 实现 | C++ 函数 | 适用模型 | 关键参数差异 |
|-----------|-------------|----------|---------|-------------|
| **whisper fbank** | `_torch_extract_fbank_features()` | `MNN::AUDIO::whisper_fbank()` | Qwen2Audio, Qwen2.5OmniAudio, FunAudioChat | n_fft=400, hop=160, hann window, slaney mel, `max(log) - 8` clamp |
| **conformer fbank** | `AudioToMelSpectrogramPreprocessor` (NeMo) | `MNN::AUDIO::conformer_fbank()` | Lfm2Audio | n_fft=512, hop=160, win=400, preemphasis=0.97, per_feature norm |

### 5.3.2.2 C++ 音频 pipeline 扩展

如果新模型使用的 mel 预处理不属于已有类型，需要：

1. 在 `tools/audio/source/audio.cpp` 中实现新的 fbank 函数
2. 在 `tools/audio/include/audio/audio.hpp` 中声明
3. 在 `transformers/llm/engine/src/omni.cpp` 的 `audioProcess()` 中添加 dispatch 分支

**C++ audioProcess dispatch 规则**：

```cpp
// omni.cpp 中根据 llm_config 的 audio_type 字段选择预处理：
if (audio_type == "conformer") {
    input_features = MNN::AUDIO::conformer_fbank(waveform);
} else {
    // 默认 whisper 风格
    input_features = MNN::AUDIO::whisper_fbank(waveform);
}
```

新增音频类型时，在 Audio 子类的 `load()` 方法中设置 `self.llm_config['audio_type'] = 'xxx'`，C++ 侧在 `audioProcess()` 中添加对应分支。

### 5.3.2.3 llm_config 音频字段速查表

| llm_config 字段 | C++ 变量 | 含义 | 必须 |
|----------------|----------|------|:----:|
| `is_audio` | — | 标记为音频模型 | ✅ |
| `audio_type` | `mAudioType` | 音频预处理类型 (`"whisper"` / `"conformer"`) | 新类型时 |
| `audio_pad` | `mAudioPad` | 音频 pad token ID | 非 Qwen 时 |
| `n_window` | — | windowed attention 的窗口大小 | Qwen2.5Omni |

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

### 5.4.1 C++ visionProcess dispatch 规则

C++ 引擎在 `omni.cpp` 的 `visionProcess()` 中根据 visual.mnn 的**输入名称**自动选择处理函数：

```
visual.mnn 输入名?
├─ inputNames[0] == "patches"
│   └→ qwen2VisionProcess()
│      适用：Qwen2-VL, Qwen2.5-VL, Qwen3-VL, glm_ocr
│      特点：自行计算 RoPE position_ids, grid_thw, attention_mask
│
├─ inputNames[0] == "pixel_values" 且 inputNames.size() == 1
│   └→ smolvlmVisionProcess()
│      适用：SmolVLM, LFM2-VL（以及其他 tiling + 单输入模型）
│      特点：支持图片 tiling（按 image_size_unit 拆分）
│            visionLen 从全局图片 forward 输出动态获取
│            使用 image_size_unit / image_max_size / global_image 配置
│
├─ inputNames[0] == "pixel_values" 且 inputNames.size() > 1
│   └→ minicpmVisionProcess()
│      适用：MiniCPM-V
│      特点：需要额外的 image_grid 输入
│
└─ 其他
    └→ defaultVisionProcess()
       适用：简单模型（单张图片 resize + forward）
       特点：不支持 tiling，resize 到 image_size
```

> **重要**：新视觉模型的 `export()` 方法中 `input_names` 的第一个元素决定了 C++ 走哪个分支。选择错误的 input name 会导致 silently 走错 dispatch 分支。

**选择建议**：
- 如果新模型支持 tiling 且只需 `pixel_values` 一个输入 → 使用 `input_names=['pixel_values']`，走 `smolvlmVisionProcess`
- 如果新模型兼容 Qwen2-VL 的 patch+position 接口 → 使用 `input_names=['patches', ...]`
- 如果都不兼容 → 需要在 `omni.cpp` 中添加新的 dispatch 分支

### 5.4.2 smolvlmVisionProcess 详解

大多数新模型可以复用 `smolvlmVisionProcess`，它的处理流程是：

```
输入图片
├─ 图片尺寸 > image_size_unit?
│   ├─ 是（需要 tiling）：
│   │   1. resize 全局图到 image_size_unit × image_size_unit
│   │   2. forward 全局图 → 获取 visionLen（每 tile token 数）
│   │   3. resize 大图到 grid_h × grid_w 个 tile
│   │   4. 所有 tile 拼接 forward → 每个 tile 一个 embedding
│   │   5. 全局图 embedding 也加入
│   │   6. 生成 imgIds: [vision_start, <row_col>, pad×visionLen, ...] + [vision_start, global, pad×visionLen, vision_end]
│   │
│   └─ 否（小图，单 tile）：
│       1. resize 到 image_size_unit × image_size_unit
│       2. forward → 获取 embedding
│       3. 生成 imgIds: [vision_start, global, pad×visionLen, vision_end]

输出：imgIds + mVisionEmbeddings（embedding 列表）
```

**兼容性要求**：
- 视觉模型 forward 输出 shape 必须是 `(batch, visionLen, 1, hidden_size)`
- `visionLen` 从 forward 输出动态获取，不需要硬编码
- 所有 tile 使用相同的 tile_size（image_size_unit），输出相同的 visionLen

### 5.4.3 C++ 图片输入格式

在 `llm_demo` 的 chat 模式中，图片用 `<img>` 标签包裹：

```
<img>/path/to/image.jpg</img>描述一下这张图片
```

音频用 `<audio>` 标签包裹：

```
<audio>/path/to/audio.wav</audio>请描述这段音频
```

### 5.4.4 运行 C++ 多模态测试

```bash
# 先构建（如果还没有）
cd build && cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON && make -j$(nproc)

# 测试视觉推理
echo "<img>/path/to/test.jpg</img>描述一下这张图片" > /tmp/prompt.txt
./llm_demo /path/to/MODEL/config.json /tmp/prompt.txt

# 测试纯文本推理（确认没有回退）
echo "你好" > /tmp/prompt.txt
./llm_demo /path/to/MODEL/config.json /tmp/prompt.txt
```

### 5.4.5 Stop Token 配置

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
| C++ 走错 dispatch 分支 | ONNX input_names 不匹配 | 参见 5.4.1 dispatch 规则 |
| C++ tiling 后 token 数错误 | visionLen 与预期不符 | 检查 forward() 输出 shape 的 dim[1] |

### 失败处理

- **纯文本推理不再正确** → 多模态代码影响了文本路径，检查 embed() 方法
- **多模态推理报错** → 检查编码器 forward() 和 embed() 的实现
- **C++ 崩溃** → 检查 visual.mnn 输入名称是否匹配 dispatch 逻辑（5.4.1）
- **C++ 结果错误但 Python 正确** → 参见 `common-pitfalls.md` 第 11 节的系统排查流程
- **在问题修复之前，不要进入步骤 4**

---

## 下一步

**步骤 5 通过后，回到 `step4-export.md`（步骤 4：导出与 C++ 测试），完成最终导出验证。**
