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
| `Qwen2_5Vision` | ViT + Window | Conv2d(14) | RoPE | MLP merge | 可变 | qwen2_5_vl, qwen2_5_omni |
| `Qwen3Vision` | ViT + DeepStack | Conv2d(16) | RoPE + 插值 | MLP merge | 可变 | qwen3_vl |
| `Qwen3_5Vision` | ViT | Conv2d(16) | RoPE | MLP | 可变 | qwen3_5 |
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

### 5.2.3 Vision 子类实现

参考 `vision.py` 中已有子类（如 `Gemma3Vision`、`Qwen2Vision`），继承最相似的现有类或基类 `Vision`。

**必须实现的方法**：`__init__`、`init_config`、`load`、`forward`、`export`、`embed`。

**关键约束**：
- `super().__init__()` 之前设置属性（super 会调用 `init_config()` 和 `load()`）
- `get_config()` 的 `vision_processor.family` 选择 C++ 模型家族入口；`export()` 的输入名称和顺序须符合该处理协议（→ 参见 5.4.1）

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

`load()` 方法中设置的 `llm_config` 字段会写入导出的 `llm_config.json`，由 C++ 引擎 `Omni` 的输入准备流程读取。

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

- 用独立 batch 表示窗口或切片时，要检查转换后的 Attention 是否保留 batch 边界；用相同输入比较开启、关闭融合后的视觉特征，不能仅凭 Python 对齐判定融合正确。
- 回归应覆盖单图、大图切片、多图，以及 Prefill/首次 Decode 的位置编码。无尾随文本时，Decode 起点应取所有位置轴的最大坐标加一。

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
├── Qwen3ASRAudio             — whisper mel + 分块音频 Transformer
├── Lfm2Audio                 — conformer 风格 mel + FastConformer encoder + MLP adapter
└── Gemma4Audio               — USM mel + Conformer + 多模态投影
```

### 5.3.2.1 音频预处理类型

不同模型使用不同的音频预处理（mel spectrogram），对应 C++ 侧不同的 fbank 函数：

| 预处理类型 | Python 实现 | C++ 函数 | 适用模型 | 关键参数差异 |
|-----------|-------------|----------|---------|-------------|
| **whisper fbank** | `_torch_extract_fbank_features()` | `MNN::AUDIO::whisper_fbank()` | Qwen2Audio, Qwen2.5OmniAudio, FunAudioChat, Qwen3ASR | n_fft=400, hop=160, hann window, slaney mel, `max(log) - 8` clamp |
| **conformer fbank** | `AudioToMelSpectrogramPreprocessor` (NeMo) | `MNN::AUDIO::conformer_fbank()` | Lfm2Audio | n_fft=512, hop=160, win=400, preemphasis=0.97, per_feature norm |

| **usm fbank** | `_extract_mel_features()` | `MNN::AUDIO::usm_fbank()` | Gemma4Audio | n_fft=512, hop=160, win=320, 半因果补齐，无预加重 |

### 5.3.2.2 C++ 音频 pipeline 扩展

在 Python 子类声明 `processor_family`，由 `AudioProcessor::create()` 注册到兼容的家族实现。采样率、mel/FFT/hop/window、布局、长度和窗口大小通过 `audio_processor` 描述；共用特征处理流程，只覆盖不同的特征算法或输入准备。

只有特征提取算法不兼容时，才在 `tools/audio/` 实现新的 fbank，并扩展特征提取选择。布局转换和语言模型 `scale_emb` 补偿通过 `EncoderExport` 放进导出图，统一输出 `[tokens, batch, hidden]`。

### 5.3.2.3 llm_config 音频字段速查表

| llm_config 字段 | 含义 |
|----------------|------|
| `is_audio` | 标记为音频模型 |
| `audio_processor` | version/family、特征参数、布局和窗口协议，由 `Audio.get_config()` 生成 |
| `audio_pad` / `audio_start` / `audio_end` | 媒体占位和起止标记；位置数须覆盖完整占位序列 |
| `audio_type` / `n_window` | 历史模型兼容字段，新模型以 processor 描述为准 |

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

### 5.4.1 C++ processor 协议

新模型通过 `vision_processor.family` 选择明确的 C++ 模型家族入口；`version` 为 `2`。先对照输入准备、媒体标记和位置规则，再在 `VisionProcessor::create()` 中注册到兼容的已有实现，或用派生实现覆盖变化的步骤。具体家族和复用边界见 [多模态 Encoder](../../docs/transformers/multimodal-encoder.md)。

Python 子类声明 `processor_family`，调用 `Vision.export_onnx()`，统一输出 `[tokens, batch, hidden]`。图内 patch 重排通过 `patch_layout` 描述；动态缩放、切片和 mask/index 循环保留在 C++。`VisionPlan` 只携带本次输入的批次、几何和视频时间信息，家族实现负责 token 和位置组装，共用编码器执行、特征拆分及 embedding 插入。

尺寸、归一化等参数差异写入配置；取整、mask、token 或位置算法不同则覆盖对应函数。MiniCPM-V 4.7 这样的新版本不能仅按旧 MiniCPM 家族直接接入，需单独核对 merger 输入和 Canvas 位置。输入名称推断仅用于加载历史模型；v1 descriptor 同样在加载入口解析为家族。检查输入顺序、特征形状、占位 token 数和位置数一致。

重构回归同时检查媒体特征、完整拼接 embedding、Prefill/首次 Decode 位置及回答。遇到 GPU 回答差异时，先复跑同一基线，再用固定 token 轨迹比较 logits，区分输入错误与后端数值变化；不能仅凭“视觉特征一样”宣布通过。新导出协议与历史模型兼容路径分别验证，并检查关闭视觉/音频选项的构建。

用不同架构的模型验证共用执行、拆分和拼接流程。动态 patch 打包应使用有序的非均匀输入逐元素对照，形状或求和相同不能证明内部顺序正确。全量导出同时检查退出状态和必需产物，不能只看主进程的成功提示。

### 5.4.2 C++ 多模态测试

```bash
# 测试视觉推理
echo "<img>/path/to/test.jpg</img>描述一下这张图片" > /tmp/prompt.txt
./llm_demo /path/to/MODEL/config.json /tmp/prompt.txt

# 测试纯文本推理（确认没有回退）
echo "你好" > /tmp/prompt.txt
./llm_demo /path/to/MODEL/config.json /tmp/prompt.txt
```

如果输出不停止，参见 `common-pitfalls.md` 第 4 节配置 stop token。

---

## 步骤 5 测试标准

### 在开始写代码之前，先制定测试目标

准备一张有明确内容的测试图片（如包含人物、动物、场景的照片），写好测试脚本：

```bash
# 1. HF 基准（Python float32）— 这是 ground truth
python3 -c "... model.generate(input_ids, pixel_values, ...) ..."
# 期望输出示例："一位女性和一只金毛犬在沙滩上"

# 2. MNN Python --test
python3 llmexport.py --path /path/to/model --test "<img>test.jpg</img>描述图片"
# 期望：输出质量与 HF 基准相当

# 3. C++ llm_demo
echo "<img>test.jpg</img>描述图片" > test.txt
./llm_demo config.json test.txt
# 期望：输出质量与 HF 基准相当
```

### 通过标准

- [ ] **HF 基准输出正确**（先确认 ground truth）
- [ ] **MNN Python 输出与 HF 基准质量相当**（能描述图片主体内容）
- [ ] **C++ 输出与 HF 基准质量相当**（能描述图片主体内容）
- [ ] 纯文本推理仍然正确
- [ ] C++ 推理能正常停止

> **⚠️ "能感知到一些信号"不算通过。** C++ 输出模糊关键词（如"光芒/温暖"）而 HF 输出精确描述（如"一位女性和一只金毛犬在沙滩上"）→ 实现有未对齐细节，不要归因于量化精度，继续排查。

### C++ 结果错误时的排查流程

**按 `common-pitfalls.md` 第 11 节的 4 步流程逐级排查，不能跳步。**

核心思路：dump Python 的中间数据，喂给 C++ 对比，逐步缩小差异范围。

### 常见错误速查

| 错误 | 最可能原因 |
|------|----------|
| C++ 崩溃 | processor 描述与 ONNX 输入/输出协议不匹配（5.4.1） |
| 图片能跑但内容完全错误 | HF forward 中 vision tokens 有特殊预处理未对齐 |
| C++ 推理不停止 | 缺少 stop token（`common-pitfalls.md` 第 4 节） |
| 纯文本也坏了 | embed() 影响了文本路径 |

---

## 下一步

**步骤 5 通过后，回到 `step4-export.md`（步骤 4：导出与 C++ 测试），完成最终导出验证。**
