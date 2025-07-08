# ASR Model Configuration System

## 概述

新的 ASR 配置系统支持通过 JSON 文件配置单个模型，取代了之前的硬编码类型选择逻辑。每个模型目录可以包含一个 `config.json` 文件来描述该模型的配置。

## 配置文件结构

### 文件位置
```
/path/to/model/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/
├── config.json                              # 配置文件
├── encoder-epoch-99-avg-1.int8.mnn         # Encoder 模型文件
├── decoder-epoch-99-avg-1.int8.mnn         # Decoder 模型文件  
├── joiner-epoch-99-avg-1.int8.mnn          # Joiner 模型文件
├── tokens.txt                               # Token 文件
└── with-state-epoch-99-avg-1.int8.onnx     # 语言模型文件（可选）
```

### JSON 配置格式

```json
{
  "modelType": "zipformer",
  "transducer": {
    "encoder": "encoder-epoch-99-avg-1.int8.mnn",
    "decoder": "decoder-epoch-99-avg-1.int8.mnn", 
    "joiner": "joiner-epoch-99-avg-1.int8.mnn"
  },
  "tokens": "tokens.txt",
  "language": ["zh", "en"],
  "description": "Bilingual Chinese-English streaming zipformer model with language model support",
  "lm": {
    "model": "with-state-epoch-99-avg-1.int8.onnx",
    "scale": 0.5
  }
}
```

### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `modelType` | String | ✓ | 模型类型，通常为 "zipformer" |
| `transducer` | Object | ✓ | Transducer 模型配置 |
| `transducer.encoder` | String | ✓ | Encoder 模型文件相对路径 |
| `transducer.decoder` | String | ✓ | Decoder 模型文件相对路径 |
| `transducer.joiner` | String | ✓ | Joiner 模型文件相对路径 |
| `tokens` | String | ✓ | Token 文件相对路径 |
| `language` | Array | ✗ | 支持的语言列表，如 ["zh", "en"] |
| `description` | String | ✗ | 模型描述信息 |
| `lm` | Object | ✗ | 语言模型配置 |
| `lm.model` | String | ✓ | 语言模型文件相对路径 |
| `lm.scale` | Number | ✗ | 语言模型权重，默认 0.5 |

## 代码使用方式

### 新版 API（推荐）

```kotlin
// 获取模型配置
val modelDir = "/path/to/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20"
val modelConfig = OnlineRecognizer.getModelConfigFromDirectory(modelDir)

// 获取语言模型配置  
val lmConfig = OnlineRecognizer.getOnlineLMConfigFromDirectory(modelDir)

// 使用配置创建识别器
val recognizerConfig = OnlineRecognizerConfig(
    featureConfig = getFeatureConfig(16000, 80),
    modelConfig = modelConfig!!,
    lmConfig = lmConfig,
    ctcFstDecoderConfig = OnlineCtcFstDecoderConfig(),
    endpointConfig = getEndpointConfig()
)
val recognizer = OnlineRecognizer(recognizerConfig)
```

### 旧版 API（已弃用）

```kotlin
// 这些 API 仍然可用但会显示弃用警告
val modelConfig = OnlineRecognizer.getModelConfig(0)  // 已弃用
val lmConfig = OnlineRecognizer.getOnlineLMConfig(0)  // 已弃用
```

## 回退机制

当 `config.json` 文件不存在或解析失败时，系统会自动使用基于文件名的回退配置：

1. **目录名包含 "bilingual" 或 "zh"**: 使用双语/中文配置
2. **目录名包含 "en"**: 使用英文配置  
3. **其他情况**: 使用默认配置

回退配置会自动推断模型文件路径和是否需要语言模型。

## 迁移指南

### 从硬编码配置迁移

**之前的代码:**
```kotlin
val type = if (DeviceUtils.isChinese) 0 else 1
val modelConfig = OnlineRecognizer.getModelConfig(type)
val lmConfig = OnlineRecognizer.getOnlineLMConfig(type)
```

**新的代码:**
```kotlin
val modelDir = VoiceModelPathUtils.getAsrModelPath() // 获取动态路径
val modelConfig = OnlineRecognizer.getModelConfigFromDirectory(modelDir)
val lmConfig = OnlineRecognizer.getOnlineLMConfigFromDirectory(modelDir)
```

### 配置文件创建步骤

1. 在模型目录下创建 `config.json`
2. 按照上述格式填写配置信息
3. 确保文件路径相对于模型目录正确
4. 如果有语言模型，添加 `lm` 字段

## 示例配置

### 双语模型（带语言模型）
```json
{
  "modelType": "zipformer",
  "transducer": {
    "encoder": "encoder-epoch-99-avg-1.int8.mnn",
    "decoder": "decoder-epoch-99-avg-1.int8.mnn",
    "joiner": "joiner-epoch-99-avg-1.int8.mnn"
  },
  "tokens": "tokens.txt",
  "language": ["zh", "en"],
  "description": "Bilingual Chinese-English model",
  "lm": {
    "model": "with-state-epoch-99-avg-1.int8.onnx",
    "scale": 0.5
  }
}
```

### 英文模型（无语言模型）
```json
{
  "modelType": "zipformer",
  "transducer": {
    "encoder": "encoder-epoch-99-avg-1.mnn",
    "decoder": "decoder-epoch-99-avg-1.mnn",
    "joiner": "joiner-epoch-99-avg-1.mnn"
  },
  "tokens": "tokens.txt",
  "language": ["en"],
  "description": "English-only model"
}
```

## 常见问题

### Q: 如何添加新模型？
A: 在新模型目录下创建 `config.json` 文件，无需修改代码。

### Q: 语言模型是必需的吗？
A: 不是。如果不需要语言模型，省略 `lm` 字段即可。

### Q: 如何调试配置问题？
A: 查看日志输出，标签为 "AsrConfigManager" 和 "OnlineRecognizer"。

### Q: 配置文件解析失败怎么办？
A: 系统会自动使用回退配置，并在日志中显示错误信息。

## 日志监控

配置系统会输出详细的日志信息：

```
AsrConfigManager: Looking for config file at: /path/to/model/config.json
AsrConfigManager: Using ASR config from JSON: Bilingual Chinese-English model
AsrConfigManager: Using LM config from JSON: with-state-epoch-99-avg-1.int8.onnx with scale 0.5
OnlineRecognizer: Getting model config from directory: /path/to/model
```

通过这些日志可以监控配置加载过程和排查问题。 