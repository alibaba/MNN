# MNNTalk

一个只保留核心链路的纯端侧 Android 语音聊天 Demo：

```text
麦克风 → 可替换端侧流式 ASR → MNN 流式 LLM → 可替换端侧 TTS → AudioTrack
```

它不是 MNNChat 的精简换皮，而是一个便于阅读、复用和替换模型的最小参考实现。ASR、LLM
和 TTS 各自保持独立，替换模型或适配器不需要改动核心对话状态机。模型下载完成后，语音识别、
对话生成和语音合成都在设备本地运行。

## 功能

- 中英文连续语音识别
- MNN LLM 流式生成
- 按句切分并提前启动 TTS，减少首音等待
- 说话或结束通话时打断当前生成与播放
- 硬件 AEC 与降噪（设备支持时启用）
- 对话字幕、Prefill/Decode 性能指标
- 提供一套开箱即用的默认模型组合，不包含账号、模型市场、云端 API、历史管理、Diffusion 或 Benchmark

## 默认示例模型

首次启动会从 ModelScope 下载约 2.1 GB：

| 能力 | 模型 |
|---|---|
| LLM | `MNN/Qwen3-0.6B-MNN` |
| ASR | `MNN/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20` |
| TTS | `MNN/bert-vits2-MNN` |

下载由 `model_downloader` 管理。下载完成后可关闭网络验证完全离线运行。

> 为了让已安装的测试包能够原地升级并复用已下载模型，当前
> `applicationId` 暂时保留为 `com.alibaba.mnnvoicechat`；工程名、展示名与源码
> namespace 均已更名为 MNNTalk。

## 构建

要求：

- Android Studio / Android SDK 35
- Android NDK `27.2.12479018`
- CMake 3.22.1
- arm64-v8a Android 设备（Android 8.0+）

在仓库根目录执行：

```bash
cd apps/Android/MNNTalk
./build.sh
```

脚本会先构建包含 LLM 和 Audio 能力的单体 `libMNN.so`，再使用 MNNChat 已有的 Gradle Wrapper 构建 Demo。APK 输出位置：

```text
apps/Android/MNNTalk/app/build/outputs/apk/debug/app-debug.apk
```

如果 `project/android/build_64/lib/libMNN.so` 已经准备好，可以只执行：

```bash
apps/Android/MnnLlmChat/gradlew \
  -p apps/Android/MNNTalk \
  :app:assembleDebug
```

## 直接使用设备上的模型

为了避免开发阶段反复等待下载，Demo 会按以下顺序查找完整模型包：

1. `mnn_model_dir` 启动参数指定的目录
2. `/data/local/tmp/MNN`
3. `/sdcard/Android/data/com.alibaba.mnnvoicechat/files/MNN`
4. App 内置的 ModelScope 下载目录

模型根目录支持平铺、`MNN/` 和 `ModelScope/MNN/` 三种布局。最简单的目录结构是：

```text
MNN/
├── Qwen3-0.6B-MNN/config.json
├── sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/
│   ├── encoder-epoch-99-avg-1.int8.mnn
│   ├── decoder-epoch-99-avg-1.int8.mnn
│   ├── joiner-epoch-99-avg-1.int8.mnn
│   └── tokens.txt
└── bert-vits2-MNN/config.json
```

每个目录中还需要保留模型仓库里的其他权重、词表和配置文件。

推荐使用 App 专属外部目录。安装并启动 App 一次后执行：

```bash
adb shell mkdir -p /sdcard/Android/data/com.alibaba.mnnvoicechat/files/MNN
adb push /path/to/MNN/. /sdcard/Android/data/com.alibaba.mnnvoicechat/files/MNN/
adb shell am start -S \
  -n com.alibaba.mnnvoicechat/com.alibaba.mnntalk.MainActivity
```

也可以推送到 `/data/local/tmp/MNN`：

```bash
adb push /path/to/MNN /data/local/tmp/
adb shell chmod -R a+rX /data/local/tmp/MNN
adb shell am start -S \
  -n com.alibaba.mnnvoicechat/com.alibaba.mnntalk.MainActivity
```

部分系统会通过 SELinux 禁止普通 App 读取 `/data/local/tmp`，即使 Unix 权限已经放开。这种设备请使用上面的 App 专属外部目录。

指定任意其他可读目录时，通过启动参数传入；该设置会持久化：

```bash
adb shell am start -S \
  -n com.alibaba.mnnvoicechat/com.alibaba.mnntalk.MainActivity \
  --es mnn_model_dir /sdcard/my-models/MNN
```

清除持久化目录并恢复自动查找：

```bash
adb shell am start -S \
  -n com.alibaba.mnnvoicechat/com.alibaba.mnntalk.MainActivity \
  --ez clear_mnn_model_dir true
```

成功使用开发者目录时，页面顶部会显示“开发目录 · 纯端侧”，不会触发模型下载。

## 代码入口

| 文件 | 职责 |
|---|---|
| `MainActivity.kt` | 单页 UI、权限与生命周期 |
| `LocalVoiceChatEngine.kt` | ASR → LLM → TTS 状态机 |
| `StreamingAsr.kt` | 默认流式 ASR 适配器、麦克风与端点检测 |
| `LocalLlm.kt` | 极小 JNI 接口 |
| `local_llm_jni.cpp` | MNN LLM 加载、流式 token、历史和打断 |
| `SentenceChunker.kt` | 流式文本的 TTS 分句 |
| `VoiceModels.kt` | 默认模型包下载和路径解析 |

## 更换模型

`VoiceModelBundleManager` 中的三个 `VoiceModelSpec` 只定义默认下载组合，不是核心链路的固定依赖。
替换同一运行时支持的模型时，调整模型 ID、目录解析和对应配置即可；接入其他 ASR/TTS
实现时，替换 `StreamingAsr` 或 TTS 适配器，`LocalVoiceChatEngine` 的 ASR → LLM → TTS
状态流无需变化。LLM 目录需要包含 MNN LLM `config.json`，ASR 与 TTS 目录需要符合各自适配器
的模型目录约定。

本 Demo 默认关闭 Qwen3 thinking，以降低语音首响延迟，并通过系统提示要求模型输出适合朗读的简短纯文本。

## 设计限制

- 默认只构建 `arm64-v8a`。
- 首次模型下载需要网络；完成后推理不需要网络。
- 回声消除效果取决于设备实现。嘈杂环境下仍可点击“结束对话”立即停止生成和播放。
- 这是集成参考实现，不包含生产应用所需的隐私协议、后台下载通知、模型许可展示和完备的异常恢复。
