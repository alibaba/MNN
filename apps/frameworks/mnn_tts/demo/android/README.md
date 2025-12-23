# MNN TTS Android Demo - 快速开始

## 一键构建

### 前置条件

1. 安装 Android Studio
2. 安装 NDK 27.2.12479018
3. 确保 MNN 库已构建 (位于 `../../../project/android/build_64/lib/libMNN.so`)

### 构建命令

```bash
cd /Users/songjinde/git/MNNX/MNN/apps/frameworks/mnn_tts/demo/android

# 清理构建
./gradlew clean

# 构建 Debug APK
./gradlew assembleDebug

# 安装到设备
./gradlew installDebug
```

### 输出位置

```
build/outputs/apk/debug/MNNTTSDemo-arm64-v8a-debug.apk
```

## 系统要求

- **最低 Android 版本**: Android 5.0 (API 21)
- **目标 Android 版本**: Android 14 (API 35)
- **支持架构**: ARM64 (arm64-v8a)
- **APK 大小**: 约 15 MB

## 功能特性

- ✅ BertVits2 TTS 语音合成
- ✅ Supertonic TTS 语音合成
- ✅ 多模型支持
- ✅ 实时音频播放
- ✅ 模型列表管理

## 项目结构

```
demo/android/
├── build.gradle          # 应用构建配置
├── settings.gradle       # Gradle 项目设置
├── src/main/
│   ├── java/            # Kotlin 源代码
│   │   └── com/alibaba/mnn/tts/demo/
│   │       ├── MainActivity.kt
│   │       ├── ModelAdapter.kt
│   │       └── audio/AudioChunksPlayer.kt
│   ├── res/             # Android 资源
│   └── AndroidManifest.xml
└── build/               # 构建输出
```

## 依赖模块

```
:app (demo)
└── :mnn_tts (库模块)
    └── MNN (原生库)
```

## 常见问题

### Q: 构建失败,提示 NDK 未找到?
A: 在项目根目录创建 `local.properties` 文件,添加:
```properties
ndk.dir=/path/to/your/ndk
sdk.dir=/path/to/your/sdk
```

### Q: 运行时提示库文件未找到?
A: 先构建 MNN 核心库:
```bash
cd ../../../project/android
./build_64.sh
```

### Q: 安装失败,提示 INSTALL_FAILED_NO_MATCHING_ABIS?
A: 确保设备是 ARM64 架构,或修改 build.gradle 添加其他 ABI 支持。

## 更多文档

详细的构建文档请查看: [BUILD.md](BUILD.md)

## 技术栈

- **语言**: Kotlin + C++17
- **构建工具**: Gradle 8.9 + CMake 3.22.1
- **框架**: MNN (Mobile Neural Network)
- **UI**: Material Design Components

## 开发者信息

基于 MNN 深度学习框架开发的 TTS 演示应用。

---

**构建时间**: 约 1 分钟
**最后验证**: 2025-12-21 ✅
