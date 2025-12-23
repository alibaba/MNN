# MNN TTS Android Demo 构建文档

## 项目概述

MNN TTS Android Demo 是基于 MNN (Mobile Neural Network) 框架的文本转语音 (Text-to-Speech) 演示应用。该应用展示了如何在 Android 平台上使用 MNN TTS SDK 进行语音合成。

## 项目结构

```
mnn_tts/
├── android/                    # MNN TTS Android 库模块
│   ├── build.gradle           # 库模块构建配置
│   ├── java/                  # Java/Kotlin 源代码
│   └── src/                   # 原生 C++ 源代码
├── demo/android/              # Android Demo 应用
│   ├── build.gradle           # 应用构建配置
│   ├── settings.gradle        # Gradle 项目设置
│   ├── src/                   # 应用源代码
│   │   └── main/
│   │       ├── java/          # Kotlin 源代码
│   │       └── res/           # Android 资源文件
│   └── build/                 # 构建输出目录
├── include/                   # C++ 头文件
├── src/                       # C++ 源代码实现
└── CMakeLists.txt            # CMake 构建配置
```

## 前置要求

### 必需的软件和工具

1. **Android Studio** (推荐版本: Arctic Fox 或更高)
   - 下载地址: https://developer.android.com/studio

2. **Android SDK**
   - Compile SDK: 35
   - Min SDK: 21 (Android 5.0)
   - Target SDK: 35
   - Build Tools: 最新版本

3. **Android NDK**
   - 版本: 27.2.12479018 (推荐)
   - NDK 用于编译 C++ 代码

4. **Java Development Kit (JDK)**
   - 版本: JDK 17 或更高
   - 用于 Gradle 构建

5. **Gradle**
   - 版本: 8.9 (通过 Gradle Wrapper 自动管理)

6. **CMake**
   - 版本: 3.22.1 或更高
   - 用于构建原生 C++ 代码

### 依赖的 MNN 库

项目依赖于预编译的 MNN 静态库,位置:
```
/Users/songjinde/git/MNNX/MNN/project/android/build_64/lib/libMNN.so
```

如果该库不存在,需要先构建 MNN 核心库:
```bash
cd /Users/songjinde/git/MNNX/MNN/project/android
./build_64.sh
```

## 构建步骤

### 方法 1: 使用 Gradle 命令行 (推荐)

1. **进入项目目录**
   ```bash
   cd /Users/songjinde/git/MNNX/MNN/apps/frameworks/mnn_tts/demo/android
   ```

2. **清理之前的构建 (可选)**
   ```bash
   ./gradlew clean
   ```

3. **构建 Debug APK**
   ```bash
   ./gradlew assembleDebug
   ```

4. **构建 Release APK**
   ```bash
   ./gradlew assembleRelease
   ```

5. **查看构建输出**
   ```bash
   ls -lh build/outputs/apk/debug/
   ```

   生成的 APK 文件:
   - **Debug**: `build/outputs/apk/debug/MNNTTSDemo-arm64-v8a-debug.apk`
   - **Release**: `build/outputs/apk/release/MNNTTSDemo-arm64-v8a-release.apk`

### 方法 2: 使用 Android Studio

1. **打开项目**
   - 启动 Android Studio
   - 选择 "Open an Existing Project"
   - 导航到 `/Users/songjinde/git/MNNX/MNN/apps/frameworks/mnn_tts/demo/android`
   - 点击 "OK"

2. **Gradle 同步**
   - Android Studio 会自动开始 Gradle 同步
   - 如果没有自动同步,点击 "File" > "Sync Project with Gradle Files"

3. **配置构建变体**
   - 在左下角选择 "Build Variants"
   - 选择 "debug" 或 "release"

4. **构建 APK**
   - 点击 "Build" > "Build Bundle(s) / APK(s)" > "Build APK(s)"
   - 或者使用快捷键: Ctrl+Shift+A (Windows/Linux) 或 Cmd+Shift+A (Mac)

5. **查看构建结果**
   - 构建成功后会显示通知
   - 点击 "locate" 查看 APK 文件位置

## 构建配置说明

### 应用配置 (demo/android/build.gradle)

```gradle
android {
    namespace 'com.alibaba.mnn.tts.demo'
    compileSdk 35                    // 编译 SDK 版本

    defaultConfig {
        applicationId "com.alibaba.mnn.tts.demo"
        minSdk 21                    // 最低支持 Android 5.0
        targetSdk 35                 // 目标 SDK
        versionCode 1                // 应用版本号
        versionName "1.0"            // 应用版本名称
    }

    splits {
        abi {
            enable true
            reset()
            include 'arm64-v8a'      // 仅构建 ARM64 版本
            universalApk false        // 不生成通用 APK
        }
    }
}
```

### 库配置 (android/build.gradle)

```gradle
android {
    namespace 'com.alibaba.mnn.tts'
    compileSdk 34
    ndkVersion "27.2.12479018"       // NDK 版本

    externalNativeBuild {
        cmake {
            path file('../CMakeLists.txt')  // CMake 配置文件
            version '3.22.1'                 // CMake 版本
        }
    }
}
```

### CMake 配置 (CMakeLists.txt)

关键配置选项:
- `BUILD_BERTVITS2`: 构建 BertVits2 TTS (默认 ON)
- `BUILD_PIPER`: 构建 PIPER TTS (默认 OFF)
- `BUILD_SUPERTONIC`: 构建 Supertonic TTS (默认 ON)
- `BUILD_ANDROID`: Android 平台标志 (自动检测)

## 依赖库说明

### Android 依赖

```gradle
dependencies {
    implementation project(':mnn_tts')              // MNN TTS 库
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'com.google.android.material:material:1.10.0'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'
    implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.7.0'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
    implementation 'androidx.core:core-ktx:1.16.0'
    implementation 'androidx.recyclerview:recyclerview:1.3.2'
    implementation 'androidx.cardview:cardview:1.0.0'
}
```

### 原生库

- **libMNN.so**: MNN 核心推理引擎
- **libmnn_tts.so**: MNN TTS SDK 实现
- **libc++_shared.so**: C++ 标准库

## 安装和运行

### 安装到设备

1. **使用 Gradle 命令**
   ```bash
   ./gradlew installDebug
   ```

2. **使用 adb 命令**
   ```bash
   adb install build/outputs/apk/debug/MNNTTSDemo-arm64-v8a-debug.apk
   ```

3. **使用 Android Studio**
   - 点击工具栏的 "Run" 按钮 (绿色三角形)
   - 选择目标设备
   - 应用会自动安装并启动

### 运行应用

1. **启动应用**
   - 在设备上找到 "MNNTTSDemo" 应用图标
   - 点击启动

2. **使用 adb 启动**
   ```bash
   adb shell am start -n com.alibaba.mnn.tts.demo/.MainActivity
   ```

## 常见问题和解决方案

### 1. NDK 未找到

**错误信息**: NDK not configured

**解决方案**:
```bash
# 在 local.properties 中配置 NDK 路径
echo "ndk.dir=/Users/songjinde/Library/Android/sdk/ndk/27.2.12479018" >> local.properties
```

### 2. MNN 库未找到

**错误信息**: libMNN.so not found

**解决方案**:
```bash
# 先构建 MNN 核心库
cd /Users/songjinde/git/MNNX/MNN/project/android
./build_64.sh
```

### 3. Gradle 同步失败

**错误信息**: Failed to sync Gradle project

**解决方案**:
```bash
# 清理 Gradle 缓存
./gradlew clean
rm -rf .gradle
./gradlew build --refresh-dependencies
```

### 4. CMake 构建失败

**错误信息**: CMake build failed

**解决方案**:
- 检查 NDK 版本是否正确
- 确保 CMake 版本 >= 3.22.1
- 检查 MNN 库是否存在

### 5. ABI 不匹配

**错误信息**: INSTALL_FAILED_NO_MATCHING_ABIS

**解决方案**:
- 应用仅支持 ARM64 (arm64-v8a) 设备
- 确保测试设备是 ARM64 架构
- 或修改 build.gradle 添加其他 ABI 支持

## 性能优化建议

### Release 构建优化

1. **启用代码混淆**
   ```gradle
   buildTypes {
       release {
           minifyEnabled true
           proguardFiles getDefaultProguardFile('proguard-android-optimize.txt')
       }
   }
   ```

2. **启用资源缩减**
   ```gradle
   buildTypes {
       release {
           shrinkResources true
       }
   }
   ```

3. **使用 Release NDK 构建**
   - Release 构建会自动使用优化的原生库

### 运行时优化

1. **模型加载**: 首次加载模型时间较长,建议使用异步加载
2. **内存管理**: 及时释放不再使用的模型资源
3. **线程池**: 使用合理的线程数量进行推理

## 技术架构

### 应用架构

```
MainActivity.kt
├── ModelAdapter.kt          # 模型列表适配器
├── AudioChunksPlayer.kt     # 音频播放器
└── MNN TTS SDK
    ├── BertVits2 TTS        # BertVits2 语音合成
    ├── Supertonic TTS       # Supertonic 语音合成
    └── MNN Engine           # MNN 推理引擎
```

### 关键功能

1. **文本转语音**: 输入文本,生成语音音频
2. **模型管理**: 支持多种 TTS 模型切换
3. **音频播放**: 实时播放生成的语音
4. **性能监控**: 显示推理时间和资源使用

## 调试技巧

### 查看日志

```bash
# 查看应用日志
adb logcat -s MNN_TTS:* AndroidRuntime:E

# 查看原生日志
adb logcat -s DEBUG:* native:*
```

### 性能分析

1. **使用 Android Profiler**
   - 在 Android Studio 中打开 "View" > "Tool Windows" > "Profiler"
   - 监控 CPU、内存和网络使用

2. **使用 Systrace**
   ```bash
   python systrace.py -t 10 -o trace.html sched freq idle
   ```

## 参考资源

- **MNN 官方文档**: https://www.yuque.com/mnn/cn
- **MNN GitHub**: https://github.com/alibaba/MNN
- **Android 开发指南**: https://developer.android.com/guide
- **NDK 开发指南**: https://developer.android.com/ndk

## 版本信息

- **应用版本**: 1.0
- **MNN 版本**: Latest
- **最低 Android 版本**: 5.0 (API 21)
- **目标 Android 版本**: 14.0 (API 35)
- **支持的架构**: ARM64 (arm64-v8a)

## 许可证

本项目遵循 MNN 项目的许可证条款。

## 联系方式

如有问题或建议,请联系 MNN 项目维护者或提交 Issue。

---

**最后更新**: 2025-12-21
**构建状态**: ✅ 成功
**生成的 APK**: `build/outputs/apk/debug/MNNTTSDemo-arm64-v8a-debug.apk` (15 MB)
