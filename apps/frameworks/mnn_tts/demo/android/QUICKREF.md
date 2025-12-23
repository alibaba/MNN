# MNN TTS Android Demo - 快速参考

## 🚀 快速构建

```bash
cd /Users/songjinde/git/MNNX/MNN/apps/frameworks/mnn_tts/demo/android

# 使用构建脚本 (推荐)
./build.sh              # 构建 Debug APK
./build.sh release      # 构建 Release APK
./build.sh install      # 构建并安装到设备
./build.sh clean        # 清理构建

# 使用 Gradle 命令
./gradlew assembleDebug     # 构建 Debug
./gradlew assembleRelease   # 构建 Release
./gradlew installDebug      # 安装 Debug
./gradlew clean             # 清理
```

## 📦 构建输出

| 构建类型 | APK 路径 | 大小 |
|---------|---------|------|
| Debug   | `build/outputs/apk/debug/MNNTTSDemo-arm64-v8a-debug.apk` | ~15 MB |
| Release | `build/outputs/apk/release/MNNTTSDemo-arm64-v8a-release-unsigned.apk` | ~8 MB |

## 📱 设备要求

- **最低版本**: Android 5.0 (API 21)
- **目标版本**: Android 14 (API 35)
- **架构**: ARM64 (arm64-v8a)
- **权限**: 无特殊权限要求

## 🛠️ 开发工具

| 工具 | 版本 |
|-----|------|
| Android Studio | Arctic Fox+ |
| Gradle | 8.9 |
| NDK | 27.2.12479018 |
| CMake | 3.22.1+ |
| Kotlin | 1.9.22 |
| JDK | 17+ |

## 📂 项目配置文件

| 文件 | 用途 |
|-----|------|
| `build.gradle` | 应用构建配置 |
| `settings.gradle` | 项目模块配置 |
| `CMakeLists.txt` | 原生代码构建配置 |
| `local.properties` | 本地 SDK/NDK 路径 |
| `gradle.properties` | Gradle 属性配置 |

## 🔧 常用命令

### Gradle 任务

```bash
./gradlew tasks                 # 查看所有任务
./gradlew build                 # 完整构建
./gradlew clean build           # 清理并构建
./gradlew assembleDebug --info  # 详细构建日志
./gradlew assembleDebug --scan  # 构建分析
```

### ADB 命令

```bash
# 安装
adb install -r build/outputs/apk/debug/MNNTTSDemo-arm64-v8a-debug.apk

# 卸载
adb uninstall com.alibaba.mnn.tts.demo

# 启动
adb shell am start -n com.alibaba.mnn.tts.demo/.MainActivity

# 停止
adb shell am force-stop com.alibaba.mnn.tts.demo

# 查看日志
adb logcat -s MNN_TTS:* AndroidRuntime:E

# 清除数据
adb shell pm clear com.alibaba.mnn.tts.demo
```

## 🐛 调试技巧

### 查看构建配置

```bash
./gradlew app:dependencies     # 查看依赖树
./gradlew :mnn_tts:tasks       # 查看库模块任务
```

### 检查 APK 内容

```bash
unzip -l build/outputs/apk/debug/MNNTTSDemo-arm64-v8a-debug.apk
```

### 查看 APK 信息

```bash
aapt dump badging build/outputs/apk/debug/MNNTTSDemo-arm64-v8a-debug.apk
```

## 🔍 故障排查

### 问题: MNN 库未找到

```bash
# 检查库是否存在
ls -la ../../../project/android/build_64/lib/libMNN.so

# 如果不存在,构建 MNN 库
cd ../../../project/android
./build_64.sh
```

### 问题: NDK 未配置

```bash
# 创建或编辑 local.properties
echo "ndk.dir=$HOME/Library/Android/sdk/ndk/27.2.12479018" >> local.properties
echo "sdk.dir=$HOME/Library/Android/sdk" >> local.properties
```

### 问题: Gradle 同步失败

```bash
# 清理并重新同步
./gradlew clean
rm -rf .gradle build
./gradlew build --refresh-dependencies
```

## 📊 构建时间

| 操作 | 预计时间 |
|-----|---------|
| Clean | ~5 秒 |
| 首次构建 | ~2-3 分钟 |
| 增量构建 | ~30-60 秒 |
| 安装到设备 | ~10 秒 |

## 🎯 关键文件

```
demo/android/
├── build.sh                    # 构建脚本 ⭐
├── BUILD.md                    # 详细构建文档 📄
├── README.md                   # 快速开始 📖
├── QUICKREF.md                 # 本文件 📋
├── build.gradle                # 构建配置 ⚙️
├── settings.gradle             # 项目设置 ⚙️
└── src/main/
    ├── java/                   # Kotlin 代码
    ├── res/                    # 资源文件
    └── AndroidManifest.xml     # 清单文件
```

## 🔗 相关链接

- **MNN 文档**: https://www.yuque.com/mnn/cn
- **Android 开发**: https://developer.android.com
- **Gradle 文档**: https://docs.gradle.org
- **Kotlin 文档**: https://kotlinlang.org

## 📝 版本历史

- **v1.0** (2025-12-21): 初始版本,支持 BertVits2 和 Supertonic TTS

---

**提示**: 详细的构建说明请参考 [BUILD.md](BUILD.md)
