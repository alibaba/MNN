# MNN CocoaPods 支持

MNN 支持通过 CMake 自动生成 iOS CocoaPods 的 podspec 文件，方便集成到 iOS 项目中。

## 概述

`MNN_GENERATE_PODSPEC` 是一个实验性选项，用于从 CMake 配置自动生成 iOS CocoaPods 的 podspec 文件。启用后，CMake 会自动扫描所有 MNN target 的源文件，生成完整的 `MNN.podspec`。

## 使用方式

### 方式一：使用脚本（推荐）

```bash
# 生成 podspec
./tools/script/generate_podspec.sh

# 生成并验证
./tools/script/generate_podspec.sh --validate

# 生成并完整验证（需要 Xcode）
./tools/script/generate_podspec.sh --validate-full

# 检查 podspec 是否与 CMake 配置同步
./tools/script/generate_podspec.sh --check

# 指定版本号
./tools/script/generate_podspec.sh -v 3.3.0

# CI 模式（无颜色输出）
./tools/script/generate_podspec.sh --validate --ci
```

### 方式二：手动 CMake 构建

```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/ios.toolchain.cmake \
    -DMNN_METAL=ON \
    -DMNN_AAPL_FMWK=ON \
    -DMNN_BUILD_SHARED_LIBS=OFF \
    -DMNN_SEP_BUILD=OFF \
    -DMNN_GENERATE_PODSPEC=ON
make -j8
```

构建完成后，`MNN.podspec` 会生成在项目根目录。

## 脚本参数说明

| 参数 | 说明 |
|------|------|
| `--version`, `-v` | 覆盖 podspec 版本号（默认使用 CMakeLists.txt 中的 MNN_VERSION） |
| `--check`, `-c` | 检查现有 podspec 是否与 CMake 配置同步（用于 CI） |
| `--validate` | 生成后执行快速语法验证 |
| `--validate-full` | 生成后执行完整验证（需要 Xcode，耗时较长） |
| `--ci` | CI 模式，禁用颜色输出，严格退出码 |
| `--help`, `-h` | 显示帮助信息 |

## 生成的 podspec 包含内容

- **平台**：iOS 11.0+
- **架构**：arm64
- **框架依赖**：Metal、Accelerate、CoreVideo、Foundation
- **弱依赖**：MetalPerformanceShaders
- **源文件**：自动从 CMake targets 提取
- **ARM82 子规范**：FP16 计算支持（需要 `-march=armv8.2-a+fp16`）
- **ARC 配置**：自动识别 .mm 文件

## 发布到 CocoaPods

```bash
# 1. 验证 podspec
pod spec lint MNN.podspec --allow-warnings

# 2. 推送到私有 spec repo
pod repo push my-specs MNN.podspec --allow-warnings

# 3. 推送到 CocoaPods trunk（需要权限）
pod trunk push MNN.podspec --allow-warnings
```

## 在 Podfile 中使用

```ruby
# 使用本地 podspec
pod 'MNN', :path => './path/to/MNN'

# 使用 Git 仓库
pod 'MNN', :git => 'https://github.com/alibaba/MNN.git', :tag => '3.3.0'

# 使用 CocoaPods trunk
pod 'MNN', '~> 3.3.0'
```

## 注意事项

1. **仅支持 iOS**：当前实现仅支持 iOS 平台，不支持 macOS 或 Catalyst
2. **Metal 后端**：生成的 podspec 默认启用 Metal 后端
3. **ARM82 子规范**：FP16 计算作为子规范提供，需要特殊编译标志
4. **同步检查**：修改 CMake 配置后，请重新生成 podspec 并保持同步
5. **错误处理**：如果生成失败，请检查 CMake 配置是否完整，特别是 MNN_TARGETS 是否正确定义