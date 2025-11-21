#!/bin/bash

# MNN 统一构建脚本
# 支持构建 Android、iOS、鸿蒙和 Python 版本的 MNN

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
BUILD_ANDROID=false
BUILD_IOS=false
BUILD_PYTHON=false
BUILD_IOS_SIMULATOR=false
BUILD_HARMONY=false
ANDROID_NDK=""
HARMONY_HOME=""
OUTPUT_DIR="mnn_builds"
CLEAN_BUILD=true
VERSION=""
PYTHON_CMD="python3"
DEPS_OPTIONS=""

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

print_usage() {
    echo "MNN 统一构建脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "构建目标:"
    echo "  --android                 构建 Android 版本 (需要 --ndk)"
    echo "  --ios                     构建 iOS 真机版本 (arm64)"
    echo "  --ios-simulator           构建 iOS 模拟器版本 (x86_64 + arm64)"
    echo "  --harmony                 构建鸿蒙版本 (需要 --harmony-home 或设置 HARMONY_HOME)"
    echo "  --python                  构建 Python 版本"
    echo ""
    echo "Android 选项:"
    echo "  --ndk PATH                Android NDK 路径 (例如: ~/Library/Android/sdk/ndk/29.0.13599879)"
    echo ""
    echo "鸿蒙选项:"
    echo "  --harmony-home PATH       鸿蒙工具链路径 (例如: ~/Library/OpenHarmony/Sdk/native)"
    echo "                           如果未指定，将优先查找 ~/Library/OpenHarmony/Sdk/native"
    echo ""
    echo "Python 选项:"
    echo "  --python-deps OPTIONS     Python 依赖选项，多个用逗号分隔"
    echo "                           可用选项: llm,opencl,cuda,torch,render,vulkan,internal,no_sse,openmp"
    echo "  --python-cmd CMD         Python 命令 (默认: python3)"
    echo ""
    echo "通用选项:"
    echo "  -o, --output DIR         输出目录 (默认: mnn_builds)"
    echo "  -v, --version VERSION    版本号 (默认: 自动从源码读取)"
    echo "  --no-clean               不清理之前的构建目录"
    echo "  -h, --help               显示帮助信息"
    echo ""
    echo "示例:"
    echo "  # 构建所有平台"
    echo "  $0 --android --ios --harmony --python --ndk ~/Library/Android/sdk/ndk/29.0.13599879"
    echo ""
    echo "  # 仅构建 Android"
    echo "  $0 --android --ndk ~/Library/Android/sdk/ndk/29.0.13599879"
    echo ""
    echo "  # 构建鸿蒙版本"
    echo "  $0 --harmony --harmony-home ~/Library/OpenHarmony/Sdk/native"
    echo ""
    echo "  # 构建 iOS 真机和模拟器"
    echo "  $0 --ios --ios-simulator"
    echo ""
    echo "  # 构建 Python (带 LLM 支持)"
    echo "  $0 --python --python-deps llm,opencl"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --android)
            BUILD_ANDROID=true
            shift
            ;;
        --ios)
            BUILD_IOS=true
            shift
            ;;
        --ios-simulator)
            BUILD_IOS_SIMULATOR=true
            shift
            ;;
        --python)
            BUILD_PYTHON=true
            shift
            ;;
        --harmony)
            BUILD_HARMONY=true
            shift
            ;;
        --ndk)
            ANDROID_NDK="$2"
            shift 2
            ;;
        --harmony-home)
            HARMONY_HOME="$2"
            shift 2
            ;;
        --python-deps)
            DEPS_OPTIONS="$2"
            shift 2
            ;;
        --python-cmd)
            PYTHON_CMD="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --no-clean)
            CLEAN_BUILD=false
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# 检查是否至少选择了一个构建目标
if [ "$BUILD_ANDROID" = false ] && [ "$BUILD_IOS" = false ] && [ "$BUILD_IOS_SIMULATOR" = false ] && [ "$BUILD_HARMONY" = false ] && [ "$BUILD_PYTHON" = false ]; then
    echo -e "${RED}错误: 请至少选择一个构建目标 (--android, --ios, --ios-simulator, --harmony, 或 --python)${NC}"
    print_usage
    exit 1
fi

# 检查 Android NDK
if [ "$BUILD_ANDROID" = true ]; then
    if [ -z "$ANDROID_NDK" ]; then
        echo -e "${RED}错误: 构建 Android 必须指定 NDK 路径 (使用 --ndk)${NC}"
        echo -e "${RED}示例: $0 --android --ndk ~/Library/Android/sdk/ndk/29.0.13599879${NC}"
        exit 1
    fi
    
    # 展开路径中的 ~
    ANDROID_NDK="${ANDROID_NDK/#\~/$HOME}"
    
    if [ ! -d "$ANDROID_NDK" ]; then
        echo -e "${RED}错误: NDK 路径不存在: $ANDROID_NDK${NC}"
        exit 1
    fi
fi

# 查找鸿蒙工具链的函数
find_harmony_toolchain() {
    # 如果环境变量已设置，优先使用
    if [ -n "$HARMONY_HOME" ]; then
        # 支持两种路径格式：直接指定或带 native 子目录
        if [ -f "$HARMONY_HOME/build/cmake/ohos.toolchain.cmake" ]; then
            echo "$HARMONY_HOME"
            return 0
        elif [ -f "$HARMONY_HOME/native/build/cmake/ohos.toolchain.cmake" ]; then
            echo "$HARMONY_HOME"
            return 0
        fi
    fi
    
    # 优先查找 ~/Library/OpenHarmony/Sdk，支持版本号目录
    local sdk_base="$HOME/Library/OpenHarmony/Sdk"
    if [ -d "$sdk_base" ]; then
        # 查找所有版本号目录下的 native 或 toolchains
        # 按版本号倒序排列，优先使用最新版本
        for version_dir in $(ls -d "$sdk_base"/* 2>/dev/null | sort -Vr); do
            if [ -d "$version_dir" ]; then
                # 尝试 native/build/cmake/ohos.toolchain.cmake
                if [ -f "$version_dir/native/build/cmake/ohos.toolchain.cmake" ]; then
                    echo "$version_dir/native"
                    return 0
                fi
                # 尝试 build/cmake/ohos.toolchain.cmake
                if [ -f "$version_dir/build/cmake/ohos.toolchain.cmake" ]; then
                    echo "$version_dir"
                    return 0
                fi
            fi
        done
    fi
    
    # 其他可能的路径
    local possible_paths=(
        "$HOME/Library/OpenHarmony/Sdk/native"
        "$HOME/HarmonyOS/Sdk/native"
        "$HOME/.ohos/native"
        "/opt/HarmonyOS/Sdk/native"
        "/usr/local/HarmonyOS/Sdk/native"
        "$HOME/Library/HarmonyOS/Sdk/native"
    )
    
    # 尝试查找
    for path in "${possible_paths[@]}"; do
        if [ -n "$path" ] && [ -f "$path/build/cmake/ohos.toolchain.cmake" ]; then
            echo "$path"
            return 0
        fi
    done
    
    # 限制搜索范围，只在 OpenHarmony/Sdk 目录下快速查找
    # 避免在整个 OpenHarmony 目录下递归查找，这可能会很慢
    local found=$(find "$HOME/Library/OpenHarmony/Sdk" -maxdepth 4 -type f -name "ohos.toolchain.cmake" 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        # 从 ohos.toolchain.cmake 向上查找 native 目录或 SDK 根目录
        found=$(dirname "$found")
        if [ "$(basename "$found")" = "cmake" ]; then
            found=$(dirname "$found")
            if [ "$(basename "$found")" = "build" ]; then
                found=$(dirname "$found")
            fi
        fi
        echo "$found"
        return 0
    fi
    
    return 1
}

# 检查鸿蒙工具链
if [ "$BUILD_HARMONY" = true ]; then
    # 展开路径中的 ~
    if [ -n "$HARMONY_HOME" ]; then
        HARMONY_HOME="${HARMONY_HOME/#\~/$HOME}"
    fi
    
    # 尝试查找工具链
    if [ -z "$HARMONY_HOME" ] || [ ! -f "$HARMONY_HOME/build/cmake/ohos.toolchain.cmake" ]; then
        # 检查是否在 native 子目录下
        if [ -n "$HARMONY_HOME" ] && [ -f "$HARMONY_HOME/native/build/cmake/ohos.toolchain.cmake" ]; then
            HARMONY_HOME="$HARMONY_HOME/native"
        else
            echo -e "${YELLOW}正在查找鸿蒙工具链...${NC}"
            HARMONY_HOME=$(find_harmony_toolchain)
            
            if [ -z "$HARMONY_HOME" ]; then
                echo -e "${RED}错误: 找不到鸿蒙工具链${NC}"
                echo -e "${RED}默认查找路径: ~/Library/OpenHarmony/Sdk/*/native${NC}"
                echo -e "${RED}请使用 --harmony-home 指定路径，或设置 HARMONY_HOME 环境变量${NC}"
                echo -e "${RED}工具链文件应位于: <HARMONY_HOME>/build/cmake/ohos.toolchain.cmake 或 <HARMONY_HOME>/native/build/cmake/ohos.toolchain.cmake${NC}"
                exit 1
            fi
            
            # 如果找到的是带 native 的路径，需要调整
            if [ -f "$HARMONY_HOME/native/build/cmake/ohos.toolchain.cmake" ]; then
                HARMONY_HOME="$HARMONY_HOME/native"
            fi
        fi
        
        echo -e "${GREEN}找到鸿蒙工具链: $HARMONY_HOME${NC}"
    fi
    
    # 验证工具链文件存在
    if [ ! -f "$HARMONY_HOME/build/cmake/ohos.toolchain.cmake" ]; then
        echo -e "${RED}错误: 鸿蒙工具链文件不存在: $HARMONY_HOME/build/cmake/ohos.toolchain.cmake${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MNN 统一构建脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo "项目根目录: $PROJECT_ROOT"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo -e "${BLUE}构建目标:${NC}"
[ "$BUILD_ANDROID" = true ] && echo "  ✓ Android"
[ "$BUILD_IOS" = true ] && echo "  ✓ iOS (真机 arm64)"
[ "$BUILD_IOS_SIMULATOR" = true ] && echo "  ✓ iOS (模拟器 x86_64 + arm64)"
[ "$BUILD_HARMONY" = true ] && echo "  ✓ 鸿蒙 (arm64-v8a)"
[ "$BUILD_PYTHON" = true ] && echo "  ✓ Python"
echo ""

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# 构建 Android 版本
# ============================================================================
if [ "$BUILD_ANDROID" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}开始构建 Android 版本${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    export ANDROID_NDK
    
    ANDROID_BUILD_DIR="project/android"
    ANDROID_OUTPUT_DIR="$OUTPUT_DIR/android"
    
    cd "$ANDROID_BUILD_DIR"
    
    # 清理
    if [ "$CLEAN_BUILD" = true ]; then
        echo -e "${YELLOW}清理之前的 Android 构建...${NC}"
        rm -rf build_32 build_64 export
    fi
    
    # 在项目根目录创建输出目录
    mkdir -p "$PROJECT_ROOT/$ANDROID_OUTPUT_DIR"
    
    # 构建 armeabi-v7a
    echo -e "${BLUE}构建 armeabi-v7a...${NC}"
    mkdir -p build_32
    cd build_32
    cmake ../../../ \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DANDROID_ABI="armeabi-v7a" \
        -DANDROID_STL=c++_static \
        -DANDROID_NATIVE_API_LEVEL=android-14 \
        -DANDROID_TOOLCHAIN=clang \
        -DMNN_USE_LOGCAT=false \
        -DMNN_USE_SSE=OFF \
        -DMNN_BUILD_TEST=ON \
        -DMNN_ARM82=OFF \
        -DMNN_LOW_MEMORY=ON \
        -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
        -DMNN_BUILD_LLM=ON \
        -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
        -DMNN_BUILD_DIFFUSION=ON \
        -DMNN_OPENCL=OFF \
        -DMNN_SEP_BUILD=OFF \
        -DLLM_SUPPORT_AUDIO=ON \
        -DMNN_BUILD_AUDIO=ON \
        -DLLM_SUPPORT_VISION=ON \
        -DMNN_BUILD_OPENCV=ON \
        -DMNN_IMGCODECS=ON \
        -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
        -DNATIVE_LIBRARY_OUTPUT=. \
        -DNATIVE_INCLUDE_OUTPUT=. \
        > /dev/null
    
    make -j4 MNN > /dev/null
    cd ..
    
    # 构建 arm64-v8a
    echo -e "${BLUE}构建 arm64-v8a...${NC}"
    mkdir -p build_64
    cd build_64
    cmake ../../../ \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DANDROID_ABI="arm64-v8a" \
        -DANDROID_STL=c++_static \
        -DANDROID_NATIVE_API_LEVEL=android-21 \
        -DANDROID_TOOLCHAIN=clang \
        -DMNN_USE_LOGCAT=false \
        -DMNN_BUILD_BENCHMARK=ON \
        -DMNN_USE_SSE=OFF \
        -DMNN_BUILD_TEST=ON \
        -DMNN_ARM82=ON \
        -DMNN_LOW_MEMORY=ON \
        -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
        -DMNN_BUILD_LLM=ON \
        -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
        -DMNN_BUILD_DIFFUSION=ON \
        -DMNN_OPENCL=OFF \
        -DMNN_SEP_BUILD=OFF \
        -DLLM_SUPPORT_AUDIO=ON \
        -DMNN_BUILD_AUDIO=ON \
        -DLLM_SUPPORT_VISION=ON \
        -DMNN_BUILD_OPENCV=ON \
        -DMNN_IMGCODECS=ON \
        -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
        -DNATIVE_LIBRARY_OUTPUT=. \
        -DNATIVE_INCLUDE_OUTPUT=. \
        > /dev/null
    
    make -j4 MNN > /dev/null
    cd ..
    
    # 导出文件
    echo -e "${BLUE}导出 Android 库文件...${NC}"
    mkdir -p export/android/{armeabi-v7a,arm64-v8a}/libs export/android/include/MNN
    
    cp build_32/*.so export/android/armeabi-v7a/libs/ 2>/dev/null || true
    cp build_64/*.so export/android/arm64-v8a/libs/ 2>/dev/null || true
    cp -r ../../include/MNN/* export/android/include/MNN/
    
    # 复制到统一输出目录
    # 如果目标路径存在但不是目录，先删除
    if [ -e "$PROJECT_ROOT/$ANDROID_OUTPUT_DIR" ] && [ ! -d "$PROJECT_ROOT/$ANDROID_OUTPUT_DIR" ]; then
        rm -f "$PROJECT_ROOT/$ANDROID_OUTPUT_DIR"
    fi
    mkdir -p "$PROJECT_ROOT/$ANDROID_OUTPUT_DIR"
    cp -r export/android/* "$PROJECT_ROOT/$ANDROID_OUTPUT_DIR/"
    
    echo -e "${GREEN}Android 构建完成！${NC}"
    echo "输出位置: $ANDROID_OUTPUT_DIR"
    cd "$PROJECT_ROOT"
fi

# ============================================================================
# 构建 iOS 真机版本
# ============================================================================
if [ "$BUILD_IOS" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}开始构建 iOS 真机版本${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # 检查是否在 macOS 上
    if [[ "$OSTYPE" != "darwin"* ]]; then
        echo -e "${RED}错误: iOS 构建只能在 macOS 上执行${NC}"
        exit 1
    fi
    
    # 检查 Xcode
    if ! command -v xcodebuild &> /dev/null; then
        echo -e "${RED}错误: 找不到 Xcode，请先安装 Xcode${NC}"
        exit 1
    fi
    
    IOS_BUILD_DIR="project/ios"
    IOS_OUTPUT_DIR="$OUTPUT_DIR/ios/device"
    
    cd "$IOS_BUILD_DIR"
    
    # 清理
    if [ "$CLEAN_BUILD" = true ]; then
        echo -e "${YELLOW}清理之前的 iOS 真机构建...${NC}"
        rm -rf MNN-iOS-CPU-GPU/Static/ios_64
        find "$PROJECT_ROOT" -name "CMakeCache.txt" -path "*/ios_64/*" 2>/dev/null | xargs rm -f 2>/dev/null || true
    fi
    
    mkdir -p MNN-iOS-CPU-GPU/Static
    cd MNN-iOS-CPU-GPU/Static
    
    # 构建真机版本 (arm64)
    echo -e "${BLUE}构建 iOS 真机版本 (arm64)...${NC}"
    rm -rf ios_64
    mkdir ios_64
    cd ios_64
    cmake "$PROJECT_ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=$PROJECT_ROOT/cmake/ios.toolchain.cmake \
        -DENABLE_BITCODE=0 \
        -DMNN_AAPL_FMWK=1 \
        -DMNN_SEP_BUILD=OFF \
        -DMNN_BUILD_SHARED_LIBS=false \
        -DMNN_USE_THREAD_POOL=OFF \
        -DPLATFORM=OS64 \
        -DARCHS="arm64" \
        -DMNN_ARM82=ON \
        -DMNN_LOW_MEMORY=ON \
        -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
        -DMNN_BUILD_LLM=ON \
        -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
        -DMNN_METAL=ON \
        -DMNN_BUILD_DIFFUSION=ON \
        -DMNN_OPENCL=OFF \
        -DLLM_SUPPORT_AUDIO=ON \
        -DMNN_BUILD_AUDIO=ON \
        -DLLM_SUPPORT_VISION=ON \
        -DMNN_BUILD_OPENCV=ON \
        -DMNN_IMGCODECS=ON \
        > /dev/null
    make MNN -j8 > /dev/null
    cd ..
    
    # 输出真机版本
    mkdir -p "$PROJECT_ROOT/$IOS_OUTPUT_DIR"
    rm -rf "$PROJECT_ROOT/$IOS_OUTPUT_DIR/MNN.framework"
    cp -R ios_64/MNN.framework "$PROJECT_ROOT/$IOS_OUTPUT_DIR/"
    
    # 清理
    rm -rf ios_64
    
    echo -e "${GREEN}iOS 真机构建完成！${NC}"
    echo "输出位置: $IOS_OUTPUT_DIR/MNN.framework"
    cd "$PROJECT_ROOT"
fi

# ============================================================================
# 构建 iOS 模拟器版本
# ============================================================================
if [ "$BUILD_IOS_SIMULATOR" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}开始构建 iOS 模拟器版本${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # 检查是否在 macOS 上
    if [[ "$OSTYPE" != "darwin"* ]]; then
        echo -e "${RED}错误: iOS 构建只能在 macOS 上执行${NC}"
        exit 1
    fi
    
    # 检查 Xcode
    if ! command -v xcodebuild &> /dev/null; then
        echo -e "${RED}错误: 找不到 Xcode，请先安装 Xcode${NC}"
        exit 1
    fi
    
    IOS_BUILD_DIR="project/ios"
    IOS_OUTPUT_DIR="$OUTPUT_DIR/ios/simulator"
    
    cd "$IOS_BUILD_DIR"
    
    # 清理
    if [ "$CLEAN_BUILD" = true ]; then
        echo -e "${YELLOW}清理之前的 iOS 模拟器构建...${NC}"
        rm -rf MNN-iOS-CPU-GPU/Static/ios_simulator*
        find "$PROJECT_ROOT" -name "CMakeCache.txt" -path "*/ios_simulator*/*" 2>/dev/null | xargs rm -f 2>/dev/null || true
    fi
    
    mkdir -p MNN-iOS-CPU-GPU/Static
    cd MNN-iOS-CPU-GPU/Static
    
    # 构建 x86_64 模拟器版本（尝试构建，失败则跳过）
    BUILD_X86_64=false
    echo -e "${BLUE}构建 iOS 模拟器版本 (x86_64)...${NC}"
    rm -rf ios_simulator_x86
    mkdir ios_simulator_x86
    cd ios_simulator_x86
    if cmake "$PROJECT_ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=$PROJECT_ROOT/cmake/ios.toolchain.cmake \
        -DENABLE_BITCODE=0 \
        -DMNN_AAPL_FMWK=1 \
        -DMNN_SEP_BUILD=OFF \
        -DMNN_BUILD_SHARED_LIBS=false \
        -DMNN_USE_THREAD_POOL=OFF \
        -DPLATFORM=SIMULATOR64 \
        -DARCHS="x86_64" \
        -DMNN_ARM82=OFF \
        -DMNN_LOW_MEMORY=ON \
        -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
        -DMNN_BUILD_LLM=ON \
        -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
        -DMNN_METAL=OFF \
        -DMNN_BUILD_DIFFUSION=ON \
        -DMNN_OPENCL=OFF \
        -DLLM_SUPPORT_AUDIO=ON \
        -DMNN_BUILD_AUDIO=ON \
        -DLLM_SUPPORT_VISION=ON \
        -DMNN_BUILD_OPENCV=ON \
        -DMNN_IMGCODECS=ON \
        && make MNN -j8; then
        if [ -f "MNN.framework/MNN" ]; then
            BUILD_X86_64=true
            echo -e "${GREEN}x86_64 模拟器构建成功${NC}"
            cd ..
        else
            echo -e "${YELLOW}警告: x86_64 模拟器构建产物不存在，跳过此架构${NC}"
            cd ..
            rm -rf ios_simulator_x86
        fi
    else
        echo -e "${YELLOW}警告: x86_64 模拟器构建失败，跳过此架构（这在 Apple Silicon Mac 上是正常的）${NC}"
        cd ..
        rm -rf ios_simulator_x86
    fi
    
    # 构建 arm64 模拟器版本
    echo -e "${BLUE}构建 iOS 模拟器版本 (arm64)...${NC}"
    rm -rf ios_simulator_arm64
    mkdir ios_simulator_arm64
    cd ios_simulator_arm64
    if ! cmake "$PROJECT_ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=$PROJECT_ROOT/cmake/ios.toolchain.cmake \
        -DENABLE_BITCODE=0 \
        -DMNN_AAPL_FMWK=1 \
        -DMNN_SEP_BUILD=OFF \
        -DMNN_BUILD_SHARED_LIBS=false \
        -DMNN_USE_THREAD_POOL=OFF \
        -DPLATFORM=SIMULATOR64 \
        -DARCHS="arm64" \
        -DMNN_ARM82=ON \
        -DMNN_LOW_MEMORY=ON \
        -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
        -DMNN_BUILD_LLM=ON \
        -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
        -DMNN_METAL=OFF \
        -DMNN_BUILD_DIFFUSION=ON \
        -DMNN_OPENCL=OFF \
        -DLLM_SUPPORT_AUDIO=ON \
        -DMNN_BUILD_AUDIO=ON \
        -DLLM_SUPPORT_VISION=ON \
        -DMNN_BUILD_OPENCV=ON \
        -DMNN_IMGCODECS=ON; then
        echo -e "${RED}错误: arm64 模拟器 cmake 配置失败${NC}"
        cd "$PROJECT_ROOT"
        exit 1
    fi
    if ! make MNN -j8; then
        echo -e "${RED}错误: arm64 模拟器编译失败${NC}"
        cd "$PROJECT_ROOT"
        exit 1
    fi
    cd ..
    
    # 验证构建产物
    if [ ! -f "ios_simulator_arm64/MNN.framework/MNN" ]; then
        echo -e "${RED}错误: 未找到 arm64 模拟器框架文件${NC}"
        cd "$PROJECT_ROOT"
        exit 1
    fi
    
    # 合并模拟器架构
    echo -e "${BLUE}合并模拟器架构...${NC}"
    rm -rf ios_simulator
    mkdir ios_simulator
    
    if [ "$BUILD_X86_64" = true ] && [ -f "ios_simulator_x86/MNN.framework/MNN" ]; then
        # 合并 x86_64 + arm64
        echo -e "${BLUE}合并 x86_64 + arm64 架构...${NC}"
        cp -R ios_simulator_x86/MNN.framework ios_simulator/MNN.framework
        mv ios_simulator/MNN.framework/MNN ios_simulator/MNN.framework/MNN_x86
        if ! lipo -create ios_simulator/MNN.framework/MNN_x86 ios_simulator_arm64/MNN.framework/MNN -output ios_simulator/MNN.framework/MNN; then
            echo -e "${RED}错误: 合并架构失败${NC}"
            cd "$PROJECT_ROOT"
            exit 1
        fi
        rm ios_simulator/MNN.framework/MNN_x86
    else
        # 仅使用 arm64
        echo -e "${BLUE}仅使用 arm64 架构（x86_64 不可用）...${NC}"
        cp -R ios_simulator_arm64/MNN.framework ios_simulator/MNN.framework
    fi
    
    # 输出模拟器版本
    mkdir -p "$PROJECT_ROOT/$IOS_OUTPUT_DIR"
    rm -rf "$PROJECT_ROOT/$IOS_OUTPUT_DIR/MNN.framework"
    cp -R ios_simulator/MNN.framework "$PROJECT_ROOT/$IOS_OUTPUT_DIR/"
    
    # 清理临时目录
    rm -rf ios_simulator ios_simulator_x86 ios_simulator_arm64
    
    echo -e "${GREEN}iOS 模拟器构建完成！${NC}"
    echo "输出位置: $IOS_OUTPUT_DIR/MNN.framework"
    cd "$PROJECT_ROOT"
fi

# ============================================================================
# 构建鸿蒙版本
# ============================================================================
if [ "$BUILD_HARMONY" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}开始构建鸿蒙版本${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    export HARMONY_HOME
    
    HARMONY_BUILD_DIR="project/harmony"
    HARMONY_OUTPUT_DIR="$OUTPUT_DIR/harmony"
    
    cd "$HARMONY_BUILD_DIR"
    
    # 清理
    if [ "$CLEAN_BUILD" = true ]; then
        echo -e "${YELLOW}清理之前的鸿蒙构建...${NC}"
        rm -rf build_64 export
    fi
    
    # 在项目根目录创建输出目录
    mkdir -p "$PROJECT_ROOT/$HARMONY_OUTPUT_DIR"
    
    # 构建 arm64-v8a
    echo -e "${BLUE}构建 arm64-v8a...${NC}"
    mkdir -p build_64
    cd build_64
    
    cmake "$PROJECT_ROOT" \
        -DCMAKE_TOOLCHAIN_FILE=$HARMONY_HOME/build/cmake/ohos.toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DOHOS_ARCH="arm64-v8a" \
        -DOHOS_STL=c++_static \
        -DMNN_USE_LOGCAT=true \
        -DMNN_BUILD_BENCHMARK=ON \
        -DMNN_USE_SSE=OFF \
        -DMNN_SUPPORT_BF16=OFF \
        -DMNN_BUILD_TEST=ON \
        -DMNN_ARM82=ON \
        -DMNN_LOW_MEMORY=ON \
        -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
        -DMNN_BUILD_LLM=ON \
        -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
        -DMNN_BUILD_DIFFUSION=ON \
        -DMNN_OPENCL=OFF \
        -DMNN_SEP_BUILD=OFF \
        -DLLM_SUPPORT_AUDIO=ON \
        -DMNN_BUILD_AUDIO=ON \
        -DLLM_SUPPORT_VISION=ON \
        -DMNN_BUILD_OPENCV=ON \
        -DMNN_IMGCODECS=ON \
        -DOHOS_PLATFORM_LEVEL=9 \
        -DNATIVE_LIBRARY_OUTPUT=. \
        -DNATIVE_INCLUDE_OUTPUT=. \
        > /dev/null
    
    make -j4 MNN > /dev/null
    cd ..
    
    # 导出文件
    echo -e "${BLUE}导出鸿蒙库文件...${NC}"
    mkdir -p export/harmony/arm64-v8a/libs export/harmony/include/MNN
    
    cp build_64/*.so export/harmony/arm64-v8a/libs/ 2>/dev/null || true
    cp -r ../../include/MNN/* export/harmony/include/MNN/
    
    # 复制到统一输出目录
    # 如果目标路径存在但不是目录，先删除
    if [ -e "$PROJECT_ROOT/$HARMONY_OUTPUT_DIR" ] && [ ! -d "$PROJECT_ROOT/$HARMONY_OUTPUT_DIR" ]; then
        rm -f "$PROJECT_ROOT/$HARMONY_OUTPUT_DIR"
    fi
    mkdir -p "$PROJECT_ROOT/$HARMONY_OUTPUT_DIR"
    cp -r export/harmony/* "$PROJECT_ROOT/$HARMONY_OUTPUT_DIR/"
    
    echo -e "${GREEN}鸿蒙构建完成！${NC}"
    echo "输出位置: $HARMONY_OUTPUT_DIR"
    cd "$PROJECT_ROOT"
fi

# ============================================================================
# 构建 Python 版本
# ============================================================================
if [ "$BUILD_PYTHON" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}开始构建 Python 版本${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # 检查 Python
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo -e "${RED}错误: 找不到 Python 命令 '$PYTHON_CMD'${NC}"
        exit 1
    fi
    
    PYTHON_OUTPUT_DIR="$OUTPUT_DIR/python"
    
    # 使用之前创建的 build_pymnn.sh 脚本
    if [ -f "$PROJECT_ROOT/build_pymnn.sh" ]; then
        PYTHON_BUILD_ARGS="-o $PYTHON_OUTPUT_DIR"
        [ -n "$VERSION" ] && PYTHON_BUILD_ARGS="$PYTHON_BUILD_ARGS -v $VERSION"
        [ -n "$DEPS_OPTIONS" ] && PYTHON_BUILD_ARGS="$PYTHON_BUILD_ARGS -d $DEPS_OPTIONS"
        [ "$CLEAN_BUILD" = false ] && PYTHON_BUILD_ARGS="$PYTHON_BUILD_ARGS --no-clean"
        PYTHON_BUILD_ARGS="$PYTHON_BUILD_ARGS --python $PYTHON_CMD"
        
        bash "$PROJECT_ROOT/build_pymnn.sh" $PYTHON_BUILD_ARGS
    else
        echo -e "${YELLOW}警告: 未找到 build_pymnn.sh，使用基本构建方式...${NC}"
        
        cd pymnn/pip_package
        
        # 构建依赖
        if [ -n "$DEPS_OPTIONS" ]; then
            $PYTHON_CMD build_deps.py $DEPS_OPTIONS
        else
            $PYTHON_CMD build_deps.py
        fi
        
        # 构建 wheel
        $PYTHON_CMD -m pip install -U numpy wheel setuptools --quiet
        rm -rf build dist
        
        BUILD_ARGS=""
        [ -n "$VERSION" ] && BUILD_ARGS="--version $VERSION"
        [ -n "$DEPS_OPTIONS" ] && BUILD_ARGS="$BUILD_ARGS --deps $DEPS_OPTIONS"
        
        $PYTHON_CMD setup.py bdist_wheel $BUILD_ARGS
        
        mkdir -p "$PROJECT_ROOT/$PYTHON_OUTPUT_DIR"
        cp dist/*.whl "$PROJECT_ROOT/$PYTHON_OUTPUT_DIR/"
        
        cd "$PROJECT_ROOT"
    fi
    
    echo -e "${GREEN}Python 构建完成！${NC}"
    echo "输出位置: $PYTHON_OUTPUT_DIR"
fi

# ============================================================================
# 总结
# ============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有构建完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "输出目录: $PROJECT_ROOT/$OUTPUT_DIR"
echo ""
[ "$BUILD_ANDROID" = true ] && echo -e "${GREEN}Android:${NC} $OUTPUT_DIR/android"
[ "$BUILD_IOS" = true ] && echo -e "${GREEN}iOS (真机):${NC} $OUTPUT_DIR/ios/device"
[ "$BUILD_IOS_SIMULATOR" = true ] && echo -e "${GREEN}iOS (模拟器):${NC} $OUTPUT_DIR/ios/simulator"
[ "$BUILD_HARMONY" = true ] && echo -e "${GREEN}鸿蒙:${NC} $OUTPUT_DIR/harmony"
[ "$BUILD_PYTHON" = true ] && echo -e "${GREEN}Python:${NC} $OUTPUT_DIR/python"
echo ""


