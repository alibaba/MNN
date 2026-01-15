#!/bin/bash
# MNN TTS Android Demo 构建脚本
# 使用方法: ./build.sh [debug|release|clean|install]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function check_prerequisites() {
    print_info "检查前置条件..."

    # 检查 MNN 库
    MNN_LIB_PATH="../../../project/android/build_64/lib/libMNN.so"
    if [ ! -f "$MNN_LIB_PATH" ]; then
        print_error "MNN 库未找到: $MNN_LIB_PATH"
        print_info "请先构建 MNN 库:"
        print_info "  cd ../../../project/android"
        print_info "  ./build_64.sh"
        exit 1
    fi

    # 检查 Java
    if ! command -v java &> /dev/null; then
        print_error "Java 未安装"
        exit 1
    fi

    # 检查 Gradle Wrapper
    if [ ! -f "./gradlew" ]; then
        print_error "Gradle Wrapper 未找到"
        exit 1
    fi

    print_info "前置条件检查通过 ✓"
}

function clean_build() {
    print_info "清理构建目录..."
    ./gradlew clean
    print_info "清理完成 ✓"
}

function build_debug() {
    print_info "开始构建 Debug APK..."
    ./gradlew assembleDebug

    APK_PATH="build/outputs/apk/debug/MNNTTSDemo-arm64-v8a-debug.apk"
    if [ -f "$APK_PATH" ]; then
        APK_SIZE=$(ls -lh "$APK_PATH" | awk '{print $5}')
        print_info "构建成功! ✓"
        print_info "APK 位置: $APK_PATH"
        print_info "APK 大小: $APK_SIZE"
    else
        print_error "构建失败,APK 未生成"
        exit 1
    fi
}

function build_release() {
    print_info "开始构建 Release APK..."
    ./gradlew assembleRelease

    APK_PATH="build/outputs/apk/release/MNNTTSDemo-arm64-v8a-release-unsigned.apk"
    if [ -f "$APK_PATH" ]; then
        APK_SIZE=$(ls -lh "$APK_PATH" | awk '{print $5}')
        print_info "构建成功! ✓"
        print_info "APK 位置: $APK_PATH"
        print_info "APK 大小: $APK_SIZE"
        print_warn "注意: Release APK 未签名,需要签名后才能发布"
    else
        print_error "构建失败,APK 未生成"
        exit 1
    fi
}

function install_debug() {
    print_info "安装 Debug APK 到设备..."

    # 检查设备连接
    if ! command -v adb &> /dev/null; then
        print_error "adb 未找到,请确保 Android SDK Platform-Tools 已安装"
        exit 1
    fi

    DEVICE_COUNT=$(adb devices | grep -v "List" | grep "device$" | wc -l)
    if [ "$DEVICE_COUNT" -eq 0 ]; then
        print_error "未检测到 Android 设备"
        print_info "请确保:"
        print_info "  1. 设备已通过 USB 连接"
        print_info "  2. 设备已开启 USB 调试"
        print_info "  3. 已授权计算机进行 USB 调试"
        exit 1
    fi

    print_info "检测到 $DEVICE_COUNT 个设备"

    APK_PATH="build/outputs/apk/debug/MNNTTSDemo-arm64-v8a-debug.apk"
    if [ ! -f "$APK_PATH" ]; then
        print_warn "APK 不存在,先构建..."
        build_debug
    fi

    print_info "正在安装..."
    ./gradlew installDebug

    print_info "安装完成! ✓"
    print_info "应用包名: com.alibaba.mnn.tts.demo"
    print_info ""
    print_info "启动应用:"
    print_info "  adb shell am start -n com.alibaba.mnn.tts.demo/.MainActivity"
}

function show_usage() {
    echo "MNN TTS Android Demo 构建脚本"
    echo ""
    echo "使用方法: $0 [command]"
    echo ""
    echo "可用命令:"
    echo "  debug    - 构建 Debug APK (默认)"
    echo "  release  - 构建 Release APK"
    echo "  clean    - 清理构建目录"
    echo "  install  - 构建并安装 Debug APK 到设备"
    echo "  help     - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0              # 构建 Debug APK"
    echo "  $0 debug        # 构建 Debug APK"
    echo "  $0 release      # 构建 Release APK"
    echo "  $0 clean        # 清理构建"
    echo "  $0 install      # 安装到设备"
    echo ""
}

# 主流程
case "${1:-debug}" in
    debug)
        check_prerequisites
        build_debug
        ;;
    release)
        check_prerequisites
        build_release
        ;;
    clean)
        clean_build
        ;;
    install)
        check_prerequisites
        install_debug
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "未知命令: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac

print_info "所有操作完成! ✓"
