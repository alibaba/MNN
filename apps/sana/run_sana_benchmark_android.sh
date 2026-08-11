#!/bin/bash
set -e

# ==============================================================================
# Sana Benchmark - Android
# 性能测试脚本，支持 CPU 和 OpenCL 后端
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
MODEL_DIR=""
BUILD_DIR="$PROJECT_ROOT/build_android"
INPUT_IMAGE="$PROJECT_ROOT/tools/cv/imgs/cat.jpg"
PROMPT="Convert to a Ghibli-style illustration: soft contrast, warm tones"
MODE="img2img"
STEPS=5
SEED=42
WIDTH=512
HEIGHT=512
USE_CFG=1
CFG_SCALE=4.5
BACKEND="opencl"  # cpu, opencl

# Device paths
DEVICE_DIR="/data/local/tmp/sana_benchmark"

# ==============================================================================
# Argument Parsing
# ==============================================================================
usage() {
    echo "Usage: $0 -m <model_dir> [options]"
    echo ""
    echo "Options:"
    echo "  -m <path>       模型目录 (必需)"
    echo "  -i <path>       输入图片 (默认: tools/cv/imgs/cat.jpg)"
    echo "  -b <backend>    后端: cpu, opencl (默认: opencl)"
    echo "  -s <steps>      推理步数 (默认: 5)"
    echo "  -M <mode>       模式: text2img, img2img (默认: img2img)"
    echo "  --build         强制重新编译"
    echo "  --push-model    强制重新推送模型"
    exit 1
}

FORCE_BUILD=false
FORCE_PUSH_MODEL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -m) MODEL_DIR="$2"; shift 2 ;;
        -i) INPUT_IMAGE="$2"; shift 2 ;;
        -b) BACKEND="$2"; shift 2 ;;
        -s) STEPS="$2"; shift 2 ;;
        -M) MODE="$2"; shift 2 ;;
        --build) FORCE_BUILD=true; shift ;;
        --push-model) FORCE_PUSH_MODEL=true; shift ;;
        -h|--help) usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

[ -z "$MODEL_DIR" ] && { echo "错误: 请指定模型目录 -m <path>"; usage; }
[ ! -d "$MODEL_DIR" ] && { echo "错误: 模型目录不存在: $MODEL_DIR"; exit 1; }
command -v adb >/dev/null 2>&1 || { echo "错误: adb 未找到"; exit 1; }

# ==============================================================================
# Helpers
# ==============================================================================
log() { echo -e "\033[1;32m[BENCHMARK] $1\033[0m"; }
warn() { echo -e "\033[1;33m[WARN] $1\033[0m"; }

# ==============================================================================
# Build (if needed)
# ==============================================================================
EXEC="$BUILD_DIR/sana_diffusion_demo"

if [ ! -f "$EXEC" ] || [ "$FORCE_BUILD" = true ]; then
    log "编译 Android 版本..."
    
    [ -z "$ANDROID_NDK" ] && { echo "错误: ANDROID_NDK 未设置"; exit 1; }
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    cmake "$PROJECT_ROOT" \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI="arm64-v8a" \
        -DANDROID_STL=c++_static \
        -DANDROID_NATIVE_API_LEVEL=android-21 \
        -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
        -DMNN_BUILD_DIFFUSION=ON \
        -DMNN_BUILD_LLM=ON \
        -DMNN_IMGCODECS=ON \
        -DMNN_BUILD_OPENCV=ON \
        -DMNN_OPENCL=ON \
        -DMNN_SEP_BUILD=OFF > build.log 2>&1
    
    make -j8 >> build.log 2>&1
    cd "$SCRIPT_DIR"
    
    [ ! -f "$EXEC" ] && { echo "编译失败，查看 $BUILD_DIR/build.log"; exit 1; }
    log "编译完成"
fi

# ==============================================================================
# Push to Device
# ==============================================================================
log "准备设备目录..."
adb shell "mkdir -p $DEVICE_DIR/libs"
adb shell "mkdir -p $DEVICE_DIR/models"

log "推送可执行文件和库..."
adb push "$EXEC" "$DEVICE_DIR/" > /dev/null
find "$BUILD_DIR" -name "*.so" -exec adb push {} "$DEVICE_DIR/libs/" \; 2>/dev/null || true

# 推送模型
MODEL_MARKER="$DEVICE_DIR/models/llm/config.json"
if [ "$FORCE_PUSH_MODEL" = true ] || ! adb shell "[ -f $MODEL_MARKER ]" 2>/dev/null; then
    log "推送模型到设备（可能需要几分钟）..."
    adb shell "rm -rf $DEVICE_DIR/models/*" 2>/dev/null || true
    adb push "$MODEL_DIR/." "$DEVICE_DIR/models/"
else
    log "模型已存在于设备"
fi

# 推送输入图片
if [ "$MODE" == "img2img" ]; then
    [ ! -f "$INPUT_IMAGE" ] && { echo "错误: 输入图片不存在: $INPUT_IMAGE"; exit 1; }
    log "推送输入图片..."
    adb push "$INPUT_IMAGE" "$DEVICE_DIR/input.jpg" > /dev/null
    DEVICE_INPUT="input.jpg"
else
    DEVICE_INPUT=""
fi

# ==============================================================================
# Run Benchmark
# ==============================================================================
log "========================================"
log "Sana Android Benchmark"
log "========================================"
log "后端: $BACKEND"
log "模式: $MODE"
log "步数: $STEPS"
log "========================================"

# 构建命令
CMD="cd $DEVICE_DIR && "
CMD+="export LD_LIBRARY_PATH=$DEVICE_DIR/libs:\$LD_LIBRARY_PATH && "
CMD+="./sana_diffusion_demo models $MODE \"$PROMPT\" \"$DEVICE_INPUT\" output.jpg $WIDTH $HEIGHT $STEPS $SEED $USE_CFG $CFG_SCALE"

log "运行推理..."
START_TIME=$(python3 -c "import time; print(int(time.time()*1000))")

OUTPUT=$(adb shell "$CMD" 2>&1) || true

END_TIME=$(python3 -c "import time; print(int(time.time()*1000))")
TOTAL_TIME=$((END_TIME - START_TIME))

echo "$OUTPUT"

# ==============================================================================
# Parse Results
# ==============================================================================
parse_time() {
    echo "$OUTPUT" | grep -i "$1" | grep -oE '[0-9]+\.?[0-9]*' | tail -1
}

LOAD_LLM=$(parse_time "Load LLM")
LOAD_DIFF=$(parse_time "Load Diffusion")
INFER_LLM=$(parse_time "LLM Inference")
INFER_DIFF=$(parse_time "Diffusion Inference")

# ==============================================================================
# Pull Output
# ==============================================================================
OUTPUT_IMAGE="$SCRIPT_DIR/benchmark_android_out.jpg"
if adb shell "[ -f $DEVICE_DIR/output.jpg ]" 2>/dev/null; then
    adb pull "$DEVICE_DIR/output.jpg" "$OUTPUT_IMAGE" > /dev/null
fi

# ==============================================================================
# Report
# ==============================================================================
echo ""
echo "================================================================="
echo "            Sana Android ($BACKEND) Benchmark Results            "
echo "================================================================="
printf "%-25s | %-15s\n" "Metric" "Time (ms)"
echo "--------------------------|-----------------"
printf "%-25s | %-15s\n" "Load LLM" "${LOAD_LLM:-N/A}"
printf "%-25s | %-15s\n" "Load Diffusion" "${LOAD_DIFF:-N/A}"
printf "%-25s | %-15s\n" "LLM Inference" "${INFER_LLM:-N/A}"
printf "%-25s | %-15s\n" "Diffusion Inference" "${INFER_DIFF:-N/A}"
echo "--------------------------|-----------------"
printf "%-25s | %-15s\n" "Total Wall Time" "$TOTAL_TIME"
echo "================================================================="

if [ -f "$OUTPUT_IMAGE" ]; then
    log "输出图片: $OUTPUT_IMAGE"
else
    warn "未生成输出图片"
fi
