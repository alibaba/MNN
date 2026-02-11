#!/bin/bash
set -e

# ==============================================================================
# Sana Diffusion Android Runner
# 独立的 Android 运行脚本，由 run_sana_demo.sh 调用或单独使用
# ==============================================================================

DEVICE_DIR="/data/local/tmp/sana_demo"

# Defaults
BUILD_DIR=""
MODEL_DIR=""
INPUT_IMAGE=""
OUTPUT_IMAGE="sana_out.jpg"
MODE="text2img"
PROMPT="A beautiful landscape"
BACKEND="opencl"
STEPS=5
SEED=42
WIDTH=512
HEIGHT=512
USE_CFG=1
CFG_SCALE=4.5

# ==============================================================================
# Argument Parsing
# ==============================================================================
usage() {
    echo "Usage: $0 -b <build_dir> -m <model_dir> [options]"
    echo ""
    echo "Required:"
    echo "  -b <build_dir>    Android 构建目录"
    echo "  -m <model_dir>    模型目录"
    echo ""
    echo "Options:"
    echo "  -M <mode>         模式: text2img 或 img2img (默认: text2img)"
    echo "  -i <input>        输入图像 (img2img 必需)"
    echo "  -o <output>       输出图像路径"
    echo "  -p <prompt>       提示词"
    echo "  -k <backend>      后端: cpu, opencl (默认: opencl)"
    echo "  -s <steps>        推理步数"
    echo "  -S <seed>         随机种子"
    echo "  -W <width>        输出宽度"
    echo "  -H <height>       输出高度"
    echo "  --cfg <scale>     CFG 强度 (0 表示禁用)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -b) BUILD_DIR="$2"; shift 2 ;;
        -m) MODEL_DIR="$2"; shift 2 ;;
        -M) MODE="$2"; shift 2 ;;
        -i) INPUT_IMAGE="$2"; shift 2 ;;
        -o) OUTPUT_IMAGE="$2"; shift 2 ;;
        -p) PROMPT="$2"; shift 2 ;;
        -k) BACKEND="$2"; shift 2 ;;
        -s) STEPS="$2"; shift 2 ;;
        -S) SEED="$2"; shift 2 ;;
        -W) WIDTH="$2"; shift 2 ;;
        -H) HEIGHT="$2"; shift 2 ;;
        --cfg) 
            if [ "$2" == "0" ]; then
                USE_CFG=0
            else
                USE_CFG=1
                CFG_SCALE="$2"
            fi
            shift 2 
            ;;
        -h|--help) usage ;;
        *) echo "Unknown: $1"; usage ;;
    esac
done

# ==============================================================================
# Helpers
# ==============================================================================
log() { echo -e "\033[1;32m[ANDROID] $1\033[0m"; }
error() { echo -e "\033[1;31m[ERROR] $1\033[0m"; exit 1; }

# ==============================================================================
# Validation
# ==============================================================================
[ -z "$BUILD_DIR" ] && error "请指定构建目录: -b <build_dir>"
[ -z "$MODEL_DIR" ] && error "请指定模型目录: -m <model_dir>"
[ ! -d "$BUILD_DIR" ] && error "构建目录不存在: $BUILD_DIR"
[ ! -f "$BUILD_DIR/sana_diffusion_demo" ] && error "可执行文件不存在: $BUILD_DIR/sana_diffusion_demo"
[ ! -d "$MODEL_DIR" ] && error "模型目录不存在: $MODEL_DIR"

if [ "$MODE" == "img2img" ]; then
    [ -z "$INPUT_IMAGE" ] && error "img2img 模式需要输入图像: -i <input>"
    [ ! -f "$INPUT_IMAGE" ] && error "输入图像不存在: $INPUT_IMAGE"
fi

command -v adb >/dev/null 2>&1 || error "adb 未找到"

# ==============================================================================
# Main
# ==============================================================================
log "准备设备目录: $DEVICE_DIR"
adb shell "mkdir -p $DEVICE_DIR/libs"

log "推送可执行文件和库..."
adb push "$BUILD_DIR/sana_diffusion_demo" "$DEVICE_DIR/"
find "$BUILD_DIR" -name "*.so" -exec adb push {} "$DEVICE_DIR/libs/" \; 2>/dev/null || true

# 推送模型
if ! adb shell "[ -f $DEVICE_DIR/models/transformer.mnn ]" 2>/dev/null; then
    log "推送模型..."
    adb shell "mkdir -p $DEVICE_DIR/models"
    adb push "$MODEL_DIR/." "$DEVICE_DIR/models/"
else
    log "模型已存在"
fi

# 推送输入图像
DEVICE_INPUT=""
if [ "$MODE" == "img2img" ]; then
    log "推送输入图像..."
    adb push "$INPUT_IMAGE" "$DEVICE_DIR/input.jpg"
    DEVICE_INPUT="input.jpg"
fi

# 运行
log "运行 Sana Diffusion..."
log "  模式: $MODE, 后端: $BACKEND, 步数: $STEPS"

CMD="cd $DEVICE_DIR && "
CMD+="export LD_LIBRARY_PATH=$DEVICE_DIR/libs:\$LD_LIBRARY_PATH && "
CMD+="./sana_diffusion_demo models $MODE \"$PROMPT\" \"$DEVICE_INPUT\" output.jpg $WIDTH $HEIGHT $STEPS $SEED $USE_CFG $CFG_SCALE"

adb shell "$CMD"

# 拉取输出
if adb shell "[ -f $DEVICE_DIR/output.jpg ]" 2>/dev/null; then
    adb pull "$DEVICE_DIR/output.jpg" "$OUTPUT_IMAGE"
    log "输出: $OUTPUT_IMAGE"
else
    error "未生成输出"
fi
