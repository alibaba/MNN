#!/bin/bash
set -e

# ==============================================================================
# Sana Diffusion Demo Script
# 支持 Mac 和 Android 平台，text2img 和 img2img 模式
# ==============================================================================

# Configuration & Defaults
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build_sana"
ANDROID_BUILD_DIR="$PROJECT_ROOT/build_android"

# User configurable
MODEL_DIR=""
INPUT_IMAGE=""
OUTPUT_IMAGE="sana_out.jpg"
BACKEND_TYPE="cpu"
PROMPT="A beautiful landscape"
TARGET="mac"
MODE="text2img"
STEPS=5
SEED=42
WIDTH=512
HEIGHT=512
USE_CFG=1
CFG_SCALE=4.5

# ==============================================================================
# Helper Functions
# ==============================================================================
log() {
    echo -e "\033[1;32m[SANA] $1\033[0m"
}

warn() {
    echo -e "\033[1;33m[WARN] $1\033[0m"
}

error() {
    echo -e "\033[1;31m[ERROR] $1\033[0m"
    exit 1
}

# ==============================================================================
# Argument Parsing
# ==============================================================================
usage() {
    echo "========================================================================"
    echo "Sana Diffusion Demo - 基于 Qwen3-0.6B 的高效文生图/图像编辑"
    echo "========================================================================"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model <path>      模型目录路径 (必需)"
    echo "  -M, --mode <mode>       模式: text2img 或 img2img (默认: text2img)"
    echo "  -i, --input <path>      输入图像路径 (img2img 模式必需)"
    echo "  -o, --output <path>     输出图像路径 (默认: sana_out.jpg)"
    echo "  -p, --prompt <text>     文本提示词"
    echo "  -b, --backend <type>    后端: cpu, metal, opencl (默认: cpu)"
    echo "  -t, --target <platform> 目标平台: mac 或 android (默认: mac)"
    echo "  -s, --steps <num>       推理步数 (默认: 5)"
    echo "  -S, --seed <num>        随机种子 (默认: 42)"
    echo "  -W, --width <num>       输出宽度 (默认: 512)"
    echo "  -H, --height <num>      输出高度 (默认: 512)"
    echo "  --cfg                   启用 CFG 引导 (默认启用)"
    echo "  --no-cfg                禁用 CFG 引导"
    echo "  --cfg-scale <num>       CFG 强度 (默认: 4.5)"
    echo "  -h, --help              显示帮助"
    echo ""
    echo "示例:"
    echo "  # Mac 文生图"
    echo "  $0 -m ~/models/sana -M text2img -p \"一只可爱的猫咪\" -o cat.jpg"
    echo ""
    echo "  # Mac 图像编辑 (img2img)"
    echo "  $0 -m ~/models/sana -M img2img -i input.jpg -p \"转换为吉卜力风格\" -o ghibli.jpg"
    echo ""
    echo "  # Android OpenCL"
    echo "  $0 -m ~/models/sana -t android -b opencl -M img2img -i photo.jpg -p \"添加彩虹\""
    echo ""
    exit 1
}

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_DIR="$2"
            shift 2
            ;;
        -M|--mode)
            MODE="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_IMAGE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_IMAGE="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -b|--backend)
            BACKEND_TYPE="$2"
            shift 2
            ;;
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -s|--steps)
            STEPS="$2"
            shift 2
            ;;
        -S|--seed)
            SEED="$2"
            shift 2
            ;;
        -W|--width)
            WIDTH="$2"
            shift 2
            ;;
        -H|--height)
            HEIGHT="$2"
            shift 2
            ;;
        --cfg)
            USE_CFG=1
            shift
            ;;
        --no-cfg)
            USE_CFG=0
            shift
            ;;
        --cfg-scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "未知参数: $1"
            usage
            ;;
        *)
            # 位置参数：输入图片或输出图片
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# 处理位置参数
if [ ${#POSITIONAL_ARGS[@]} -ge 1 ]; then
    # 第一个位置参数作为输入图片
    INPUT_IMAGE="${POSITIONAL_ARGS[0]}"
fi
if [ ${#POSITIONAL_ARGS[@]} -ge 2 ]; then
    # 第二个位置参数作为输出图片
    OUTPUT_IMAGE="${POSITIONAL_ARGS[1]}"
fi

# 如果有输入图片，自动切换到 img2img 模式
if [ -n "$INPUT_IMAGE" ] && [ -f "$INPUT_IMAGE" ] && [ "$MODE" == "text2img" ]; then
    warn "检测到输入图片，自动切换到 img2img 模式"
    MODE="img2img"
fi

# ==============================================================================
# Validation
# ==============================================================================
if [ -z "$MODEL_DIR" ]; then
    error "请指定模型目录: -m <model_path>"
fi

if [ ! -d "$MODEL_DIR" ]; then
    error "模型目录不存在: $MODEL_DIR"
fi

if [ "$MODE" != "text2img" ] && [ "$MODE" != "img2img" ]; then
    error "无效的模式: $MODE (必须是 text2img 或 img2img)"
fi

if [ "$MODE" == "img2img" ] && [ -z "$INPUT_IMAGE" ]; then
    error "img2img 模式需要指定输入图像: -i <input_image>"
fi

if [ "$MODE" == "img2img" ] && [ ! -f "$INPUT_IMAGE" ]; then
    error "输入图像不存在: $INPUT_IMAGE"
fi

# Android 不支持 metal
if [ "$TARGET" == "android" ] && [ "$BACKEND_TYPE" == "metal" ]; then
    warn "Android 不支持 Metal 后端，自动切换到 opencl"
    BACKEND_TYPE="opencl"
fi

# ==============================================================================
# Setup Build Directory
# ==============================================================================
if [ "$TARGET" == "android" ]; then
    EXEC_PATH="$ANDROID_BUILD_DIR/sana_diffusion_demo"
else
    EXEC_PATH="$BUILD_DIR/sana_diffusion_demo"
fi

if [ ! -f "$EXEC_PATH" ]; then
    error "可执行文件不存在: $EXEC_PATH\n请先构建项目。"
fi

# ==============================================================================
# Run Demo
# ==============================================================================
log "========================================"
log "Sana Diffusion Demo"
log "========================================"
log "目标平台: $TARGET"
log "模式: $MODE"
log "模型: $MODEL_DIR"
log "后端: $BACKEND_TYPE"
log "提示词: $PROMPT"
log "输出: $OUTPUT_IMAGE"
log "尺寸: ${WIDTH}x${HEIGHT}"
log "步数: $STEPS"
log "种子: $SEED"
log "CFG: $([ $USE_CFG -eq 1 ] && echo "启用 (scale=$CFG_SCALE)" || echo "禁用")"
if [ "$MODE" == "img2img" ]; then
    log "输入图像: $INPUT_IMAGE"
fi
log "========================================"

if [ "$TARGET" == "android" ]; then
    # Android 运行
    log "在 Android 设备上运行..."
    
    # 检查 adb
    if ! command -v adb >/dev/null 2>&1; then
        error "adb 未找到，请安装 Android Platform Tools"
    fi
    
    DEVICE_DIR="/data/local/tmp/sana_demo"
    
    # 准备设备目录
    adb shell "mkdir -p $DEVICE_DIR/libs"
    
    # 推送可执行文件和库
    log "推送可执行文件..."
    adb push "$EXEC_PATH" "$DEVICE_DIR/"
    find "$ANDROID_BUILD_DIR" -name "*.so" -exec adb push {} "$DEVICE_DIR/libs/" \; 2>/dev/null || true
    
    # 推送模型（如果需要）
    if ! adb shell "[ -f $DEVICE_DIR/models/transformer.mnn ]" 2>/dev/null; then
        log "推送模型到设备（可能需要几分钟）..."
        adb shell "mkdir -p $DEVICE_DIR/models"
        adb push "$MODEL_DIR/." "$DEVICE_DIR/models/"
    else
        log "模型已存在于设备"
    fi
    
    # 推送输入图像（如果是 img2img）
    if [ "$MODE" == "img2img" ]; then
        log "推送输入图像..."
        adb push "$INPUT_IMAGE" "$DEVICE_DIR/input.jpg"
        DEVICE_INPUT="input.jpg"
    else
        DEVICE_INPUT=""
    fi
    
    # 构建命令
    CMD="cd $DEVICE_DIR && export LD_LIBRARY_PATH=$DEVICE_DIR/libs:\$LD_LIBRARY_PATH && "
    CMD+="./sana_diffusion_demo models $MODE \"$PROMPT\" \"$DEVICE_INPUT\" output.jpg $WIDTH $HEIGHT $STEPS $SEED $USE_CFG $CFG_SCALE"
    
    log "执行命令..."
    adb shell "$CMD"
    
    # 拉取输出
    if adb shell "[ -f $DEVICE_DIR/output.jpg ]" 2>/dev/null; then
        adb pull "$DEVICE_DIR/output.jpg" "$OUTPUT_IMAGE"
        log "输出已保存: $OUTPUT_IMAGE"
    else
        error "未生成输出图像"
    fi
else
    # Mac 本地运行
    if [ "$MODE" == "img2img" ]; then
        "$EXEC_PATH" "$MODEL_DIR" "$MODE" "$PROMPT" "$INPUT_IMAGE" "$OUTPUT_IMAGE" "$WIDTH" "$HEIGHT" "$STEPS" "$SEED" "$USE_CFG" "$CFG_SCALE"
    else
        "$EXEC_PATH" "$MODEL_DIR" "$MODE" "$PROMPT" "" "$OUTPUT_IMAGE" "$WIDTH" "$HEIGHT" "$STEPS" "$SEED" "$USE_CFG" "$CFG_SCALE"
    fi
    
    if [ -f "$OUTPUT_IMAGE" ]; then
        log "输出已保存: $OUTPUT_IMAGE"
        # Mac 上自动打开图片
        if command -v open >/dev/null 2>&1; then
            open "$OUTPUT_IMAGE" || warn "自动打开图片失败（不影响生成结果）"
        fi
    else
        error "未生成输出图像"
    fi
fi

log "完成！"
