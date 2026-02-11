#!/bin/bash
set -e

# ==============================================================================
# Sana Benchmark - Host (Mac)
# 性能测试脚本，支持 CPU 和 Metal 后端
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
MODEL_DIR=""
BUILD_DIR="$PROJECT_ROOT/build_sana"
INPUT_IMAGE="$PROJECT_ROOT/tools/cv/imgs/cat.jpg"
PROMPT="Convert to a Ghibli-style illustration: soft contrast, warm tones"
MODE="img2img"
STEPS=5
SEED=42
WIDTH=512
HEIGHT=512
USE_CFG=1
CFG_SCALE=4.5
BACKEND="cpu"  # cpu, metal

# ==============================================================================
# Argument Parsing
# ==============================================================================
usage() {
    echo "Usage: $0 -m <model_dir> [options]"
    echo ""
    echo "Options:"
    echo "  -m <path>       模型目录 (必需)"
    echo "  -i <path>       输入图片 (默认: tools/cv/imgs/cat.jpg)"
    echo "  -b <backend>    后端: cpu, metal (默认: cpu)"
    echo "  -s <steps>      推理步数 (默认: 5)"
    echo "  -M <mode>       模式: text2img, img2img (默认: img2img)"
    echo "  --build         强制重新编译"
    exit 1
}

FORCE_BUILD=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -m) MODEL_DIR="$2"; shift 2 ;;
        -i) INPUT_IMAGE="$2"; shift 2 ;;
        -b) BACKEND="$2"; shift 2 ;;
        -s) STEPS="$2"; shift 2 ;;
        -M) MODE="$2"; shift 2 ;;
        --build) FORCE_BUILD=true; shift ;;
        -h|--help) usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

[ -z "$MODEL_DIR" ] && { echo "错误: 请指定模型目录 -m <path>"; usage; }
[ ! -d "$MODEL_DIR" ] && { echo "错误: 模型目录不存在: $MODEL_DIR"; exit 1; }

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
    log "编译 sana_diffusion_demo..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    cmake "$PROJECT_ROOT" \
        -DMNN_BUILD_DIFFUSION=ON \
        -DMNN_BUILD_LLM=ON \
        -DMNN_IMGCODECS=ON \
        -DMNN_BUILD_OPENCV=ON \
        -DMNN_METAL=ON \
        -DMNN_SEP_BUILD=OFF > build.log 2>&1
    
    make -j$(sysctl -n hw.logicalcpu 2>/dev/null || nproc) >> build.log 2>&1
    cd "$SCRIPT_DIR"
    
    [ ! -f "$EXEC" ] && { echo "编译失败，查看 $BUILD_DIR/build.log"; exit 1; }
    log "编译完成"
fi

# ==============================================================================
# Run Benchmark
# ==============================================================================
OUTPUT_IMAGE="$SCRIPT_DIR/benchmark_out.jpg"

log "========================================"
log "Sana Host Benchmark"
log "========================================"
log "模型: $MODEL_DIR"
log "后端: $BACKEND"
log "模式: $MODE"
log "步数: $STEPS"
log "========================================"

# 准备输入参数
if [ "$MODE" == "img2img" ]; then
    [ ! -f "$INPUT_IMAGE" ] && { echo "错误: 输入图片不存在: $INPUT_IMAGE"; exit 1; }
    INPUT_ARG="$INPUT_IMAGE"
else
    INPUT_ARG=""
fi

# 运行并捕获输出
log "运行推理..."
START_TIME=$(date +%s%3N)

OUTPUT=$("$EXEC" "$MODEL_DIR" "$MODE" "$PROMPT" "$INPUT_ARG" "$OUTPUT_IMAGE" "$WIDTH" "$HEIGHT" "$STEPS" "$SEED" "$USE_CFG" "$CFG_SCALE" 2>&1) || true

END_TIME=$(date +%s%3N)
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
# Report
# ==============================================================================
echo ""
echo "================================================================="
echo "              Sana Host ($BACKEND) Benchmark Results             "
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
    # Mac 上自动打开
    command -v open >/dev/null 2>&1 && open "$OUTPUT_IMAGE"
else
    warn "未生成输出图片"
fi
