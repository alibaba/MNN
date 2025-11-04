#!/bin/bash

# MNNCLI Build Script
# This script builds the mnncli executable

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building MNNCLI...${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Stage 1: Build MNN static library
MNN_BUILD_DIR="$PROJECT_ROOT/build_mnn_static"
echo -e "${YELLOW}Stage 1: Building MNN static library...${NC}"
echo -e "${YELLOW}MNN build directory: $MNN_BUILD_DIR${NC}"

# Clean MNN build directory if requested
if [ "$1" == "--clean" ]; then
    if [ -d "$MNN_BUILD_DIR" ]; then
        echo -e "${YELLOW}Cleaning MNN build directory...${NC}"
        rm -rf "$MNN_BUILD_DIR"
    fi
fi

# Create and build MNN
mkdir -p "$MNN_BUILD_DIR"

echo -e "${YELLOW}Configuring MNN...${NC}"
cmake -B "$MNN_BUILD_DIR" -S "$PROJECT_ROOT" \
    -DMNN_BUILD_LLM=ON \
    -DMNN_BUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_LOW_MEMORY=ON \
    -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
    -DMNN_METAL=ON \
    -DLLM_SUPPORT_VISION=ON \
    -DMNN_BUILD_OPENCV=ON \
    -DMNN_IMGCODECS=ON \
    -DLLM_SUPPORT_AUDIO=ON \
    -DMNN_BUILD_AUDIO=ON \
    -DMNN_BUILD_DIFFUSION=ON \
    -DMNN_SEP_BUILD=OFF \
    -DMNN_USE_OPENCV=ON

echo -e "${YELLOW}Building MNN...${NC}"
if command -v nproc &> /dev/null; then
    cmake --build "$MNN_BUILD_DIR" -- -j$(nproc)
else
    cmake --build "$MNN_BUILD_DIR" -- -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi

# Verify MNN library was built
if [ ! -f "$MNN_BUILD_DIR/libMNN.a" ]; then
    echo -e "${RED}Failed to build MNN static library!${NC}"
    exit 1
fi

echo -e "${GREEN}MNN static library built successfully!${NC}"
ls -lh "$MNN_BUILD_DIR/libMNN.a"

# Stage 2: Build mnncli executable
MNNCLI_BUILD_DIR="$SCRIPT_DIR/build_mnncli"
echo -e "${YELLOW}Stage 2: Building mnncli executable...${NC}"
echo -e "${YELLOW}mnncli build directory: $MNNCLI_BUILD_DIR${NC}"

mkdir -p "$MNNCLI_BUILD_DIR"

echo -e "${YELLOW}Configuring mnncli...${NC}"
cmake -B "$MNNCLI_BUILD_DIR" -S "$SCRIPT_DIR" \
    -DMNN_BUILD_DIR="$MNN_BUILD_DIR" \
    -DMNN_SOURCE_DIR="$PROJECT_ROOT" \
    -DCMAKE_BUILD_TYPE=Release

echo -e "${YELLOW}Building mnncli...${NC}"
if command -v nproc &> /dev/null; then
    cmake --build "$MNNCLI_BUILD_DIR" -j$(nproc)
else
    cmake --build "$MNNCLI_BUILD_DIR" -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi

# Check if build was successful
if [ -f "$MNNCLI_BUILD_DIR/mnncli" ]; then
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}Executable location: $MNNCLI_BUILD_DIR/mnncli${NC}"
    
    # Show file size
    ls -lh "$MNNCLI_BUILD_DIR/mnncli"
    
    # Test basic functionality
    echo -e "${YELLOW}Testing basic functionality...${NC}"
    if "$MNNCLI_BUILD_DIR/mnncli" --help &> /dev/null; then
        echo -e "${GREEN}Basic functionality test passed!${NC}"
    else
        echo -e "${RED}Basic functionality test failed!${NC}"
        exit 1
    fi
    
    # Show dependencies only on macOS with --check parameter
    if [[ "$OSTYPE" == "darwin"* ]] && [[ "$1" == "--check" ]]; then
        echo -e "${YELLOW}Checking dependencies...${NC}"
        otool -L "$MNNCLI_BUILD_DIR/mnncli"
    fi
else
    echo -e "${RED}Build failed! Executable not found.${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"

