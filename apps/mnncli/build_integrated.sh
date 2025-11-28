#!/bin/bash

# MNNCLI Integrated Build Script
# This script builds mnncli as part of the MNN project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building MNNCLI as part of MNN project...${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Create build directory
BUILD_DIR="$PROJECT_ROOT/build_integrated"
echo -e "${YELLOW}Build directory: $BUILD_DIR${NC}"

# Clean build directory if it exists to avoid CMake cache conflicts
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning existing build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake with mnncli enabled
echo -e "${YELLOW}Configuring CMake with MNNCLI enabled...${NC}"
cmake \
    -DBUILD_MNNCLI=ON \
    -DMNN_BUILD_LLM=ON \
    -DMNN_BUILD_SHARED_LIBS=ON \
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
    "$PROJECT_ROOT"

# Build
echo -e "${YELLOW}Building MNN and MNNCLI...${NC}"
if command -v nproc &> /dev/null; then
    make -j$(nproc)
else
    make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi

# Check if build was successful
if [ -f "apps/mnncli/mnncli" ]; then
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}Executable location: $BUILD_DIR/apps/mnncli/mnncli${NC}"
    
    # Show file size
    ls -lh "apps/mnncli/mnncli"
    
    # Test basic functionality
    echo -e "${YELLOW}Testing basic functionality...${NC}"
    if ./apps/mnncli/mnncli --help &> /dev/null; then
        echo -e "${GREEN}Basic functionality test passed!${NC}"
    else
        echo -e "${RED}Basic functionality test failed!${NC}"
        exit 1
    fi
else
    echo -e "${RED}Build failed! Executable not found.${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${YELLOW}Note: This build includes the complete MNN project with mnncli integrated.${NC}"
