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

# Detect OS
OS_NAME=$(uname -s)
echo -e "${YELLOW}Detected OS: $OS_NAME${NC}"

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

# Common CMake args
CMAKE_ARGS=(
    "-DMNN_BUILD_LLM=ON"
    "-DMNN_BUILD_SHARED_LIBS=OFF"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DMNN_LOW_MEMORY=ON"
    "-DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON"
    "-DMNN_SUPPORT_TRANSFORMER_FUSE=ON"
    "-DLLM_SUPPORT_VISION=ON"
    "-DMNN_BUILD_OPENCV=ON"
    "-DMNN_IMGCODECS=ON"
    "-DLLM_SUPPORT_AUDIO=ON"
    "-DMNN_BUILD_AUDIO=ON"
    "-DMNN_BUILD_DIFFUSION=ON"
    "-DMNN_SEP_BUILD=OFF"
    "-DMNN_USE_OPENCV=ON"
    "-DLLM_SUPPORT_HTTP_RESOURCE=OFF"
)

# OS-specific CMake args
if [ "$OS_NAME" == "Darwin" ]; then
    CMAKE_ARGS+=("-DMNN_METAL=ON")
elif [ "$OS_NAME" == "Linux" ]; then
    CMAKE_ARGS+=("-DMNN_OPENCL=ON") # Enable OpenCL for Linux if available/needed, otherwise can be omitted
    # You might want to add -DMNN_VULKAN=ON if Vulkan is needed
else
    echo -e "${YELLOW}Warning: Unknown OS $OS_NAME, using default configuration${NC}"
fi

cmake -B "$MNN_BUILD_DIR" -S "$PROJECT_ROOT" "${CMAKE_ARGS[@]}"

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

MNNCLI_CMAKE_ARGS=(
    "-DMNN_BUILD_DIR=$MNN_BUILD_DIR"
    "-DMNN_SOURCE_DIR=$PROJECT_ROOT"
    "-DCMAKE_BUILD_TYPE=Release"
)

if [ "$OS_NAME" == "Darwin" ]; then
    SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)
    MNNCLI_CMAKE_ARGS+=("-DCMAKE_OSX_SYSROOT=$SDK_PATH")
fi

cmake -B "$MNNCLI_BUILD_DIR" -S "$SCRIPT_DIR" "${MNNCLI_CMAKE_ARGS[@]}"

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
    
    # Check dependencies (OS specific)
    if [ "$1" == "--check" ]; then
        echo -e "${YELLOW}Checking dependencies...${NC}"
        if [ "$OS_NAME" == "Darwin" ]; then
            otool -L "$MNNCLI_BUILD_DIR/mnncli"
        elif [ "$OS_NAME" == "Linux" ]; then
            ldd "$MNNCLI_BUILD_DIR/mnncli"
        fi
    fi
else
    echo -e "${RED}Build failed! Executable not found.${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"