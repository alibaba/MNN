#!/bin/bash

# Test script for ModelRepository
# This script builds and tests the ModelRepository functionality

set -e

echo "=========================================="
echo "Testing ModelRepository"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please run this script from the mnncli root directory"
    exit 1
fi

# Create build directory
BUILD_DIR="build_test"
if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Building with tests enabled..."
cmake -DBUILD_MNNCLI_TEST=ON ..

echo "Compiling..."
make -j$(nproc)

echo "=========================================="
echo "Running ModelRepository tests..."
echo "=========================================="

# Check if test executable exists
if [ -f "test/model_repository_test" ]; then
    echo "Running model_repository_test..."
    ./test/model_repository_test
else
    echo "Warning: model_repository_test executable not found"
fi

echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "You can now run the example program:"
echo "  ./examples/model_repository_example"
echo ""
echo "Or test the integration with mnncli:"
echo "  ./mnncli model download gpt-oss-20b-MNN"
echo ""
echo "Note: Make sure model_market.json is available in one of these locations:"
echo "  - ./model_market.json"
echo "  - ./assets/model_market.json"
echo "  - ../assets/model_market.json"
