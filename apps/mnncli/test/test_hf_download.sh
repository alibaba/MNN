#!/bin/bash

# Test script for Hugging Face Model Download
# This script builds and tests the Hugging Face model download functionality

set -e

echo "=========================================="
echo "Testing Hugging Face Model Download"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please run this script from the mnncli root directory"
    exit 1
fi

# Parse command line arguments
VERBOSE=false
CLEAN_BUILD=false
RUN_TEST=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        --build-only)
            RUN_TEST=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -v, --verbose     Enable verbose output"
            echo "  -c, --clean       Clean build directory before building"
            echo "  --build-only      Only build, don't run tests"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Create build directory
BUILD_DIR="build_test"
if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

echo "Configuring CMake..."
if [ "$VERBOSE" = true ]; then
    cmake -DBUILD_MNNCLI_TEST=ON -DCMAKE_VERBOSE_MAKEFILE=ON ..
else
    cmake -DBUILD_MNNCLI_TEST=ON ..
fi

echo "Building HF model download test..."
if [ "$VERBOSE" = true ]; then
    make -j$(nproc) VERBOSE=1
else
    make -j$(nproc)
fi

if [ "$RUN_TEST" = false ]; then
    echo "=========================================="
    echo "Build completed successfully!"
    echo "=========================================="
    echo ""
    echo "To run the test manually:"
    echo "  cd $BUILD_DIR"
    echo "  ./test/hf_model_download_test"
    echo ""
    echo "To run with verbose output:"
    echo "  ./test/hf_model_download_test -v"
    exit 0
fi

echo "=========================================="
echo "Running Hugging Face Model Download Tests..."
echo "=========================================="

# Check if test executable exists
if [ -f "test/hf_model_download_test" ]; then
    echo "Running hf_model_download_test..."
    echo ""
    
    # Run the test with appropriate flags
    if [ "$VERBOSE" = true ]; then
        ./test/hf_model_download_test -v
    else
        ./test/hf_model_download_test
    fi
    
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "🎉 All tests passed successfully!"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "❌ Some tests failed (exit code: $TEST_EXIT_CODE)"
        echo "=========================================="
        exit $TEST_EXIT_CODE
    fi
else
    echo "❌ Error: hf_model_download_test executable not found"
    echo "Build may have failed. Check the build output above."
    exit 1
fi

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "✅ Build completed successfully"
echo "✅ HF model download test executed"
echo "✅ Model: taobao-mnn/SmolLM2-135M-Instruct-MNN"
echo ""
echo "Test cache location: /tmp/mnncli_test_cache"
echo "You can inspect downloaded files there."
echo ""
echo "To clean up test cache:"
echo "  rm -rf /tmp/mnncli_test_cache"
echo ""
echo "To run individual tests:"
echo "  cd $BUILD_DIR"
echo "  ./test/hf_model_download_test -v"

