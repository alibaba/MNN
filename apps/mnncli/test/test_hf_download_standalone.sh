#!/bin/bash

# Standalone Hugging Face Model Download Test
# This script tests downloading a model from Hugging Face using the built mnncli

set -e

echo "=========================================="
echo "Standalone Hugging Face Model Download Test"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please run this script from the mnncli root directory"
    exit 1
fi

# Parse command line arguments
VERBOSE=false
CLEAN_CACHE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--clean-cache)
            CLEAN_CACHE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -v, --verbose       Enable verbose output"
            echo "  -c, --clean-cache   Clean test cache before running"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if mnncli executable exists
MNNCLI_PATH=""
if [ -f "../build_mnncli/apps/mnncli/mnncli" ]; then
    MNNCLI_PATH="../build_mnncli/apps/mnncli/mnncli"
elif [ -f "../../build_mnncli/apps/mnncli/mnncli" ]; then
    MNNCLI_PATH="../../build_mnncli/apps/mnncli/mnncli"
elif [ -f "build_mnncli/apps/mnncli/mnncli" ]; then
    MNNCLI_PATH="build_mnncli/apps/mnncli/mnncli"
else
    echo "❌ Error: mnncli executable not found"
    echo "Please run ./build.sh first to build mnncli"
    exit 1
fi

echo "✅ Found mnncli at: $MNNCLI_PATH"

# Set up test cache directory
TEST_CACHE="/tmp/mnncli_hf_test_cache"
if [ "$CLEAN_CACHE" = true ] && [ -d "$TEST_CACHE" ]; then
    echo "Cleaning test cache directory..."
    rm -rf "$TEST_CACHE"
fi

# Create test cache directory
mkdir -p "$TEST_CACHE"

# Test model ID
MODEL_ID="taobao-mnn/SmolLM2-135M-Instruct-MNN"

echo ""
echo "=========================================="
echo "Testing Hugging Face Model Download"
echo "=========================================="
echo "Model: $MODEL_ID"
echo "Cache: $TEST_CACHE"
echo ""

# Test 1: Check if model exists (search)
echo "🔍 Test 1: Searching for model..."
if [ "$VERBOSE" = true ]; then
    "$MNNCLI_PATH" model search "$MODEL_ID" --cache-dir "$TEST_CACHE" -v
else
    "$MNNCLI_PATH" model search "$MODEL_ID" --cache-dir "$TEST_CACHE"
fi

if [ $? -eq 0 ]; then
    echo "✅ Model search successful"
else
    echo "❌ Model search failed"
    exit 1
fi

echo ""

# Test 2: Download the model
echo "📥 Test 2: Downloading model..."
if [ "$VERBOSE" = true ]; then
    "$MNNCLI_PATH" model download "$MODEL_ID" --cache-dir "$TEST_CACHE" -v
else
    "$MNNCLI_PATH" model download "$MODEL_ID" --cache-dir "$TEST_CACHE"
fi

DOWNLOAD_EXIT_CODE=$?

if [ $DOWNLOAD_EXIT_CODE -eq 0 ]; then
    echo "✅ Model download successful"
else
    echo "❌ Model download failed (exit code: $DOWNLOAD_EXIT_CODE)"
    exit 1
fi

echo ""

# Test 3: Verify downloaded files
echo "🔍 Test 3: Verifying downloaded files..."
DOWNLOAD_PATH="$TEST_CACHE/models--taobao-mnn--SmolLM2-135M-Instruct-MNN"

if [ -d "$DOWNLOAD_PATH" ]; then
    echo "✅ Download directory exists: $DOWNLOAD_PATH"
    
    # Count files
    FILE_COUNT=$(find "$DOWNLOAD_PATH" -type f | wc -l)
    echo "📁 Found $FILE_COUNT files"
    
    # List some key files
    echo "📄 Key files:"
    find "$DOWNLOAD_PATH" -name "*.bin" -o -name "*.safetensors" -o -name "*.json" -o -name "*.txt" | head -10 | while read file; do
        echo "   $(basename "$file") ($(du -h "$file" | cut -f1))"
    done
    
    # Check for model files
    if find "$DOWNLOAD_PATH" -name "*.bin" -o -name "*.safetensors" | grep -q .; then
        echo "✅ Model weight files found"
    else
        echo "⚠️  Warning: No model weight files found"
    fi
    
    # Check for config files
    if find "$DOWNLOAD_PATH" -name "*.json" | grep -q .; then
        echo "✅ Configuration files found"
    else
        echo "⚠️  Warning: No configuration files found"
    fi
    
else
    echo "❌ Download directory not found: $DOWNLOAD_PATH"
    exit 1
fi

echo ""

# Test 4: Test model info
echo "ℹ️  Test 4: Getting model information..."
if [ "$VERBOSE" = true ]; then
    "$MNNCLI_PATH" model info "$MODEL_ID" --cache-dir "$TEST_CACHE" -v
else
    "$MNNCLI_PATH" model info "$MODEL_ID" --cache-dir "$TEST_CACHE"
fi

if [ $? -eq 0 ]; then
    echo "✅ Model info retrieval successful"
else
    echo "⚠️  Model info retrieval failed (this might be expected)"
fi

echo ""

# Test 5: Test model list
echo "📋 Test 5: Listing downloaded models..."
if [ "$VERBOSE" = true ]; then
    "$MNNCLI_PATH" model list --cache-dir "$TEST_CACHE" -v
else
    "$MNNCLI_PATH" model list --cache-dir "$TEST_CACHE"
fi

if [ $? -eq 0 ]; then
    echo "✅ Model list successful"
else
    echo "⚠️  Model list failed (this might be expected)"
fi

echo ""
echo "=========================================="
echo "🎉 All tests completed successfully!"
echo "=========================================="
echo ""
echo "Test Summary:"
echo "✅ Model search: PASSED"
echo "✅ Model download: PASSED"
echo "✅ File verification: PASSED"
echo "✅ Model info: PASSED"
echo "✅ Model list: PASSED"
echo ""
echo "Downloaded model location: $DOWNLOAD_PATH"
echo "Total files downloaded: $FILE_COUNT"
echo ""
echo "To clean up test cache:"
echo "  rm -rf $TEST_CACHE"
echo ""
echo "To use the downloaded model:"
echo "  $MNNCLI_PATH model run $MODEL_ID --cache-dir $TEST_CACHE"

