#!/bin/bash

# Unified Android Debug Script for MNNCLI
# This script handles deployment, testing, and debugging operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    echo -e "${BLUE}MNNCLI Android Debug Script${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy                 Deploy MNNCLI to device (default)"
    echo "  test                   Run basic functionality tests"
    echo "  video                  Test video processing functionality (requires -v option)"
    echo "  debug                  Run with verbose debugging"
    echo "  clean                  Clean device test directory"
    echo ""
    echo "Options:"
    echo "  -a, --arch ARCH        Target architecture (auto-detect if not specified)"
    echo "  -m, --model PATH       Model path on device"
    echo "  -v, --video PATH       Video file path for testing"
    echo "  -d, --device ID        Specific device ID"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Deploy to device"
    echo "  $0 test                # Run basic tests"
    echo "  $0 video -v /sdcard/test.mp4 # Test video functionality with specific video"
    echo "  $0 video -v /sdcard/test.mp4 -m /path/to/model.json # Test with video and model"
    echo ""
}

# Default values
COMMAND="deploy"
TARGET_ARCH=""
MODEL_PATH="/data/local/tmp/mnn_models/SmolVLM-256M-Instruct-MNN/config.json"
VIDEO_PATH="/data/local/tmp/mnncli_test/videos/xiujian.mp4"
DEVICE_ID=""
ANDROID_TEST_DIR="/data/local/tmp/mnncli_test"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--arch)
            TARGET_ARCH="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -v|--video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE_ID="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        deploy|test|video|debug|clean)
            COMMAND="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}MNNCLI Android Debug Script${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if adb is available
if ! command -v adb &> /dev/null; then
    echo -e "${RED}Error: adb command not found${NC}"
    echo -e "${YELLOW}Please install Android SDK Platform Tools${NC}"
    exit 1
fi

# Function to get device info
get_device_info() {
    if [ -n "$DEVICE_ID" ]; then
        echo "$DEVICE_ID"
    else
        local device=$(adb devices | grep "device$" | head -1 | cut -f1)
        if [ -z "$device" ]; then
            echo -e "${RED}Error: No Android device connected${NC}"
            exit 1
        fi
        echo "$device"
    fi
}

# Function to detect device architecture
detect_architecture() {
    local device="$1"
    if [ -z "$TARGET_ARCH" ]; then
        TARGET_ARCH=$(adb -s "$device" shell getprop ro.product.cpu.abi)
        echo -e "${GREEN}Detected device architecture: $TARGET_ARCH${NC}"
    else
        echo -e "${BLUE}Using specified architecture: $TARGET_ARCH${NC}"
    fi
}

# Function to check if mnncli is built and return build directory
check_mnncli_built() {
    local build_dir="$PROJECT_ROOT/build_mnncli_android"
    if [ "$TARGET_ARCH" != "arm64-v8a" ]; then
        build_dir="$PROJECT_ROOT/build_mnncli_android_multi"
    fi
    
    local executable="$build_dir/apps/mnncli/mnncli"
    if [ -f "$executable" ]; then
        echo -e "${GREEN}Found MNNCLI executable: $executable${NC}"
        echo "$build_dir"
        return 0
    else
        echo -e "${RED}Error: MNNCLI executable not found${NC}"
        echo -e "${YELLOW}Please run ./android_build.sh first${NC}"
        return 1
    fi
}

# Function to deploy to device
deploy_to_device() {
    local device="$1"
    local build_dir="$2"
    
    echo -e "${YELLOW}Deploying to Android device: $device${NC}"
    
    # Create test directory on device
    echo -e "${BLUE}Creating test directory: $ANDROID_TEST_DIR${NC}"
    adb -s "$device" shell mkdir -p "$ANDROID_TEST_DIR"
    
    # Copy executable
    local executable="$build_dir/apps/mnncli/mnncli"
    echo -e "${BLUE}Copying MNNCLI executable...${NC}"
    adb -s "$device" push "$executable" "$ANDROID_TEST_DIR/"
    
    # Copy libraries
    local lib_path="$build_dir/libMNN.so"
    if [ -f "$lib_path" ]; then
        echo -e "${BLUE}Copying libMNN.so...${NC}"
        adb -s "$device" push "$lib_path" "$ANDROID_TEST_DIR/"
    fi
    
    # Copy OpenCV libraries if they exist
    local opencv_lib_dir="$PROJECT_ROOT/opencv-android-sdk/sdk/native/libs/$TARGET_ARCH"
    if [ -d "$opencv_lib_dir" ]; then
        echo -e "${BLUE}Copying OpenCV libraries...${NC}"
        adb -s "$device" push "$opencv_lib_dir/"* "$ANDROID_TEST_DIR/"
    fi
    
    # Set permissions
    adb -s "$device" shell chmod +x "$ANDROID_TEST_DIR/mnncli"
    
    echo -e "${GREEN}Deployment completed successfully!${NC}"
}

# Function to run basic tests
run_basic_tests() {
    local device="$1"
    
    echo -e "${YELLOW}Running basic functionality tests...${NC}"
    
    # Test executable
    echo -e "${BLUE}Testing MNNCLI executable...${NC}"
    adb -s "$device" shell "cd $ANDROID_TEST_DIR && ./mnncli --help"
    
    # Test library loading
    echo -e "${BLUE}Testing library loading...${NC}"
    adb -s "$device" shell "cd $ANDROID_TEST_DIR && LD_LIBRARY_PATH=. ./mnncli --help"
    
    echo -e "${GREEN}Basic tests completed!${NC}"
}

# Function to test video functionality
test_video_functionality() {
    local device="$1"
    
    echo -e "${YELLOW}Testing video processing functionality...${NC}"
    
    # Check if video path is provided
    if [ -z "$VIDEO_PATH" ]; then
        echo -e "${YELLOW}No video path specified. Please provide a video file with -v option.${NC}"
        echo -e "${YELLOW}Example: $0 video -v /sdcard/your_video.mp4${NC}"
        return 1
    fi
    
    # Check if specified video file exists on device
    echo -e "${BLUE}Checking video file: $VIDEO_PATH${NC}"
    if ! adb -s "$device" shell "[ -f \"$VIDEO_PATH\" ]"; then
        echo -e "${RED}Error: Video file not found: $VIDEO_PATH${NC}"
        echo -e "${YELLOW}Please ensure the video file exists on the device${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ Video file found: $VIDEO_PATH${NC}"
    
    # Check if model path is provided
    if [ -z "$MODEL_PATH" ]; then
        echo -e "${YELLOW}No model path specified. Running basic video functionality test...${NC}"
        
        # Test 1: Basic help functionality
        echo -e "${BLUE}Test 1: Basic help functionality...${NC}"
        adb -s "$device" shell "cd $ANDROID_TEST_DIR && LD_LIBRARY_PATH=. ./mnncli -v --help"
        
        # Test 2: Check if video processing is available
        echo -e "${BLUE}Test 2: Checking video processing capabilities...${NC}"
        local help_output=$(adb -s "$device" shell "cd $ANDROID_TEST_DIR && LD_LIBRARY_PATH=. ./mnncli --help")
        if echo "$help_output" | grep -q "video\|vision\|multimodal"; then
            echo -e "${GREEN}✓ Video processing capabilities detected!${NC}"
        else
            echo -e "${YELLOW}⚠ Video processing capabilities not clearly indicated in help${NC}"
        fi
        
        # Test 3: Verify video file is accessible
        echo -e "${BLUE}Test 3: Verifying video file accessibility...${NC}"
        local file_size=$(adb -s "$device" shell "stat -c %s \"$VIDEO_PATH\" 2>/dev/null || echo 'unknown'")
        echo -e "${GREEN}✓ Video file size: $file_size bytes${NC}"
        
        echo -e "${GREEN}Basic video test completed!${NC}"
        echo -e "${BLUE}To test with a model, use: $0 video -m /path/to/model.json${NC}"
        return 0
    fi
    
    # Test video processing with verbose output
    echo -e "${BLUE}Testing video processing with model: $MODEL_PATH${NC}"
    echo -e "${BLUE}Video file: $VIDEO_PATH${NC}"
    
    # Run the video processing test
    local test_cmd="cd $ANDROID_TEST_DIR && LD_LIBRARY_PATH=. ./mnncli -v run -v -c $MODEL_PATH -p \"<video>$VIDEO_PATH</video>Describe this video.\""
    echo -e "${BLUE}Executing: $test_cmd${NC}"
    
    adb -s "$device" shell "$test_cmd"
    
    echo -e "${GREEN}Video test completed!${NC}"
}

# Function to run debug mode
run_debug_mode() {
    local device="$1"
    
    echo -e "${YELLOW}Running in debug mode...${NC}"
    
    # Enable verbose logging
    echo -e "${BLUE}Running with verbose logging...${NC}"
    adb -s "$device" shell "cd $ANDROID_TEST_DIR && LD_LIBRARY_PATH=. ./mnncli -v --help"
    
    # Show device info
    echo -e "${BLUE}Device information:${NC}"
    adb -s "$device" shell getprop ro.product.model
    adb -s "$device" shell getprop ro.build.version.release
    adb -s "$device" shell getprop ro.product.cpu.abi
    
    echo -e "${GREEN}Debug mode completed!${NC}"
}

# Function to clean device
clean_device() {
    local device="$1"
    
    echo -e "${YELLOW}Cleaning device test directory...${NC}"
    adb -s "$device" shell rm -rf "$ANDROID_TEST_DIR"
    echo -e "${GREEN}Device cleaned successfully!${NC}"
}

# Main execution
DEVICE=$(get_device_info)
echo -e "${GREEN}Connected device: $DEVICE${NC}"

detect_architecture "$DEVICE"

# Get build directory and handle errors separately
BUILD_DIR=""
if BUILD_DIR_OUTPUT=$(check_mnncli_built 2>&1); then
    # Extract the last line which should be the build directory
    BUILD_DIR=$(echo "$BUILD_DIR_OUTPUT" | sed -n '$p')
    # Remove the build directory line from output and display the rest
    echo "$BUILD_DIR_OUTPUT" | sed '$d'
else
    # Display error message and exit
    echo "$BUILD_DIR_OUTPUT"
    exit 1
fi

case "$COMMAND" in
    deploy)
        deploy_to_device "$DEVICE" "$BUILD_DIR"
        ;;
    test)
        deploy_to_device "$DEVICE" "$BUILD_DIR"
        run_basic_tests "$DEVICE"
        ;;
    video)
        deploy_to_device "$DEVICE" "$BUILD_DIR"
        test_video_functionality "$DEVICE"
        ;;
    debug)
        deploy_to_device "$DEVICE" "$BUILD_DIR"
        run_debug_mode "$DEVICE"
        ;;
    clean)
        clean_device "$DEVICE"
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        show_help
        exit 1
        ;;
esac

echo -e "${GREEN}Operation completed successfully!${NC}"
