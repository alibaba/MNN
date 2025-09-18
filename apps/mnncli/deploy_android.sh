#!/bin/bash

# Android deployment script for MNNCLI with OpenCV
# This script copies the executable and all required libraries to the Android device
# Based on android_video_test.sh for comprehensive video testing setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}MNNCLI Android Deployment Script with OpenCV Support${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
ANDROID_TEST_DIR="/data/local/tmp/mnncli_test"
MODEL_PATH="/data/local/tmp/mnn_models/SmolVLM-256M-Instruct-MNN"
DEFAULT_VIDEO_PATH="/data/local/tmp/mnncli_test/videos/xiujian.mp4"

# Check if adb is available
if ! command -v adb &> /dev/null; then
    echo -e "${RED}Error: adb command not found${NC}"
    echo -e "${YELLOW}Please install Android SDK Platform Tools or add them to your PATH${NC}"
    exit 1
fi

# Check if device is connected
echo -e "${YELLOW}Checking for connected Android devices...${NC}"
if ! adb devices | grep -q "device$"; then
    echo -e "${RED}Error: No Android device connected${NC}"
    echo -e "${YELLOW}Please connect a device via USB or start an emulator${NC}"
    exit 1
fi

echo -e "${GREEN}Android device connected successfully${NC}"

# Get device info
DEVICE=$(adb devices | grep "device$" | head -1 | cut -f1)
echo -e "${GREEN}Connected device: $DEVICE${NC}"

# Check device architecture
echo -e "${YELLOW}Checking device architecture...${NC}"
DEVICE_ARCH=$(adb shell getprop ro.product.cpu.abi)
echo -e "${GREEN}Device architecture: $DEVICE_ARCH${NC}"

# Set paths
BUILD_DIR="$PROJECT_ROOT/build_mnncli_android"
OPENCV_SDK_DIR="$PROJECT_ROOT/opencv-android-sdk"

# Check if build exists
if [ ! -f "$BUILD_DIR/apps/mnncli/mnncli" ]; then
    echo -e "${RED}Error: MNNCLI executable not found${NC}"
    echo -e "${YELLOW}Please run ./build_android.sh first${NC}"
    exit 1
fi

# Function to check if mnncli is built
check_mnncli_built() {
    local executable="$BUILD_DIR/apps/mnncli/mnncli"
    
    if [ -f "$executable" ]; then
        echo -e "${GREEN}Found existing Android mnncli executable: $executable${NC}"
        return 0
    else
        echo -e "${YELLOW}Android mnncli executable not found${NC}"
        return 1
    fi
}

# Function to find libMNN.so
find_libmnn() {
    local lib_path="$BUILD_DIR/libMNN.so"
    
    if [ -f "$lib_path" ]; then
        echo "$lib_path"
        return 0
    else
        echo -e "${YELLOW}Warning: libMNN.so not found at $lib_path${NC}"
        echo -e "${YELLOW}You may need to manually locate and copy it${NC}"
        return 1
    fi
}

# Check if build exists
if ! check_mnncli_built; then
    echo -e "${RED}Error: MNNCLI executable not found${NC}"
    echo -e "${YELLOW}Please run ./build_android.sh first${NC}"
    exit 1
fi

# Check if OpenCV SDK exists
if [ ! -d "$OPENCV_SDK_DIR" ]; then
    echo -e "${RED}Error: OpenCV Android SDK not found${NC}"
    echo -e "${YELLOW}Please run ./build_android.sh first to download OpenCV${NC}"
    exit 1
fi

# Determine OpenCV library path based on device architecture
case "$DEVICE_ARCH" in
    "arm64-v8a")
        OPENCV_LIB_DIR="$OPENCV_SDK_DIR/sdk/native/libs/arm64-v8a"
        ;;
    "armeabi-v7a")
        OPENCV_LIB_DIR="$OPENCV_SDK_DIR/sdk/native/libs/armeabi-v7a"
        ;;
    "x86")
        OPENCV_LIB_DIR="$OPENCV_SDK_DIR/sdk/native/libs/x86"
        ;;
    "x86_64")
        OPENCV_LIB_DIR="$OPENCV_SDK_DIR/sdk/native/libs/x86_64"
        ;;
    *)
        echo -e "${YELLOW}Unknown architecture: $DEVICE_ARCH, using arm64-v8a${NC}"
        OPENCV_LIB_DIR="$OPENCV_SDK_DIR/sdk/native/libs/arm64-v8a"
        ;;
esac

# Check if OpenCV library exists for this architecture
if [ ! -d "$OPENCV_LIB_DIR" ]; then
    echo -e "${RED}Error: OpenCV library directory not found for architecture: $DEVICE_ARCH${NC}"
    echo -e "${YELLOW}Available architectures:${NC}"
    ls -la "$OPENCV_SDK_DIR/sdk/native/libs/"
    exit 1
fi

echo -e "${GREEN}Using OpenCV library directory: $OPENCV_LIB_DIR${NC}"

# Create test directory on device
echo -e "${YELLOW}Creating test directory on device...${NC}"
adb shell "mkdir -p $ANDROID_TEST_DIR"

# Copy MNNCLI executable
echo -e "${YELLOW}Copying MNNCLI executable...${NC}"
adb push "$BUILD_DIR/apps/mnncli/mnncli" "$ANDROID_TEST_DIR/"

# Copy OpenCV library
echo -e "${YELLOW}Copying OpenCV library...${NC}"
adb push "$OPENCV_LIB_DIR/libopencv_java4.so" "$ANDROID_TEST_DIR/"

# Copy libMNN.so if found
lib_path=$(find_libmnn)
if [ -n "$lib_path" ] && [ -f "$lib_path" ]; then
    echo -e "${BLUE}Copying libMNN.so...${NC}"
    adb push "$lib_path" "$ANDROID_TEST_DIR/libMNN.so"
    adb shell "chmod 644 $ANDROID_TEST_DIR/libMNN.so"
else
    echo -e "${YELLOW}libMNN.so not found, skipping library deployment${NC}"
    echo -e "${YELLOW}You may need to manually copy it to $ANDROID_TEST_DIR${NC}"
fi

# Set permissions
echo -e "${YELLOW}Setting permissions...${NC}"
adb shell "chmod +x $ANDROID_TEST_DIR/mnncli"
adb shell "chmod 644 $ANDROID_TEST_DIR/libopencv_java4.so"

# Function to generate video test script
generate_video_test_script() {
    echo -e "${YELLOW}Generating video test script...${NC}"
    
    local test_script_content="#!/system/bin/sh

# MNNCLI Video Test Script for Android
# Generated by deploy_android.sh

set -e

# Configuration
MNNCLI_PATH=\"/data/local/tmp/mnncli_test/mnncli\"
MODEL_CONFIG=\"$MODEL_PATH/config.json\"
DEFAULT_VIDEO=\"$DEFAULT_VIDEO_PATH\"

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m'

echo -e \"\${BLUE}MNNCLI Video Test Script\${NC}\"

# Check if mnncli exists
if [ ! -f \"\$MNNCLI_PATH\" ]; then
    echo -e \"\${RED}Error: mnncli not found at \$MNNCLI_PATH\${NC}\"
    exit 1
fi

# Check if model config exists
if [ ! -f \"\$MODEL_CONFIG\" ]; then
    echo -e \"\${YELLOW}Warning: Model config not found at \$MODEL_CONFIG\${NC}\"
    echo -e \"\${YELLOW}You can still test basic functionality without a model\${NC}\"
fi

# Set library path
export LD_LIBRARY_PATH=\"/data/local/tmp/mnncli_test:\$LD_LIBRARY_PATH\"
echo -e \"\${BLUE}Set LD_LIBRARY_PATH: \$LD_LIBRARY_PATH\${NC}\"

# Get video path from command line or use default
VIDEO_PATH=\"\${1:-\$DEFAULT_VIDEO}\"
echo -e \"\${BLUE}Using video: \$VIDEO_PATH\${NC}\"

# Check if video exists
if [ ! -f \"\$VIDEO_PATH\" ]; then
    echo -e \"\${YELLOW}Warning: Video file not found at \$VIDEO_PATH\${NC}\"
    echo -e \"\${YELLOW}You can specify a different video path as an argument\${NC}\"
    echo -e \"\${YELLOW}Usage: \$0 [video_path]\${NC}\"
    echo -e \"\${YELLOW}Example: \$0 /sdcard/Download/test_video.mp4\${NC}\"
fi

# Test basic functionality first
echo -e \"\${YELLOW}Testing basic mnncli functionality...\${NC}\"
if \"\$MNNCLI_PATH\" --help; then
    echo -e \"\${GREEN}Basic functionality test passed!\${NC}\"
else
    echo -e \"\${RED}Basic functionality test failed!\${NC}\"
    exit 1
fi

# Test video processing capability if model is available
if [ -f \"\$MODEL_CONFIG\" ]; then
    echo -e \"\${YELLOW}Testing video processing capability...\${NC}\"
    echo -e \"\${BLUE}Running: mnncli run -c \$MODEL_CONFIG -p \\\"what is in the video:<video>\$VIDEO_PATH</video>\\\"\${NC}\"

    # Execute the command
    if \"\$MNNCLI_PATH\" run -c \"\$MODEL_CONFIG\" -p \"what is in the video:<video>\$VIDEO_PATH</video>\"; then
        echo -e \"\${GREEN}Video test completed successfully!\${NC}\"
    else
        echo -e \"\${RED}Video test failed!\${NC}\"
        echo -e \"\${YELLOW}This might be due to:\${NC}\"
        echo -e \"\${YELLOW}1. Model not properly loaded\${NC}\"
        echo -e \"\${YELLOW}2. Video file not accessible\${NC}\"
        echo -e \"\${YELLOW}3. Insufficient device memory\${NC}\"
        echo -e \"\${YELLOW}4. Model configuration issues\${NC}\"
        exit 1
    fi
else
    echo -e \"\${YELLOW}Skipping video processing test (no model config)\${NC}\"
    echo -e \"\${BLUE}To test video processing, deploy a model to: \$MODEL_PATH\${NC}\"
fi

echo -e \"\${GREEN}Video test script execution completed!\${NC}\"
"
    
    # Write script to device
    echo "$test_script_content" | adb shell "cat > $ANDROID_TEST_DIR/mnn_test_video.sh"
    
    # Make script executable
    adb shell chmod +x "$ANDROID_TEST_DIR/mnn_test_video.sh"
    
    echo -e "${GREEN}Video test script generated successfully!${NC}"
    echo -e "${BLUE}Script location: $ANDROID_TEST_DIR/mnn_test_video.sh${NC}"
}

# Function to test the setup
test_setup() {
    echo -e "${YELLOW}Testing the setup...${NC}"
    
    # Test if mnncli can run
    echo -e "${BLUE}Testing mnncli basic functionality...${NC}"
    if adb shell "cd $ANDROID_TEST_DIR && export LD_LIBRARY_PATH=\"/data/local/tmp/mnncli_test:\$LD_LIBRARY_PATH\" && ./mnncli --help"; then
        echo -e "${GREEN}mnncli basic functionality test passed!${NC}"
    else
        echo -e "${RED}mnncli basic functionality test failed!${NC}"
        return 1
    fi
    
    # Test if video test script exists and is executable
    echo -e "${BLUE}Testing video test script...${NC}"
    if adb shell "[ -x $ANDROID_TEST_DIR/mnn_test_video.sh ]"; then
        echo -e "${GREEN}Video test script is executable!${NC}"
    else
        echo -e "${RED}Video test script is not executable!${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Setup test completed successfully!${NC}"
    return 0
}

# Set library path and test
echo -e "${YELLOW}Testing deployment...${NC}"
adb shell "cd $ANDROID_TEST_DIR && export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$ANDROID_TEST_DIR && ./mnncli --help"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Deployment successful!${NC}"
    
    # Generate video test script
    generate_video_test_script
    
    # Test the setup
    if test_setup; then
        echo -e "${GREEN}All setup steps completed successfully!${NC}"
    else
        echo -e "${RED}Setup test failed!${NC}"
        exit 1
    fi
    
    # Final instructions
    echo -e "${BLUE}Deployment Summary:${NC}"
    echo -e "${GREEN}✓ mnncli deployed to: $ANDROID_TEST_DIR${NC}"
    echo -e "${GREEN}✓ OpenCV library deployed: libopencv_java4.so${NC}"
    echo -e "${GREEN}✓ Video test script generated: $ANDROID_TEST_DIR/mnn_test_video.sh${NC}"
    echo -e "${GREEN}✓ Environment configured for video testing${NC}"
    
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "${YELLOW}1. Deploy your video model to: $MODEL_PATH${NC}"
    echo -e "${YELLOW}2. Copy a test video to the device${NC}"
    echo -e "${YELLOW}3. Run video test: adb shell $ANDROID_TEST_DIR/mnn_test_video.sh [video_path]${NC}"
    echo -e "${YELLOW}4. Example: adb shell $ANDROID_TEST_DIR/mnn_test_video.sh /sdcard/test.mp4${NC}"
    
    echo -e ""
    echo -e "${GREEN}To run MNNCLI on your device:${NC}"
    echo -e "${YELLOW}adb shell \"cd $ANDROID_TEST_DIR && export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$ANDROID_TEST_DIR && ./mnncli\"${NC}"
    echo -e ""
    echo -e "${GREEN}To test video processing:${NC}"
    echo -e "${YELLOW}adb shell \"cd $ANDROID_TEST_DIR && export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$ANDROID_TEST_DIR && echo 'What is in this video? <video>/sdcard/test.mp4</video>' | ./mnncli\"${NC}"
else
    echo -e "${RED}Deployment test failed${NC}"
    echo -e "${YELLOW}Checking device logs...${NC}"
    adb logcat -d | tail -20
    exit 1
fi

echo -e "${GREEN}Deployment completed successfully!${NC}"
