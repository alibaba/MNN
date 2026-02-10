#!/bin/bash

# Unified Android Build Script for MNNCLI
# This script supports both single and multi-architecture builds with OpenCV support

set -e

# Check and clone libyuv if it doesn't exist
LIBYUV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../3rd_party/libyuv"
LIBYUV_REPO="https://github.com/bilibili/libyuv.git"
LIBYUV_COMMIT="1f337a783783776e2735d88628e3674681284d72"

if [ ! -d "$LIBYUV_DIR" ]; then
    echo "Cloning libyuv from $LIBYUV_REPO..."
    git clone "$LIBYUV_REPO" "$LIBYUV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clone libyuv."
        exit 1
    fi
    # (cd "$LIBYUV_DIR" && git checkout "$LIBYUV_COMMIT")
    # if [ $? -ne 0 ]; then
    #     echo "Error: Failed to checkout libyuv commit."
    #     exit 1
    # fi
else
    echo "libyuv directory already exists. Skipping clone."
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    echo -e "${BLUE}MNNCLI Android Build Script${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -a, --arch ARCH        Target architecture (arm64-v8a, armeabi-v7a, x86, x86_64)"
    echo "  -m, --multi            Build for multiple architectures"
    echo "  -c, --clean            Clean build directory before building"
    echo "  -v, --verbose          Enable verbose output"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Build for default architecture (arm64-v8a)"
    echo "  $0 -a arm64-v8a       # Build for specific architecture"
    echo "  $0 -m                  # Build for all supported architectures"
    echo "  $0 -c                  # Clean build and rebuild"
    echo ""
}

# Default values
TARGET_ARCH="arm64-v8a"
BUILD_MULTI=false
CLEAN_BUILD=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--arch)
            TARGET_ARCH="$2"
            shift 2
            ;;
        -m|--multi)
            BUILD_MULTI=true
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Building MNNCLI for Android with OpenCV support...${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if ANDROID_NDK is set
if [ -z "$ANDROID_NDK" ]; then
    echo -e "${RED}Error: ANDROID_NDK environment variable is not set${NC}"
    echo -e "${YELLOW}Please set ANDROID_NDK to your Android NDK path${NC}"
    echo -e "${YELLOW}Example: export ANDROID_NDK=/path/to/android-ndk${NC}"
    exit 1
fi

# Check if ANDROID_NDK exists
if [ ! -d "$ANDROID_NDK" ]; then
    echo -e "${RED}Error: ANDROID_NDK directory does not exist: $ANDROID_NDK${NC}"
    exit 1
fi

# OpenCV configuration
OPENCV_VERSION="4.8.0"
OPENCV_ANDROID_SDK_DIR="$PROJECT_ROOT/opencv-android-sdk"
OPENCV_DOWNLOAD_URL="https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/opencv-${OPENCV_VERSION}-android-sdk.zip"

echo -e "${YELLOW}OpenCV version: $OPENCV_VERSION${NC}"
echo -e "${YELLOW}OpenCV SDK directory: $OPENCV_ANDROID_SDK_DIR${NC}"

# Download and extract OpenCV Android SDK if not already present
if [ ! -d "$OPENCV_ANDROID_SDK_DIR" ]; then
    echo -e "${YELLOW}Downloading OpenCV Android SDK...${NC}"
    
    # Create temporary directory for download
    TEMP_DIR="$PROJECT_ROOT/temp_opencv_download"
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    # Download OpenCV Android SDK
    echo -e "${YELLOW}Downloading from: $OPENCV_DOWNLOAD_URL${NC}"
    if command -v wget &> /dev/null; then
        wget -O opencv-android-sdk.zip "$OPENCV_DOWNLOAD_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o opencv-android-sdk.zip "$OPENCV_DOWNLOAD_URL"
    else
        echo -e "${RED}Error: Neither wget nor curl is available. Please install one of them.${NC}"
        exit 1
    fi
    
    # Check if download was successful
    if [ ! -f "opencv-android-sdk.zip" ]; then
        echo -e "${RED}Error: Failed to download OpenCV Android SDK${NC}"
        exit 1
    fi
    
    # Extract OpenCV Android SDK
    echo -e "${YELLOW}Extracting OpenCV Android SDK...${NC}"
    unzip -q opencv-android-sdk.zip
    
    # Move to final location
    if [ -d "OpenCV-android-sdk" ]; then
        mv "OpenCV-android-sdk" "$OPENCV_ANDROID_SDK_DIR"
    else
        echo -e "${RED}Error: Unexpected archive structure${NC}"
        exit 1
    fi
    
    # Clean up temporary files
    cd "$PROJECT_ROOT"
    rm -rf "$TEMP_DIR"
    
    echo -e "${GREEN}OpenCV Android SDK downloaded and extracted successfully!${NC}"
else
    echo -e "${GREEN}OpenCV Android SDK already exists at: $OPENCV_ANDROID_SDK_DIR${NC}"
fi

# Verify OpenCV Android SDK structure
if [ ! -d "$OPENCV_ANDROID_SDK_DIR/sdk/native" ]; then
    echo -e "${RED}Error: Invalid OpenCV Android SDK structure${NC}"
    echo -e "${YELLOW}Expected directory: $OPENCV_ANDROID_SDK_DIR/sdk/native${NC}"
    exit 1
fi

# Set OpenCV environment variable for CMake
export OPENCV_ANDROID_SDK="$OPENCV_ANDROID_SDK_DIR"
echo -e "${GREEN}OpenCV Android SDK path set: $OPENCV_ANDROID_SDK${NC}"

# Build configuration
if [ "$BUILD_MULTI" = true ]; then
    echo -e "${BLUE}Building for multiple architectures...${NC}"
    BUILD_DIR="$PROJECT_ROOT/build_mnncli_android_multi"
    
    # Clean build directory if requested
    if [ "$CLEAN_BUILD" = true ]; then
        echo -e "${YELLOW}Cleaning build directory...${NC}"
        rm -rf "$BUILD_DIR"
    fi
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure for multiple architectures
    cmake .. \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="arm64-v8a;armeabi-v7a;x86;x86_64" \
        -DANDROID_PLATFORM=android-24 \
        -DCMAKE_BUILD_TYPE=Release \
        -DMNN_BUILD_TRAIN=OFF \
        -DMNN_BUILD_TRAIN_MINI=OFF \
        -DMNN_BUILD_DEMO=OFF \
        -DMNN_BUILD_QUANTOOLS=OFF \
        -DMNN_BUILD_TEST=OFF \
        -DMNN_BUILD_BENCHMARK=OFF \
        -DMNN_BUILD_FOR_ANDROID=ON \
        -DMNN_USE_OPENCV=ON \
        -DOPENCV_ANDROID_SDK="$OPENCV_ANDROID_SDK" \
        -DCMAKE_VERBOSE_MAKEFILE="$VERBOSE"
    
    # Build
    echo -e "${BLUE}Building MNNCLI for multiple architectures...${NC}"
    make -j$(nproc)
    
else
    echo -e "${BLUE}Building for single architecture: $TARGET_ARCH${NC}"
    BUILD_DIR="$PROJECT_ROOT/build_mnncli_android"
    
    # Clean build directory if requested
    if [ "$CLEAN_BUILD" = true ]; then
        echo -e "${YELLOW}Cleaning build directory...${NC}"
        rm -rf "$BUILD_DIR"
    fi
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure for single architecture
    cmake .. \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="$TARGET_ARCH" \
        -DANDROID_PLATFORM=android-24 \
        -DCMAKE_BUILD_TYPE=Release \
        -DMNN_BUILD_TRAIN=OFF \
        -DMNN_BUILD_TRAIN_MINI=OFF \
        -DMNN_BUILD_DEMO=OFF \
        -DMNN_BUILD_QUANTOOLS=OFF \
        -DMNN_BUILD_TEST=OFF \
        -DMNN_BUILD_BENCHMARK=OFF \
        -DMNN_BUILD_FOR_ANDROID=ON \
        -DMNN_USE_OPENCV=ON \
        -DOPENCV_ANDROID_SDK="$OPENCV_ANDROID_SDK" \
        -DCMAKE_VERBOSE_MAKEFILE="$VERBOSE"
    
    # Build
    echo -e "${BLUE}Building MNNCLI for $TARGET_ARCH...${NC}"
    make -j$(nproc)
fi

# Check if build was successful
if [ -f "apps/mnncli/mnncli" ]; then
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo -e "${GREEN}Executable location: $BUILD_DIR/apps/mnncli/mnncli${NC}"
    
    # Show file size and permissions
    ls -lh "apps/mnncli/mnncli"
    
    # Check if libMNN.so was created
    if [ -f "libMNN.so" ]; then
        echo -e "${GREEN}libMNN.so created successfully${NC}"
        ls -lh "libMNN.so"
    else
        echo -e "${YELLOW}Warning: libMNN.so not found${NC}"
    fi
    
else
    echo -e "${RED}Build failed! Executable not found${NC}"
    exit 1
fi

echo -e "${GREEN}Android build completed!${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo -e "${BLUE}1. Run ./android_debug.sh to deploy and test on device${NC}"
echo -e "${BLUE}2. Or run ./android_debug.sh --help for more options${NC}"
