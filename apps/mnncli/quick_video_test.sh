#!/bin/bash

# Quick Video Test Script for MNNCLI
# This script quickly tests video processing functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Quick Video Test for MNNCLI${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if we're on Android or desktop
if [ -f "/system/bin/sh" ] || [ -n "$ANDROID_ROOT" ]; then
    echo -e "${YELLOW}Android environment detected${NC}"
    IS_ANDROID=true
else
    echo -e "${BLUE}Desktop environment detected${NC}"
    IS_ANDROID=false
fi

if [ "$IS_ANDROID" = true ]; then
    # Android environment
    MNNCLI_PATH="/data/local/tmp/mnncli_test/mnncli"
    TEST_VIDEO="/sdcard/Download/test_video.mp4"
    
    # Check if mnncli exists
    if [ ! -f "$MNNCLI_PATH" ]; then
        echo -e "${RED}Error: MNNCLI not found at $MNNCLI_PATH${NC}"
        echo -e "${YELLOW}Please run the deployment script first${NC}"
        exit 1
    fi
    
    # Set library path
    export LD_LIBRARY_PATH="/data/local/tmp/mnncli_test:$LD_LIBRARY_PATH"
    
    # Test video processing with verbose mode
    echo -e "${YELLOW}Testing video processing on Android...${NC}"
    echo "Command: echo 'What is in this video? <video>$TEST_VIDEO</video>' | $MNNCLI_PATH -v run -c /path/to/model/config.json -p"
    
    # Check if test video exists
    if [ -f "$TEST_VIDEO" ]; then
        echo -e "${GREEN}Test video found, running test...${NC}"
        echo "What is in this video? <video>$TEST_VIDEO</video>" | "$MNNCLI_PATH" -v run -c /path/to/model/config.json -p 2>&1 || {
            echo -e "${RED}Video processing test failed${NC}"
            echo "Check the debug output above for specific error details"
        }
    else
        echo -e "${YELLOW}Test video not found at $TEST_VIDEO${NC}"
        echo "Please place a test video file at that location"
    fi
else
    # Desktop environment
    MNNCLI_PATH="./mnncli"
    
    # Check if mnncli exists
    if [ ! -f "$MNNCLI_PATH" ]; then
        echo -e "${RED}Error: MNNCLI not found in current directory${NC}"
        echo -e "${YELLOW}Please build the project first${NC}"
        exit 1
    fi
    
    # Test video processing with verbose mode
    echo -e "${YELLOW}Testing video processing on desktop...${NC}"
    echo "Command: echo 'What is in this video? <video>test_video.mp4</video>' | $MNNCLI_PATH -v run -c /path/to/model/config.json -p"
    
    # Check if test video exists
    if [ -f "test_video.mp4" ]; then
        echo -e "${GREEN}Test video found, running test...${NC}"
        echo "What is in this video? <video>test_video.mp4</video>" | "$MNNCLI_PATH" -v run -c /path/to/model/config.json -p 2>&1 || {
            echo -e "${RED}Video processing test failed${NC}"
            echo "Check the debug output above for specific error details"
        }
    else
        echo -e "${YELLOW}Test video not found in current directory${NC}"
        echo "Please place a test video file named 'test_video.mp4' in the current directory"
    fi
fi

echo -e "${GREEN}Quick video test completed!${NC}"
