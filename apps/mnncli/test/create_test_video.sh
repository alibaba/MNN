#!/bin/bash

# Create a simple test video for Android testing
# This script creates a minimal MP4 file for testing purposes

set -e

echo "Creating test video for Android testing..."

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg not found. Please install ffmpeg first."
    echo "On macOS: brew install ffmpeg"
    echo "On Ubuntu: sudo apt install ffmpeg"
    exit 1
fi

# Create a simple test video (1 second, 320x240, black frame with text)
echo "Generating test video..."
ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=1 \
       -f lavfi -i sine=frequency=1000:duration=1 \
       -c:v libx264 -preset ultrafast -crf 23 \
       -c:a aac -b:a 128k \
       -y test_video.mp4

if [ -f "test_video.mp4" ]; then
    echo "Test video created successfully: test_video.mp4"
    echo "File size: $(ls -lh test_video.mp4 | awk '{print $5}')"
    echo ""
    echo "To use this video for testing:"
    echo "1. Copy to device: adb push test_video.mp4 /sdcard/DCIM/Camera/"
    echo "2. Run test: ./android_debug.sh -v /sdcard/DCIM/Camera/test_video.mp4 video"
else
    echo "Error: Failed to create test video"
    exit 1
fi
