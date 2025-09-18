#!/bin/bash

echo "Testing AndroidVideoDecoder fix..."
echo "Starting video decoding test with enhanced debugging..."

# Use interactive mode to test video processing with detailed logs
adb shell "cd /data/local/tmp/mnncli_test && LD_LIBRARY_PATH=/data/local/tmp/mnncli_test:\$LD_LIBRARY_PATH ./mnncli --version" 2>&1

echo -e "\nTesting video decoding..."
echo "Note: Look for the following in logs to verify fix:"
echo "  - 'ImageReader callback triggered' - confirms callback is working"
echo "  - 'Fed X input buffers to codec' - shows input feeding"
echo "  - 'Drained X output buffers' - shows output processing"
echo "  - 'Frame added to queue' - shows frame queue working"
echo "  - 'Manually acquired frame' - shows fallback working if needed"
echo "  - Reduced 'No input buffer available yet' loops"
echo ""

# Test with a simple prompt that triggers video processing
echo "Creating test prompt..."
adb shell "cd /data/local/tmp/mnncli_test && echo '<video>xiujian.mp4</video> What do you see in this video?' > test_prompt.txt"

echo "Starting video processing test..."
echo "This will show detailed debug logs. Press Ctrl+C to stop if it runs too long."
echo ""

# Run with verbose logging to see the debug information
adb shell "cd /data/local/tmp/mnncli_test && timeout 30 sh -c 'LD_LIBRARY_PATH=/data/local/tmp/mnncli_test:\$LD_LIBRARY_PATH ./mnncli -v < test_prompt.txt 2>&1 | head -100'" || echo "Test completed (timeout or manual stop)"


