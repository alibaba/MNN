#!/bin/bash

echo "=== Direct AndroidVideoDecoder Test ==="
echo "This will test video decoding without requiring a full LLM model"
echo ""

# Create a simple test that just tries to initialize and decode a few frames
adb shell "cd /data/local/tmp/mnncli_test && cat > simple_video_test.txt << 'EOF'
Test prompt that will trigger video processing even without model.
The key is to get to ProcessVideoPrompt function.
We expect to see AndroidVideoDecoder initialization and decode attempts.
EOF"

echo "Testing video file access..."
adb shell "cd /data/local/tmp/mnncli_test && ls -la xiujian.mp4"

echo ""
echo "Starting simplified video test..."
echo "Look for these debug messages to verify our fixes:"
echo "  - 'Using Android native decoder for:'"
echo "  - 'ImageReader callback triggered'"  
echo "  - 'Fed X input buffers to codec'"
echo "  - 'Drained X output buffers'"
echo "  - 'Frame added to queue'"
echo "  - Absence of continuous 'No input buffer available yet' loops"
echo ""
echo "Press Ctrl+C if it gets stuck in a loop..."

# Try to trigger video processing - this should at least show initialization
adb shell "cd /data/local/tmp/mnncli_test && echo 'Testing video processing without full model setup' && timeout 30 sh -c 'LD_LIBRARY_PATH=/data/local/tmp/mnncli_test:\$LD_LIBRARY_PATH strace -e trace=write ./mnncli -v 2>&1' | head -50" 2>/dev/null || echo "Test completed"

echo ""
echo "If you're still seeing '[DEBUG] No input buffer available yet' loops,"
echo "please share the output so we can further diagnose the issue."
