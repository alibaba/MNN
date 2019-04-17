#!/bin/bash
adb push ./resizeTest.out /data/local/tmp/MNN/resizeTest.out

adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH&& ./resizeTest.out 36 18 40 256 144"
adb pull /data/local/tmp/MNN/output.txt output_resize.txt
