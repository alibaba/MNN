#!/bin/bash

set -e

# 1. Build native test
cd "$(dirname "$0")"
mkdir -p build_test && cd build_test
cmake ../.. -DBUILD_TESTING=ON -DBUILD_ANDROID=ON -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21
make mnn_tts_test

# 2. Find generated so and test executable
SO_PATH=$(find . -name 'libmnn_tts.so' | head -n 1)
TEST_BIN=$(find . -name 'mnn_tts_test' | head -n 1)

if [[ ! -f "$SO_PATH" || ! -f "$TEST_BIN" ]]; then
  echo "Build failed: .so or test executable not found."
  exit 1
fi

# 3. Push to Android device
adb push "$SO_PATH" /data/local/tmp/
adb push "$TEST_BIN" /data/local/tmp/
adb shell chmod +x /data/local/tmp/mnn_tts_test

# 4. Set LD_LIBRARY_PATH and run test
adb shell "LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/mnn_tts_test"
