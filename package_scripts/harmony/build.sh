#!/bin/bash
#
# Package MNN for OpenHarmony / HarmonyOS NEXT.
#
# Requires the OpenHarmony SDK installed and either:
#   * $HARMONY_HOME  pointing at the SDK root (e.g. /path/to/sdk where
#     `native/build/cmake/ohos.toolchain.cmake` lives), OR
#   * $OHOS_SDK_NATIVE pointing directly at the SDK `native/` directory.
#
# Usage:
#   ./package_scripts/harmony/build.sh -o <output_dir>
#
# Default behaviour mirrors package_scripts/android/build.sh: enables LLM
# (+ multi-modal vision / audio) and arm64 CPU acceleration (ARM82 + SME2 +
# BF16). Output layout:
#
#   <output_dir>/
#     arm64-v8a/
#       libMNN.so
#       libllm.so                (when MNN_BUILD_LLM=ON)
#       libMNN_Express.so        (when MNN_SEP_BUILD=ON)
#       libMNNOpenCV.so          (when MNN_BUILD_OPENCV=ON)
#       libMNNAudio.so           (when MNN_BUILD_AUDIO=ON)

set -e

usage() {
    echo "Usage: $0 -o <output_dir>"
    echo -e "\t-o package files output directory"
    exit 1
}

while getopts "o:h" opt; do
    case "$opt" in
        o ) path=$OPTARG ;;
        h|? ) usage ;;
    esac
done

if [ -z "$path" ]; then
    usage
fi

# Resolve OpenHarmony SDK toolchain file.
# Prefer $OHOS_SDK_NATIVE (set by openharmony-rs/setup-ohos-sdk), then
# fall back to $HARMONY_HOME (the variable used by project/harmony/build_64.sh).
if [ -n "$OHOS_SDK_NATIVE" ]; then
    OHOS_TOOLCHAIN="$OHOS_SDK_NATIVE/build/cmake/ohos.toolchain.cmake"
elif [ -n "$HARMONY_HOME" ]; then
    OHOS_TOOLCHAIN="$HARMONY_HOME/native/build/cmake/ohos.toolchain.cmake"
else
    echo "[harmony/build.sh] ERROR: set OHOS_SDK_NATIVE or HARMONY_HOME first." >&2
    exit 1
fi

if [ ! -f "$OHOS_TOOLCHAIN" ]; then
    echo "[harmony/build.sh] ERROR: ohos toolchain not found: $OHOS_TOOLCHAIN" >&2
    exit 1
fi

rm -rf "$path" && mkdir -p "$path"
PACKAGE_PATH=$(realpath "$path")
mkdir -p "$PACKAGE_PATH/arm64-v8a"

# Common cmake args (kept consistent with package_scripts/android/build.sh):
# - MNN_BUILD_LLM=ON      auto-enables MNN_LOW_MEMORY + MNN_SUPPORT_TRANSFORMER_FUSE
# - MNN_BUILD_LLM_OMNI=ON auto-enables MNN_BUILD_OPENCV / MNN_BUILD_AUDIO / MNN_IMGCODECS
#   (vision + audio multi-modal capabilities of LLM)
CMAKEARGS="-DMNN_BUILD_LLM=ON \
-DMNN_BUILD_LLM_OMNI=ON"

# build harmony arm64
# - MNN_ARM82=ON       fp16 acceleration
# - MNN_SME2=ON        Arm SME2 instructions (default ON in CMakeLists.txt)
# - MNN_SUPPORT_BF16=ON  bf16 ops
rm -rf build_harmony && mkdir build_harmony
pushd build_harmony
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$OHOS_TOOLCHAIN" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOHOS_ARCH=arm64-v8a \
    -DOHOS_STL=c++_shared \
    -DOHOS_PLATFORM_LEVEL=12 \
    -DMNN_USE_LOGCAT=ON \
    -DMNN_USE_SSE=OFF \
    -DMNN_ARM82=ON \
    -DMNN_SME2=ON \
    -DMNN_SUPPORT_BF16=ON \
    -DMNN_SEP_BUILD=OFF \
    -DMNN_BUILD_SHARED_LIBS=ON \
    -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. \
    ${CMAKEARGS}

make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

cp libMNN.so "$PACKAGE_PATH/arm64-v8a/"
# Optional companion libs (best-effort; ignore if static-linked into libMNN.so).
cp libllm.so                  "$PACKAGE_PATH/arm64-v8a/" 2>/dev/null || true
cp express/libMNN_Express.so  "$PACKAGE_PATH/arm64-v8a/" 2>/dev/null || true
cp tools/cv/libMNNOpenCV.so   "$PACKAGE_PATH/arm64-v8a/" 2>/dev/null || true
cp tools/audio/libMNNAudio.so "$PACKAGE_PATH/arm64-v8a/" 2>/dev/null || true
popd