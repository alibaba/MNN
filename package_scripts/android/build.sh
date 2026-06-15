#!/bin/bash

set -e

usage() {
    echo "Usage: $0 -o path [-c]"
    echo -e "\t-o package files output directory"
    exit 1
}

while getopts "o:c:h" opt; do
  case "$opt" in
    o ) path=$OPTARG ;;
    h|? ) usage ;;
  esac
done

rm -rf $path && mkdir -p $path
PACKAGE_PATH=$(realpath $path)
pushd $PACKAGE_PATH && mkdir -p arm64-v8a && popd
pushd $PACKAGE_PATH && mkdir -p armeabi-v7a && popd

# Common cmake args
# - MNN_BUILD_LLM=ON implicitly enables MNN_LOW_MEMORY and MNN_SUPPORT_TRANSFORMER_FUSE
# - MNN_BUILD_LLM_OMNI=ON implicitly enables MNN_BUILD_OPENCV, MNN_BUILD_AUDIO, MNN_IMGCODECS
#   so vision / audio multi-modal capabilities of LLM are available.
# - ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON: NDK r27+ official option to enable 16KB page
#   size support (required by Android 15+ devices and Google Play from Nov 2025).
#   It transparently sets `-Wl,-z,max-page-size=16384` for the linker.
CMAKEARGS="-DMNN_BUILD_LLM=ON \
-DMNN_BUILD_LLM_OMNI=ON \
-DLLM_SUPPORT_VISION=ON \
-DLLM_SUPPORT_AUDIO=ON \
-DMNN_OPENCL=ON \
-DMNN_VULKAN=ON \
-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON"

# build android_32
rm -rf build_32 && mkdir build_32
pushd build_32
cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="armeabi-v7a" \
-DANDROID_STL=c++_shared \
-DANDROID_NATIVE_API_LEVEL=android-21  \
-DANDROID_TOOLCHAIN=clang \
-DMNN_USE_LOGCAT=ON \
-DMNN_USE_SSE=OFF \
-DMNN_JNI=ON \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. \
${CMAKEARGS}

make -j8
libc_32=`find $ANDROID_NDK -name "libc++_shared.so" | grep "arm-linux-androideabi/libc++_shared.so" | head -n 1`
cp *.so source/jni/libmnncore.so tools/cv/libMNNOpenCV.so tools/audio/libMNNAudio.so $libc_32 $PACKAGE_PATH/armeabi-v7a
popd

# build android_64
# - MNN_ARM82=ON enables fp16 acceleration
# - MNN_SME2=ON (default ON in CMakeLists.txt) enables Arm SME2 instructions for new Arm v9 devices
# - MNN_SUPPORT_BF16=ON enables bf16 ops
rm -rf build_64 && mkdir build_64
pushd build_64
cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_shared \
-DANDROID_NATIVE_API_LEVEL=android-21  \
-DANDROID_TOOLCHAIN=clang \
-DMNN_USE_LOGCAT=ON \
-DMNN_USE_SSE=OFF \
-DMNN_ARM82=ON \
-DMNN_SME2=ON \
-DMNN_SUPPORT_BF16=ON \
-DMNN_JNI=ON \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. \
${CMAKEARGS}

make -j8
libc_64=`find $ANDROID_NDK -name "libc++_shared.so" | grep "aarch64-linux-android/libc++_shared.so" | head -n 1`
cp *.so source/jni/libmnncore.so tools/cv/libMNNOpenCV.so tools/audio/libMNNAudio.so $libc_64 $PACKAGE_PATH/arm64-v8a
popd