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

# build android_32
rm -rf build_32 && mkdir build_32
pushd build_32
cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="armeabi-v7a" \
-DANDROID_STL=c++_shared \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_NATIVE_API_LEVEL=android-14  \
-DANDROID_TOOLCHAIN=clang \
-DMNN_USE_LOGCAT=ON \
-DMNN_USE_SSE=OFF \
-DMNN_OPENCL=ON \
-DMNN_VULKAN=ON \
-DMNN_BUILD_OPENCV=ON \
-DMNN_IMGCODECS=ON \
-DMNN_JNI=ON \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=.

make -j8
libc_32=`find $ANDROID_NDK -name "libc++_shared.so" | grep "arm-linux-androideabi/libc++_shared.so" | head -n 1`
cp *.so source/jni/libmnncore.so tools/cv/libMNNOpenCV.so $libc_32 $PACKAGE_PATH/armeabi-v7a
popd

# build android_64
rm -rf build_64 && mkdir build_64
pushd build_64
cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_shared \
-DMNN_USE_LOGCAT=ON \
-DMNN_USE_SSE=OFF \
-DMNN_ARM82=ON \
-DMNN_OPENCL=ON \
-DMNN_VULKAN=ON \
-DMNN_JNI=ON \
-DMNN_BUILD_OPENCV=ON \
-DMNN_IMGCODECS=ON \
-DMNN_SUPPORT_BF16=ON \
-DANDROID_NATIVE_API_LEVEL=android-21  \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=.

make -j8
libc_64=`find $ANDROID_NDK -name "libc++_shared.so" | grep "aarch64-linux-android/libc++_shared.so" | head -n 1`
cp *.so source/jni/libmnncore.so tools/cv/libMNNOpenCV.so $libc_64 $PACKAGE_PATH/arm64-v8a
popd
