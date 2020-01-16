#!/bin/bash

basepath=$(cd `dirname $0`; pwd)

cd $basepath

cd ../../source/backend/vulkan/compiler/

python makeshader.py

cd ../../../

rm -rf build_vulkan

mkdir build_vulkan

cd build_vulkan

# for cpp cache
# macOS `brew install ccache` ubuntu `apt-get install ccache` windows `not care`
cmake ../../ \
-DNDK_CCACHE=ccache \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DANDROID_ABI="armeabi-v7a" \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_STL=c++_static \
-DANDROID_NATIVE_API_LEVEL=android-21  \
-DANDROID_TOOLCHAIN=gcc \
-DMNN_OPENGL=OFF \
-DMNN_OPENCL=OFF \
-DMNN_VULKAN=ON \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2

make -j4
