#!/bin/bash
cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="armeabi-v7a" \
-DANDROID_STL=c++_static \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_NATIVE_API_LEVEL=android-24  \
-DANDROID_TOOLCHAIN=clang \
-DMNN_USE_LOGCAT=false \
-DMNN_VULKAN=true \
-DMNN_USE_SYSTEM_LIB=true \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3

make -j4
