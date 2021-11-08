#!/bin/bash

# Release compile work until ndk-r21e (clang 9.0.9svn), Debug compile work until ndk-r22 (clang 11.0.5)
# https://github.com/android/ndk/wiki/Changelog-r22#changes Issues 1303
# https://github.com/android/ndk/wiki/Changelog-r21#r21e Issues 1248
# export ANDROID_NDK=/path/to/ndk-r21e

cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="armeabi-v7a" \
-DANDROID_STL=c++_static \
-DANDROID_NATIVE_API_LEVEL=android-14  \
-DANDROID_TOOLCHAIN=clang \
-DMNN_USE_LOGCAT=false \
-DMNN_USE_SSE=OFF \
-DMNN_SUPPORT_BF16=OFF \
-DMNN_BUILD_TEST=ON \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3 \
-DMNN_ARM82=ON \
-DMNN_BUILD_BENCHMARK=ON

make -j8
