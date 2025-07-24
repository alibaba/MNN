#!/usr/bin/env bash
set -ex

# If BUILD_SHARED_LIBS is ON, we use libonnxruntime.so
# If BUILD_SHARED_LIBS is OFF, we use libonnxruntime.a
#
# In any case, we will have libsherpa-onnx-jni.so
#
# If BUILD_SHARED_LIBS is OFF, then libonnxruntime.a is linked into libsherpa-onnx-jni.so
# and you only need to copy libsherpa-onnx-jni.so to your Android projects.
#
# If BUILD_SHARED_LIBS is ON, then you need to copy both libsherpa-onnx-jni.so
# and libonnxruntime.so to your Android projects
#
BUILD_SHARED_LIBS=ON

if [ $BUILD_SHARED_LIBS == ON ]; then
  dir=$PWD/build-android-arm64-v8a
else
  dir=$PWD/build-android-arm64-v8a-static
fi

mkdir -p $dir
cd $dir

# Note from https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android
# (optional) remove the hardcoded debug flag in Android NDK android-ndk
# issue: https://github.com/android/ndk/issues/243
#
# open $ANDROID_NDK/build/cmake/android.toolchain.cmake for ndk < r23
# or $ANDROID_NDK/build/cmake/android-legacy.toolchain.cmake for ndk >= r23
#
# delete "-g" line
#
# list(APPEND ANDROID_COMPILER_FLAGS
#   -g
#   -DANDROID

if [ -z $ANDROID_NDK ]; then
  ANDROID_NDK=/star-fj/fangjun/software/android-sdk/ndk/22.1.7171670
  if [ $BUILD_SHARED_LIBS == OFF ]; then
    ANDROID_NDK=/star-fj/fangjun/software/android-sdk/ndk/27.0.11718014
  fi
  # or use
  # ANDROID_NDK=/star-fj/fangjun/software/android-ndk
  #
  # Inside the $ANDROID_NDK directory, you can find a binary ndk-build
  # and some other files like the file "build/cmake/android.toolchain.cmake"

  if [ ! -d $ANDROID_NDK ]; then
    # For macOS, I have installed Android Studio, select the menu
    # Tools -> SDK manager -> Android SDK
    # and set "Android SDK location" to /Users/fangjun/software/my-android
    ANDROID_NDK=/Users/fangjun/software/my-android/ndk/22.1.7171670

    if [ $BUILD_SHARED_LIBS == OFF ]; then
      ANDROID_NDK=/Users/fangjun/software/my-android/ndk/27.0.11718014
    fi
  fi
fi

if [ ! -d $ANDROID_NDK ]; then
  echo Please set the environment variable ANDROID_NDK before you run this script
  exit 1
fi

echo "ANDROID_NDK: $ANDROID_NDK"
sleep 1

if [ -z $SHERPA_MNN_ENABLE_TTS ]; then
  SHERPA_MNN_ENABLE_TTS=ON
fi

if [ -z $SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION ]; then
  SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION=ON
fi

if [ -z $SHERPA_MNN_ENABLE_BINARY ]; then
  SHERPA_MNN_ENABLE_BINARY=OFF
fi

if [ -z $SHERPA_MNN_ENABLE_C_API ]; then
  SHERPA_MNN_ENABLE_C_API=OFF
fi

if [ -z $SHERPA_MNN_ENABLE_JNI ]; then
  SHERPA_MNN_ENABLE_JNI=ON
fi

if [ -z $MNN_LIB_DIR ]; then
  MNN_LIB_DIR=/Users/xtjiang/alicnn/AliNNPrivate/project/android/build_64
fi

if [ -z $SHERPA_MNN_ENABLE_16K_PAGE_SIZE ]; then
  SHERPA_MNN_ENABLE_16K_PAGE_SIZE=OFF
fi

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DSHERPA_MNN_ENABLE_TTS=$SHERPA_MNN_ENABLE_TTS \
    -DSHERPA_MNN_ENABLE_SPEAKER_DIARIZATION=$SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION \
    -DSHERPA_MNN_ENABLE_BINARY=$SHERPA_MNN_ENABLE_BINARY \
    -DBUILD_PIPER_PHONMIZE_EXE=OFF \
    -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
    -DBUILD_ESPEAK_NG_EXE=OFF \
    -DBUILD_ESPEAK_NG_TESTS=OFF \
    $([ "$SHERPA_MNN_ENABLE_16K_PAGE_SIZE" = "ON" ] && echo "-DCMAKE_SHARED_LINKER_FLAGS=\"-Wl,-z,max-page-size=16384\"") \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_LIB_DIR=$MNN_LIB_DIR \
    -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS \
    -DSHERPA_MNN_ENABLE_PYTHON=OFF \
    -DSHERPA_MNN_ENABLE_TESTS=OFF \
    -DSHERPA_MNN_ENABLE_CHECK=OFF \
    -DSHERPA_MNN_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_MNN_ENABLE_JNI=$SHERPA_MNN_ENABLE_JNI \
    -DSHERPA_MNN_LINK_LIBSTDCPP_STATICALLY=OFF \
    -DSHERPA_MNN_ENABLE_C_API=$SHERPA_MNN_ENABLE_C_API \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-21 ..

    # By default, it links to libc++_static.a
    # -DANDROID_STL=c++_shared \

# Please use -DANDROID_PLATFORM=android-27 if you want to use Android NNAPI

# make VERBOSE=1 -j4
make -j4
make install/strip
rm -rf install/share
rm -rf install/lib/pkgconfig
rm -rf install/lib/lib*.a
if [ -f install/lib/libsherpa-onnx-c-api.so ]; then
  cat >install/lib/README.md <<EOF
# Introduction

Note that if you use Android Studio, then you only need to
copy libonnxruntime.so and libsherpa-onnx-jni.so
to your jniLibs, and you don't need libsherpa-onnx-c-api.so or
libsherpa-onnx-cxx-api.so.

libsherpa-onnx-c-api.so and libsherpa-onnx-cxx-api.so are for users
who don't use JNI. In that case, libsherpa-onnx-jni.so is not needed.

In any case, libonnxruntime.is is always needed.
EOF
  ls -lh install/lib/README.md
fi

# To run the generated binaries on Android, please use the following steps.
#
#
# 1. Copy sherpa-onnx and its dependencies to Android
#
#   cd build-android-arm64-v8a/install/lib
#   adb push ./lib*.so /data/local/tmp
#   cd ../bin
#   adb push ./sherpa-onnx /data/local/tmp
#
# 2. Login into Android
#
#   adb shell
#   cd /data/local/tmp
#   ./sherpa-onnx
#
# It should show the help message of sherpa-onnx.
#
# Please use the above approach to copy model files to your phone.
