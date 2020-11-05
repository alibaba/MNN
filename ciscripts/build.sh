#!/usr/bin/env bash
. ./parse_options.sh || exit 1;

CMAKE=cmake
MAKE=make
ANDROID_NDK=/home/android-ndk-r18b

BUILD_ROOT=`pwd`

# Clean the exist directory other than remove it in order to solve
# the problem "Current working directory cannot be established".
function make_or_clean_dir {
  if [ -d $1 ]; then
    rm -rf $1/*
  else
    mkdir $1
  fi
}

function build_arm_android_32 {
  make_or_clean_dir build_arm_android_32 && cd build_arm_android_32
  $CMAKE ../.. \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_ABI="armeabi-v7a" \
      -DANDROID_STL=c++_static \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_NATIVE_API_LEVEL=android-21 \
      -DANDROID_TOOLCHAIN=clang \
      -DMNN_USE_LOGCAT=true \
      -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
      -DNATIVE_LIBRARY_OUTPUT=. \
      -DNATIVE_INCLUDE_OUTPUT=. \
      -DMNN_VULKAN=$USE_VULKAN \
      -DMNN_OPENCL=$USE_OPENCL \
      -DMNN_OPENGL=$USE_OPENGL \
      -DMNN_USE_THREAD_POOL=$USE_THREAD_POOL || exit 1;
  $MAKE -j $build_threads  || exit 1;
  cd $BUILD_ROOT; true;
}

function build_arm_android_64 {
  make_or_clean_dir build_arm_android_64 && cd build_arm_android_64
  $CMAKE ../.. \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_STL=c++_static \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_NATIVE_API_LEVEL=android-21 \
      -DANDROID_TOOLCHAIN=clang \
      -DMNN_USE_LOGCAT=true \
      -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
      -DNATIVE_LIBRARY_OUTPUT=. \
      -DNATIVE_INCLUDE_OUTPUT=. \
      -DMNN_ARM82=ON \
      -DMNN_VULKAN=$USE_VULKAN \
      -DMNN_OPENCL=$USE_OPENCL \
      -DMNN_OPENGL=$USE_OPENGL \
      -DMNN_USE_THREAD_POOL=$USE_THREAD_POOL || exit 1;
  $MAKE -j $build_threads || exit 1;
  cd $BUILD_ROOT; true;
}

function build_arm_linux_32 {
  cd $BUILD_ROOT; true;
}

function build_arm_linux_64 {
  cd $BUILD_ROOT; true;
}

function build_x86_linux {
  make_or_clean_dir build_x86_linux && cd build_x86_linux
  $CMAKE ../.. \
      -DCMAKE_BUILD_TYPE=Release \
      -DMNN_BUILD_TRAIN=ON \
      -DMNN_SEP_BUILD=OFF \
      -DMNN_BUILD_DEMO=ON \
      -DMNN_BUILD_QUANTOOLS=ON \
      -DMNN_EVALUATION=ON \
      -DMNN_BUILD_CONVERTER=ON \
      -DMNN_SUPPORT_TFLITE_QUAN=ON \
      -DMNN_BUILD_TEST=ON \
      -DMNN_OPENCL=$USE_OPENCL \
      -DMNN_VULKAN=$USE_VULKAN \
      -DMNN_OPENMP=$USE_OPENMP \
      -DMNN_USE_THREAD_POOL=OFF \
      -DMNN_BUILD_BENCHMARK=ON  || exit 1;
  $MAKE -j $build_threads || exit 1;
  cd $BUILD_ROOT; true;
}

function build_all {
  build_arm_android_32 || exit 1;
  build_arm_android_64 || exit 1;
  build_arm_linux_32 || exit 1;
  build_arm_linux_64 || exit 1;
  build_x86_linux || exit 1;
  true;
}

function clean {
  rm -rf build_arm_android_32
  rm -rf build_arm_android_64
  rm -rf build_arm_linux_32
  rm -rf build_arm_linux_64
  rm -rf build_x86_linux
}

function build {
  case $platform in
    "arm_linux_32")
      build_arm_linux_32 || exit 1;
      ;;
    "arm_linux_64")
      build_arm_linux_64 || exit 1;
      ;;
    "x86_linux")
      build_x86_linux || exit 1;
      ;;
    "arm_android_32")
      build_arm_android_32 || exit 1;
      ;;
    "arm_android_64")
      build_arm_android_64 || exit 1;
      ;;
    "all")
      build_all || exit 1;
      ;;
  *) echo "Invalid platform: $platform" && exit 1;
  esac
}

if [ $clean == 1 ]; then
  clean
else
  build $@
fi
true;
