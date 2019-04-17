#!/bin/sh

set -x
set -e

mkdir -p project/android/build
pushd project/android/build

../build_64.sh -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_BUILD_TYPE=Release

popd

adb push project/android/build/libMNN.so /data/local/tmp/nntest
adb push project/android/build/project/android/OpenCL/libMNN_CL.so /data/local/tmp/nntest
adb push project/android/build/MNNV2Basic.out /data/local/tmp/nntest

adb push /Users/xixi/DL/mobilenet_pose_7_27/posev2_out.MNN /data/local/tmp/nntest
adb push /Users/xixi/DL/mobilenet_pose_7_27/testinput.txt /data/local/tmp/nntest/input_0.txt
adb shell LD_LIBRARY_PATH=/data/local/tmp/nntest /data/local/tmp/nntest/MNNV2Basic.out \
  /data/local/tmp/nntest/posev2_out.MNN 100 0 3 1x3x320x320
