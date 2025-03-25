#!/bin/sh
echo "Change directory to MNN_SOURCE_ROOT/project/ios before running this script"
echo "Current PWD: ${PWD}"

rm -rf MNN-iOS-CPU-GPU
mkdir MNN-iOS-CPU-GPU
cd MNN-iOS-CPU-GPU
# Static Begin
mkdir Static 
cd Static

rm -rf ios_64
mkdir ios_64
cd ios_64
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios.toolchain.cmake -DMNN_METAL=ON -DARCHS="arm64" -DENABLE_BITCODE=0 -DMNN_AAPL_FMWK=1 -DMNN_SEP_BUILD=0 -DMNN_ARM82=true -DMNN_BUILD_SHARED_LIBS=false -DMNN_USE_THREAD_POOL=OFF $1
echo "Building AArch64"
make MNN -j16
echo "End Building AArch64"
cd ../

mv ios_64/MNN.framework MNN.framework

rm -rf ios_64
