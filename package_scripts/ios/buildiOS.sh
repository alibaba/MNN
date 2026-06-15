#!/bin/sh
echo "Change directory to MNN_SOURCE_ROOT/project/ios before running this script"
echo "Current PWD: ${PWD}"

# Common cmake args
# - MNN_BUILD_LLM=ON implicitly enables MNN_LOW_MEMORY / MNN_SUPPORT_TRANSFORMER_FUSE
# - MNN_BUILD_LLM_OMNI=ON implicitly enables MNN_BUILD_OPENCV / MNN_BUILD_AUDIO / MNN_IMGCODECS
# - MNN_SME2=ON (default ON) enables Arm SME2 instructions on newer Apple silicon
LLM_ARGS="-DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=ON -DLLM_SUPPORT_VISION=ON -DLLM_SUPPORT_AUDIO=ON"

rm -rf MNN-iOS-CPU-GPU
mkdir MNN-iOS-CPU-GPU
cd MNN-iOS-CPU-GPU
# Static Begin
mkdir Static 
cd Static

rm -rf ios_64
mkdir ios_64
cd ios_64
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios.toolchain.cmake -DMNN_METAL=ON -DARCHS="arm64" -DENABLE_BITCODE=0 -DMNN_AAPL_FMWK=1 -DMNN_SEP_BUILD=0 -DMNN_ARM82=ON -DMNN_SME2=ON -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_USE_THREAD_POOL=OFF $LLM_ARGS $*
echo "Building AArch64"
make MNN -j16
echo "End Building AArch64"
cd ../

mv ios_64/MNN.framework MNN.framework

rm -rf ios_64