#!/bin/sh
echo "Change directory to MNN_SOURCE_ROOT/project/ios before running this script"
echo "Current PWD: ${PWD}"

rm -rf MNN-iOS-CPU-GPU
mkdir MNN-iOS-CPU-GPU
cd MNN-iOS-CPU-GPU
# Static Begin
mkdir Static 
cd Static

COMMON="-DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios.toolchain.cmake -DENABLE_BITCODE=0 -DMNN_AAPL_FMWK=1 -DMNN_SEP_BUILD=0 -DMNN_BUILD_SHARED_LIBS=false -DMNN_USE_THREAD_POOL=OFF"

rm -rf ios_64
mkdir ios_64
cd ios_64
cmake ../../../ ${COMMON} -DMNN_METAL=ON -DARCHS="arm64" $*
echo "Building AArch64"
make MNN -j16
echo "End Building AArch64"
cd ../

rm -rf ios_32
mkdir ios_32
cd ios_32
cmake ../../../ ${COMMON} -DMNN_METAL=OFF -DPLATFORM=SIMULATOR64 -DARCHS="x86_64" $*
echo "Building Simulator64"
make MNN -j16
echo "End Building Simulator64"
cd ../

mv ios_32/MNN.framework/MNN ios_32/MNN.framework/MNN_32

echo "Creating Fat Binary"
lipo -create ios_32/MNN.framework/MNN_32 ios_64/MNN.framework/MNN -output ios_32/MNN.framework/MNN
rm ios_32/MNN.framework/MNN_32
echo "Patching Framework Headers"
rm -rf ./MNN.framework
cp -R ios_32/MNN.framework ./MNN.framework
rm -rf ios_32
rm -rf ios_64
