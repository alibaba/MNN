#!/bin/sh
export MACOSX_DEPLOYMENT_TARGET=10.11
echo "Change directory to MNN_SOURCE_ROOT/project/mac before running this script"
echo "Current PWD: ${PWD}"

# ARM
rm -rf mac_a64
mkdir mac_a64
cd mac_a64
cmake ../../../ -DMNN_USE_SSE=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON -DMNN_METAL=ON -DARCHS="arm64" -DMNN_AAPL_FMWK=ON -DMNN_SEP_BUILD=OFF -DMNN_ARM82=ON -DCMAKE_OSX_ARCHITECTURES=arm64 -DMNN_BUILD_SHARED_LIBS=OFF
echo "Building ARM64"
make MNN -j16
echo "End Building ARM64"
cd ../

# X86
rm -rf mac_x64
mkdir mac_x64
cd mac_x64
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON -DMNN_METAL=ON -DARCHS="x86_64" -DMNN_AAPL_FMWK=ON -DMNN_SEP_BUILD=OFF $1 $2 -DCMAKE_OSX_ARCHITECTURES=x86_64 -DMNN_BUILD_SHARED_LIBS=OFF
echo "Building x86"
make MNN -j16
echo "End Building x86"
cd ../


echo "Creating Fat Binary"
lipo -create mac_x64/MNN.framework/Versions/A/MNN mac_a64/MNN.framework/Versions/A/MNN -output mac_x64/MNN.framework/Versions/A/MNN
mv mac_x64/MNN.framework MNN.framework
echo "Patching Framework Headers"
rm -rf mac_a64
rm -rf mac_x64

