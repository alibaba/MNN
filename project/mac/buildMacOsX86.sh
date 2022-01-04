#!/bin/sh
echo "Change directory to MNN_SOURCE_ROOT/project/mac before running this script"
echo "Current PWD: ${PWD}"

# ARM
rm -rf mac_a64
mkdir mac_a64
cd mac_a64
cmake ../../../ -DMNN_USE_SSE=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON -DMNN_METAL=ON -DARCHS="arm64" -DMNN_AAPL_FMWK=1 -DMNN_SEP_BUILD=0 -DCMAKE_OSX_ARCHITECTURES=arm64 $1 $2
echo "Building ARM64"
make MNN -j16
echo "End Building ARM64"
cd ../

# X86
rm -rf mac_x64
mkdir mac_x64
cd mac_x64
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON -DMNN_METAL=ON -DARCHS="x86_64" -DMNN_AAPL_FMWK=1 -DMNN_SEP_BUILD=0 $1 $2
echo "Building x86"
make MNN -j16
echo "End Building x86"
cd ../



find mac_x64 -name "MNN*framework"
find mac_a64 -name "MNN*framework"
mv mac_x64/MNN.framework/MNN mac_x64/MNN.framework/MNN_32

echo "Creating Fat Binary"
lipo -create mac_x64/MNN.framework/MNN_32 mac_a64/MNN.framework/MNN -output mac_x64/MNN.framework/MNN
rm mac_x64/MNN.framework/MNN_32
echo "Patching Framework Headers"
rm -rf ./MNN.framework
cp -R mac_x64/MNN.framework ./MNN.framework

