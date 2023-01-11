#!/bin/sh
export MACOSX_DEPLOYMENT_TARGET=10.11
echo "Change directory to MNN_SOURCE_ROOT/project/mac before running this script"
echo "Current PWD: ${PWD}"
echo "Extra Build Config: $1"
# MNN-MacOS-CPU-GPU
# |--- Dynamic
# |--- Static

rm -rf MNN-MacOS-CPU-GPU
mkdir MNN-MacOS-CPU-GPU
cd MNN-MacOS-CPU-GPU

# Static Begin
mkdir Static 
cd Static

# ARM
mkdir mac_a64
cd mac_a64
cmake ../../../ -DMNN_USE_SSE=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON -DMNN_METAL=ON -DARCHS="arm64" -DMNN_AAPL_FMWK=ON -DMNN_SEP_BUILD=OFF -DMNN_ARM82=ON -DCMAKE_OSX_ARCHITECTURES=arm64 -DMNN_BUILD_SHARED_LIBS=OFF $1
echo "Building ARM64"
make MNN -j16
echo "End Building ARM64"
cd ../

# X86
mkdir mac_x64
cd mac_x64
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON -DMNN_METAL=ON -DARCHS="x86_64" -DMNN_AAPL_FMWK=ON -DMNN_SEP_BUILD=OFF -DCMAKE_OSX_ARCHITECTURES=x86_64 -DMNN_BUILD_SHARED_LIBS=OFF $1
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

cd ../

# Static End

# Dynamic Begin
mkdir Dynamic
cd Dynamic

# ARM
mkdir mac_a64
cd mac_a64
cmake ../../../ -DMNN_USE_SSE=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON -DMNN_METAL=ON -DARCHS="arm64" -DMNN_AAPL_FMWK=ON -DMNN_SEP_BUILD=OFF -DMNN_ARM82=ON -DCMAKE_OSX_ARCHITECTURES=arm64 $1
echo "Building ARM64"
make MNN -j16
echo "End Building ARM64"
cd ../

# X86
mkdir mac_x64
cd mac_x64
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON -DMNN_METAL=ON -DARCHS="x86_64" -DMNN_AAPL_FMWK=ON -DMNN_SEP_BUILD=OFF -DCMAKE_OSX_ARCHITECTURES=x86_64 $1
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

cd ../

# Dynamic End
