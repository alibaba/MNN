#!/bin/sh
echo "Change directory to MNN_SOURCE_ROOT/project/ios before running this script"
echo "Current PWD: ${PWD}"

rm -rf ios_64
mkdir ios_64
cd ios_64
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios.toolchain.cmake -DMNN_METAL=ON -DIOS_ARCH="arm64" -DENABLE_BITCODE=0 -G Xcode
echo "Building AArch64"
xcodebuild ONLY_ACTIVE_ARCH=NO -configuration Release -scheme MNN -target MNN -sdk iphoneos -quiet
cd ../

rm -rf ios_32
mkdir ios_32
cd ios_32
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios.toolchain.cmake -DMNN_METAL=ON -DIOS_ARCH="armv7;armv7s" -DENABLE_BITCODE=0 -G Xcode
echo "Building AArch32"
xcodebuild ONLY_ACTIVE_ARCH=NO -configuration Release -scheme MNN -target MNN -sdk iphoneos -quiet
cd ../

mv ios_32/Release-iphoneos/MNN.framework/MNN ios_32/Release-iphoneos/MNN.framework/MNN_32

echo "Creating Fat Binary"
lipo -create ios_32/Release-iphoneos/MNN.framework/MNN_32 ios_64/Release-iphoneos/MNN.framework/MNN -output ios_32/Release-iphoneos/MNN.framework/MNN
rm ios_32/Release-iphoneos/MNN.framework/MNN_32
echo "Patching Framework Headers"
rm -rf ./MNN.framework
cp -R ios_32/Release-iphoneos/MNN.framework ./MNN.framework
cp -R ../../include/MNN/expr ./MNN.framework/Headers/expr
