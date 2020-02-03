#!/bin/sh
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
pushd ${SCRIPT_DIR}

rm -rf iOS_64
mkdir iOS_64
cd iOS_64
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios.toolchain.cmake -DMNN_METAL=ON -DIOS_ARCH="arm64" -DMNN_AAPL_FMWK=ON -DENABLE_BITCODE=0 -G Xcode
echo "Building AArch64"
xcodebuild ONLY_ACTIVE_ARCH=NO CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO -configuration Release -scheme MNN -target MNN -sdk iphoneos -quiet
cd ../

rm -rf iOS_32
mkdir iOS_32
cd iOS_32
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios.toolchain.cmake -DMNN_METAL=ON -DIOS_ARCH="armv7;armv7s" -DMNN_AAPL_FMWK=ON -DENABLE_BITCODE=0 -G Xcode
echo "Building AArch32"
xcodebuild ONLY_ACTIVE_ARCH=NO CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO -configuration Release -scheme MNN -target MNN -sdk iphoneos -quiet
cd ../

rm -rf SIM_32
mkdir SIM_32
cd SIM_32
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios.toolchain.cmake -DMNN_METAL=ON -DPLATFORM=SIMULATOR -DENABLE_BITCODE=0 -DMNN_AAPL_FMWK=ON -G Xcode
echo "Building Simulator32"
xcodebuild ONLY_ACTIVE_ARCH=NO CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO -configuration Release -scheme MNN -target MNN -sdk iphonesimulator -quiet
cd ../

rm -rf SIM_64
mkdir SIM_64
cd SIM_64
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios.toolchain.cmake -DMNN_METAL=ON -DPLATFORM=SIMULATOR64 -DENABLE_BITCODE=0 -DMNN_AAPL_FMWK=ON -G Xcode
echo "Building Simulator64"
xcodebuild ONLY_ACTIVE_ARCH=NO CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO -configuration Release -scheme MNN -target MNN -sdk iphonesimulator -quiet
cd ../



echo "Moving Slices"
rm -rf output/
mkdir output
mv iOS_64/Release-iphoneos/MNN.framework/MNN ./MNN_iOS64
mv iOS_32/Release-iphoneos/MNN.framework/MNN ./MNN_iOS32
mv SIM_32/Release-iphonesimulator/MNN.framework/MNN ./MNN_SIM_32
mv SIM_64/Release-iphonesimulator/MNN.framework/MNN ./MNN_SIM_64
mv iOS_32/Release-iphoneos/MNN.framework output/
echo "Creating Fat Binary"
lipo -create ./MNN_iOS64 ./MNN_iOS32 ./MNN_SIM_64 ./MNN_SIM_32 -output output/MNN.framework/MNN
echo "Cleaning up"
rm ./MNN_iOS64 ./MNN_iOS32 ./MNN_SIM_64 ./MNN_SIM_32
echo "Patching framework Headers"
cp -R ../../include/MNN/expr output/MNN.framework/Headers/expr
echo "Unified Framework built at ${PWD}/output/"

popd
