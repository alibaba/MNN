#!/bin/sh
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
pushd ${SCRIPT_DIR}

rm -rf iOSCOMBINED
mkdir iOSCOMBINED
cd iOSCOMBINED
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../..//cmake/ios.toolchain.cmake -DMNN_METAL=ON -DPLATFORM=OS -DENABLE_BITCODE=0 -G Xcode
echo "Building for iOS Device"
xcodebuild ONLY_ACTIVE_ARCH=NO CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO -configuration Release -scheme MNN -target MNN -sdk iphoneos -quiet
cd ../

rm -rf iOSSIMULATOR64
mkdir iOSSIMULATOR64
cd iOSSIMULATOR64
cmake ../../..// -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../..//cmake/ios.toolchain.cmake -DMNN_METAL=ON -DPLATFORM=SIMULATOR64 -DENABLE_BITCODE=0 -G Xcode
echo "Building for 64Bit Simulator"
xcodebuild ONLY_ACTIVE_ARCH=NO CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO -configuration Release -scheme MNN -target MNN -sdk iphoneos -quiet
cd ../

rm -rf iOSSIMULATOR32
mkdir iOSSIMULATOR32
cd iOSSIMULATOR32
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../..//cmake/ios.toolchain.cmake -DMNN_METAL=ON -DPLATFORM=SIMULATOR -DENABLE_BITCODE=0 -G Xcode
echo "Building for 32Bit Simulator"
xcodebuild ONLY_ACTIVE_ARCH=NO CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO -configuration Release -scheme MNN -target MNN -sdk iphoneos -quiet
cd ../

echo "Moving Slices"
mv iOSCOMBINED/Release-iphoneos/MNN.framework/MNN ./MNN_IOS_COMBINED
mv iOSSIMULATOR64/Release-iphoneos/MNN.framework/MNN ./MNN_iOSSIMULATOR64
mv iOSSIMULATOR32/Release-iphoneos/MNN.framework/MNN ./MNN_iOSSIMULATOR32
mv iOSCOMBINED/Release-iphoneos/MNN.framework output/
echo "Creating Fat Binary"
lipo -create ./MNN_IOS_COMBINED ./MNN_iOSSIMULATOR64 ./MNN_iOSSIMULATOR32 -output output/MNN.framework/MNN
echo "Cleaning up"
rm ./MNN_IOS_COMBINED
rm ./MNN_iOSSIMULATOR64
rm ./MNN_iOSSIMULATOR32
echo "Patching framework Headers"
cp -R ../../..//include/MNN/expr output/MNN.framework/Headers/expr


echo "Creating Fat Binary"
lipo -create ios_32/Release-iphoneos/MNN.framework/MNN_32 ios_64/Release-iphoneos/MNN.framework/MNN -output ios_32/Release-iphoneos/MNN.framework/MNN
rm ios_32/Release-iphoneos/MNN.framework/MNN_32
echo "Patching Framework Headers"
rm -rf ./MNN.framework
cp -R ios_32/Release-iphoneos/MNN.framework ./MNN.framework
cp -R ../../include/MNN/expr ./MNN.framework/Headers/expr

popd
