#!/bin/sh
echo "Change directory to MNN_SOURCE_ROOT/project/mac before running this script"
echo "Current PWD: ${PWD}"

rm -rf mac_x64
mkdir mac_x64
cd mac_x64
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON -DMNN_METAL=ON -DARCHS="x86_64" -DMNN_AAPL_FMWK=1 -DMNN_SEP_BUILD=0 $1 $2 -G Xcode
echo "Building x86"
xcodebuild ONLY_ACTIVE_ARCH=NO -configuration Release -scheme MNN -target MNN -sdk macosx -quiet
echo "End Building x86"
cd ../

find mac_x64 -name "MNN*framework"

echo "Patching Framework Headers"
rm -rf ./MNN.framework
cp -R mac_x64/Release/MNN.framework ./MNN.framework
