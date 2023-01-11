#!/bin/bash
set -e

usage() {
    echo "Usage: $0 -o path [-c]"
    echo -e "\t-o package files output directory"
    exit 1
}

while getopts "o:c:h" opt; do
  case "$opt" in
    o ) path=$OPTARG ;;
    h|? ) usage ;;
  esac
done

rm -rf $path && mkdir -p $path
PACKAGE_PATH=$(realpath $path)

# build framework
xcodebuild build \
-project project/ios/MNN.xcodeproj \
-target MNN \
-destination "platform=iOS,id=dvtdevice-DVTiPhonePlaceholder-iphoneos:placeholder,name=Any iOS Device" \
SYMROOT=$PACKAGE_PATH \
CODE_SIGN_IDENTITY="" \
CODE_SIGNING_REQUIRED=NO
