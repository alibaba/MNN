#!/bin/bash
rm -rf outputs
source ~/third/hexagon/setup_sdk_env.source
build_cmake android
build_cmake hexagon DSP_ARCH=$1
mkdir outputs
cp android_ReleaseG_aarch64/libMNN_htpops.so outputs/
cp hexagon_ReleaseG_toolv19_$1/libMNN_htpops_skel.so outputs/

# Check for unexpected undefined symbols in the DSP dynamic library
./check_so_symbols.sh outputs/libMNN_htpops_skel.so
if [ $? -ne 0 ]; then
    echo "Build failed due to unresolved symbols in DSP library."
    exit 1
fi
