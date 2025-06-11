#!/usr/bin/env  bash

set -e

dir=build-ios
mkdir -p $dir
cd $dir


if [ -z ${MNN_LIB_DIR} ]; then
  echo "Please export MNN_LIB_DIR=/path/to/MNN"
  exit 1
fi

# First, for simulator
echo "Building for simulator (x86_64)"


# Note: We use -DENABLE_ARC=1 here to fix the linking error:
#
# The symbol _NSLog is not defined
#

cmake \
  -DSHERPA_MNN_ENABLE_BINARY=OFF \
  -DMNN_LIB_DIR=${MNN_LIB_DIR} \
  -DBUILD_PIPER_PHONMIZE_EXE=OFF \
  -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
  -DBUILD_ESPEAK_NG_EXE=OFF \
  -DBUILD_ESPEAK_NG_TESTS=OFF \
  -S .. \
  -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
  -DPLATFORM=SIMULATOR64 \
  -DENABLE_BITCODE=0 \
  -DENABLE_ARC=1 \
  -DENABLE_VISIBILITY=0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_MNN_ENABLE_PYTHON=OFF \
  -DSHERPA_MNN_ENABLE_TESTS=OFF \
  -DSHERPA_MNN_ENABLE_CHECK=OFF \
  -DSHERPA_MNN_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_MNN_ENABLE_JNI=OFF \
  -DSHERPA_MNN_ENABLE_C_API=ON \
  -DSHERPA_MNN_ENABLE_WEBSOCKET=OFF \
  -DDEPLOYMENT_TARGET=13.0 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -B build/simulator_x86_64

cmake --build build/simulator_x86_64 -j $(nproc) --verbose

echo "Building for simulator (arm64)"

cmake \
  -DSHERPA_MNN_ENABLE_BINARY=OFF \
  -DMNN_LIB_DIR=${MNN_LIB_DIR} \
  -DBUILD_PIPER_PHONMIZE_EXE=OFF \
  -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
  -DBUILD_ESPEAK_NG_EXE=OFF \
  -DBUILD_ESPEAK_NG_TESTS=OFF \
  -S .. \
  -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
  -DPLATFORM=SIMULATORARM64 \
  -DENABLE_BITCODE=0 \
  -DENABLE_ARC=1 \
  -DENABLE_VISIBILITY=0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_MNN_ENABLE_PYTHON=OFF \
  -DSHERPA_MNN_ENABLE_TESTS=OFF \
  -DSHERPA_MNN_ENABLE_CHECK=OFF \
  -DSHERPA_MNN_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_MNN_ENABLE_JNI=OFF \
  -DSHERPA_MNN_ENABLE_C_API=ON \
  -DSHERPA_MNN_ENABLE_WEBSOCKET=OFF \
  -DDEPLOYMENT_TARGET=13.0 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -B build/simulator_arm64

cmake --build build/simulator_arm64 -j $(nproc) --verbose

echo "Building for arm64"


cmake \
  -DSHERPA_MNN_ENABLE_BINARY=OFF \
  -DMNN_LIB_DIR=${MNN_LIB_DIR} \
  -DBUILD_PIPER_PHONMIZE_EXE=OFF \
  -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
  -DBUILD_ESPEAK_NG_EXE=OFF \
  -DBUILD_ESPEAK_NG_TESTS=OFF \
  -S .. \
  -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
  -DPLATFORM=OS64 \
  -DENABLE_BITCODE=0 \
  -DENABLE_ARC=1 \
  -DENABLE_VISIBILITY=0 \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_MNN_ENABLE_PYTHON=OFF \
  -DSHERPA_MNN_ENABLE_TESTS=OFF \
  -DSHERPA_MNN_ENABLE_CHECK=OFF \
  -DSHERPA_MNN_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_MNN_ENABLE_JNI=OFF \
  -DSHERPA_MNN_ENABLE_C_API=ON \
  -DSHERPA_MNN_ENABLE_WEBSOCKET=OFF \
  -DDEPLOYMENT_TARGET=13.0 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -B build/os64

cmake --build build/os64 -j $(nproc)
# Generate headers for sherpa-mnn.xcframework
cmake --build build/os64 --target install

echo "Generate xcframework"

mkdir -p "build/simulator/lib"
for f in libkaldi-native-fbank-core.a libsherpa-mnn-c-api.a libsherpa-mnn-core.a \
         libsherpa-mnn-fstfar.a libssentencepiece_core.a \
         libsherpa-mnn-fst.a libsherpa-mnn-kaldifst-core.a libkaldi-decoder-core.a \
         libucd.a libpiper_phonemize.a libespeak-ng.a; do
  lipo -create build/simulator_arm64/lib/${f} \
               build/simulator_x86_64/lib/${f} \
       -output build/simulator/lib/${f}
done

# Merge archive first, because the following xcodebuild create xcframework
# cannot accept multi archive with the same architecture.
libtool -static -o build/simulator/sherpa-mnn.a \
  build/simulator/lib/libkaldi-native-fbank-core.a \
  build/simulator/lib/libsherpa-mnn-c-api.a \
  build/simulator/lib/libsherpa-mnn-core.a  \
  build/simulator/lib/libsherpa-mnn-fstfar.a   \
  build/simulator/lib/libsherpa-mnn-fst.a   \
  build/simulator/lib/libsherpa-mnn-kaldifst-core.a \
  build/simulator/lib/libkaldi-decoder-core.a \
  build/simulator/lib/libucd.a \
  build/simulator/lib/libpiper_phonemize.a \
  build/simulator/lib/libespeak-ng.a \
  build/simulator/lib/libssentencepiece_core.a

libtool -static -o build/os64/sherpa-mnn.a \
  build/os64/lib/libkaldi-native-fbank-core.a \
  build/os64/lib/libsherpa-mnn-c-api.a \
  build/os64/lib/libsherpa-mnn-core.a \
  build/os64/lib/libsherpa-mnn-fstfar.a   \
  build/os64/lib/libsherpa-mnn-fst.a   \
  build/os64/lib/libsherpa-mnn-kaldifst-core.a \
  build/os64/lib/libkaldi-decoder-core.a \
  build/os64/lib/libucd.a \
  build/os64/lib/libpiper_phonemize.a \
  build/os64/lib/libespeak-ng.a \
  build/os64/lib/libssentencepiece_core.a

rm -rf sherpa-mnn.xcframework

xcodebuild -create-xcframework \
      -library "build/os64/sherpa-mnn.a" \
      -library "build/simulator/sherpa-mnn.a" \
      -output sherpa-mnn.xcframework

# Copy Headers
mkdir -p sherpa-mnn.xcframework/Headers
cp -av install/include/* sherpa-mnn.xcframework/Headers

pushd sherpa-mnn.xcframework/ios-arm64_x86_64-simulator
ln -s sherpa-mnn.a libsherpa-mnn.a
popd

pushd sherpa-mnn.xcframework/ios-arm64
ln -s sherpa-mnn.a libsherpa-mnn.a
