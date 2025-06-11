#!/usr/bin/env  bash

set -ex

dir=build-swift-macos
mkdir -p $dir
cd $dir

cmake \
  -DSHERPA_MNN_ENABLE_BINARY=OFF \
  -DMNN_LIB_DIR=/Users/xtjiang/alicnn/AliNNPrivate \
  -DSHERPA_MNN_BUILD_C_API_EXAMPLES=OFF \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
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
  ../

make VERBOSE=1 -j4
make install
rm -fv ./install/include/cargs.h

libtool -static -o ./install/lib/libsherpa-mnn.a \
  ./install/lib/libsherpa-mnn-c-api.a \
  ./install/lib/libsherpa-mnn-core.a \
  ./install/lib/libkaldi-native-fbank-core.a \
  ./install/lib/libsherpa-mnn-fstfar.a \
  ./install/lib/libsherpa-mnn-fst.a \
  ./install/lib/libsherpa-mnn-kaldifst-core.a \
  ./install/lib/libkaldi-decoder-core.a \
  ./install/lib/libucd.a \
  ./install/lib/libpiper_phonemize.a \
  ./install/lib/libespeak-ng.a \
  ./install/lib/libssentencepiece_core.a

xcodebuild -create-xcframework \
  -library install/lib/libsherpa-mnn.a \
  -headers install/include \
  -output sherpa-mnn.xcframework
