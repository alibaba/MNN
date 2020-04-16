MAC_PACKAGE_NAME=MNN_Mac_lib_and_tools

# clear and create package directory
MAC_PACKAGE_PATH=$(pwd)/$MAC_PACKAGE_NAME
rm -rf $MAC_PACKAGE_PATH
mkdir $MAC_PACKAGE_PATH && cd $MAC_PACKAGE_PATH
mkdir Dynamic_Library
mkdir Static_Library
mkdir tools
cd ..

rm -rf build
mkdir build && cd build
# tools without dependency, static library without sep_build
cmake -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_SEP_BUILD=OFF -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_DEMO=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON .. && make -j8
pushd ${MAC_PACKAGE_PATH}
cp ../build/*.out tools
cp ../build/MNNConvert tools
cp ../build/MNNDump2Json tools
cp ../build/OnnxClip tools
cp ../build/libMNN.a Static_Library
popd

# dynamic library without sep_build
rm CMakeCache.txt
cmake -DMNN_SEP_BUILD=OFF .. && make -j8
cd ..
cp build/libMNN.dylib ${MAC_PACKAGE_PATH}/Dynamic_Library
