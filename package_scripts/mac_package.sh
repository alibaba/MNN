#   MNN_Mac
#     |------- MNN_Mac_lib
#                   |---------- Dynamic_Library
#                   |---------- Static_Library
#                   |---------- MNN.framework
#     |------- MNN_Mac_tools

MAC_PACKAGE_NAME="MNN_Mac"

# clear and create package directory
./schema/generate.sh
MAC_PACKAGE_PATH=$(pwd)/$MAC_PACKAGE_NAME
rm -rf $MAC_PACKAGE_PATH
mkdir $MAC_PACKAGE_PATH && cd $MAC_PACKAGE_PATH
mkdir MNN_Mac_lib && cd MNN_Mac_lib
mkdir Dynamic_Library
mkdir Static_Library
cd ..
mkdir MNN_Mac_tools
cd ..

rm -rf build
mkdir build && cd build
# tools without dependency, static library without sep_build
cmake -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_SEP_BUILD=OFF -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_DEMO=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON .. && make -j8
pushd ${MAC_PACKAGE_PATH}
cp ../build/*.out MNN_Mac_tools
cp ../build/MNNConvert MNN_Mac_tools
cp ../build/MNNDump2Json MNN_Mac_tools
cp ../build/OnnxClip MNN_Mac_tools
cp ../build/libMNN.a MNN_Mac_lib/Static_Library
popd

# dynamic library without sep_build
rm CMakeCache.txt
cmake -DMNN_SEP_BUILD=OFF -DCMAKE_BUILD_TYPE=Release .. && make -j8
cd ..
cp build/libMNN.dylib ${MAC_PACKAGE_PATH}/MNN_Mac_lib/Dynamic_Library

# mac framework without sep_build
cd build
rm CMakeCache.txt
cmake -DMNN_SEP_BUILD=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_AAPL_FMWK=ON .. && make -j8
cd $MAC_PACKAGE_PATH
cp -r ../build/MNN.framework MNN_Mac_lib

# auto zip MNN_Mac_lib MNN_Mac_tools
zip -r MNN_Mac_lib.zip MNN_Mac_lib
zip -r MNN_Mac_tools.zip MNN_Mac_tools
