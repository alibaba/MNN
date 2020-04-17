#   MNN_Linux
#     |------- MNN_Linux_lib
#                   |---------- Dynamic_Library
#                   |---------- Static_Library
#     |------- MNN_Linux_tools

LINUX_PACKAGE_NAME="MNN_Linux"

# clear and create package directory
./schema/generate.sh
LINUX_PACKAGE_PATH="$(pwd)/$LINUX_PACKAGE_NAME"
rm -rf $LINUX_PACKAGE_PATH
mkdir $LINUX_PACKAGE_PATH && cd $LINUX_PACKAGE_PATH
mkdir MNN_Linux_lib && cd MNN_Linux_lib
mkdir Dynamic_Library
mkdir Static_Library
cd ..
mkdir MNN_Linux_tools
cd ..

rm -rf build
mkdir build && cd build
# tools without dependency, static library without sep_build
cmake -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_SEP_BUILD=OFF -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_DEMO=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON .. && make -j$(nproc)
pushd ${LINUX_PACKAGE_PATH}
cp ../build/*.out MNN_Linux_tools
cp ../build/MNNConvert MNN_Linux_tools
cp ../build/MNNDump2Json MNN_Linux_tools
cp ../build/OnnxClip MNN_Linux_tools
cp ../build/libMNN.a MNN_Linux_lib/Static_Library
popd

# dynamic library without sep_build
rm CMakeCache.txt
cmake -DMNN_SEP_BUILD=OFF -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
cd $LINUX_PACKAGE_PATH
cp ../build/libMNN.so MNN_Linux_lib/Dynamic_Library

# auto zip MNN_Linux_lib MNN_Linux_tools
zip -r MNN_Linux_lib.zip MNN_Linux_lib
zip -r MNN_Linux_tools.zip MNN_Linux_tools
