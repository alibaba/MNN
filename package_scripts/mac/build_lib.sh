# MNN
#  |--- Debug
#  |      |--- Dynamic
#  |      |--- Static
#  |
#  |--- Release
#         |--- Dynamic
#         |--- Static
# Only have MNN.framework

set -e

usage() {
    echo "Usage: $0 -o path [-b]"
    echo -e "\t-o package files output directory"
    echo -e "\t-b opencl backend"
    exit 1
}

while getopts "o:hb" opt; do
  case "$opt" in
    o ) path=$OPTARG ;;
    b ) opencl=true ;;
    h|? ) usage ;;
  esac
done

# clear and create package directory
./schema/generate.sh
rm -rf $path && mkdir -p $path
pushd $path
mkdir -p Debug/Dynamic
mkdir -p Debug/Static
mkdir -p Release/Dynamic
mkdir -p Release/Static
popd

PACKAGE_PATH=$(realpath $path)

CMAKE_ARGS="-DMNN_SEP_BUILD=OFF -DMNN_AAPL_FMWK=ON"
if [ ! -z $opencl ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DMNN_OPENCL=ON"
fi

rm -rf build && mkdir build
pushd build

# Debug Dynamic MNN.framework
[ -f CMakeCache.txt ] && rm CMakeCache.txt
cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_BUILD_SHARED_LIBS=ON .. && make -j8
cp -R MNN.framework $PACKAGE_PATH/Debug/Dynamic

# Debug Static MNN.framework
[ -f CMakeCache.txt ] && rm CMakeCache.txt
cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_BUILD_SHARED_LIBS=OFF .. && make -j8
cp -R MNN.framework $PACKAGE_PATH/Debug/Static

# Release Dynamic MNN.framework
[ -f CMakeCache.txt ] && rm CMakeCache.txt
cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=ON .. && make -j8
cp -R MNN.framework $PACKAGE_PATH/Release/Dynamic

# Release Static MNN.framework
[ -f CMakeCache.txt ] && rm CMakeCache.txt
cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF .. && make -j8
cp -R MNN.framework $PACKAGE_PATH/Release/Static

popd
