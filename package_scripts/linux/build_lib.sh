# MNN
#  |--- Debug
#  |      |--- libMNN.a
#  |      |--- libMNN.so
#  |
#  |--- Release
#         |--- libMNN.a
#         |--- libMNN.so

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
mkdir -p $path/Debug
mkdir -p $path/Release

PACKAGE_PATH=$(realpath $path)

CMAKE_ARGS="-DMNN_SEP_BUILD=OFF"
if [ ! -z $opencl ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DMNN_OPENCL=ON"
fi

rm -rf build && mkdir build
pushd build

# Debug Dynamic MNN.framework
[ -f CMakeCache.txt ] && rm CMakeCache.txt
cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_BUILD_SHARED_LIBS=ON .. && make -j24
cp libMNN.so $PACKAGE_PATH/Debug

# Debug Static MNN.framework
[ -f CMakeCache.txt ] && rm CMakeCache.txt
cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_BUILD_SHARED_LIBS=OFF .. && make -j24
cp libMNN.a $PACKAGE_PATH/Debug

# Release Dynamic MNN.framework
[ -f CMakeCache.txt ] && rm CMakeCache.txt
cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=ON .. && make -j24
cp libMNN.so $PACKAGE_PATH/Release

# Release Static MNN.framework
[ -f CMakeCache.txt ] && rm CMakeCache.txt
cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF .. && make -j24
cp libMNN.a $PACKAGE_PATH/Release

popd
