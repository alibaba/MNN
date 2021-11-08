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
TOOLS_PATH=$(realpath $path)

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DMNN_SEP_BUILD=OFF -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TRAIN=ON -DMNN_PORTABLE_BUILD=ON -DMNN_BUILD_TOOLS=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TEST=ON"
if [ ! -z $opencl ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DMNN_OPENCL=ON"
fi

rm -rf build && mkdir build
pushd build

[ -f CMakeCache.txt ] && rm CMakeCache.txt
cmake $CMAKE_ARGS .. && make -j24
cp *.out $TOOLS_PATH

popd
