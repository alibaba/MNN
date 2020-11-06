# ./package_scripts/mac/build_whl.sh -o MNN-CPU/py_whl -v 2.7.17,3.5.7,3.6.9,3.7.4,3.8.0
# ./package_scripts/mac/build_whl.sh -o MNN-CPU-OPENCL/py_whl -v 2.7.17,3.5.7,3.6.9,3.7.4,3.8.0 -b

set -e

usage() {
    echo "Usage: $0 -o path -v python_versions [-b]"
    echo -e "\t-o package files output directory"
    echo -e "\t-v python versions in pyenv"
    echo -e "\t-b opencl backend"
    exit 1
}

while getopts "o:v:hb" opt; do
  case "$opt" in
    o ) path=$OPTARG ;;
    v ) IFS="," read -a python_versions <<< $OPTARG ;;
    b ) opencl=true ;;
    h|? ) usage ;;
  esac
done

./schema/generate.sh
rm -rf $path && mkdir -p $path
PACKAGE_PATH=$(realpath $path)

CMAKE_ARGS="-DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TRAIN=ON -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_AAPL_FMWK=OFF -DMNN_SEP_BUILD=OFF -DMNN_EXPR_SHAPE_EAGER=ON -DMNN_TRAIN_DEBUG=ON"
if [ ! -z $opencl ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DMNN_OPENCL=ON"
fi

rm -rf pymnn_build && mkdir pymnn_build
pushd pymnn_build
cmake $CMAKE_ARGS .. && make MNN MNNTrain MNNConvert -j8
popd

pushd pymnn/pip_package
rm -rf dist && mkdir dist
for env in $python_versions; do
    pyenv global $env
    python build_wheel.py
done
cp dist/* $PACKAGE_PATH

popd
