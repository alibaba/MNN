set -e

usage() {
    echo "Usage: $0 -o path [-b]"
    echo -e "\t-o package files output directory"
    echo -e "\t-v MNN dist version"
    echo -e "\t-b opencl backend"
    exit 1
}

while getopts "o:v:hb" opt; do
  case "$opt" in
    o ) path=$OPTARG ;;
    v ) mnn_version=$OPTARG ;;
    b ) opencl=true ;;
    h|? ) usage ;;
  esac
done

./schema/generate.sh
rm -rf $path && mkdir -p $path
PACKAGE_PATH=$(realpath $path)

CMAKE_ARGS="-DMNN_BUILD_CONVERTER=on -DMNN_BUILD_TRAIN=ON -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_SEP_BUILD=OFF -DMNN_USE_THREAD_POOL=OFF -DMNN_OPENMP=ON"
if [ ! -z $opencl ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DMNN_OPENCL=ON"
fi
rm -rf pymnn_build && mkdir pymnn_build
pushd pymnn_build
cmake $CMAKE_ARGS .. && make MNN MNNTrain MNNConvert -j24
popd

pushd pymnn/pip_package
echo -e "__version__ = '$mnn_version'" > MNN/version.py
rm -rf build && mkdir build
rm -rf dist && mkdir dist
rm -rf wheelhouse && mkdir wheelhouse

#Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -U numpy
    "${PYBIN}/python" setup.py bdist_wheel --version $mnn_version
done

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat manylinux2014_x86_64 -w wheelhouse
done
cp wheelhouse/* $PACKAGE_PATH
rm MNN/version.py
popd
