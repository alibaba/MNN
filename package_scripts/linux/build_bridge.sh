#!/bin/bash

# MNN
#  |--- Debug
#  |      |--- libmnnpybridge.a
#  |      |--- libmnnpybridge.so
#  |
#  |--- Release
#         |--- libmnnpybridge.a
#         |--- libmnnpybridge.so

set -e

usage() {
    echo "Usage: $0 -i mnn_path -o path [-t build_type -t lib_type]"
    echo -e "\t-i MNN library path"
    echo -e "\t-o package files output directory"
    echo -e "\t-t build type (debug/release), lib_type (dynamic/static), build all when unspecify"
    exit 1
}

build_all=true
while getopts "i:o:ft:h" opt; do
  case "$opt" in
    i ) mnn_path=$OPTARG ;;
    o ) path=$OPTARG ;;
    t ) build_all=""
        case "$OPTARG" in
            "debug"|"release" ) build_type=$OPTARG ;;
            "dynamic"|"static" ) lib_type=$OPTARG ;;
        esac ;;
    h|? ) usage ;;
  esac
done

if [ -z $build_all ] && ([ -z $build_type ] || [ -z $lib_type ]); then
    echo "build_type(debug/release) and lib_type(dynamic/static) should be set or not-set together"
    exit 1
fi

rm -rf $path && mkdir -p $path
pushd $path
mkdir -p include wrapper lib/Debug lib/Release
popd
PACKAGE_PATH=$(realpath $path)
MNN_PACKAGE_PATH=$(realpath $mnn_path)

pushd pymnn/3rd_party
rm -rf MNN && mkdir -p MNN/lib
cp -r $MNN_PACKAGE_PATH/* MNN/lib
cp -r ../../include MNN
popd

cp pymnn/src/MNNPyBridge.h $PACKAGE_PATH/include
rm -rf /tmp/mnn_py && mkdir -p /tmp/mnn_py
cp -r pymnn/pip_package/MNN /tmp/mnn_py
pushd /tmp/mnn_py
find . -name __pycache__ | xargs rm -rf
pushd MNN
rm -rf tools
cat __init__.py | sed '/from . import tools/d' > __init__.py.tmp
mv __init__.py.tmp __init__.py
rm -rf data
cat __init__.py | sed '/from . import data/d' > __init__.py.tmp
mv __init__.py.tmp __init__.py
rm -rf optim
cat __init__.py | sed '/from . import optim/d' > __init__.py.tmp
mv __init__.py.tmp __init__.py
python -c "import compileall; compileall.compile_dir('/tmp/mnn_py/MNN', force=True)"
find . -name "*.py" | xargs rm -rf
popd
cp -r MNN $PACKAGE_PATH/wrapper
popd

CMAKE_ARGS="-DPYMNN_USE_ALINNPYTHON=ON -DPYMNN_RUNTIME_CHECK_VM=ON -DPYMNN_EXPR_API=ON -DPYMNN_NUMPY_USABLE=ON -DPYMNN_TRAIN_API=OFF"

rm -rf mnnpybridge_build && mkdir mnnpybridge_build
pushd mnnpybridge_build

log() {
    echo "==================================="
    echo "Build mnnpybridge $1"
    echo "==================================="
}

# Debug Dynamic
if [ $build_all ] || [ $build_type = "debug" -a $lib_type = "dynamic" ]; then
    log "debug + dynamic"
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_BUILD_SHARED_LIBS=ON ../pymnn && make -j8
    cp libmnnpybridge.so $PACKAGE_PATH/lib/Debug
fi

# Debug Static
if [ $build_all ] || [ $build_type = "debug" -a $lib_type = "static" ]; then
    log "debug + static"
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_BUILD_SHARED_LIBS=OFF ../pymnn && make -j8
    cp libmnnpybridge.a $PACKAGE_PATH/lib/Debug
fi

# Release Dynamic
if [ $build_all ] || [ $build_type = "release" -a $lib_type = "dynamic" ]; then
    log "release + dynamic"
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=ON ../pymnn && make -j8
    cp libmnnpybridge.so $PACKAGE_PATH/lib/Release
fi

# Release Static
if [ $build_all ] || [ $build_type = "release" -a $lib_type = "static" ]; then
    log "release + static"
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF ../pymnn && make -j8
    cp libmnnpybridge.a $PACKAGE_PATH/lib/Release
fi

popd
