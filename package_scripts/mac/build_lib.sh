#!/bin/bash

# MNN
#  |--- Debug
#  |      |--- Dynamic
#  |      |--- Static
#  |
#  |--- Release
#         |--- Dynamic
#         |--- Static

set -e

export MACOSX_DEPLOYMENT_TARGET=10.11

usage() {
    echo "Usage: $0 -o path [-b backends] [-f] [-s] [-c] [-t build_type -t lib_type [-c]]"
    echo -e "\t-o package files output directory"
    echo -e "\t-b extra backends to support (opencl, opengl, vulkan, onednn, avx512, coreml)"
    echo -e "\t-f MNN.framework, otherwise .dylib or .a"
    echo -e "\t-s re-generate schema"
    echo -e "\t-c clean build folder"
    echo -e "\t-t build type (debug/release), lib_type (dynamic/static), build all when unspecify"
    exit 1
}

build_all=true
while getopts "o:b:fsct:h" opt; do
  case "$opt" in
    o ) path=$OPTARG ;;
    b ) IFS="," read -a backends <<< $OPTARG ;;
    f ) fmwk=true ;;
    s ) clean_schema=true ;;
    c ) clean_build=true ;;
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

# clear and create package directory
if [ $clean_schema ]; then
    ./schema/generate.sh
fi
rm -rf $path && mkdir -p $path

PACKAGE_PATH=$(realpath $path)

CMAKE_ARGS="-DMNN_SEP_BUILD=OFF"
if [ $fmwk ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DMNN_AAPL_FMWK=ON"
fi
if [ "$backends" ]; then
    for backend in $backends; do
        case $backend in
            "opencl" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_OPENCL=ON" ;;
            "opengl" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_OPENGL=ON" ;;
            "vulkan" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_VULKAN=ON" ;;
            "onednn" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_ONEDNN=ON" ;;
            "avx512" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_AVX512=ON" ;;
            "coreml" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_COREML=ON" ;;
            "metal" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_METAL=ON" ;;
        esac
    done
fi

if [ $clean_build ]; then
    rm -rf build && mkdir build
fi
pushd build

deploy() {
    _path=$1
    if [ $fmwk ]; then
        cp -R MNN.framework $_path
        return
    fi
    _lib_type=$2
    if [ $_lib_type = "dynamic" ]; then
        cp libMNN.dylib $_path
    else
        cp libMNN.a $_path
    fi
}

log() {
    echo "==================================="
    echo "Build MNN (CPU $backends) $1"
    echo "==================================="
}

# Debug Dynamic
if [ $build_all ] || [ $build_type = "debug" -a $lib_type = "dynamic" ]; then
    log "debug + dynamic"
    pushd $PACKAGE_PATH && mkdir -p Debug/Dynamic && popd
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_BUILD_SHARED_LIBS=ON .. && make -j8
    deploy $PACKAGE_PATH/Debug/Dynamic "dynamic"
fi

# Debug Static
if [ $build_all ] || [ $build_type = "debug" -a $lib_type = "static" ]; then
    log "debug + static"
    pushd $PACKAGE_PATH && mkdir -p Debug/Static && popd
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_BUILD_SHARED_LIBS=OFF .. && make -j8
    deploy $PACKAGE_PATH/Debug/Static "static"
fi

# Release Dynamic
if [ $build_all ] || [ $build_type = "release" -a $lib_type = "dynamic" ]; then
    log "release + dynamic"
    pushd $PACKAGE_PATH && mkdir -p Release/Dynamic && popd
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=ON .. && make -j8
    deploy $PACKAGE_PATH/Release/Dynamic "dynamic"
fi

# Release Static
if [ $build_all ] || [ $build_type = "release" -a $lib_type = "static" ]; then
    log "release + static"
    pushd $PACKAGE_PATH && mkdir -p Release/Static && popd
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF .. && make -j8
    deploy $PACKAGE_PATH/Release/Static "static"
fi

popd
