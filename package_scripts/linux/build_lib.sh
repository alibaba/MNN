#!/bin/bash

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
    echo "Usage: $0 -o path [-b backends] [-s] [-c] [-t build_type -t lib_type [-c]]"
    echo -e "\t-o package files output directory"
    echo -e "\t-b extra backends to support (opencl, opengl, vulkan, onednn, avx512, coreml)"
    echo -e "\t-s re-generate schema"
    echo -e "\t-c clean build folder"
    echo -e "\t-t build type (debug/release), lib_type (dynamic/static), build all when unspecify"
    exit 1
}

build_all=true
while getopts "o:b:sct:h" opt; do
  case "$opt" in
    o ) path=$OPTARG ;;
    b ) IFS="," read -a backends <<< $OPTARG ;;
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
mkdir -p $path/Debug
mkdir -p $path/Release

PACKAGE_PATH=$(realpath $path)

CMAKE_ARGS="-DMNN_SEP_BUILD=OFF -DMNN_BUILD_TOOLS=OFF"
if [ "$backends" ]; then
    for backend in $backends; do
        case $backend in
            "opencl" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_OPENCL=ON" ;;
            "opengl" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_OPENGL=ON" ;;
            "vulkan" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_VULKAN=ON" ;;
            "onednn" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_ONEDNN=ON" ;;
            "avx512" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_AVX512=ON" ;;
            "coreml" ) CMAKE_ARGS="$CMAKE_ARGS -DMNN_COREML=ON" ;;
        esac
    done
fi

if [ $clean_build ]; then
    rm -rf build && mkdir build
fi
pushd build

log() {
    echo "==================================="
    echo "Build MNN (CPU $backends) $1"
    echo "==================================="
}

# Debug Dynamic
if [ $build_all ] || [ $build_type = "debug" -a $lib_type = "dynamic" ]; then
    log "debug + dynamic"
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_BUILD_SHARED_LIBS=ON .. && make -j24
    cp libMNN.so $PACKAGE_PATH/Debug
fi

# Debug Static
if [ $build_all ] || [ $build_type = "debug" -a $lib_type = "static" ]; then
    log "debug + static"
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_BUILD_SHARED_LIBS=OFF .. && make -j24
    cp libMNN.a $PACKAGE_PATH/Debug
fi

# Release Dynamic
if [ $build_all ] || [ $build_type = "release" -a $lib_type = "dynamic" ]; then
    log "release + dynamic"
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=ON .. && make -j24
    cp libMNN.so $PACKAGE_PATH/Release
fi

# Release Static
if [ $build_all ] || [ $build_type = "release" -a $lib_type = "static" ]; then
    log "release + static"
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    cmake $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF .. && make -j24
    cp libMNN.a $PACKAGE_PATH/Release
fi

popd
