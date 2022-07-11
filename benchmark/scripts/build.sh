#!/bin/bash
set -e

usage() {
    echo "Usage: $0 [--compile] [--pc] [--android] [--ios] [--mnn-tag MNN_TAG] [--torch-tag TORCH_TAG] [--tf-tag TF_TAG]"
    echo -e "\t--compile force re-compile MNN/pytorch/tf"
    echo -e "\t--pc build for pc"
    echo -e "\t--android build for android"
    echo -e "\t--ios build for ios"
    echo -e "\t--mnn_tag mnn git tag"
    echo -e "\t--torch_tag torch git tag"
    echo -e "\t--tf_tag tf git tag"
    exit 1
}

re_compile=false
build_for_pc=false
build_for_android=false
build_for_ios=false
while getopts ":h-:" opt; do
  case "$opt" in
    -)
        case "$OPTARG" in
            compile) re_compile=true ;;
            mnn-tag) MNN_TAG=${!OPTIND} ; OPTIND=$(( $OPTIND + 1 )) ;;
            torch-tag) TORCH_TAG=${!OPTIND} ; OPTIND=$(( $OPTIND + 1 )) ;;
            tf-tag) TF_TAG=${!OPTIND} ; OPTIND=$(( $OPTIND + 1 )) ;;
            pc) build_for_pc=true ;;
            android) build_for_android=true ;;
            ios) build_for_ios=true ;;
            *) usage ;;
        esac ;;
    h|? ) usage ;;
  esac
done

pip install typing-extensions pyyaml numpy
if [[ "$OSTYPE" == "darwin"* ]] && ! command -v realpath &> /dev/null ; then
    brew install coreutils
fi

DIST_DIR=$(dirname $(realpath $0))/dist
rm -rf ./dist
###### MNN ######
if [ ! -z "$MNN_TAG" ]; then
    #MNN_REPO=https://github.com/alibaba/MNN.git
    MNN_REPO=git@gitlab.alibaba-inc.com:AliNN/AliNNPrivate.git
    MNN_HOME=$DIST_DIR/android/MNN
    if [ ! -d MNN ] || ([ "$(cd MNN && git describe --tags)" != "$MNN_TAG" ] && [[ "$(cd MNN && git rev-parse HEAD)" != "${MNN_TAG}"* ]]); then
        rm -rf MNN
        set +e
        git clone --depth 1 --branch $MNN_TAG $MNN_REPO MNN
        set -e
        if [ ! -d MNN ]; then
            git clone $MNN_REPO MNN
            pushd MNN
            git checkout $MNN_TAG
            popd
        fi
    fi
    pushd MNN
    # android
    push_mnn_tools() {
        mkdir -p $1
        find . -name "*.so" | while read solib; do
            cp $solib $1
        done
        cp benchmark.out $1
        return 0
    }
    if $build_for_pc; then
        if [ ! -d build ] || $re_compile; then
            rm -rf build && mkdir build && pushd build
            cmake -DMNN_SEP_BUILD=OFF -DMNN_BUILD_BENCHMARK=ON -DMNN_AVX512=ON -DMNN_CUDA=OFF ..
            make -j benchmark.out
            popd
        fi
        pushd build && push_mnn_tools "$MNN_HOME/pc" && popd
    fi
    if $build_for_android; then
        pushd project/android
        # aarch32
        if [ ! -d build_32 ] || $re_compile; then
            rm -rf build_32 && mkdir build_32 && pushd build_32
            cmake ../../../ \
              -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
              -DCMAKE_BUILD_TYPE=Release \
              -DANDROID_ABI="armeabi-v7a" \
              -DANDROID_STL=c++_static \
              -DCMAKE_BUILD_TYPE=Release \
              -DANDROID_NATIVE_API_LEVEL=android-14  \
              -DANDROID_TOOLCHAIN=clang \
              -DMNN_VULKAN:BOOL=ON \
              -DMNN_OPENCL:BOOL=ON \
              -DMNN_OPENMP:BOOL=ON \
              -DMNN_OPENGL:BOOL=ON \
              -DMNN_DEBUG:BOOL=OFF \
              -DMNN_BUILD_BENCHMARK:BOOL=ON \
              -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
              -DNATIVE_LIBRARY_OUTPUT=.
            make -j benchmark.out
            popd
        fi
        pushd build_32 && push_mnn_tools "$MNN_HOME/arm32" && popd
        # aarch64 with fp16
        if [ ! -d build_64 ] || $re_compile; then
            rm -rf build_64 && mkdir build_64 && pushd build_64
                cmake ../../../ \
                  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
                  -DCMAKE_BUILD_TYPE=Release \
                  -DANDROID_ABI="arm64-v8a" \
                  -DANDROID_STL=c++_static \
                  -DCMAKE_BUILD_TYPE=Release \
                  -DANDROID_NATIVE_API_LEVEL=android-14  \
                  -DANDROID_TOOLCHAIN=clang \
                  -DMNN_VULKAN:BOOL=ON \
                  -DMNN_OPENCL:BOOL=ON \
                  -DMNN_OPENMP:BOOL=ON \
                  -DMNN_OPENGL:BOOL=ON \
                  -DMNN_DEBUG:BOOL=OFF \
                  -DMNN_BUILD_BENCHMARK:BOOL=ON \
                  -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
                  -DNATIVE_LIBRARY_OUTPUT=.
            make -j benchmark.out
            popd
        fi
        pushd build_64 && push_mnn_tools "$MNN_HOME/arm64" && popd
        popd
    fi
    popd
fi

###### pytorch ######
if [ ! -z "$TORCH_TAG" ]; then
    if [ ! -d pytorch ] || [ "$(cd pytorch && git describe --tags)" != "$TORCH_TAG" ]; then
        rm -rf pytorch
        git clone --depth 1 --branch $TORCH_TAG https://github.com/pytorch/pytorch.git
        pushd pytorch
        git submodule sync --recursive
        git submodule update --init --recursive
        popd
    fi
    pushd pytorch
    if $build_for_android; then
        TORCH_HOME=$DIST_DIR/android/torch
        # android arm32
        if [ ! -d build_android_arm32 ] || $re_compile; then
            rm -rf build_android_arm32
            BUILD_PYTORCH_MOBILE=1 ANDROID_ABI=armeabi-v7a ./scripts/build_android.sh -DBUILD_BINARY=ON
            cp -r build_android build_android_arm32
            rm -rf build_android
        fi
        mkdir -p $TORCH_HOME/arm32
        cp build_android_arm32/bin/speed_benchmark_torch $TORCH_HOME/arm32
        # android arm64
        if [ ! -d build_android_arm64 ] || $re_compile; then
            rm -rf build_android_arm64
            BUILD_PYTORCH_MOBILE=1 ANDROID_ABI=arm64-v8a ./scripts/build_android.sh -DBUILD_BINARY=ON
            cp -r build_android build_android_arm64
            rm -rf build_android
        fi
        mkdir -p $TORCH_HOME/arm64
        cp build_android_arm64/bin/speed_benchmark_torch $TORCH_HOME/arm64
    fi
    # iOS app generated by ios/TestApp/benchmark/setup.rb be failed even a empty test case, say: Unknown custom class type quantized.Conv2dPackedParamsBase
    if false; then
    if $build_for_ios; then
        TORCH_HOME=$DIST_DIR/ios/torch
        brew install automake libtool
        if [ ! -d build_ios_arm64 ] || $re_compile; then
            rm -rf build_ios build_ios_arm64
            BUILD_PYTORCH_MOBILE=1 IOS_ARCH="arm64" ./scripts/build_ios.sh
            cp -r build_ios build_ios_arm64 && rm -rf build_ios
        fi
        # ios arm32
        #if [ ! -d build_ios_arm32 ] || $re_compile; then
        #    rm -rf build_ios build_ios_arm32
        #    BUILD_PYTORCH_MOBILE=1 IOS_ARCH="armv7k;arm64_32" ./scripts/build_ios.sh
        #    cp -r build_ios build_ios_arm32 && rm -rf build_ios
        #fi
    fi
    fi
    popd
fi

###### tensorflow ######
if [ ! -z "$TF_TAG" ]; then
    if [ ! -d tensorflow ] || [ "$(cd tensorflow && git describe --tags)" != "$TF_TAG" ]; then
        rm -rf tensorflow
        git clone --depth 1 --branch $TF_TAG https://github.com/tensorflow/tensorflow.git
    fi
    pushd tensorflow
    BAZEL_FLAGS="--discard_analysis_cache --notrack_incremental_state --nokeep_state_after_build" # avoid out of memory
    if [ -z $(which bazel) ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install bazelisk
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -Lo /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
            chmod +x /usr/local/bin/bazel
        fi
    fi
    TF_NEED_ROCM=false TF_NEED_CUDA=false TF_SET_ANDROID_WORKSPACE=$build_for_android TF_CONFIGURE_IOS=$build_for_ios ANDROID_SDK_HOME=$ANDROID_SDK ANDROID_NDK_HOME=$ANDROID_NDK ./configure
    if $build_for_android; then
        TFLITE_HOME=$DIST_DIR/android/tflite
        # android arm32
        if [ ! -d build_android_arm32 ] || $re_compile; then
            bazel build $BAZEL_FLAGS -c opt --config=android_arm --config=monolithic tensorflow/lite/tools/benchmark:benchmark_model_plus_flex
            rm -rf build_android_arm32 && mkdir build_android_arm32
            cp bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_plus_flex build_android_arm32
        fi
        mkdir -p $TFLITE_HOME/arm32
        cp build_android_arm32/benchmark_model_plus_flex $TFLITE_HOME/arm32
        # android arm64
        if [ ! -d build_android_arm64 ] || $re_compile; then
            bazel build $BAZEL_FLAGS -c opt --config=android_arm64 --config=monolithic tensorflow/lite/tools/benchmark:benchmark_model_plus_flex
            rm -rf build_android_arm64 && mkdir build_android_arm64
            cp bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_plus_flex build_android_arm64
        fi
        mkdir -p $TFLITE_HOME/arm64
        cp build_android_arm64/benchmark_model_plus_flex $TFLITE_HOME/arm64
    fi
    if $build_for_ios; then
        brew install automake libtool
        TFLITE_IOS_PRODUCT=tensorflow/lite/tools/benchmark/ios/TFLiteBenchmark/TFLiteBenchmark/Frameworks
        if [ ! -d $TFLITE_IOS_PRODUCT ] || $re_compile; then
            ./tensorflow/lite/tools/benchmark/ios/build_benchmark_framework.sh
        fi
    fi
    popd
fi
