set -e
ABI="arm64-v8a"
OPENMP="OFF"
VULKAN="ON"
OPENCL="ON"
OPENGL="OFF"
RUN_LOOP=10
FORWARD_TYPE=0
CLEAN=""
PUSH_MODEL=""

WORK_DIR=`pwd`
BUILD_DIR=build
BENCHMARK_MODEL_DIR=$WORK_DIR/../models
ANDROID_DIR=/data/local/tmp

function usage() {
    echo "-32\tBuild 32bit."
    echo "-c\tClean up build folders."
    echo "-p\tPush models to device"
}
function die() {
    echo $1
    exit 1
}

function clean_build() {
    echo $1 | grep "$BUILD_DIR\b" > /dev/null
    if [[ "$?" != "0" ]]; then
        die "Warnning: $1 seems not to be a BUILD folder."
    fi
    rm -rf $1
    mkdir $1
}

function build_android_bench() {
    if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
    fi
    mkdir -p build
    cd $BUILD_DIR
    cmake ../../../ \
          -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DANDROID_ABI="${ABI}" \
          -DANDROID_STL=c++_static \
          -DANDROID_NATIVE_API_LEVEL=android-21  \
          -DMNN_USE_LOGCAT:BOOL=OFF \
          -DMNN_VULKAN:BOOL=$VULKAN \
          -DMNN_OPENCL:BOOL=$OPENCL \
          -DMNN_OPENMP:BOOL=$OPENMP \
          -DMNN_OPENGL:BOOL=$OPENGL \
          -DMNN_ARM82:BOOL=ON \
          -DMNN_BUILD_BENCHMARK:BOOL=ON \
          -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
          -DNATIVE_LIBRARY_OUTPUT=.
    make -j8 benchmark.out timeProfile.out
}

function bench_android() {
    build_android_bench
    find . -name "*.so" | while read solib; do
        adb push $solib  $ANDROID_DIR
    done
    adb push benchmark.out $ANDROID_DIR
    adb push timeProfile.out $ANDROID_DIR
    adb shell chmod 0777 $ANDROID_DIR/benchmark.out

    if [ "" != "$PUSH_MODEL" ]; then
        adb shell "rm -rf $ANDROID_DIR/benchmark_models"
        adb push $BENCHMARK_MODEL_DIR $ANDROID_DIR/benchmark_models
    fi
    adb shell "cat /proc/cpuinfo > $ANDROID_DIR/benchmark.txt"
    adb shell "echo >> $ANDROID_DIR/benchmark.txt"
    adb shell "echo Build Flags: ABI=$ABI  OpenMP=$OPENMP Vulkan=$VULKAN OpenCL=$OPENCL >> $ANDROID_DIR/benchmark.txt"
    #benchmark  CPU
    adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/benchmark.out $ANDROID_DIR/benchmark_models $RUN_LOOP 5 $FORWARD_TYPE 2>$ANDROID_DIR/benchmark.err >> $ANDROID_DIR/benchmark.txt"
    #benchmark  Vulkan
    adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/benchmark.out $ANDROID_DIR/benchmark_models $RUN_LOOP 5 7 2>$ANDROID_DIR/benchmark.err >> $ANDROID_DIR/benchmark.txt"
    #benchmark OpenCL
    adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/benchmark.out $ANDROID_DIR/benchmark_models 100 20 3 2>$ANDROID_DIR/benchmark.err >> $ANDROID_DIR/benchmark.txt"
}

while [ "$1" != "" ]; do
    case $1 in
        -32)
            shift
            ABI="armeabi-v7a"
            ;;
        -c)
            shift
            CLEAN="-c"
            ;;
        -p)
            shift
            PUSH_MODEL="-p"
            ;;
        *)
            usage
            exit 1
    esac
done

bench_android
adb pull $ANDROID_DIR/benchmark.txt .
cat benchmark.txt
