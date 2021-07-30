# test script for MNN-Release
# steps: 0. build for andorid 1. build for linux; 2. unit-test; 3. module-test; 4. convert-test;

# 0. build for android
USER_NAME=`whoami`
USER_HOME="$(echo -n $(bash -c "cd ~${USER_NAME} && pwd"))"

android_build() {
    BASH_FILE="$USER_HOME/.zshrc"
    if [ -f "$BASH_FILE" ]; then
        source $BASH_FILE
    fi
    if [ ! $ANDROID_NDK ] || [ ! -d $ANDROID_NDK ]; then
        export ANDROID_NDK="$USER_HOME/android-ndk-r21"
    fi
    mkdir android_build
    pushd android_build
    cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_STL=c++_static \
    -DMNN_USE_LOGCAT=false \
    -DMNN_BUILD_BENCHMARK=ON \
    -DANDROID_NATIVE_API_LEVEL=android-21  \
    -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
    -DMNN_OPENGL=true \
    -DMNN_BUILD_TRAIN=true \
    -DMNN_VULKAN=true \
    -DMNN_SUPPORT_BF16=true \
    -DMNN_OPENCL=true -DMNN_ARM82=true \
    -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3
    make -j16
    android_build_wrong=$[$? > 0]
    printf "TEST_NAME_ANDROID: Android编译测试\nTEST_CASE_AMOUNT_ANDROID: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" $android_build_wrong $[1 - $android_build_wrong]
    if [ $android_build_wrong -ne 0 ]; then
        echo '### Android编译失败，测试终止！'
        exit 0
    fi
    popd

    mkdir android_build_32
    pushd android_build_32
    cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="armeabi-v7a" \
    -DANDROID_STL=c++_shared \
    -DMNN_USE_LOGCAT=false \
    -DMNN_BUILD_BENCHMARK=ON \
    -DANDROID_NATIVE_API_LEVEL=android-21  \
    -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
    -DMNN_OPENGL=true \
    -DMNN_BUILD_TRAIN=true \
    -DMNN_VULKAN=true \
    -DMNN_BUILD_MINI=true \
    -DMNN_SUPPORT_BF16=true \
    -DMNN_OPENCL=true\
    -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=.
    make -j16
    android_build_wrong=$[$? > 0]
    printf "TEST_NAME_ANDROID_32: Android 32-Mini 编译测试\nTEST_CASE_AMOUNT_ANDROID_32: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" $android_build_wrong $[1 - $android_build_wrong]
    if [ $android_build_wrong -ne 0 ]; then
        echo '### Android编译失败，测试终止！'
        exit 0
    fi
    popd
}

linux_build() {
    if [ $# -gt 0 ]; then
        COVERAGE=ON
    else
        COVERAGE=OFF
    fi

    mkdir build_non_sse
    pushd build_non_sse
    cmake .. -DMNN_USE_SSE=OFF && make -j16

    linux_build_wrong=$[$? > 0]
    popd

    mkdir build
    pushd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DMNN_BUILD_TEST=ON \
        -DMNN_CUDA=ON \
        -DMNN_BUILD_QUANTOOLS=ON \
        -DMNN_BUILD_DEMO=ON \
        -DMNN_BUILD_CONVERTER=ON \
        -DMNN_BUILD_TORCH=ON \
        -DMNN_ENABLE_COVERAGE=$COVERAGE
    make -j16

    linux_build_wrong+=$[$? > 0]
    printf "TEST_NAME_LINUX: Linux编译测试\nTEST_CASE_AMOUNT_LINUX: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" $linux_build_wrong $[2 - $linux_build_wrong]
    if [ $linux_build_wrong -ne 0 ]; then
        echo '### Linux编译失败，测试终止！'
        exit 0
    fi

    # Don't remove this! It turn off MNN_CUDA and MNN_TENSORRT in build, workaround some bug in PTQTest
    cmake .. -DMNN_CUDA=OFF -DMNN_TENSORRT=OFF && make -j16
}

unit_test() {
    ./run_test.out
    if [ $? -ne 0 ]; then
        echo '### 单元测试失败，测试终止！'
        exit 0
    fi

    ./run_test.out op 0 0 4
    if [ $? -ne 0 ]; then
        echo '### 多线程单元测试失败，测试终止！'
        exit 0
    fi
}

model_test() {
    ../tools/script/modelTest.py ~/AliNNModel 0 0.002
    if [ $? -ne 0 ]; then
        echo '### 模型测试失败，测试终止！'
        exit 0
    fi

    ../tools/script/modelTest.py ~/AliNNModel 0 0.002 0 1
    if [ $? -ne 0 ]; then
        echo '### 静态模型测试失败，测试终止！'
        exit 0
    fi
}

onnx_convert_test() {
    ../tools/script/convertOnnxTest.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### ONNXConvert测试失败，测试终止！'
        exit 0
    fi
}

tf_convert_test() {
    ../tools/script/convertTfTest.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### TFConvert测试失败，测试终止！'
        exit 0
    fi
}

tflite_convert_test() {
    ../tools/script/convertTfliteTest.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### TFLITEConvert测试失败，测试终止！'
        exit 0
    fi
}

torch_convert_test() {
    ../tools/script/convertTorchTest.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### TORCHConvert测试失败，测试终止！'
        exit 0
    fi
}

ptq_test() {
    ../tools/script/testPTQ.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### PTQ测试失败，测试终止！'
        exit 0
    fi
}

coverage_init() {
    popd
    lcov -c -i -d ./ -o init.info
    pushd build
}

coverage_report() {
    popd
    cover_report_dir="../../../../CoverageReport"
    lcov -c -d ./ -o cover.info
    lcov -a init.info -a cover.info -o total.info
    lcov --remove total.info \
    '*/usr/include/*' '*/usr/lib/*' '*/usr/lib64/*' '*/usr/local/include/*' '*/usr/local/lib/*' '*/usr/local/lib64/*' \
    '*/3rd_party/*' '*/build/*' '*/schema/*' '*/test/*' '/tmp/*' \
    -o final.info
    commitId=$(git log | head -n1 | awk '{print $2}')
    genhtml -o cover_report --legend --title "MNN Coverage Report [commit SHA1:${commitId}]" --prefix=`pwd` final.info
    coverage_wrong=$[$? > 0]
    printf "TEST_NAME_COVERAGE: 代码覆盖率(点击\"通过\"查看报告)\nTEST_CASE_AMOUNT_COVERAGE: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" $coverage_wrong $[1 - $coverage_wrong]
    if [ $coverage_wrong -ne 0 ]; then
        echo '### 代码覆盖率生成失败，测试终止！'
        exit 0
    else
        hostIp=$(cat .aoneci.yml | grep host | awk '{print $2}')
        testId=$(pwd | awk -F "/" '{print $(NF-1)}')
        mv cover_report $cover_report_dir/$testId
        echo "TEST_REPORT_COVERAGE: http://$hostIp/$testId"
    fi
    # clean test dir
    cd ../.. && rm -rf $testId
}

android_test() {
    pushd project/android
    mkdir build_32
    pushd build_32
    ../build_32.sh
    python3 ../../../tools/script/AndroidTest.py ~/AliNNModel 32
    if [ $? -ne 0 ]; then
        echo '### AndroidTest32测试失败，测试终止！'
        exit 0
    fi
    popd
    mkdir build_64
    pushd build_64
    ../build_64.sh
    python3 ../../../tools/script/AndroidTest.py ~/AliNNModel 64
    if [ $? -ne 0 ]; then
        echo '### AndroidTest64测试失败，测试终止！'
        exit 0
    fi
    popd
    popd
}

case "$1" in
    build)
        linux_build
        ;;
    test)
        android_build
        linux_build
        unit_test
        model_test
        onnx_convert_test
        tf_convert_test
        tflite_convert_test
        torch_convert_test
        ptq_test
        ;;
    coverage)
        android_build
        linux_build 1
        coverage_init
        unit_test
        model_test
        onnx_convert_test
        tf_convert_test
        tflite_convert_test
        torch_convert_test
        ptq_test
        coverage_report
        ;;
    test_local)
        pushd build
        unit_test
        model_test
        onnx_convert_test
        tf_convert_test
        tflite_convert_test
        torch_convert_test
        ptq_test
        ;;
    test_android)
        android_test
        ;;
    *)
        echo $"Usage: $0 {build|test|coverage|test_local|test_android}"
        exit 2
esac
exit $?
