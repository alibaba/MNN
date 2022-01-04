# test script for MNN-Release
#
# 0. arg = local: [ test for your local build ]
#       1. unit-test;
#       2. model-test;
#       3. onnx convert test
#       4. tf convert test
#       5. tflite convert test
#       6. torch convert test
#       7. ptq test
#       8. pymnn test
#
# 1. arg = linux: [ all test on linux with coverage ]
#       0. static check (if source change)
#       1. build for linux;
#       2. unit-test;
#       3. model-test;
#       4. onnx convert test
#       5. tf convert test
#       6. tflite convert test
#       7. torch convert test
#       8. ptq test
#       9. pymnn test (if pymnn change)
#      10. opencv test (if opencv change)
#      11. convert-report;
#
# 2. arg = android: [ simple test on android ]
#       1. build Android with static_stl
#       2. build Android arm64
#       3. unit-test for Android arm64
#       4. build Android arm32
#       5. unit-test for Android arm32

# 0. build for android
USER_NAME=`whoami`
USER_HOME="$(echo -n $(bash -c "cd ~${USER_NAME} && pwd"))"

# detect change
SOURCE_CHANGE=$(git show --name-only | grep -E "^source/(internal|backend|core|common|cv|geometry|math|plugin|shape|utils)/.*\.(cpp|cc|c|hpp)$" | grep -v "aliyun-log-c-sdk")
PYMNN_CHANGE=$(git show --name-only | grep -E "^pymnn/.*\.(cpp|cc|c|h|hpp|py)$")
OPENCV_CHANGE=$(git show --name-only | grep -E "^tools/cv/.*\.(cpp|cc|c|h|hpp)$")

failed() {
    printf "TEST_NAME_EXCEPTION: Exception\nTEST_CASE_AMOUNT_EXCEPTION: {\"blocked\":0,\"failed\":1,\"passed\":0,\"skipped\":0}\n"
    exit 1
}

static_check() {
    if [ -z "$SOURCE_CHANGE" ]; then
        return
    fi
    cppcheck --error-exitcode=1 --language=c++ --std=c++14 --addon=tools/script/mnn_rules.py $SOURCE_CHANGE 1> /dev/null
    static_check_wrong=$[$? > 0]
    printf "TEST_NAME_STATIC_CHECK: cppcheck静态分析\nTEST_CASE_AMOUNT_STATIC_CHECK: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
           $static_check_wrong $[1 - $static_check_wrong]
    if [ $static_check_wrong -ne 0 ]; then
        echo '### cppcheck静态分析失败，测试终止！'
        failed
    fi
}

android_static_build() {
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
    printf "TEST_NAME_ANDROID_STATIC: AndroidStatic编译测试\nTEST_CASE_AMOUNT_ANDROID_STATIC: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
           $android_build_wrong $[1 - $android_build_wrong]
    if [ $android_build_wrong -ne 0 ]; then
        echo '### AndroidStatic编译失败，测试终止！'
        failed
    fi
    popd

:<<!
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
        failed
    fi
    popd
!
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
        -DMNN_BUILD_OPENCV=ON \
        -DMNN_IMGCODECS=ON \
        -DMNN_ENABLE_COVERAGE=$COVERAGE
    make -j16

    linux_build_wrong+=$[$? > 0]
    printf "TEST_NAME_LINUX: Linux编译测试\nTEST_CASE_AMOUNT_LINUX: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" $linux_build_wrong $[2 - $linux_build_wrong]
    if [ $linux_build_wrong -ne 0 ]; then
        echo '### Linux编译失败，测试终止！'
        failed
    fi

    # Don't remove this! It turn off MNN_CUDA and MNN_TENSORRT in build, workaround some bug in PTQTest
    cmake .. -DMNN_CUDA=OFF -DMNN_TENSORRT=OFF && make -j16
}

unit_test() {
    ./run_test.out
    if [ $? -ne 0 ]; then
        echo '### 单元测试失败，测试终止！'
        failed
    fi

    ./run_test.out op 0 0 4
    if [ $? -ne 0 ]; then
        echo '### 多线程单元测试失败，测试终止！'
        failed
    fi
}

model_test() {
    ../tools/script/modelTest.py ~/AliNNModel 0 0.002
    if [ $? -ne 0 ]; then
        echo '### 模型测试失败，测试终止！'
        failed
    fi

    ../tools/script/modelTest.py ~/AliNNModel 0 0.002 0 1
    if [ $? -ne 0 ]; then
        echo '### 静态模型测试失败，测试终止！'
        failed
    fi
}

onnx_convert_test() {
    ../tools/script/convertOnnxTest.py ~/AliNNModel
    if [ $? -eq 0 ] && [ -f ~/AliNNModel/TestOnnx/ops/run.py ]; then
        ~/AliNNModel/TestOnnx/ops/run.py --mnndir $(pwd) --aone-mode
    fi
    if [ $? -ne 0 ]; then
        echo '### ONNXConvert测试失败，测试终止！'
        failed
    fi
}

tf_convert_test() {
    ../tools/script/convertTfTest.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### TFConvert测试失败，测试终止！'
        failed
    fi
}

tflite_convert_test() {
    ../tools/script/convertTfliteTest.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### TFLITEConvert测试失败，测试终止！'
        failed
    fi
}

torch_convert_test() {
    ../tools/script/convertTorchTest.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### TORCHConvert测试失败，测试终止！'
        failed
    fi
}

ptq_test() {
    ../tools/script/testPTQ.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### PTQ测试失败，测试终止！'
        failed
    fi
}

pymnn_test() {
    if [ -z "$PYMNN_CHANGE" ]; then
        return
    fi
    popd
    pushd pymnn
    # 1. build pymnn
    pushd pip_package
    python3 build_deps.py
    # uninstall original MNN
    pip uninstall --yes MNN MNN-Internal
    python3 setup.py install --version 1.0
    pymnn_build_wrong=$[$? > 0]
    printf "TEST_NAME_PYMNN_BUILD: PYMNN编译测试\nTEST_CASE_AMOUNT_PYMNN_BUILD: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
            $pymnn_build_wrong $[1 - $pymnn_build_wrong]
    if [ $pymnn_build_wrong -ne 0 ]; then
        echo '### PYMNN编译失败，测试终止！'
        failed
    fi
    popd
    # 2. unit test
    pushd test
    python3 unit_test.py
    if [ $? -ne 0 ]; then
        echo '### PYMNN单元测试失败，测试终止！'
        failed
    fi
    # 3. model test
    python3 model_test.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### PYMNN模型测试失败，测试终止！'
        failed
    fi
    # 4. uninstall pymnn
    pip uninstall --yes MNN-Internal
    popd
    popd
    pushd build
}

opencv_test() {
    if [ -z "$OPENCV_CHANGE" ]; then
        return
    fi
    # 1. build opencv-test
    cmake -DMNN_OPENCV_TEST=ON ..
    make -j8
    opencv_build_wrong=$[$? > 0]
    printf "TEST_NAME_OPENCV_BUILD: OPENCV编译测试\nTEST_CASE_AMOUNT_OPENCV_BUILD: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
            $opencv_build_wrong $[1 - $opencv_build_wrong]
    if [ $opencv_build_wrong -ne 0 ]; then
        echo '### OPENCV编译失败，测试终止！'
        failed
    fi
    # 2. run opencv unit test
    ./opencv_test
    if [ $? -gt 0 ]; then
        echo '### OPENCV单元测试失败，测试终止！'
        failed
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
    '*/usr/include/*' '*/usr/lib/*' '*/usr/lib64/*' '*/usr/local/*'  \
    '*/3rd_party/*' '*/build/*' '*/schema/*' '*/test/*' '/tmp/*' \
    '*/demo/*' '*/tools/cpp/*' '*/tools/train/*' '*/source/backend/cuda/*' \
    -o final.info
    commitId=$(git log | head -n1 | awk '{print $2}')
    genhtml -o cover_report --legend --title "MNN Coverage Report [commit SHA1:${commitId}]" --prefix=`pwd` final.info
    coverage_wrong=$[$? > 0]
    printf "TEST_NAME_COVERAGE: 代码覆盖率(点击\"通过\"查看报告)\nTEST_CASE_AMOUNT_COVERAGE: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" $coverage_wrong $[1 - $coverage_wrong]
    if [ $coverage_wrong -ne 0 ]; then
        echo '### 代码覆盖率生成失败，测试终止！'
        failed
    else
        hostIp=$(cat .aoneci.yml | grep host -m 1 | awk '{print $2}')
        testId=$(pwd | awk -F "/" '{print $(NF-1)}')
        mv cover_report $cover_report_dir/$testId
        echo "TEST_REPORT_COVERAGE: http://$hostIp/$testId"
    fi
    # clean test dir
    cd ../.. && rm -rf $testId
}

android_test() {
    pushd project/android
    # 1. build Android32
    mkdir build_32
    pushd build_32
    ../build_32.sh
    android32_build_wrong=$[$? > 0]
    mnn32_size=$(ls -lh libMNN.so | awk '{print $5}')
    expr32_size=$(ls -lh libMNN_Express.so | awk '{print $5}')
    printf "TEST_NAME_ANDROID_32: Android32编译测试(libMNN.so - %s, libMNN_Express.so - %s)\nTEST_CASE_AMOUNT_ANDROID_32: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
           $mnn32_size $expr32_size $android32_build_wrong $[1 - $android32_build_wrong]
    if [ $android32_build_wrong -ne 0 ]; then
        echo '### Android32编译失败，测试终止！'
        failed
    fi

    # 2. test Androird32
    python3 ../../../tools/script/AndroidTest.py ~/AliNNModel 32 unit
    if [ $? -ne 0 ]; then
        echo '### AndroidTest32测试失败，测试终止！'
        failed
    fi
    popd

    # 3. build Android64
    mkdir build_64
    pushd build_64
    ../build_64.sh
    android64_build_wrong=$[$? > 0]
    mnn64_size=$(ls -lh libMNN.so | awk '{print $5}')
    expr64_size=$(ls -lh libMNN_Express.so | awk '{print $5}')
    printf "TEST_NAME_ANDROID_64: Android64编译测试(libMNN.so - %s, libMNN_Express.so - %s)\nTEST_CASE_AMOUNT_ANDROID_64: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
            $mnn64_size $expr64_size $android64_build_wrong $[1 - $android64_build_wrong]
    if [ $android64_build_wrong -ne 0 ]; then
        echo '### Android64编译失败，测试终止！'
        failed
    fi

    # 4. test Android64
    python3 ../../../tools/script/AndroidTest.py ~/AliNNModel 64 unit
    if [ $? -ne 0 ]; then
        echo '### AndroidTest64测试失败，测试终止！'
        failed
    fi
    popd
    popd
}

case "$1" in
    local)
        pushd build
        unit_test
        model_test
        onnx_convert_test
        tf_convert_test
        tflite_convert_test
        torch_convert_test
        ptq_test
        pymnn_test
        ;;
    linux)
        static_check
        linux_build 1
        coverage_init
        unit_test
        model_test
        onnx_convert_test
        tf_convert_test
        tflite_convert_test
        torch_convert_test
        ptq_test
        pymnn_test
        opencv_test
        coverage_report
        ;;
    android)
        android_static_build
        android_test
        ;;
    *)
        echo $"Usage: $0 {local|linux|android}"
        exit 2
esac
exit $?
